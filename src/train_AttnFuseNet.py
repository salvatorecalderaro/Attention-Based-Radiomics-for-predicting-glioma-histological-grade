import argparse
import random 
import os
import platform
import cpuinfo
import numpy as np
import torch
import pandas as pd
from AttnFuseNet import AggregatorConvClassifier,train_binary,predict_binary,train_multiclass,predict_multiclass
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset,WeightedRandomSampler
from sklearn.metrics import f1_score,roc_auc_score,recall_score,accuracy_score,balanced_accuracy_score,confusion_matrix
from sklearn.preprocessing import label_binarize
import yaml
from sklearn.metrics import  roc_curve, auc
import seaborn as sns 

seed = 2025
d_model = 107
n_head = 32
conv_out = 64
hidden_DIM = 32
batch_size = 16
epochs = 50
lr = 1e-4
dpi = 1000

plt.rcParams["text.usetex"] = True


def set_seed(seed):
    """
    Set seeds for reproducibility.

    Parameters
    ----------
    seed : int
        The seed to use for reproducibility.

    Notes
    -----
    This function sets seeds for Python's random module, NumPy, PyTorch
    (including CUDA), and PyTorch's cuDNN backend. It also sets
    PyTorch's cuDNN backend to use deterministic and benchmark
    modes.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)


def identify_device():
    """
    Identify the device to use for PyTorch computations.

    This function checks the operating system and selects either
    the Metal Performance Shader (MPS) device or the CPU
    device if running on macOS, or the CUDA device or the
    CPU device if running on another operating system. It
    also sets the seed for reproducibility if using a CUDA
    device.

    Returns
    -------
    device : torch.device
        The device to use for PyTorch computations.
    dev_name : str
        The name of the device.
    """
    so = platform.system()
    if (so == "Darwin"):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        dev_name = cpuinfo.get_cpu_info()["brand_raw"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d = str(device)
        if d == 'cuda':
            dev_name = torch.cuda.get_device_name()
            set_seed(seed)
        else:
            dev_name = cpuinfo.get_cpu_info()["brand_raw"]
    return device, dev_name


def parse_args():
    """
    Parse command line arguments.

    This function parses the command line arguments and returns
    the type of classification task to perform.

    Parameters
    ----------
    None

    Returns
    -------
    t : str
        The type of classification task to perform. Either
        "binary" or "multiclass".
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c" , "--classification", type=str, required=True,choices=["binary", "multiclass"], help="Type of classification task.")
    args = parser.parse_args()
    t = args.classification
    return t



def find_features(data, pid):
    """
    Find the radiomic features associated with a given patient ID.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing the radiomic features.
    pid : int
        The patient ID to find the radiomic features for.

    Returns
    -------
    radiomics_feats : numpy.ndarray
        The radiomic features associated with the given patient ID.
    """
    df = data[data['PazienteID'] == pid]
    radiomics_feats = df.loc[:, ~df.columns.str.startswith("diagnostics_")].drop(columns=["label","label_value","PazienteID"])
    radiomics_feats = np.array(radiomics_feats)
    radiomics_feats = radiomics_feats[:, 1:]
    if radiomics_feats.shape[0] !=8:
        missing_info = np.zeros((8 - radiomics_feats.shape[0], radiomics_feats.shape[1]))
        radiomics_feats = np.vstack((radiomics_feats, missing_info))
    radiomics_feats = radiomics_feats.astype(np.float32)
    return radiomics_feats

def create_train_test_split(t):
    """
    Create a train and test split based on the given task type.

    Parameters
    ----------
    t : str
        The type of the task to perform. Either "binary" or "multiclass".

    Returns
    -------
    train_data : tuple
        A tuple containing the training data and labels.
    test_data : tuple
        A tuple containing the test data and labels.
    """
    raw_data = pd.read_csv("../data/UCSF_Features.csv")
    split = pd.read_csv("../data/patient_splits.csv")
    
    x_train, x_test = [], []
    y_train, y_test = [], []
    
    for index, row in split.iterrows():
        pid = row['patient_id']
        group = row['set']
        raw_label = row["label"]
        
        if t == "binary":
            if raw_label == 1 or raw_label == 2:
                label = 0
            elif raw_label == 3 or raw_label == 4:
                label = 1
        else:
            label = raw_label-2
        
        feats = find_features(raw_data, pid)
        
        if group == "train":
            x_train.append(feats)
            y_train.append(label)
        else:
            x_test.append(feats)
            y_test.append(label)

    x_train = np.stack(x_train, axis=0)
    x_test = np.stack(x_test, axis=0)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print(f"Number of patients in training set: {len(y_train)}")
    print(f"Number of patients in test set: {len(y_test)}")
    train_data = (x_train, y_train)
    test_data = (x_test, y_test)
    return train_data, test_data   


def create_loaders(train_data, test_data):
    """
    Create a train and test loader based on the given data.

    Parameters
    ----------
    train_data : tuple
        A tuple containing the training data and labels.
    test_data : tuple
        A tuple containing the test data and labels.

    Returns
    -------
    train_loader : DataLoader
        A DataLoader for the training set.
    test_loader : DataLoader
        A DataLoader for the test set.
    """
    x_train, y_train = train_data
    x_test, y_test = test_data
    
    # Convert to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Compute class weights
    classes, counts = np.unique(y_train.numpy(), return_counts=True)
    class_weights = 1.0 / counts
    sample_weights = class_weights[y_train.numpy()]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
    
    # Weighted sampler for training
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Create datasets
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def plot_data_dist(y_train, y_test, t):

    """
    Plot a bar chart showing the distribution of the training and test sets.

    Parameters
    ----------
    y_train : np.ndarray
        The labels of the training set.
    y_test : np.ndarray
        The labels of the test set.
    t : str
        The type of the classification task. Either "binary" or "multiclass".
    """
    cmap = plt.cm.Set1

    if t == "binary":
        labels = {0: "LGG", 1: "HGG"}
        class_colors = {
            "LGG": cmap(0),
            "HGG": cmap(1)
        }
    else:
        labels = {0: "Grade 2", 1: "Grade 3", 2: "Grade 4"}
        class_colors = {
            "Grade 2": cmap(0),
            "Grade 3": cmap(1),
            "Grade 4": cmap(2)
        }
    text_labels_train = [labels[label] for label in y_train]
    text_labels_test = [labels[label] for label in y_test]

    train_dist = Counter(text_labels_train)
    test_dist = Counter(text_labels_test)

    plt.figure(figsize=(12, 5))
    plt.suptitle(f"Data Distribution - {t} classification task")

    # ---- TRAIN ----
    plt.subplot(1, 2, 1)
    plt.bar(
        train_dist.keys(),
        train_dist.values(),
        color=[class_colors[k] for k in train_dist.keys()],
        alpha=0.8
    )
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Training")

    # ---- TEST ----
    plt.subplot(1, 2, 2)
    plt.bar(
        test_dist.keys(),
        test_dist.values(),
        color=[class_colors[k] for k in test_dist.keys()],
        alpha=0.8
    )
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Test")

    plt.tight_layout()
    path = f"../plots/data_dist_{t}.png"
    plt.savefig(path, dpi=dpi)
    plt.close()
    plt.clf()
    
    print(f"Training set distribution: {train_dist}")
    print(f"Test set distribution: {test_dist}")



def evaluate_model(t, targets, predictions, proba):
    """
    Evaluate model performance.

    Parameters:
    t (str): Type of classification task ("binary" or "multiclass").
    targets (numpy array): True labels.
    predictions (numpy array): Predicted labels.
    proba (numpy array): Predicted probabilities.

    Returns:
    dict: A dictionary containing the evaluation metrics.
    """
    f1s = f1_score(targets, predictions, average='micro')
    print(f"F1 (micro): {f1s:.4f}")
    
    acc = accuracy_score(targets, predictions)
    print(f"Accuracy: {acc:.4f}")

    reports = {}

    if t == "binary":
        # AUC
        auc = roc_auc_score(targets, proba, average="macro")
        
        # Sensitivity / Recall
        recall = recall_score(targets, predictions, pos_label=1)
        
        # Specificity
        spec = recall_score(targets, predictions, pos_label=0)
        
        # Balanced Accuracy
        balanced = balanced_accuracy_score(targets, predictions)
        print(f"Balanced accuracy: {balanced:.4f}")

        print(f"AUC: {auc:.4f}")
        print(f"Sensitivity (Recall): {recall:.4f}")
        print(f"Specificity: {spec:.4f}")

        reports["f1"] = float(f1s)
        reports["auc"] = float(auc)
        reports["recall"] = float(recall)
        reports["balanced"] = float(balanced)
        reports["specificity"] = float(spec)
        
        cm = confusion_matrix(targets, predictions, normalize='true')
        print(f"Confusion matrix:\n{cm}")

    else:  # multiclass
        n_classes = proba.shape[1]
        class_names = ["Grade 2", "Grade 3", "Grade 4"]

        cm = confusion_matrix(targets, predictions, normalize='true')

        f1_per_class = f1_score(targets, predictions, average=None)

        recall_per_class = recall_score(targets, predictions, average=None)

        spec_per_class = []
        for i in range(n_classes):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            spec = tn / (tn + fp)
            spec_per_class.append(spec)
            

        targets_bin = label_binarize(targets, classes=np.arange(n_classes))
        auc_per_class = roc_auc_score(
            targets_bin,
            proba,
            multi_class="ovr",
            average=None
        )

        print("\nMetriche per classe:")
        for i, c in enumerate(class_names):
            print(f"\n{c}")
            print(f"  F1-score:     {f1_per_class[i]:.4f}")
            print(f"  Sensitivity:  {recall_per_class[i]:.4f}")
            print(f"  Specificity:  {spec_per_class[i]:.4f}")
            print(f"  AUC (OvR):    {auc_per_class[i]:.4f}")

        f1_micro = f1_score(targets, predictions, average="micro")
        f1_macro = f1_score(targets, predictions, average="macro")
        auc_micro = roc_auc_score(targets_bin, proba, multi_class="ovr", average="micro")
        auc_macro = roc_auc_score(targets_bin, proba, multi_class="ovr", average="macro")

        print("\nMetriche globali:")
        print(f"F1 micro: {f1_micro:.4f}")
        print(f"F1 macro: {f1_macro:.4f}")
        print(f"AUC micro: {auc_micro:.4f}")
        print(f"AUC macro: {auc_macro:.4f}")

        reports["f1_per_class"] = f1_per_class.tolist()
        reports["sensitivity_per_class"] = recall_per_class.tolist()
        reports["specificity_per_class"] = [float(x) for x in spec_per_class]
        reports["auc_per_class"] = auc_per_class.tolist()

        reports["f1_micro"] = float(f1_micro)
        reports["f1_macro"] = float(f1_macro)
        reports["auc_micro"] = float(auc_micro)
        reports["auc_macro"] = float(auc_macro)
        reports["acc"] = float(acc)
        reports["confusion_matrix"] = cm.tolist()

    plot_confusion_matrix(t, cm)
    path = f"../results/results_{t}.yaml"
    with open(path, 'w') as file:
        yaml.dump(reports, file)
    return

def plot_roc_curve(t, targets, proba):

    """
    Plot the ROC curve of a binary or multiclass classification task.

    Parameters:
    t (str): type of classification task, either "binary" or "multiclass".
    targets (numpy array): true labels of the data.
    proba (numpy array): predicted probabilities of the data.
    predictions (numpy array): predicted labels of the data.

    Returns:
    None
    """
    plt.figure()
    if t == "binary":
        fpr, tpr, _ = roc_curve(targets, proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')

    else:
        n_classes = proba.shape[1]
        class_names = ["Grade 2", "Grade 3", "Grade 4"]

        for i in range(n_classes):
            binary_targets = (targets == i).astype(int)
            fpr, tpr, _ = roc_curve(binary_targets, proba[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(
                fpr, tpr,
                lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.4f})'
            )

    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    path = f"../plots/roc_curve_{t}.png"
    plt.savefig(path, dpi=dpi)
    plt.close()


def plot_confusion_matrix(t, cm):
    """
    Plot a confusion matrix for the given classification task.

    Parameters
    ----------
    t : str
        The type of the classification task. Either "binary" or "multiclass".
    cm : numpy array
        The confusion matrix.

    Returns
    -------
    None

    """
    if t == "binary":
        classes = ["LGG", "HGG"]
    else:
        classes = ["Grade II", "Grade III", "Grade IV"]
    
    plt.figure()
    plt.title(f"Confusion Matrix - {t} classification task")
    sns.heatmap(cm, annot=True, fmt=".3f", cmap="Blues", xticklabels=classes, yticklabels=classes,cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    path = f"../plots/confusion_matrix_{t}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()

def main():
    """
    Main entry point of the program.

    It performs the following steps:
    1. Identify the device to use for computation.
    2. Parse the command line arguments.
    3. Print information about the device and the classification task.
    4. Split the data into training and testing sets.
    5. Plot the distribution of the data.
    6. Create a model and move it to the device.
    7. Train the model on the training data.
    8. Evaluate the model on the testing data.
    9. Plot the ROC curve of the model.
    """
    device, dev_name = identify_device()
    t = parse_args()
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"Using {device} - {dev_name}")
    print(f"AttnFuseNet Glioma Grade Prediction")
    print(f"Dataset: UCSF")
    print(f"Classification task: {t}")
    train_data, test_data = create_train_test_split(t)
    plot_data_dist(train_data[1], test_data[1], t)
    nclasses = 2 if t == "binary" else 3
    model = AggregatorConvClassifier(input_features=d_model, conv_out=conv_out, hidden_dim=hidden_DIM, n_heads=n_head, n_classes=nclasses).to(device)
    print(model)
    train_loader, test_loader = create_loaders(train_data, test_data)
    
    if t == "binary":
        model = train_binary(device, model, train_loader, epochs, lr=1e-3)
        preds, probs = predict_binary(device, model, test_loader)
    else:
        model = train_multiclass(device, model, train_loader, epochs, lr=1e-3)
        preds, probs = predict_multiclass(device, model, test_loader)
    
    evaluate_model(t, test_data[1], preds, probs)
    plot_roc_curve(t, test_data[1], probs)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return

if __name__ == "__main__":
    main()