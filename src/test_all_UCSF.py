import argparse
from pyexpat import features
import random 
import os
import platform
import cpuinfo
import numpy as np
import torch
import pandas as pd
from AttnFuseNet import AggregatorConvClassifier,predict_binary,predict_multiclass
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score,roc_auc_score,recall_score,accuracy_score,balanced_accuracy_score,confusion_matrix
from sklearn.preprocessing import label_binarize
import yaml


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

def parse_args():
    """
    Parse the command line arguments for the task type.

    Parameters
    ----------
    None

    Returns
    -------
    task : str
        The type of the classification task. Either "binary" or "multiclass".
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, default="binary", choices=["binary", "multiclass"])
    return parser.parse_args().task

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

def load_data(t):
    """
    Load the data from the UCSF dataset.

    Parameters
    ----------
    t : str
        Type of task to perform (either "binary" or "multiclass").

    Returns
    -------
    features : numpy.ndarray
        The radiomic features associated with the dataset.
    labels : numpy.ndarray
        The labels associated with the dataset.
    """
    raw_data = pd.read_csv("../data/UCSF_Features.csv")
    split = pd.read_csv("../data/patient_splits.csv")
    features , labels = [],[]
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
        features.append(feats)
        labels.append(label)
        

    features = np.stack(features, axis=0)
    labels = np.array(labels)
    return features,labels

def create_loader(features,labels):
    """
    Create a DataLoader from the given features and labels.

    Parameters
    ----------
    features : numpy.ndarray
        The radiomic features associated with the dataset.
    labels : numpy.ndarray
        The labels associated with the dataset.

    Returns
    -------
    loader : DataLoader
        A DataLoader that can be used to iterate over the dataset in batches.
    """
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

def load_net(device,t):
    """
    Load the pre-trained model for the given task.

    Parameters
    ----------
    device : torch.device
        The device to use for computation.
    t : str
        The type of task to perform (either "binary" or "multiclass").

    Returns
    -------
    net : nn.Module
        The pre-trained model.

    """
    if t == "binary":
        nclasses = 2
        net = AggregatorConvClassifier(input_features=d_model, conv_out=conv_out, hidden_dim=hidden_DIM, n_heads=n_head, n_classes=nclasses).to(device)

    else:
        nclasses = 3
        net  = AggregatorConvClassifier(input_features=d_model, conv_out=conv_out, hidden_dim=hidden_DIM, n_heads=n_head, n_classes=nclasses).to(device)

    net.to(device)
    net.load_state_dict(torch.load(f"../models/AttnFuseNet_{t}.pt"))
    return net


        

def evaluate_model(t, targets, predictions, proba):

    """
    Evaluate the model on the given data.

    Parameters
    ----------
    t : str
        The type of task to perform (either "binary" or "multiclass").
    targets : numpy array
        The true labels of the data.
    predictions : numpy array
        The predicted labels of the data.
    proba : numpy array
        The predicted probabilities of the data.

    Returns
    -------
    dict: A dictionary containing the evaluation metrics.
    """
    reports = {}

    acc = accuracy_score(targets, predictions)
    f1_micro = f1_score(targets, predictions, average='micro')
    f1_macro = f1_score(targets, predictions, average='macro')
    ba = balanced_accuracy_score(targets, predictions)

    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {ba:.4f}")
    print(f"F1 micro: {f1_micro:.4f}, F1 macro: {f1_macro:.4f}")

    cm = confusion_matrix(targets, predictions)

    if t == "binary":
        # AUC
        auc = roc_auc_score(targets, proba, average="micro")
        sens = recall_score(targets, predictions, pos_label=1)
        spec = recall_score(targets, predictions, pos_label=0)

        print(f"AUC: {auc:.4f}")
        print(f"Sensitivity (Recall): {sens:.4f}")
        print(f"Specificity: {spec:.4f}")

        reports.update({
            "acc": float(acc),
            "balanced": float(ba),
            "f1_micro": float(f1_micro),
            "f1_macro": float(f1_macro),
            "auc": float(auc),
            "sensitivity": float(sens),
            "specificity": float(spec),
        })

    else:  # multiclass
        n_classes = proba.shape[1]
        targets_bin = label_binarize(targets, classes=np.arange(n_classes))

        # ===== AUC micro e macro =====
        auc_micro = roc_auc_score(targets_bin, proba, multi_class="ovr", average="micro")
        auc_macro = roc_auc_score(targets_bin, proba, multi_class="ovr", average="macro")

        # ===== Sensitivity micro e macro =====
        micro_sens = recall_score(targets, predictions, average="micro")
        macro_sens = recall_score(targets, predictions, average="macro")

        # ===== Specificity per classe =====
        spec_per_class = []
        tn_total = 0
        fp_total = 0
        for i in range(n_classes):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            spec_per_class.append(tn / (tn + fp))
            tn_total += tn
            fp_total += fp

        # ===== Micro-specificity =====
        micro_spec = tn_total / (tn_total + fp_total)
        macro_spec = np.mean(spec_per_class)

        print(f"AUC micro: {auc_micro:.4f}, AUC macro: {auc_macro:.4f}")
        print(f"Sensitivity micro: {micro_sens:.4f}, macro: {macro_sens:.4f}")
        print(f"Specificity micro: {micro_spec:.4f}, macro: {macro_spec:.4f}")
        acc = accuracy_score(targets, predictions)
        print(f"Accuracy: {acc:.4f}")

        reports.update({
            "acc": float(acc),
            "balanced": float(ba),
            "f1_micro": float(f1_micro),
            "f1_macro": float(f1_macro),
            "auc_micro": float(auc_micro),
            "auc_macro": float(auc_macro),
            "sensitivity_micro": float(micro_sens),
            "sensitivity_macro": float(macro_sens),
            "specificity_micro": float(micro_spec),
            "specificity_macro": float(macro_spec),
        })

    # Confusion matrix
    print(f"Confusion matrix:\n{cm}")
    reports["confusion_matrix"] = cm.tolist()

    # Salvataggio YAML
    path = f"../results/results_{t}.yaml"
    with open(path, 'w') as file:
        yaml.dump(reports, file)

    return

def main():
    """
    Main entry point of the program.

    It performs the following steps:
    1. Identify the device to use for computation.
    2. Parse the command line arguments.
    3. Print information about the device and the classification task.
    4. Load the data.
    5. Create a model and move it to the device.
    6. Perform predictions on the data.
    7. Evaluate the model on the data.
    8. Print the results.
    """
    device, dev_name = identify_device()
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"Using {device} - {dev_name}")
    print(f"AttnFuseNet Glioma Grade Prediction")
    print(f"Dataset: UCSF")
    t = parse_args()
    print(f"Classification task: {t}")
    features, labels = load_data(t)
    loader = create_loader(features,labels)
    model = load_net(device,t)
    if t == "binary":
        preds, probs = predict_binary(device, model, loader)
    else:
        preds, probs = predict_multiclass(device, model, loader)
    evaluate_model(t, labels, preds, probs)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return 

if __name__ == "__main__":
    main()