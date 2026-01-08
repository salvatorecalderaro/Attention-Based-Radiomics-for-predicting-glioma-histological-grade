import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

class MultiHeadFeatureAggregator(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        """
        Initializes the MultiHeadFeatureAggregator module.

        Parameters:
            d_model (int): Number of input features.
            n_heads (int): Number of attention heads. Defaults to 4.
            dropout (float): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        d_model_proj = 128
        self.d_model_proj = d_model_proj
        self.fc_in = nn.Linear(d_model, d_model_proj)
        self.mha = nn.MultiheadAttention(embed_dim=d_model_proj, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model_proj)
        self.fc_out = nn.Linear(d_model_proj, d_model_proj)
        self.dropout = nn.Dropout(dropout)
        self.att_fc = nn.Linear(d_model_proj, 1)

    def forward(self, X):
        """
        Forward pass of the MultiHeadFeatureAggregator module.

        Parameters:
            X (torch.Tensor): Input tensor of shape (B, L, d_model).

        Returns:
            z (torch.Tensor): Patient-level embedding tensor of shape (B, 128).
            weights (torch.Tensor): Attention weights tensor of shape (B, L, 1).
        """
        # X: (B, L, d_model)
        X_proj = self.fc_in(X)                          # (B, L, 128)
        attn_output, _ = self.mha(X_proj, X_proj, X_proj)
        X_norm = self.norm(X_proj + self.dropout(attn_output))
        X_out = F.relu(self.fc_out(X_norm))            # (B, L, 128)
        
        # Attention pooling scalare per scansione
        scores = self.att_fc(X_out)                    # (B, L, 1)
        weights = torch.softmax(scores, dim=1)         # (B, L, 1)
        z = torch.sum(X_out * weights, dim=1)          # (B, 128) → patient-level embedding
        
        return z, weights


class AggregatorConvClassifier(nn.Module):
    def __init__(self, input_features, conv_out=128, hidden_dim=64, n_classes=2, n_heads=4, dropout=0.1):
        """
        Initializes an AggregatorConvClassifier instance.

        Parameters:
            input_features (int): Number of input features.
            conv_out (int): Number of output features for the convolutional block. Defaults to 128.
            hidden_dim (int): Number of hidden units in the MLP. Defaults to 64.
            n_classes (int): Number of output classes. Defaults to 2.
            n_heads (int): Number of attention heads. Defaults to 4.
            dropout (float): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.n_classes = n_classes

        self.aggregator = MultiHeadFeatureAggregator(
            d_model=input_features,
            n_heads=n_heads,
            dropout=dropout
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(self.aggregator.d_model_proj, conv_out, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_out),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(conv_out, conv_out, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_out),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(conv_out, conv_out, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_out),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.mlp = nn.Sequential(
            nn.Linear(conv_out, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1 if n_classes == 2 else n_classes)
        )

        self.activation = nn.Identity()

    def forward(self, X):
        
        """
        Forward pass of the AggregatorConvClassifier module.

        Parameters:
            X (torch.Tensor): Input tensor of shape (B, L, d_model).

        Returns:
            out (torch.Tensor): Output tensor of shape (B, 1) or (B, n_classes).
        """
        z, _ = self.aggregator(X)  # z: (B, 128), weights: (B, L, 1)

        z = z.unsqueeze(2)                # (B, 128, 1)
        z = self.conv_block(z)            # (B, conv_out, 1)
        z = z.squeeze(2)                  # (B, conv_out)

        out = self.mlp(z)                 # (B, 1) o (B, n_classes)
        return out

def train_binary(device, model, trainloader, epochs, lr, pos_weight=None):
    """
    Train a binary classification model.

    Parameters:
        device (torch.device): Device to perform computation on.
        model (nn.Module): Model to be trained.
        trainloader (DataLoader): Data loader for the training set.
        epochs (int): Number of epochs to train for.
        lr (float): Learning rate.
        pos_weight (float, optional): Weight for positive class. Defaults to None.

    Returns:
        nn.Module: Trained model.
    """
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight, dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            logits = model(inputs).squeeze()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader.dataset):.4f}")

    print("Training complete.")
    path = "../models/AttnFuseNet_binary.pt"
    torch.save(model.state_dict(), path)
    return model

def train_multiclass(device, model, trainloader, epochs, lr):
    """
    Train a multiclass classification classification model.

    Parameters:
        device (torch.device): Device to perform computation on.
        model (nn.Module): Model to be trained.
        trainloader (DataLoader): Data loader for the training set.
        epochs (int): Number of epochs to train for.
        lr (float): Learning rate.

    Returns:
        nn.Module: Trained model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader.dataset):.4f}")

    print("Training complete.")
    path = "../models/AttnFuseNet_multiclass.pt"
    #torch.save(model.state_dict(), path)
    return model

def predict_binary(device, model, testloader):
    """
    Make predictions on a test set using a binary classification model.

    Parameters:
        device (torch.device): Device to perform computation on.
        model (nn.Module): Model to be used for prediction.
        testloader (DataLoader): Data loader for the test set.

    Returns:
        tuple: A tuple containing two numpy arrays. The first array contains the predicted classes, the second array contains the predicted probabilities.
    """
    model.to(device)
    model.eval()
    all_probs, all_preds = [], []

    with torch.no_grad():
        for inputs, _ in testloader:
            inputs = inputs.to(device)
            logits = model(inputs).squeeze()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)

def predict_multiclass(device, model, testloader):
    """
    Make predictions on a test set using a multiclass classification model.

    Parameters:
        device (torch.device): Device to perform computation on.
        model (nn.Module): Model to be used for prediction.
        testloader (DataLoader): Data loader for the test set.

    Returns:
        tuple: A tuple containing two numpy arrays. The first array contains the predicted classes, the second array contains the predicted probabilities.
    """
    model.to(device)
    model.eval()
    all_probs, all_preds = [], []

    with torch.no_grad():
        for inputs, _ in testloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)
