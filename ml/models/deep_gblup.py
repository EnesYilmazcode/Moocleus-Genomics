"""
deepGBLUP — Deep Learning + GBLUP Hybrid for Genomic Prediction.

Combines a locally-connected Conv1d pathway that learns haplotype patterns
with pre-computed GBLUP breeding values through a learnable alpha parameter.

Reference:
    Lee, J. et al. (2023). "deepGBLUP: Joint deep learning and GBLUP framework
    for genomic prediction." Genetics Selection Evolution 55:25.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class DeepGBLUPModel(nn.Module):
    """
    deepGBLUP architecture:
        Input SNPs -> Conv1d(1, 4, k=50, s=50) -> ReLU
        -> FC(local_out, 512) -> BN -> ReLU -> Dropout(0.3)
        -> FC(512, 128) -> BN -> ReLU -> Dropout(0.2)
        -> FC(128, 1) -> g_dl

        alpha = sigmoid(learned_param)
        output = alpha * g_dl + (1 - alpha) * g_gblup
    """

    def __init__(self, n_snps: int, group_size: int = 50, hidden: int = 512):
        super().__init__()
        self.n_snps = n_snps
        self.group_size = group_size

        self.local_conv = nn.Conv1d(1, 4, kernel_size=group_size, stride=group_size)
        local_out = (n_snps // group_size) * 4

        self.fc = nn.Sequential(
            nn.Linear(local_out, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, snp_input: torch.Tensor, gblup_value: torch.Tensor) -> torch.Tensor:
        x = snp_input.unsqueeze(1)  # (batch, 1, n_snps)
        x = F.relu(self.local_conv(x))
        x = x.flatten(1)
        g_dl = self.fc(x)
        alpha = torch.sigmoid(self.alpha)
        return alpha * g_dl + (1 - alpha) * gblup_value.unsqueeze(1)


def train_deep_gblup(
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    gblup_values: np.ndarray,
    n_epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 20,
    n_folds: int = 5,
    batch_size: int = 32,
    seed: int = 42,
) -> dict:
    """
    Train deepGBLUP with k-fold cross-validation.

    Args:
        genotypes: (n_animals, n_snps) matrix, 0/1/2 encoded
        phenotypes: (n_animals,) trait values
        gblup_values: (n_animals,) pre-computed GBLUP GEBVs
        n_epochs: max training epochs per fold
        lr: learning rate
        patience: early stopping patience
        n_folds: number of CV folds
        batch_size: training batch size
        seed: random seed

    Returns:
        dict with predictions (out-of-fold), cv_correlations, final_alpha, model_state_dict
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_animals, n_snps = genotypes.shape

    # Pad SNPs to nearest multiple of group_size (50)
    group_size = 50
    pad_to = ((n_snps + group_size - 1) // group_size) * group_size
    if pad_to > n_snps:
        genotypes = np.pad(genotypes, ((0, 0), (0, pad_to - n_snps)), mode="constant")
        n_snps = pad_to

    # Convert to tensors
    X = torch.tensor(genotypes, dtype=torch.float32)
    y = torch.tensor(phenotypes, dtype=torch.float32)
    g = torch.tensor(gblup_values, dtype=torch.float32)

    # K-fold CV
    rng = np.random.default_rng(seed)
    indices = np.arange(n_animals)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    oof_predictions = np.zeros(n_animals)
    cv_losses = []
    best_state = None
    best_alpha = 0.5

    for fold_idx in range(n_folds):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        g_train, g_val = g[train_idx], g[val_idx]

        train_ds = TensorDataset(X_train, g_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model = DeepGBLUPModel(n_snps, group_size=group_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_fold_state = None

        for epoch in range(n_epochs):
            # Training
            model.train()
            for batch_X, batch_g, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_g = batch_g.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                pred = model(batch_X, batch_g).squeeze()
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

            scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val.to(device), g_val.to(device)).squeeze()
                val_loss = criterion(val_pred, y_val.to(device)).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_fold_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        # Predict validation fold with best model
        model.load_state_dict(best_fold_state)
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device), g_val.to(device)).squeeze().cpu().numpy()
        oof_predictions[val_idx] = val_pred
        cv_losses.append(best_val_loss)

        # Keep the last fold's best state as the final model
        best_state = best_fold_state
        best_alpha = torch.sigmoid(best_fold_state["alpha"]).item()

    # Train final model on all data
    full_ds = TensorDataset(X, g, y)
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True)
    final_model = DeepGBLUPModel(n_snps, group_size=group_size).to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    for epoch in range(n_epochs // 2):  # Fewer epochs for final model
        final_model.train()
        for batch_X, batch_g, batch_y in full_loader:
            batch_X = batch_X.to(device)
            batch_g = batch_g.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = final_model(batch_X, batch_g).squeeze()
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    final_state = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}
    final_alpha = torch.sigmoid(final_state["alpha"]).item()

    return {
        "oof_predictions": oof_predictions,
        "cv_losses": cv_losses,
        "final_alpha": final_alpha,
        "model_state_dict": final_state,
        "n_snps": n_snps,
    }


def predict_deep_gblup(
    model_result: dict,
    genotypes: np.ndarray,
    gblup_values: np.ndarray,
) -> np.ndarray:
    """
    Predict GEBVs for new animals using a trained deepGBLUP model.

    Args:
        model_result: dict from train_deep_gblup()
        genotypes: (n_new, n_snps) genotype matrix
        gblup_values: (n_new,) GBLUP predictions for new animals

    Returns:
        (n_new,) predicted GEBVs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_snps = model_result["n_snps"]

    # Pad if needed
    if genotypes.shape[1] < n_snps:
        genotypes = np.pad(
            genotypes, ((0, 0), (0, n_snps - genotypes.shape[1])), mode="constant"
        )

    model = DeepGBLUPModel(n_snps).to(device)
    model.load_state_dict(model_result["model_state_dict"])
    model.eval()

    X = torch.tensor(genotypes, dtype=torch.float32).to(device)
    g = torch.tensor(gblup_values, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(X, g).squeeze().cpu().numpy()

    return predictions
