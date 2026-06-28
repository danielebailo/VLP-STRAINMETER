"""Supervised 1D CNN classifier for VLP waveform discrimination."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

import yaml

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class WaveformDataset(Dataset):
    def __init__(self, waveforms: np.ndarray, labels: np.ndarray) -> None:
        self.waveforms = np.asarray(waveforms, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x = torch.from_numpy(self.waveforms[idx]).float().unsqueeze(0)
        return x, self.labels[idx]


def _resolve_path(path_value: str | Path) -> Path:
    """Resolve path relative to project root (where src/ lives)."""
    p = Path(path_value)
    if p.is_absolute():
        return p
    # Try relative to this module's parent (project root)
    module_dir = Path(__file__).resolve().parent.parent
    resolved = module_dir / p
    if resolved.exists():
        return resolved
    # Fallback: relative to cwd
    if p.exists():
        return p
    raise FileNotFoundError(f"{path_value} not found (tried {resolved} and {p})")


def load_and_split(
    type1_path: str | Path,
    type2_path: str | Path,
    waveform_length: int = 128,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load both classes, combine, split into train/val/test."""
    rng = np.random.default_rng(seed)

    type1_path = _resolve_path(type1_path)
    type2_path = _resolve_path(type2_path)
    print(f"  Type1: {type1_path}")
    print(f"  Type2: {type2_path}")

    # Load
    type1 = np.load(type1_path) if str(type1_path).endswith(".npy") else np.loadtxt(type1_path, dtype=np.float32)
    type2 = np.load(type2_path) if str(type2_path).endswith(".npy") else np.loadtxt(type2_path, dtype=np.float32)

    # Coerce to (n, length)
    if type1.ndim == 1:
        type1 = type1.reshape(1, -1)
    if type2.ndim == 1:
        type2 = type2.reshape(1, -1)

    # Normalize by global mean/std
    all_data = np.vstack([type1, type2])
    global_mean = all_data.mean()
    global_std = all_data.std() + 1e-8
    type1 = (type1 - global_mean) / global_std
    type2 = (type2 - global_mean) / global_std

    # Labels: 0 = type1, 1 = type2
    X = np.vstack([type1, type2])
    y = np.concatenate([np.zeros(len(type1)), np.ones(len(type2))]).astype(np.int64)

    # Shuffle
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # Split
    n_train = int(len(X) * train_ratio)
    n_val = int(len(X) * val_ratio)

    train_X, train_y = X[:n_train], y[:n_train]
    val_X, val_y = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    test_X, test_y = X[n_train + n_val:], y[n_train + n_val:]

    def make_loader(X, y, shuffle=False):
        ds = WaveformDataset(X, y)
        return DataLoader(ds, batch_size=64, shuffle=shuffle, num_workers=0)

    return make_loader(train_X, train_y, shuffle=True), \
           make_loader(val_X, val_y, shuffle=False), \
           make_loader(test_X, test_y, shuffle=False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class VLPClassifier(nn.Module):
    """1D CNN classifier for binary waveform classification."""

    def __init__(
        self,
        waveform_length: int = 128,
        n_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 128 -> 64
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),

            # Block 2: 32 -> 16
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),

            # Global average pool
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    test_acc: float = 0.0
    test_auc: float = 0.0
    best_val_acc: float = 0.0
    best_epoch: int = 0


def train_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    device: torch.device | None = None,
    save_path: str | Path | None = None,
    plot: bool = True,
) -> tuple[VLPClassifier, TrainingHistory]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VLPClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    history = TrainingHistory()
    best_val_acc = 0.0
    best_state = None

    # Create real-time training plot
    show_realtime = True
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    line_train_loss, = axes[0].plot([], [], label='Train', linewidth=1.5)
    line_val_loss, = axes[0].plot([], [], label='Validation', linewidth=1.5)
    line_train_acc, = axes[1].plot([], [], label='Train', linewidth=1.5)
    line_val_acc, = axes[1].plot([], [], label='Validation', linewidth=1.5)
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.canvas.draw()
    plt.pause(0.01)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += len(y)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        scheduler.step(1.0 - val_acc)

        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)

        # Real-time plot update
        if show_realtime:
            # Update loss plot
            epochs_x = list(range(1, epoch + 1))
            line_train_loss.set_data(epochs_x, history.train_loss)
            line_val_loss.set_data(epochs_x, history.val_loss)
            axes[0].relim()
            axes[0].autoscale_view()
            axes[0].set_title('Loss (epoch %d/%d)' % (epoch, epochs))

            # Update accuracy plot
            line_train_acc.set_data(epochs_x, history.train_acc)
            line_val_acc.set_data(epochs_x, history.val_acc)
            axes[1].relim()
            axes[1].autoscale_view()
            axes[1].set_title('Accuracy (epoch %d/%d)' % (epoch, epochs))

            fig.canvas.draw()
            plt.pause(0.01)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            history.best_val_acc = best_val_acc
            history.best_epoch = epoch

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                  f"best={best_val_acc:.4f} (epoch {history.best_epoch})")

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    if val_loader is not None:
        test_loader = val_loader  # Use val as test proxy
    else:
        test_loader = train_loader

    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())

    history.test_acc = accuracy_score(all_labels, all_preds)
    try:
        history.test_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        history.test_auc = 0.0

    # Save
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "model_state_dict": best_state,
            "history": {
                "train_loss": history.train_loss,
                "train_acc": history.train_acc,
                "val_loss": history.val_loss,
                "val_acc": history.val_acc,
                "test_acc": history.test_acc,
                "test_auc": history.test_auc,
                "best_epoch": history.best_epoch,
            },
        }, save_path)
        print(f"Model saved to {save_path}")

    if plot:
        _plot_training(history)

    return model, history


def _plot_training(history: TrainingHistory) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.train_loss, label="train loss")
    axes[0].plot(history.val_loss, label="val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.train_acc, label="train acc")
    axes[1].plot(history.val_acc, label="val acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: VLPClassifier,
    test_loader: DataLoader,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    all_waveforms = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
            all_waveforms.extend(X.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["Type1", "Type2"])
    cm = confusion_matrix(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test AUC: {auc:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")

    return {
        "accuracy": acc,
        "auc": auc,
        "report": report,
        "confusion_matrix": cm,
        "predictions": np.array(all_preds),
        "probabilities": np.array(all_probs),
        "labels": np.array(all_labels),
        "waveforms": np.array(all_waveforms),
    }


def plot_confusion_matrix(
    cm: np.ndarray, ax: plt.Axes | None = None, title: str = "Confusion Matrix"
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Type1", "Type2"],
           yticklabels=["Type1", "Type2"],
           title=title)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    return ax


def plot_roc_curve(
    all_labels: np.ndarray, all_probs: np.ndarray, ax: plt.Axes | None = None
) -> plt.Axes:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc = roc_auc_score(all_labels, all_probs)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, label="ROC (AUC = %.4f)" % auc)
    ax.plot([0, 1], [0, 1], linestyle="--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Predict on new waveforms
# ---------------------------------------------------------------------------

def predict(
    model: VLPClassifier,
    waveforms: np.ndarray,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Predict class probabilities for new waveforms."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if waveforms.ndim == 1:
        waveforms = waveforms.reshape(1, -1)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(len(waveforms)):
            x = torch.from_numpy(waveforms[i]).float().unsqueeze(0).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.item())  # Extract scalar from tensor

    all_probs = np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)

    return {
        "prob_type2": all_probs,
        "prob_type1": 1.0 - all_probs,
        "predictions": preds,
    }
