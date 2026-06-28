#!/usr/bin/env python3
"""Generate all report figures and create a Word document."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import src.classifier as classifier_module

# ============================================================
# CONFIGURAZIONE
# ============================================================
OUTPUT_DIR = PROJECT_ROOT / "report_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# 1. CARICAMENTO DATI E MODELLO
# ============================================================
print("\n=== Caricamento dati ===")
TYPE1_PATH = PROJECT_ROOT / "data/processed/Waveforms_type1_extended_128samples.txt"
TYPE2_PATH = PROJECT_ROOT / "data/processed/Waveforms_type2_extended_128samples.txt"

train_loader, val_loader, test_loader = classifier_module.load_and_split(
    TYPE1_PATH, TYPE2_PATH, waveform_length=128, seed=42
)

print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

# Load trained model
print("\n=== Caricamento modello ===")
model_path = PROJECT_ROOT / "outputs/cnn_classifier.pt"
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = classifier_module.VLPClassifier().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Modello caricato da {model_path}")

# Training history from checkpoint
history = checkpoint["history"]
best_val_acc = max(history["val_acc"])
print(f"Best epoch: {history['best_epoch']} | Best val acc: {best_val_acc:.4f}")

# ============================================================
# FIGURA 1: Curve di training
# ============================================================
print("\n=== Generazione Figura 1: Curve di training ===")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

epochs = list(range(1, len(history["train_loss"]) + 1))

# Loss
axes[0].plot(epochs, history["train_loss"], label="Training", linewidth=2, color="#2196F3")
axes[0].plot(epochs, history["val_loss"], label="Validation", linewidth=2, color="#FF9800")
axes[0].axvline(x=history["best_epoch"], color="red", linestyle="--", alpha=0.5, label=f"Best epoch ({history['best_epoch']})")
axes[0].set_title("Loss", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("Loss", fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(epochs, history["train_acc"], label="Training", linewidth=2, color="#2196F3")
axes[1].plot(epochs, history["val_acc"], label="Validation", linewidth=2, color="#FF9800")
axes[1].axvline(x=history["best_epoch"], color="red", linestyle="--", alpha=0.5, label=f"Best epoch ({history['best_epoch']})")
axes[1].set_title("Accuracy", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("Accuracy", fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0.9, 1.0])

plt.suptitle("Training Curves — CNN Classifier (50 epochs, Adam, seed=42)", fontsize=16, y=1.02)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "01_training_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Salvato: {OUTPUT_DIR / '01_training_curves.png'}")

# ============================================================
# FIGURA 2: Confusion Matrix (ri-generata con stile migliore)
# ============================================================
print("\n=== Generazione Figura 2: Confusion Matrix ===")

# Get predictions on test set
test_preds = classifier_module.evaluate_model(model, test_loader, device=device)
cm = test_preds["confusion_matrix"]

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

ax.set(xticks=[0, 1], yticks=[0, 1],
       xticklabels=["Type 1", "Type 2"],
       yticklabels=["Type 1", "Type 2"],
       title="Confusion Matrix — Test Set",
       ylabel="True Label",
       xlabel="Predicted Label")

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16, fontweight="bold")

# Add accuracy annotation
acc = test_preds["accuracy"]
ax.text(0.5, -0.15, f"Accuracy: {acc:.4f} | AUC: {test_preds['auc']:.4f}",
        transform=ax.transAxes, ha="center", fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "02_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Salvato: {OUTPUT_DIR / '02_confusion_matrix.png'}")

# ============================================================
# FIGURA 3: ROC Curve
# ============================================================
print("\n=== Generazione Figura 3: ROC Curve ===")

from sklearn.metrics import roc_curve, auc

all_labels = test_preds["labels"]
all_probs = test_preds["probabilities"]

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color="#2196F3", lw=2.5, label=f"ROC (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", alpha=0.5, label="Random")
ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — CNN Classifier", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "03_roc_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Salvato: {OUTPUT_DIR / '03_roc_curve.png'}")

# ============================================================
# FIGURA 4: Timeline di predizione sui dati reali
# ============================================================
print("\n=== Generazione Figura 4: Timeline predizione ===")

real_path = PROJECT_ROOT / "data/measured/VLPs-2019_low-pass.dat"
real_data = np.loadtxt(real_path, dtype=np.float32)

# Center-crop to 128
target_len = 128
current_len = real_data.shape[1] if real_data.ndim == 2 else len(real_data)
if current_len > target_len:
    excess = current_len - target_len
    left = excess // 2
    real_data = real_data[:, left : current_len - left]
elif current_len < target_len:
    deficit = target_len - current_len
    left = deficit // 2
    real_data = np.pad(real_data, ((0, 0), (left, deficit - left)), mode="constant")

# Normalize with training stats
all_train = train_loader.dataset.waveforms
global_mean = float(all_train.mean())
global_std = float(all_train.std()) + 1e-8
real_data = (real_data - global_mean) / global_std

print(f"  Dati reali: {len(real_data)} waveform")

# Predict
all_preds = classifier_module.predict(model, real_data, device=device)
n_type1 = int((all_preds["predictions"] == 0).sum())
n_type2 = int((all_preds["predictions"] == 1).sum())
n_uncertain = int(((all_preds["prob_type1"] > 0.3) & (all_preds["prob_type2"] > 0.3)).sum())

print(f"  Type1: {n_type1} | Type2: {n_type2} | Uncertain: {n_uncertain}")

# Create timeline plot
fig, ax = plt.subplots(figsize=(16, 6))

x = np.arange(len(real_data))
y = []
colors = []
sizes = []

for i in range(len(real_data)):
    p1 = float(all_preds["prob_type1"][i])
    p2 = float(all_preds["prob_type2"][i])
    conf = abs(p1 - p2)
    if all_preds["predictions"][i] == 0:
        y.append(p1)
        colors.append("#2196F3")  # Blue for Type1
    else:
        y.append(p2)
        colors.append("#FF9800")  # Orange for Type2
    sizes.append(5 + conf * 15)

ax.scatter(x, y, c=colors, s=sizes, alpha=0.6, edgecolors="none")
ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3, linewidth=1.5)

ax.set_xlabel("Waveform Index", fontsize=12)
ax.set_ylabel("Confidence (predicted class)", fontsize=12)
ax.set_title("Prediction Timeline — Real Data (VLPs-2019_low-pass.dat)", fontsize=14, fontweight="bold")
ax.legend(
    ["Type 1", "Type 2", "Threshold (0.5)"],
    loc="upper right",
    fontsize=10,
    facecolor="white",
    edgecolor="gray"
)
ax.grid(True, alpha=0.2)
ax.set_ylim([0.6, 1.0])

# Add summary text
summary_text = (
    f"Total: {len(real_data)} waveforms\n"
    f"Type 1: {n_type1} ({100.0*n_type1/len(real_data):.1f}%)\n"
    f"Type 2: {n_type2} ({100.0*n_type2/len(real_data):.1f}%)\n"
    f"Uncertain: {n_uncertain} ({100.0*n_uncertain/len(real_data):.1f}%)"
)
ax.text(0.02, 0.02, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray"))

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "04_prediction_timeline.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Salvato: {OUTPUT_DIR / '04_prediction_timeline.png'}")

# ============================================================
# Riepilogo
# ============================================================
print("\n" + "="*60)
print("✅ Tutte le figure generate in:", OUTPUT_DIR)
print("="*60)
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
print("\nOra posso creare il documento Word con queste figure.")
