# STRAINMETER Project Guide — v2.0

This repository implements a **supervised 1D CNN classifier** for VLP waveform discrimination.

## Goal

The scientific objective is to classify strainmeter waveforms into two classes:

1. **Type1** (class 0) — very long period waveforms of type 1
2. **Type2** (class 1) — very long period waveforms of type 2

The classifier learns directly from labeled training data and outputs class probabilities for each input waveform.

## Repository Map

### `config/`

- `config/config.yaml` — full project configuration (model, training, data paths)

### `data/`

Training data:
- `data/processed/Waveforms_type1_extended_128samples.txt` — type1 training set (~7.2MB, ~100k waveforms)
- `data/processed/Waveforms_type2_extended_128samples.txt` — type2 training set (~5.3MB, ~75k waveforms)

Real data:
- `data/measured/VLPs-2019_low-pass.dat` — real measured waveforms (~115MB, ~150k waveforms)

Format: one waveform per row, 128 samples each (training), 151 samples each (measured → center-cropped to 128).

### `src/`

Core implementation:
- `src/classifier.py` — **main module**: `VLPClassifier`, training, evaluation, prediction
- `src/config.py` — YAML parsing and validation
- `src/data.py` — data loading (legacy autoencoder, kept for compatibility)
- `src/plotting.py` — plotting utilities (legacy autoencoder)
- `src/models.py` — autoencoder model (legacy, v1.0)
- `src/train.py` — autoencoder training (legacy, v1.0)
- `src/discriminator.py` — autoencoder discriminator (legacy, v1.0)

### `notebooks/`

- `notebooks/classifier.ipynb` — **primary notebook**: complete classifier workflow
- `notebooks/strainmeter_workflow.ipynb` — legacy autoencoder workflow (v1.0)

### `outputs/`

Generated artifacts:
- `outputs/cnn_classifier.pt` — trained classifier model
- `notebooks/outputs/confusion_matrix.png` — confusion matrix plot
- `notebooks/outputs/roc_curve.png` — ROC curve plot

### `tests/`

Reserved for unit tests (currently empty).

## Technical Architecture

### Model: VLPClassifier

A deep 1D convolutional neural network for binary classification:

```
Input: (batch, 1, 128)
  │
  ├── Conv1d(1→32, k=7, s=2, p=3) → BatchNorm1d → ReLU → Dropout1d(0.3)
  ├── Conv1d(32→64, k=5, s=2, p=2) → BatchNorm1d → ReLU → Dropout1d(0.3)
  ├── Conv1d(64→128, k=5, s=2, p=2) → BatchNorm1d → ReLU → Dropout1d(0.3)
  ├── Conv1d(128→256, k=3, s=2, p=1) → BatchNorm1d → ReLU → Dropout1d(0.3)
  ├── AdaptiveAvgPool1d(1)
  │
  ├── Flatten
  ├── Linear(256→64) → ReLU → Dropout(0.3)
  └── Linear(64→2) → Softmax → [P(Type1), P(Type2)]
```

### Data Pipeline

1. **Load** type1 and type2 waveforms from `.txt` files
2. **Normalize** globally: `(x - mean) / std` computed on all training data
3. **Label**: type1=0, type2=1
4. **Shuffle** with fixed seed (42)
5. **Split**: 80% train, 10% val, 10% test
6. **Batch**: DataLoader with batch_size=64

### Training Loop

- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (mode=max, factor=0.5, patience=5)
- **Epochs**: 50 (configurable)
- **Checkpoint**: best model saved by validation accuracy
- **Live plots**: real-time loss/accuracy curves during training

### Evaluation

- **Accuracy**: overall correct predictions
- **AUC-ROC**: area under ROC curve
- **Confusion Matrix**: TP, TN, FP, FN per class
- **Classification Report**: precision, recall, F1 per class

### Prediction

- **Single waveforms**: `predict(model, waveforms)` returns probabilities
- **Batch prediction**: works on entire measured file
- **Confidence**: `|P(Type1) - P(Type2)|` measures prediction certainty

## How to Run

### Notebook (recommended)

```bash
cd /Users/danielebailo/Hermes/workspace/projects/VLP-STRAINMETER
jupyter notebook notebooks/classifier.ipynb
```

Run cells in order:
1. Setup (device, imports)
2. Load data (train/val/test split)
3. Train CNN (real-time plots)
4. Evaluate (metrics, confusion matrix, ROC)
5. Predict on first 20 real waveforms
6. Predict on all real waveforms + Plotly interactive chart

### CLI

```bash
python3 main_classifier.py \
  --type1 data/processed/Waveforms_type1_extended_128samples.txt \
  --type2 data/processed/Waveforms_type2_extended_128samples.txt \
  --epochs 50 \
  --lr 1e-3 \
  --device auto
```

### Programmatic

```python
from src.classifier import load_and_split, train_classifier, evaluate_model, predict

train_loader, val_loader, test_loader = load_and_split("data/processed/Waveforms_type1_extended_128samples.txt",
    "data/processed/Waveforms_type2_extended_128samples.txt")

model, history = train_classifier(train_loader, val_loader, epochs=50)
results = evaluate_model(model, test_loader)
preds = predict(model, new_waveforms)
```

## Configuration

Edit `config/config.yaml` to control:

| Section | Parameters |
|---------|-----------|
| `data` | waveform_length, batch_size, seed, normalize |
| `split` | train_ratio, val_ratio, test_ratio |
| `classifier` | dropout, encoder_channels, kernel_sizes, strides, paddings, activation |
| `training` | epochs, learning_rate, weight_decay, device, save_path |
| `discriminator` | measured_path, confidence_threshold |

## What to Modify

### Change training data
Edit `classifier_runs` paths in `config/config.yaml` or CLI arguments

### Change the model architecture
Edit `src/classifier.py` → class `VLPClassifier`

### Change training hyperparameters
Edit `config/config.yaml` → `training` section or CLI arguments

### Change the measured file
Edit `config/config.yaml` → `discriminator.measured_path`

### Change the plots
Edit `notebooks/classifier.ipynb` — Plotly charts are defined in the last cells

## Output Artifacts

Trained model:
- `outputs/cnn_classifier.pt` — model weights + training history

Evaluation plots:
- `notebooks/outputs/confusion_matrix.png`
- `notebooks/outputs/roc_curve.png`

## Legacy (v1.0 — Autoencoder)

The original autoencoder-based approach is deprecated but preserved for reference:

- `src/models.py` — Conv1dAutoencoder
- `src/train.py` — autoencoder training
- `src/discriminator.py` — reconstruction error discriminator
- `notebooks/strainmeter_workflow.ipynb` — autoencoder workflow
- `outputs/ae1_type1/` — AE1 checkpoints
- `outputs/ae2_type2/` — AE2 checkpoints
- `outputs/discriminator/` — timeline outputs

Do not build new work on legacy modules. Use the classifier instead.

## Dependencies

```
torch>=2.0
numpy
matplotlib
scikit-learn
pyyaml
plotly
jupyter
```

## Version History

See [`CHANGELOG.md`](../CHANGELOG.md) for complete version history.

- **v2.0 (current)** — Supervised CNN classifier
- **v1.0 (deprecated)** — 1D Autoencoder with reconstruction error discrimination
