# Waveform Data — v2.0 Classifier

This folder contains waveform data for the CNN classifier project.

## `processed/` — Training Data

Training datasets for the classifier. Each file contains waveforms of a single class.

### Files

| File | Size | Description |
|------|------|-------------|
| `Waveforms_type1_extended_128samples.txt` | ~7.2MB | Type1 waveforms (~100k samples) |
| `Waveforms_type2_extended_128samples.txt` | ~5.3MB | Type2 waveforms (~75k samples) |

### Format

- One waveform per row
- 128 samples per row (float values)
- Space-separated values
- No header row

### Usage

The classifier loads both files, normalizes globally (mean/std), and splits into train/val/test.

## `measured/` — Real Data

Real measured waveforms to classify.

### Files

| File | Size | Description |
|------|------|-------------|
| `VLPs-2019_low-pass.dat` | ~115MB | Real VLP waveforms (~150k samples) |

### Format

- One waveform per row
- 151 samples per row (float values)
- The classifier center-crops each row to 128 samples before prediction

### Usage

Loaded in the notebook's last cells and classified with the trained model.

## Normalization

All waveforms are normalized using global statistics computed on the combined training data:

```python
global_mean = all_train_data.mean()
global_std = all_train_data.std() + 1e-8
normalized = (waveforms - global_mean) / global_std
```

The same normalization is applied to measured data using training statistics.
