# Measured Data — v2.0 Classifier

Real waveform file to classify with the trained CNN classifier.

## File

`VLPs-2019_low-pass.dat`

- **Size:** ~115MB
- **Waveforms:** ~150k
- **Samples per waveform:** 151 (center-cropped to 128 for classification)
- **Format:** one waveform per row, space-separated float values

## Usage

The classifier loads this file, normalizes using training data statistics, center-crops to 128 samples, and predicts class probabilities for each waveform.

## Notes

- This file is NOT used during training
- It represents real measured strainmeter data from 2019
- Low-pass filtered version of the original data
