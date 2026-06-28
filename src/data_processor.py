"""Compatibility layer for the old module path."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    WaveformDataset,
    build_dataloaders,
    create_dataloaders,
    load_waveforms,
    normalize_waveforms,
    split_waveforms,
)


def preprocess_waveforms(waveforms):
    return normalize_waveforms(waveforms, mode="minmax")
