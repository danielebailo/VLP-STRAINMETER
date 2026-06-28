"""Waveform loading, validation, preprocessing, and data loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def _coerce_waveform_array(array: np.ndarray, expected_length: int) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        if array.shape[0] != expected_length:
            raise ValueError(
                f"Expected waveform length {expected_length}, got {array.shape[0]}."
            )
        return array.reshape(1, expected_length)

    if array.ndim != 2:
        raise ValueError("Input data must be 1D or 2D.")

    if array.shape[1] == expected_length:
        return array
    if array.shape[0] == expected_length and array.shape[1] != expected_length:
        return array.T
    if 1 in array.shape:
        flattened = array.reshape(-1)
        if flattened.shape[0] != expected_length:
            raise ValueError(
                f"Expected waveform length {expected_length}, got {flattened.shape[0]}."
            )
        return flattened.reshape(1, expected_length)

    raise ValueError(
        f"Could not interpret waveform matrix shape {array.shape} for expected length {expected_length}."
    )


def _load_numpy(path: Path, expected_length: int) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    return _coerce_waveform_array(array, expected_length)


def _load_tabular(path: Path, expected_length: int) -> np.ndarray:
    try:
        frame = pd.read_csv(path, header=None, sep=None, engine="python")
        array = frame.to_numpy(dtype=np.float32)
    except Exception:
        array = np.loadtxt(path, dtype=np.float32)
    return _coerce_waveform_array(array, expected_length)


def load_waveforms(path: str | Path, expected_length: int) -> np.ndarray:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Waveform file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".npy":
        waveforms = _load_numpy(input_path, expected_length)
    elif suffix in {".csv", ".txt", ".tsv", ".dat"}:
        waveforms = _load_tabular(input_path, expected_length)
    else:
        raise ValueError("Supported input formats are .npy, .csv, .txt, .tsv, and .dat")

    if waveforms.shape[1] != expected_length:
        raise ValueError(
            f"Waveform length mismatch. Expected {expected_length}, got {waveforms.shape[1]}."
        )
    return waveforms.astype(np.float32, copy=False)


def normalize_waveforms(waveforms: np.ndarray, mode: str = "none") -> np.ndarray:
    normalized = np.asarray(waveforms, dtype=np.float32)
    mode = mode.lower()
    if mode == "none":
        return normalized
    if mode == "minmax":
        minimum = normalized.min(axis=1, keepdims=True)
        maximum = normalized.max(axis=1, keepdims=True)
        denominator = np.where((maximum - minimum) == 0, 1.0, maximum - minimum)
        return (normalized - minimum) / denominator
    if mode == "zscore":
        mean = normalized.mean(axis=1, keepdims=True)
        std = normalized.std(axis=1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        return (normalized - mean) / std
    raise ValueError("normalization must be one of: none, minmax, zscore")


class WaveformDataset(Dataset):
    def __init__(self, waveforms: np.ndarray) -> None:
        self.waveforms = np.asarray(waveforms, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.waveforms.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        waveform = torch.from_numpy(self.waveforms[index]).float().unsqueeze(0)
        return waveform


def split_waveforms(
    waveforms: np.ndarray,
    validation_split: float,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray | None]:
    if validation_split <= 0:
        return waveforms, None

    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1.")

    total = len(waveforms)
    if total < 2:
        return waveforms, None

    indices = np.arange(total)
    if shuffle:
        generator = np.random.default_rng(seed)
        generator.shuffle(indices)

    split_index = int(round(total * (1.0 - validation_split)))
    split_index = min(max(split_index, 1), total - 1)
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]
    return waveforms[train_indices], waveforms[val_indices]


def create_dataloaders(
    train_waveforms: np.ndarray,
    *,
    val_waveforms: np.ndarray | None = None,
    batch_size: int = 32,
    validation_split: float = 0.2,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader | None]:
    if val_waveforms is None:
        train_waveforms, val_waveforms = split_waveforms(
            train_waveforms,
            validation_split=validation_split,
            shuffle=shuffle,
            seed=seed,
        )

    train_dataset = WaveformDataset(train_waveforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = None
    if val_waveforms is not None and len(val_waveforms) > 0:
        val_dataset = WaveformDataset(val_waveforms)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader


def build_dataloaders(
    *,
    train_path: str | Path,
    expected_length: int,
    batch_size: int,
    val_path: str | Path | None = None,
    validation_split: float = 0.2,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    normalization: str = "none",
    seed: int = 42,
) -> tuple[DataLoader, DataLoader | None]:
    train_waveforms = normalize_waveforms(load_waveforms(train_path, expected_length), normalization)
    val_waveforms = None
    if val_path:
        val_waveforms = normalize_waveforms(load_waveforms(val_path, expected_length), normalization)
    return create_dataloaders(
        train_waveforms,
        val_waveforms=val_waveforms,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )
