"""Discriminator pipeline built on top of two trained autoencoders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import AutoencoderRunConfig, ProjectConfig
from .data import WaveformDataset, build_dataloaders, normalize_waveforms
from .models import Conv1dAutoencoder
from .plotting import plot_discriminator_timeline
from .utils import ensure_directory, save_csv_rows, save_json


@dataclass
class DiscriminatorArtifacts:
    output_dir: Path
    scores_csv_path: Path
    scores_json_path: Path
    timeline_plot_path: Path
    thresholds: dict[str, float]


def _resolve_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def _align_rows_to_length(array: np.ndarray, target_length: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {array.shape}.")

    current_length = array.shape[1]
    if current_length == target_length:
        return array

    if current_length > target_length:
        excess = current_length - target_length
        left = excess // 2
        right = excess - left
        print(
            f"Measured waveforms are longer than the model input ({current_length} -> {target_length}); "
            f"using a centered crop."
        )
        return array[:, left : current_length - right]

    deficit = target_length - current_length
    left = deficit // 2
    right = deficit - left
    print(
        f"Measured waveforms are shorter than the model input ({current_length} -> {target_length}); "
        f"using symmetric zero padding."
    )
    return np.pad(array, ((0, 0), (left, right)), mode="constant")


def _load_measured_waveforms(path: Path, target_length: int) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        array = np.loadtxt(path, dtype=np.float32)
    aligned = _align_rows_to_length(array, target_length)
    return normalize_waveforms(aligned, mode="none")


def _load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> Conv1dAutoencoder:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]["model"]
    model_name = model_config.get("name") or checkpoint.get("model_name", "AE")
    model = Conv1dAutoencoder(
        input_channels=model_config["input_channels"],
        waveform_length=model_config["waveform_length"],
        encoder_channels=list(model_config["encoder_channels"]),
        kernel_size=model_config["kernel_size"],
        stride=model_config["stride"],
        padding=model_config["padding"],
        latent_dim=model_config["latent_dim"],
        activation=model_config["activation"],
        batch_norm=model_config["batch_norm"],
        dropout=model_config["dropout"],
        decoder_type=model_config["decoder_type"],
        output_activation=model_config["output_activation"],
        name=model_name,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def _as_waveform_batch(batch) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _batch_reconstruction_errors(
    reconstruction: torch.Tensor,
    waveforms: torch.Tensor,
    *,
    loss_name: str,
    reduction: str,
) -> torch.Tensor:
    normalized = loss_name.lower()
    reduction = reduction.lower()
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be mean or sum")

    def reduce(tensor: torch.Tensor) -> torch.Tensor:
        if reduction == "mean":
            return torch.mean(tensor, dim=(1, 2))
        return torch.sum(tensor, dim=(1, 2))

    if normalized == "mse":
        return reduce((reconstruction - waveforms) ** 2)
    if normalized in {"l1", "mae"}:
        return reduce(torch.abs(reconstruction - waveforms))
    if normalized in {"smooth_l1", "huber"}:
        return reduce(
            torch.nn.functional.smooth_l1_loss(
                reconstruction,
                waveforms,
                reduction="none",
                beta=1.0,
            )
        )
    raise ValueError(f"Unsupported loss: {loss_name}")


def collect_reconstruction_errors(
    model: Conv1dAutoencoder,
    loader,
    *,
    device: torch.device,
    loss_name: str,
    reduction: str = "mean",
) -> np.ndarray:
    errors: list[float] = []
    with torch.no_grad():
        for batch in loader:
            waveforms = _as_waveform_batch(batch).to(device)
            reconstruction, _ = model(waveforms)
            batch_errors = _batch_reconstruction_errors(
                reconstruction,
                waveforms,
                loss_name=loss_name,
                reduction=reduction,
            )
            errors.extend(batch_errors.detach().cpu().tolist())
    return np.asarray(errors, dtype=np.float32)


def estimate_threshold(errors: np.ndarray, quantile: float) -> float:
    return float(np.quantile(errors, quantile))


def score_errors(errors: np.ndarray, threshold: float) -> np.ndarray:
    normalized = errors / max(threshold, 1e-8)
    return 1.0 / (1.0 + normalized)


def build_discrimination_rows(
    score_type1: np.ndarray,
    score_type2: np.ndarray,
    outlier_score: np.ndarray,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for index, (type1, type2, outlier) in enumerate(
        zip(score_type1, score_type2, outlier_score, strict=True)
    ):
        rows.append(
            {
                "waveform_index": int(index),
                "type1_score": float(type1),
                "type2_score": float(type2),
                "outlier_score": float(outlier),
            }
        )
    return rows


def run_discriminator(
    config: ProjectConfig,
    autoencoder_runs: list[tuple[AutoencoderRunConfig, Path]],
    *,
    project_root: Path,
    device: torch.device,
) -> DiscriminatorArtifacts:
    output_dir = ensure_directory(_resolve_path(config.discriminator.output_dir, project_root))
    measured_path = _resolve_path(config.discriminator.measured_path, project_root)
    measured_waveforms = _load_measured_waveforms(measured_path, config.data.expected_length)
    measured_waveforms = normalize_waveforms(measured_waveforms, mode=config.data.normalization)
    measured_loader = DataLoader(
        WaveformDataset(measured_waveforms),
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    all_models: list[dict[str, object]] = []
    thresholds: dict[str, float] = {}

    for run_config, run_output_dir in autoencoder_runs:
        checkpoint_path = run_output_dir / "best_model.pt"
        model = _load_checkpoint_model(checkpoint_path, device)

        train_path = _resolve_path(run_config.train_path, project_root)
        val_path = _resolve_path(run_config.val_path, project_root) if run_config.val_path else None
        train_loader, val_loader = build_dataloaders(
            train_path=train_path,
            val_path=val_path,
            expected_length=config.data.expected_length,
            batch_size=config.data.batch_size,
            validation_split=config.data.validation_split,
            shuffle=config.data.shuffle,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            normalization=config.data.normalization,
            seed=config.training.seed,
        )
        threshold_source = val_loader if val_loader is not None else train_loader
        calibration_errors = collect_reconstruction_errors(
            model,
            threshold_source,
            device=device,
            loss_name=config.loss.name,
            reduction=config.loss.reduction,
        )
        threshold = estimate_threshold(calibration_errors, config.discriminator.threshold_quantile)
        thresholds[run_config.name] = threshold
        all_models.append(
            {
                "name": run_config.name,
                "model": model,
                "threshold": threshold,
            }
        )

    all_errors: dict[str, np.ndarray] = {}
    for entry in all_models:
        model = entry["model"]
        name = entry["name"]
        errors: list[float] = []
        with torch.no_grad():
            for batch in measured_loader:
                waveforms = batch.to(device)
                reconstruction, _ = model(waveforms)
                batch_errors = _batch_reconstruction_errors(
                    reconstruction,
                    waveforms,
                    loss_name=config.loss.name,
                    reduction=config.loss.reduction,
                )
                errors.extend(batch_errors.detach().cpu().tolist())
        all_errors[name] = np.asarray(errors, dtype=np.float32)

    type1_name = autoencoder_runs[0][0].name
    type2_name = autoencoder_runs[1][0].name
    type1_errors = all_errors[type1_name]
    type2_errors = all_errors[type2_name]

    type1_scores = score_errors(type1_errors, thresholds[type1_name])
    type2_scores = score_errors(type2_errors, thresholds[type2_name])
    outlier_scores = 1.0 - np.maximum(type1_scores, type2_scores)
    outlier_scores = np.clip(outlier_scores, 0.0, 1.0)

    rows = build_discrimination_rows(type1_scores, type2_scores, outlier_scores)
    scores_csv_path = output_dir / "discriminator_scores.csv"
    scores_json_path = output_dir / "discriminator_scores.json"
    save_csv_rows(rows, scores_csv_path)
    save_json(rows, scores_json_path)

    timeline_plot_path = output_dir / "discriminator_timeline.png"
    figure = plot_discriminator_timeline(
        type1_scores=type1_scores,
        type2_scores=type2_scores,
        outlier_scores=outlier_scores,
        title="Waveform discrimination timeline",
    )
    figure.savefig(timeline_plot_path, dpi=150, bbox_inches="tight")

    print("\nDiscriminator summary")
    print(f"Measured file: {measured_path}")
    for name, threshold in thresholds.items():
        print(f"{name} threshold: {threshold:.6f}")
    print(f"Scores CSV: {scores_csv_path}")
    print(f"Timeline plot: {timeline_plot_path}")

    return DiscriminatorArtifacts(
        output_dir=output_dir,
        scores_csv_path=scores_csv_path,
        scores_json_path=scores_json_path,
        timeline_plot_path=timeline_plot_path,
        thresholds=thresholds,
    )
