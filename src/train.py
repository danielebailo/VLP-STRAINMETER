"""Training loop for waveform autoencoders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .config import ProjectConfig
from .models import Conv1dAutoencoder
from .plotting import LiveTrainingPlot
from .utils import ensure_directory, save_csv_rows, save_json


@dataclass
class TrainingArtifacts:
    output_dir: Path
    best_model_path: Path | None
    last_model_path: Path | None
    history_csv_path: Path
    history_json_path: Path


def _as_waveform_batch(batch: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _first_batch(loader) -> torch.Tensor | None:
    for batch in loader:
        return _as_waveform_batch(batch)
    return None


def _save_checkpoint(
    *,
    path: Path,
    model: Conv1dAutoencoder,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    train_loss: float,
    val_loss: float | None,
    config: ProjectConfig,
    ) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_name": model.name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": config.to_dict(),
        },
        path,
    )


def build_reconstruction_loss(config: ProjectConfig) -> nn.Module:
    loss_name = config.loss.name.lower()
    reduction = config.loss.reduction

    if loss_name == "mse":
        return nn.MSELoss(reduction=reduction)
    if loss_name in {"l1", "mae"}:
        return nn.L1Loss(reduction=reduction)
    if loss_name in {"smooth_l1", "huber"}:
        return nn.SmoothL1Loss(reduction=reduction, beta=config.loss.beta)
    raise ValueError(f"Unsupported loss: {config.loss.name}")


def fit_autoencoder(
    model: Conv1dAutoencoder,
    train_loader,
    val_loader,
    config: ProjectConfig,
    *,
    output_dir: str | Path,
    plotter: LiveTrainingPlot | None = None,
    device: torch.device | None = None,
) -> TrainingArtifacts:
    run_dir = ensure_directory(output_dir)
    device = device or torch.device("cpu")
    model = model.to(device)

    criterion = build_reconstruction_loss(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    history: list[dict[str, float]] = []
    best_metric = float("inf")
    best_model_path = run_dir / "best_model.pt"
    last_model_path = run_dir / "last_model.pt"
    history_csv_path = run_dir / "history.csv"
    history_json_path = run_dir / "history.json"
    preview_batch = _first_batch(val_loader) if val_loader is not None else _first_batch(train_loader)

    print(f"Device: {device}")
    print(f"Saving artifacts to: {run_dir}")
    print(model.architecture_summary())
    print(f"Reconstruction loss: {config.loss.name} (reduction={config.loss.reduction})")
    print("Starting training...")

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            waveforms = _as_waveform_batch(batch).to(device)
            reconstruction, _ = model(waveforms)
            loss = criterion(reconstruction, waveforms)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

        train_loss = total_train_loss / max(train_batches, 1)
        val_loss: float | None = None

        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    waveforms = _as_waveform_batch(batch).to(device)
                    reconstruction, _ = model(waveforms)
                    loss = criterion(reconstruction, waveforms)
                    total_val_loss += loss.item()
                    val_batches += 1
            val_loss = total_val_loss / max(val_batches, 1)

        metric = val_loss if val_loss is not None else train_loss
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss) if val_loss is not None else None,
            }
        )

        _save_checkpoint(
            path=last_model_path,
            model=model,
            epoch=epoch,
            optimizer=optimizer,
            train_loss=train_loss,
            val_loss=val_loss,
            config=config,
        )

        if config.training.save_best_model and metric < best_metric:
            best_metric = metric
            _save_checkpoint(
                path=best_model_path,
                model=model,
                epoch=epoch,
                optimizer=optimizer,
                train_loss=train_loss,
                val_loss=val_loss,
                config=config,
            )

        save_csv_rows(history, history_csv_path)
        save_json(history, history_json_path)

        preview_original = None
        preview_reconstructed = None
        if preview_batch is not None:
            preview_original = preview_batch[:1].to(device)
            model.eval()
            with torch.no_grad():
                preview_reconstructed, _ = model(preview_original)

        print(
            f"Epoch {epoch:03d}/{config.training.epochs} | "
            f"train_loss={train_loss:.6f}"
            + (f" | val_loss={val_loss:.6f}" if val_loss is not None else "")
        )

        if plotter is not None:
            plotter.update(
                epoch=epoch,
                history=history,
                original=preview_original.cpu() if preview_original is not None else None,
                reconstructed=preview_reconstructed.cpu() if preview_reconstructed is not None else None,
                title=model.name,
            )

    print("Training completed.")
    return TrainingArtifacts(
        output_dir=run_dir,
        best_model_path=best_model_path if best_model_path.exists() else None,
        last_model_path=last_model_path if last_model_path.exists() else None,
        history_csv_path=history_csv_path,
        history_json_path=history_json_path,
    )
