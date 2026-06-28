#!/usr/bin/env python3

"""Command line entrypoint for autoencoder training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data import build_dataloaders
from src.discriminator import run_discriminator
from src.models import Conv1dAutoencoder
from src.plotting import LiveTrainingPlot, plot_reconstructions
from src.train import fit_autoencoder
from src.utils import ensure_directory, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a strainmeter waveform autoencoder.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to the YAML config.")
    parser.add_argument(
        "--device",
        default=None,
        help="Override device selection: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--no-live-plot",
        action="store_true",
        help="Disable live plotting even if enabled in the config.",
    )
    return parser.parse_args()


def _resolve_path(path_value: str | None, project_root: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def _run_single_autoencoder(config, run_config, project_root: Path, device):
    train_path = _resolve_path(run_config.train_path, project_root)
    val_path = _resolve_path(run_config.val_path, project_root)
    output_dir = _resolve_path(run_config.output_dir, project_root) or (project_root / config.output_dir / run_config.name.lower())
    output_dir = ensure_directory(output_dir)

    print("\n" + "=" * 80)
    print(f"Training {run_config.name}")
    print(f"Train file: {train_path}")
    if val_path is not None:
        print(f"Validation file: {val_path}")
    print(f"Output dir: {output_dir}")
    print("=" * 80)

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

    model = Conv1dAutoencoder(
        input_channels=config.model.input_channels,
        waveform_length=config.model.waveform_length,
        encoder_channels=config.model.encoder_channels,
        kernel_size=config.model.kernel_size,
        stride=config.model.stride,
        padding=config.model.padding,
        latent_dim=config.model.latent_dim,
        activation=config.model.activation,
        batch_norm=config.model.batch_norm,
        dropout=config.model.dropout,
        decoder_type=config.model.decoder_type,
        output_activation=config.model.output_activation,
        name=run_config.name,
    )
    config.model.name = run_config.name

    plotter = LiveTrainingPlot(enabled=config.training.live_plot)
    artifacts = fit_autoencoder(
        model,
        train_loader,
        val_loader,
        config,
        output_dir=output_dir,
        plotter=plotter if config.training.live_plot else None,
        device=device,
    )

    preview_batch = next(iter(val_loader if val_loader is not None else train_loader))
    figure = plot_reconstructions(model.to(device), preview_batch, device=device, max_items=4)
    figure.savefig(output_dir / "final_reconstructions.png", dpi=150, bbox_inches="tight")
    print(f"- final reconstruction plot: {output_dir / 'final_reconstructions.png'}")
    return artifacts, output_dir


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config.training.device = args.device
    if args.no_live_plot:
        config.training.live_plot = False

    set_seed(config.training.seed)
    device = resolve_device(config.training.device)
    project_root = PROJECT_ROOT

    all_artifacts = []
    trained_runs = []
    for run_config in config.autoencoders:
        artifacts, output_dir = _run_single_autoencoder(config, run_config, project_root, device)
        all_artifacts.append((run_config.name, artifacts))
        trained_runs.append((run_config, output_dir))

    discriminator_artifacts = run_discriminator(
        config,
        trained_runs,
        project_root=project_root,
        device=device,
    )

    print("\nArtifacts:")
    for run_name, artifacts in all_artifacts:
        print(f"[{run_name}] best model: {artifacts.best_model_path}")
        print(f"[{run_name}] last model: {artifacts.last_model_path}")
        print(f"[{run_name}] history csv: {artifacts.history_csv_path}")
        print(f"[{run_name}] history json: {artifacts.history_json_path}")
    print(f"[discriminator] scores csv: {discriminator_artifacts.scores_csv_path}")
    print(f"[discriminator] timeline plot: {discriminator_artifacts.timeline_plot_path}")


if __name__ == "__main__":
    main()
