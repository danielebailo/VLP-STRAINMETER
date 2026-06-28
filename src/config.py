"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    expected_length: int = 131
    batch_size: int = 32
    validation_split: float = 0.2
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    normalization: str = "none"


@dataclass
class ModelConfig:
    name: str = "AE1"
    type: str = "convolutional_1d"
    input_channels: int = 1
    waveform_length: int = 131
    encoder_channels: list[int] = field(default_factory=lambda: [16, 32, 64])
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1
    latent_dim: int = 30
    activation: str = "relu"
    batch_norm: bool = True
    dropout: float = 0.0
    decoder_type: str = "convtranspose"
    output_activation: str = "none"


@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: str = "auto"
    live_plot: bool = True
    seed: int = 42
    save_best_model: bool = True
    save_last_model: bool = True


@dataclass
class NotebookConfig:
    run_mode: str = "train + discriminate"


@dataclass
class LossConfig:
    name: str = "mse"
    reduction: str = "mean"
    beta: float = 1.0


@dataclass
class AutoencoderRunConfig:
    name: str
    train_path: str
    val_path: str | None = None
    output_dir: str | None = None


@dataclass
class DiscriminatorConfig:
    measured_path: str = "data/measured/VLPs-2019_low-pass.dat"
    output_dir: str = "outputs/discriminator"
    threshold_quantile: float = 0.95


@dataclass
class ProjectConfig:
    project_name: str = "strainmeter_autoencoder"
    output_dir: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    notebook: NotebookConfig = field(default_factory=NotebookConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    autoencoders: list[AutoencoderRunConfig] = field(default_factory=list)

    def validate(self) -> None:
        if self.model.type != "convolutional_1d":
            raise ValueError("The default supported model type is 'convolutional_1d'.")
        if self.data.expected_length != self.model.waveform_length:
            raise ValueError(
                "data.expected_length and model.waveform_length must match."
            )
        if not self.model.encoder_channels:
            raise ValueError("model.encoder_channels must contain at least one channel.")
        if self.model.input_channels != 1:
            raise ValueError("This phase expects single-channel waveform input.")
        if self.model.kernel_size <= 0 or self.model.stride <= 0:
            raise ValueError("kernel_size and stride must be positive.")
        if self.model.latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if not 0.0 <= self.data.validation_split < 1.0:
            raise ValueError("validation_split must be in [0, 1).")
        if self.training.epochs <= 0:
            raise ValueError("training.epochs must be positive.")
        if self.training.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive.")
        if self.notebook.run_mode not in {
            "train only",
            "train + discriminate",
            "solo timeline",
            "solo ricostruzioni",
        }:
            raise ValueError(
                "notebook.run_mode must be one of: "
                "train only, train + discriminate, solo timeline, solo ricostruzioni"
            )
        if self.loss.name.lower() not in {"mse", "l1", "mae", "smooth_l1", "huber"}:
            raise ValueError("loss.name must be one of: mse, l1, mae, smooth_l1, huber")
        if self.loss.reduction not in {"mean", "sum"}:
            raise ValueError("loss.reduction must be one of: mean, sum")
        if len(self.autoencoders) != 2:
            raise ValueError("Exactly two autoencoder runs must be configured.")
        names = [run.name for run in self.autoencoders]
        if len(names) != len(set(names)):
            raise ValueError("Autoencoder run names must be unique.")
        for run in self.autoencoders:
            if not run.train_path:
                raise ValueError("Each autoencoder run must define train_path.")
        if not self.discriminator.measured_path:
            raise ValueError("discriminator.measured_path must be set.")
        if not 0.0 < self.discriminator.threshold_quantile < 1.0:
            raise ValueError("discriminator.threshold_quantile must be in (0, 1).")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(dc_cls, payload: dict[str, Any]):
    fields = {field.name for field in dc_cls.__dataclass_fields__.values()}
    kwargs = {key: payload[key] for key in payload if key in fields}
    return dc_cls(**kwargs)


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    data = _merge_dataclass(DataConfig, raw.get("data", {}))
    model = _merge_dataclass(ModelConfig, raw.get("model", {}))
    training_raw = raw.get("training", {})
    training = _merge_dataclass(TrainingConfig, training_raw)
    notebook = _merge_dataclass(NotebookConfig, raw.get("notebook", {}))
    if "loss" in training_raw:
        loss = _merge_dataclass(LossConfig, training_raw.get("loss", {}))
    else:
        loss = LossConfig(name=training_raw.get("criterion", "mse"))

    autoencoders_raw = raw.get("autoencoders", [])
    autoencoders = [
        _merge_dataclass(AutoencoderRunConfig, run_raw)
        for run_raw in autoencoders_raw
    ]

    config = ProjectConfig(
        project_name=raw.get("project_name", "strainmeter_autoencoder"),
        output_dir=raw.get("output_dir", "outputs"),
        data=data,
        model=model,
        training=training,
        notebook=notebook,
        loss=loss,
        discriminator=_merge_dataclass(DiscriminatorConfig, raw.get("discriminator", {})),
        autoencoders=autoencoders,
    )
    config.validate()
    return config
