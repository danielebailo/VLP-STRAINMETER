"""Core package for the strainmeter autoencoder training stack."""

from .config import (
    AutoencoderRunConfig,
    DataConfig,
    DiscriminatorConfig,
    LossConfig,
    ModelConfig,
    ProjectConfig,
    TrainingConfig,
    load_config,
)
from .data import build_dataloaders, load_waveforms, normalize_waveforms
from .models import Conv1dAutoencoder, FullyConnectedAutoencoder
from .train import TrainingArtifacts, fit_autoencoder
