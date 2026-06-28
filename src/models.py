"""PyTorch models for waveform autoencoding."""

from __future__ import annotations

import logging
import warnings
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


LOGGER = logging.getLogger(__name__)


def conv1d_output_length(
    input_length: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> int:
    return ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


def convtranspose1d_output_length(
    input_length: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int = 1,
) -> int:
    return (
        (input_length - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )


def align_length(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    """Crop or pad symmetrically so the last dimension matches target_length."""

    current_length = tensor.shape[-1]
    if current_length == target_length:
        return tensor

    if current_length > target_length:
        excess = current_length - target_length
        left = excess // 2
        right = excess - left
        return tensor[..., left : current_length - right]

    deficit = target_length - current_length
    left = deficit // 2
    right = deficit - left
    return F.pad(tensor, (left, right))


def make_activation(name: str | None) -> nn.Module | None:
    if name is None:
        return None

    normalized = name.lower()
    if normalized == "none":
        return None
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1)
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "sigmoid":
        return nn.Sigmoid()
    if normalized == "selu":
        return nn.SELU()
    raise ValueError(f"Unsupported activation: {name}")


class ConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int,
        padding: int,
        activation: str,
        batch_norm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        ]
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        act = make_activation(activation)
        if act is not None:
            layers.append(act)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock1d(nn.Module):
    def __init__(self, layers: Iterable[nn.Module]) -> None:
        super().__init__()
        self.block = nn.Sequential(*list(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Conv1dAutoencoder(nn.Module):
    """Configurable convolutional autoencoder for 1D waveform reconstruction."""

    def __init__(
        self,
        *,
        input_channels: int,
        waveform_length: int,
        encoder_channels: list[int],
        kernel_size: int,
        stride: int,
        padding: int,
        latent_dim: int,
        activation: str = "relu",
        batch_norm: bool = True,
        dropout: float = 0.0,
        decoder_type: str = "convtranspose",
        output_activation: str = "none",
        name: str = "AE",
    ) -> None:
        super().__init__()
        if not encoder_channels:
            raise ValueError("encoder_channels cannot be empty.")

        self.name = name
        self.input_channels = input_channels
        self.waveform_length = waveform_length
        self.encoder_channels = list(encoder_channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.latent_dim = latent_dim
        self.activation_name = activation
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.decoder_type = decoder_type.lower()
        self.output_activation_name = output_activation

        if self.decoder_type not in {"convtranspose", "upsample"}:
            raise ValueError("decoder_type must be 'convtranspose' or 'upsample'.")

        self.encoder_lengths = [waveform_length]
        current_length = waveform_length
        current_channels = input_channels
        encoder_blocks: list[nn.Module] = []
        for out_channels in self.encoder_channels:
            next_length = conv1d_output_length(
                current_length,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
            if next_length <= 0:
                raise ValueError(
                    "The configured encoder reduces the waveform to a non-positive length. "
                    "Reduce the number of layers/stride or increase the input length."
                )
            encoder_blocks.append(
                ConvBlock1d(
                    current_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    activation=self.activation_name,
                    batch_norm=self.batch_norm,
                    dropout=self.dropout,
                )
            )
            self.encoder_lengths.append(next_length)
            current_channels = out_channels
            current_length = next_length

        self.encoded_length = self.encoder_lengths[-1]
        self.bottleneck_channels = self.encoder_channels[-1]
        self.flattened_features = self.bottleneck_channels * self.encoded_length

        self.encoder = nn.ModuleList(encoder_blocks)
        self.encoder_projection = nn.Linear(self.flattened_features, latent_dim)
        self.decoder_projection = nn.Linear(latent_dim, self.flattened_features)

        decoder_blocks: list[nn.Module] = []
        decoder_input_channels = self.encoder_channels[-1]
        decoder_channel_targets = list(reversed(self.encoder_channels[:-1])) + [input_channels]
        decoder_length_targets = list(reversed(self.encoder_lengths[:-1]))
        current_length = self.encoded_length
        current_channels = decoder_input_channels

        for stage_index, (target_channels, target_length) in enumerate(
            zip(decoder_channel_targets, decoder_length_targets)
        ):
            is_last_stage = stage_index == len(decoder_channel_targets) - 1
            layers: list[nn.Module] = []

            if self.decoder_type == "convtranspose":
                base_length = convtranspose1d_output_length(
                    current_length,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    output_padding=0,
                )
                output_padding = target_length - base_length
                if output_padding < 0 or output_padding >= self.stride:
                    warnings.warn(
                        f"{self.name}: exact ConvTranspose1d inversion is not possible at stage "
                        f"{stage_index + 1}; using output_padding=0 and cropping/padding at the end."
                    )
                    output_padding = 0
                layers.append(
                    nn.ConvTranspose1d(
                        current_channels,
                        target_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        output_padding=output_padding,
                    )
                )
            else:
                layers.extend(
                    [
                        nn.Upsample(size=target_length, mode="linear", align_corners=False),
                        nn.Conv1d(
                            current_channels,
                            target_channels,
                            kernel_size=self.kernel_size,
                            stride=1,
                            padding=self.padding,
                        ),
                    ]
                )

            if not is_last_stage:
                if self.batch_norm:
                    layers.append(nn.BatchNorm1d(target_channels))
                act = make_activation(self.activation_name)
                if act is not None:
                    layers.append(act)
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))

            decoder_blocks.append(DecoderBlock1d(layers))
            current_channels = target_channels
            current_length = target_length

        self.decoder = nn.ModuleList(decoder_blocks)
        self.output_activation = make_activation(self.output_activation_name)

        LOGGER.info(self.architecture_summary())

    def architecture_summary(self) -> str:
        lines = [
            f"{self.name} | Conv1dAutoencoder",
            f"  input shape: (batch, {self.input_channels}, {self.waveform_length})",
            f"  encoder channels: {self.encoder_channels}",
            f"  encoder lengths: {self.encoder_lengths}",
            f"  latent dim: {self.latent_dim}",
            f"  decoder type: {self.decoder_type}",
            f"  output activation: {self.output_activation_name}",
        ]
        return "\n".join(lines)

    def _check_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError("Expected input shape (batch, channels, waveform_length).")
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {x.shape[1]}."
            )
        if x.shape[2] != self.waveform_length:
            raise ValueError(
                f"Expected waveform length {self.waveform_length}, got {x.shape[2]}."
            )
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._check_input(x)
        for block in self.encoder:
            x = block(x)
        x = x.reshape(x.shape[0], -1)
        return self.encoder_projection(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.decoder_projection(latent)
        x = x.view(latent.shape[0], self.bottleneck_channels, self.encoded_length)
        for block in self.decoder:
            x = block(x)
        x = align_length(x, self.waveform_length)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        if reconstruction.shape[-1] != self.waveform_length:
            raise RuntimeError("Decoder did not return the expected waveform length.")
        return reconstruction, latent


class FullyConnectedAutoencoder(nn.Module):
    """Optional future model, kept for completeness but not used by default."""

    def __init__(
        self,
        *,
        waveform_length: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        output_activation: str = "none",
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        encoder_layers: list[nn.Module] = []
        decoder_layers: list[nn.Module] = []

        encoder_dims = [waveform_length] + hidden_dims
        for in_dim, out_dim in zip(encoder_dims[:-1], encoder_dims[1:]):
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(hidden_dims[-1], latent_dim))

        decoder_dims = [latent_dim] + list(reversed(hidden_dims))
        for in_dim, out_dim in zip(decoder_dims[:-1], decoder_dims[1:]):
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(hidden_dims[0], waveform_length))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        self.output_activation = make_activation(output_activation)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.squeeze(1)
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        if self.output_activation is not None:
            reconstruction = self.output_activation(reconstruction)
        return reconstruction.unsqueeze(1), latent
