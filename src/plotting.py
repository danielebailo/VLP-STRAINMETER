"""Matplotlib helpers for live training and reconstruction plots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output, display
import plotly.graph_objects as go

from .models import Conv1dAutoencoder
from .utils import to_numpy


@dataclass
class LivePlotState:
    fig: plt.Figure | None = None
    axes: Sequence[plt.Axes] | None = None


class LiveTrainingPlot:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.state = LivePlotState()

    def update(
        self,
        *,
        epoch: int,
        history: list[dict[str, float]],
        original: torch.Tensor | None,
        reconstructed: torch.Tensor | None,
        title: str,
    ) -> None:
        if not self.enabled:
            return

        if self.state.fig is None or self.state.axes is None:
            fig, axes = plt.subplots(2, 1, figsize=(11, 8))
            self.state = LivePlotState(fig=fig, axes=axes)
        else:
            fig = self.state.fig
            axes = self.state.axes
            for axis in axes:
                axis.clear()

        loss_axis, waveform_axis = self.state.axes
        epochs = [entry["epoch"] for entry in history]
        train_losses = [entry["train_loss"] for entry in history]
        loss_axis.plot(epochs, train_losses, label="train loss", color="tab:blue")
        if any(entry.get("val_loss") is not None for entry in history):
            val_epochs = [entry["epoch"] for entry in history if entry.get("val_loss") is not None]
            val_losses = [entry["val_loss"] for entry in history if entry.get("val_loss") is not None]
            loss_axis.plot(val_epochs, val_losses, label="val loss", color="tab:orange")
        loss_axis.set_title(f"{title} - Training progress (epoch {epoch})")
        loss_axis.set_xlabel("Epoch")
        loss_axis.set_ylabel("Loss")
        loss_axis.grid(True, alpha=0.3)
        loss_axis.legend()

        waveform_axis.set_title("Reference waveform and reconstruction")
        waveform_axis.set_xlabel("Sample index")
        waveform_axis.set_ylabel("Amplitude")
        waveform_axis.grid(True, alpha=0.3)
        if original is not None:
            original_np = to_numpy(original.squeeze())
            waveform_axis.plot(original_np, label="original", color="tab:green", linewidth=1.5)
        if reconstructed is not None:
            reconstructed_np = to_numpy(reconstructed.squeeze())
            waveform_axis.plot(
                reconstructed_np,
                label="reconstructed",
                color="tab:red",
                linewidth=1.2,
                alpha=0.9,
            )
        waveform_axis.legend()

        clear_output(wait=True)
        display(fig)
        plt.close(fig)


def plot_reconstructions(
    model: Conv1dAutoencoder,
    waveforms: torch.Tensor,
    *,
    device: torch.device,
    max_items: int = 4,
) -> plt.Figure:
    model.eval()
    selected = waveforms[:max_items].to(device)
    with torch.no_grad():
        reconstructed, _ = model(selected)

    selected = selected.cpu()
    reconstructed = reconstructed.cpu()
    count = selected.shape[0]
    figure, axes = plt.subplots(count, 1, figsize=(11, 2.8 * count), sharex=True)
    if count == 1:
        axes = [axes]
    for index, axis in enumerate(axes):
        axis.plot(to_numpy(selected[index].squeeze()), label="original", color="tab:green")
        axis.plot(
            to_numpy(reconstructed[index].squeeze()),
            label="reconstructed",
            color="tab:red",
            alpha=0.9,
        )
        axis.set_title(f"Sample {index + 1}")
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    return figure


def plot_discriminator_timeline(
    *,
    type1_scores: torch.Tensor | np.ndarray,
    type2_scores: torch.Tensor | np.ndarray,
    outlier_scores: torch.Tensor | np.ndarray,
    title: str = "Discrimination timeline",
) -> plt.Figure:
    import numpy as np

    type1_scores = np.asarray(type1_scores, dtype=np.float32)
    type2_scores = np.asarray(type2_scores, dtype=np.float32)
    outlier_scores = np.asarray(outlier_scores, dtype=np.float32)

    x_axis = np.arange(len(type1_scores))
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.plot(x_axis, type1_scores, label="Type 1 score", color="tab:blue", linewidth=1.5)
    axis.plot(x_axis, type2_scores, label="Type 2 score", color="tab:orange", linewidth=1.5)
    axis.plot(x_axis, outlier_scores, label="Outlier score", color="tab:red", linewidth=1.5)
    axis.set_title(title)
    axis.set_xlabel("Waveform index")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.05)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right")
    figure.tight_layout()
    return figure


def plot_discriminator_timeline_interactive(
    *,
    type1_scores: torch.Tensor | np.ndarray,
    type2_scores: torch.Tensor | np.ndarray,
    outlier_scores: torch.Tensor | np.ndarray,
    title: str = "Discrimination timeline",
    downsample_step: int = 1,
    default_style: str = "markers",
) -> go.Figure:
    """Interactive Plotly timeline with zoom and range slider.

    The full scores stay available in the exported CSV; this function can
    downsample the visible view to keep the notebook responsive when the
    measured file contains many waveforms.
    """

    type1_scores = np.asarray(type1_scores, dtype=np.float32)
    type2_scores = np.asarray(type2_scores, dtype=np.float32)
    outlier_scores = np.asarray(outlier_scores, dtype=np.float32)

    if downsample_step < 1:
        raise ValueError("downsample_step must be >= 1")
    if default_style not in {"lines", "markers", "lines+markers"}:
        raise ValueError("default_style must be one of: lines, markers, lines+markers")

    x_axis = np.arange(len(type1_scores))
    if downsample_step > 1:
        x_axis = x_axis[::downsample_step]
        type1_scores = type1_scores[::downsample_step]
        type2_scores = type2_scores[::downsample_step]
        outlier_scores = outlier_scores[::downsample_step]

    figure = go.Figure()
    styles = ["lines", "markers", "lines+markers"]
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    score_sets = [
        ("Type 1 score", type1_scores),
        ("Type 2 score", type2_scores),
        ("Outlier score", outlier_scores),
    ]

    for style_index, style in enumerate(styles):
        for score_index, (label, scores) in enumerate(score_sets):
            figure.add_trace(
                go.Scattergl(
                    x=x_axis,
                    y=scores,
                    mode=style,
                    name=f"{label} ({style})",
                    line=dict(color=colors[score_index], width=1.6),
                    marker=dict(color=colors[score_index], size=4),
                    visible=style == default_style,
                    legendgroup=label,
                    showlegend=style == default_style,
                )
            )

    visibility_map = {
        style: [trace_style == style for trace_style in styles for _ in score_sets]
        for style in styles
    }

    buttons = [
        dict(
            label="Markers",
            method="update",
            args=[
                {
                    "visible": visibility_map["markers"],
                },
                {"title": f"{title} — markers"},
            ],
        ),
        dict(
            label="Lines",
            method="update",
            args=[
                {
                    "visible": visibility_map["lines"],
                },
                {"title": f"{title} — lines"},
            ],
        ),
        dict(
            label="Lines + markers",
            method="update",
            args=[
                {
                    "visible": visibility_map["lines+markers"],
                },
                {"title": f"{title} — lines + markers"},
            ],
        ),
    ]

    figure.update_layout(
        title=f"{title} — use the menu to change style",
        xaxis_title="Waveform index",
        yaxis_title="Score",
        hovermode="x unified",
        template="plotly_white",
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.0,
                y=1.18,
                xanchor="left",
                yanchor="top",
                buttons=buttons,
                bgcolor="white",
                bordercolor="#cccccc",
                borderwidth=1,
            )
        ],
    )
    figure.update_yaxes(range=[0.0, 1.05])
    figure.update_xaxes(
        rangeslider=dict(visible=True),
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
    )
    figure.update_layout(
        annotations=[
            dict(
                text="Plot style",
                x=0.0,
                y=1.22,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12),
            )
        ]
    )
    return figure
