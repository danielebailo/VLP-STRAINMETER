#!/usr/bin/env python3
"""Train a 1D CNN classifier for VLP waveform discrimination."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.classifier import (
    VLPClassifier,
    evaluate_model,
    load_and_split,
    train_classifier,
    plot_confusion_matrix,
    plot_roc_curve,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VLP waveform CNN classifier")
    parser.add_argument("--type1", default="data/processed/Waveforms_type1_extended_128samples.txt")
    parser.add_argument("--type2", default="data/processed/Waveforms_type2_extended_128samples.txt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save", default="outputs/cnn_classifier.pt")
    args = parser.parse_args()

    device_str = args.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    # Load data
    print(f"\nLoading data...")
    print(f"  Type1: {args.type1}")
    print(f"  Type2: {args.type2}")
    train_loader, val_loader, test_loader = load_and_split(
        args.type1, args.type2, waveform_length=128, seed=42
    )
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Train
    print(f"\nTraining CNN classifier ({args.epochs} epochs, lr={args.lr})...")
    model, history = train_classifier(
        train_loader, val_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        save_path=args.save,
        plot=True,
    )

    # Evaluate on test set
    print(f"\n=== Evaluating on test set ({len(test_loader.dataset)} samples) ===")
    results = evaluate_model(model, test_loader, device=device)

    # Plots
    plot_confusion_matrix(results["confusion_matrix"])
    plot_roc_curve(results["labels"], results["probabilities"])

    print(f"\nDone! Model saved to {args.save}")


if __name__ == "__main__":
    main()
