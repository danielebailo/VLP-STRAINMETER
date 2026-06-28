#!/usr/bin/env python3

"""
VLP-STRAINMETER - Waveform Discriminator Implementation
"""

import os
import sys
from pathlib import Path

def create_project_structure():
    """Create the project structure for the current strainmeter training stack."""

    repo_root = Path(__file__).resolve().parent

    dirs = [
        "config",
        'src',
        'data',
        'models',
        'outputs',
        'notebooks',
        'tests',
    ]

    for dir_path in dirs:
        os.makedirs(repo_root / dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    init_file = repo_root / 'src' / '__init__.py'
    if not os.path.exists(init_file):
        with open(init_file, 'w', encoding='utf-8') as handle:
            handle.write('"""Core package for the strainmeter autoencoder stack."""\n')

    print("Project structure created successfully")

if __name__ == "__main__":
    create_project_structure()
