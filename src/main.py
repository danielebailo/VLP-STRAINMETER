#!/usr/bin/env python3

"""Compatibility wrapper for the old entrypoint."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main_train import main


if __name__ == "__main__":
    main()
