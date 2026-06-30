"""Shared utilities: seeding, device selection, lightweight logging."""
from __future__ import annotations

import importlib
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch


def ensure_deps(pip_map: dict[str, str], log=None) -> None:
    """Best-effort runtime install for real-model experiments.

    ``pip_map`` maps import-name -> pip-name. Tries to import each; pip-installs the
    missing ones (Kaggle usually preinstalls them, but the watch loop can't be restarted
    mid-session, so a runtime install keeps the autonomous loop unblocked).
    """
    for import_name, pip_name in pip_map.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            msg = f"installing missing dep '{pip_name}' (import '{import_name}')"
            (log or print)(msg)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=False)


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    """Pick a device. 'auto' uses CUDA when present, else CPU."""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" or (prefer == "auto" and torch.cuda.is_available()):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


class Logger:
    """Tee-style logger: prints to stdout and appends to a file."""

    def __init__(self, path: str | os.PathLike | None = None):
        self.path = str(path) if path is not None else None
        self._t0 = time.time()

    def __call__(self, msg: str) -> None:
        line = f"[{time.time() - self._t0:8.1f}s] {msg}"
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode(), flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


@dataclass
class RunContext:
    """Bundle passed to every experiment's run() function."""

    cfg: dict
    outdir: str
    log: Logger
    device: torch.device
