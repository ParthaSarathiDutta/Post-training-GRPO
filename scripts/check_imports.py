#!/usr/bin/env python3
"""Lightweight import smoke test (no model load, no dataset download)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def version(name: str) -> str:
    mod = importlib.import_module(name)
    return getattr(mod, "__version__", "unknown")


def main() -> int:
    for pkg in ("torch", "transformers", "trl", "peft", "datasets", "math_verify"):
        print(f"{pkg}: {version(pkg)}")

    for mod in ("data", "rewards", "train", "inference"):
        importlib.import_module(mod)
        print(f"import ok: {mod}")

    print("All imports succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
