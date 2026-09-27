#!/usr/bin/env python3
"""Synthetic tests for format_reward (including multiline)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rewards import format_reward


def conv(text: str) -> list:
    return [[{"role": "assistant", "content": text}]]


def score(text: str) -> float:
    return format_reward(completions=conv(text))[0]


def main() -> int:
    cases = [
        (
            "one-line valid",
            "<think>add</think><answer>4</answer>",
            1.0,
        ),
        (
            "multiline valid",
            "<think>\nline 1\nline 2\n</think>\n"
            "<answer>\n\\frac{63}{400}\n</answer>",
            1.0,
        ),
        (
            "missing closing think tag",
            "<think>oops<answer>4</answer>",
            0.0,
        ),
        (
            "missing answer tag",
            "<think>done</think>",
            0.0,
        ),
        (
            "extra text outside tags",
            "prefix<think>t</think><answer>4</answer>",
            0.0,
        ),
    ]
    ok = True
    for name, text, expected in cases:
        got = score(text)
        print(f"{name}: {got} (expected {expected})")
        ok &= got == expected
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
