#!/usr/bin/env python3
"""CPU-only sanity checks for format_reward and accuracy_reward."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rewards import accuracy_reward, format_reward


def conv(text: str) -> list:
    return [[{"role": "assistant", "content": text}]]


def main() -> int:
    ok = True

    good_fmt = (
        "<think>add 2 and 2</think>"
        "<answer>4</answer>"
    )
    bad_fmt = "<answer>4</answer> without thinking tags"
    fr = format_reward(completions=conv(good_fmt) + conv(bad_fmt))
    print("format_reward good:", fr[0], "expected 1.0")
    print("format_reward bad:", fr[1], "expected 0.0")
    ok &= fr[0] == 1.0 and fr[1] == 0.0

    # math_verify extracts LaTeX from text; plain "4" in <answer> does not parse.
    gold = r"Final answer: \boxed{\frac{63}{400}}"
    right = (
        "<think>binomial expansion</think>"
        r"<answer>\boxed{\frac{63}{400}}</answer>"
    )
    # Wrong math in thinking, correct answer tag only (accuracy uses answer block).
    think_trap = (
        "<think>\n\\frac{1}{2}\n</think>"
        r"<answer>\boxed{\frac{63}{400}}</answer>"
    )
    wrong = (
        "<think>binomial expansion</think>"
        r"<answer>\boxed{\frac{1}{2}}</answer>"
    )
    ar = accuracy_reward(
        completions=conv(right) + conv(wrong) + conv(think_trap),
        solution=[gold, gold, gold],
    )
    print("accuracy_reward right:", ar[0], "expected 1.0")
    print("accuracy_reward wrong:", ar[1], "expected 0.0")
    print("accuracy_reward think_trap:", ar[2], "expected 1.0")
    ok &= ar[0] == 1.0 and ar[1] == 0.0 and ar[2] == 1.0

    if ok:
        print("All synthetic reward checks passed.")
        return 0
    print("Synthetic reward checks FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
