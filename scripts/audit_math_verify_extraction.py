#!/usr/bin/env python3
"""Audit what math_verify first_match extracts from gold vs generated completions."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import load_smoke_train
from math_verify import LatexExtractionConfig, parse, verify
from rewards import _extract_answer_for_scoring, accuracy_reward


def main() -> None:
    ext = [LatexExtractionConfig()]
    row = load_smoke_train(n=1)[0]
    solution = row["solution"]

    gold_full = parse(solution, extraction_mode="first_match", extraction_config=ext)
    print("=== Gold NuminaMath solution (full text) ===")
    print("parsed (first_match):", gold_full)
    tail = solution[-120:]
    print("solution tail:", tail)

    wrong_think_first = (
        "<think>\n\\frac{1}{2}\n\\frac{1}{3}\n</think>\n"
        r"<answer>\boxed{\frac{63}{400}}</answer>"
    )
    full_parse = parse(wrong_think_first, extraction_mode="first_match", extraction_config=ext)
    answer_only = _extract_answer_for_scoring(wrong_think_first)
    answer_parse = parse(answer_only, extraction_mode="first_match", extraction_config=ext)

    print("\n=== Generated-style completion (math in thinking + correct answer tag) ===")
    print("full completion parse:", full_parse)
    print("answer block text:", repr(answer_only))
    print("answer-only parse:", answer_parse)
    print(
        "verify full vs gold:",
        float(verify(full_parse, gold_full)) if gold_full and full_parse else "n/a",
    )
    print(
        "verify answer-only vs gold:",
        float(verify(answer_parse, gold_full)) if gold_full and answer_parse else "n/a",
    )
    ar = accuracy_reward(completions=[[{"content": wrong_think_first}]], solution=[solution])
    print("accuracy_reward (after answer-tag extraction):", ar[0])


if __name__ == "__main__":
    main()
