"""GRPO reward functions: format compliance and math accuracy."""

from __future__ import annotations

import re

from math_verify import LatexExtractionConfig, parse, verify

# re.DOTALL: '.' matches newlines inside thinking/answer blocks.
_FORMAT_PATTERN = re.compile(
    r"^<think>.*?</think>\s*<answer>.*?</answer>\s*$",
    re.DOTALL,
)

_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _completion_text(completion) -> str:
    """TRL conversational GRPO passes completions as message lists, not plain strings."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return completion[0].get("content", "")
    return str(completion)


def _extract_answer_for_scoring(completion_text: str) -> str:
    """Score only the final answer region, not intermediate math in thinking."""
    match = _ANSWER_BLOCK.search(completion_text)
    if match:
        return match.group(1).strip()
    return completion_text


def format_reward(completions, **kwargs) -> list[float]:
    """
    Reward 1.0 if the completion matches <think>...</think><answer>...</answer>.

    Used by GRPOTrainer together with other reward_funcs; scalar rewards drive group-relative advantages.
    """
    del kwargs
    texts = [_completion_text(c) for c in completions]
    return [1.0 if _FORMAT_PATTERN.match(text.strip()) else 0.0 for text in texts]


def accuracy_reward(completions, **kwargs) -> list[float]:
    """
    Reward 1.0 when math_verify agrees with the dataset solution column.

    Generated completions are parsed from <answer>...</answer> only so thinking-step LaTeX
    does not override the final result. Gold solutions use the full reference text (boxed final answer).
    """
    solutions = kwargs.get("solution")
    if solutions is None:
        raise ValueError("accuracy_reward requires a 'solution' column in the training dataset.")

    texts = [_completion_text(c) for c in completions]
    rewards: list[float] = []
    extraction = [LatexExtractionConfig()]

    for content, solution in zip(texts, solutions):
        gold_parsed = parse(
            solution,
            extraction_mode="first_match",
            extraction_config=extraction,
        )
        answer_text = _extract_answer_for_scoring(content)
        answer_parsed = parse(
            answer_text,
            extraction_mode="first_match",
            extraction_config=extraction,
        )
        if len(gold_parsed) == 0:
            rewards.append(1.0)
            continue
        try:
            rewards.append(float(verify(answer_parsed, gold_parsed)))
        except Exception:
            rewards.append(0.0)
    return rewards
