"""Shared generation and scoring utilities for GRPO evaluation."""

from __future__ import annotations

import re
import time
from typing import Any

import torch
from math_verify import LatexExtractionConfig, parse, verify
from peft import PeftModel
from transformers import AutoModelForCausalLM

from rewards import format_reward

_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_EXTRACTION = [LatexExtractionConfig()]
_GSM8K_HASH = re.compile(r"####\s*(.+)", re.DOTALL)
_NUM = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def use_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def completion_format_score(text: str) -> float:
    return format_reward(completions=[[{"role": "assistant", "content": text}]])[0]


def answer_text_for_eval(completion: str) -> str:
    match = _ANSWER_BLOCK.search(completion)
    if match:
        return match.group(1).strip()
    return completion.strip()


def _verify_parsed(gold_text: str, pred_text: str) -> tuple[bool, bool]:
    """Returns (correct, parse_failed)."""
    gn = normalize_final_number(gold_text)
    pn = normalize_final_number(pred_text)
    if gn is not None and pn is not None:
        return abs(gn - pn) < 1e-6, False

    gold_parsed = parse(
        gold_text,
        extraction_mode="first_match",
        extraction_config=_EXTRACTION,
    )
    pred_parsed = parse(
        pred_text,
        extraction_mode="first_match",
        extraction_config=_EXTRACTION,
    )
    if len(gold_parsed) == 0 or len(pred_parsed) == 0:
        return False, True
    try:
        return bool(verify(pred_parsed, gold_parsed)), False
    except Exception:
        return False, False


def math_accuracy_and_parse(completion: str, gold: str) -> tuple[bool, bool]:
    pred_text = answer_text_for_eval(completion)
    return _verify_parsed(gold, pred_text)


def gsm8k_gold_answer(answer_field: str) -> str:
    m = _GSM8K_HASH.search(answer_field)
    if m:
        return m.group(1).strip()
    return answer_field.strip()


def gsm8k_pred_text(completion: str) -> str:
    text = answer_text_for_eval(completion)
    m = _GSM8K_HASH.search(text)
    if m:
        return m.group(1).strip()
    return text


def normalize_final_number(text: str) -> float | None:
    cleaned = text.replace(",", "").strip()
    matches = _NUM.findall(cleaned)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def gsm8k_accuracy_and_parse(completion: str, gold_answer_field: str) -> tuple[bool, bool]:
    gold = gsm8k_gold_answer(gold_answer_field)
    pred = gsm8k_pred_text(completion)
    return _verify_parsed(gold, pred)


def batched_generate(
    model,
    tokenizer,
    prompts: list[list[dict]],
    *,
    max_new_tokens: int,
    batch_size: int,
) -> tuple[list[str], list[int], list[bool]]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    completions: list[str] = []
    token_counts: list[int] = []
    clipped: list[bool] = []

    for start in range(0, len(prompts), batch_size):
        batch_msgs = prompts[start : start + batch_size]
        prompt_texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_msgs
        ]
        inputs = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for row in range(out.shape[0]):
            full = out[row]
            attn = inputs["attention_mask"][row]
            plen = int(attn.sum().item())
            new_ids = full[plen:]
            if tokenizer.eos_token_id is not None:
                eos_pos = (new_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_pos) > 0:
                    new_ids = new_ids[: int(eos_pos[0].item()) + 1]
            n_tok = int(new_ids.numel())
            hit_cap = n_tok >= max_new_tokens
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            completions.append(text)
            token_counts.append(n_tok)
            clipped.append(hit_cap)

    return completions, token_counts, clipped


def load_base_model(model_id: str, device: torch.device):
    dtype = torch.bfloat16 if use_bf16() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    return model.to(device).eval()


def load_adapter_model(model_id: str, adapter_dir: str, device: torch.device):
    base = load_base_model(model_id, device)
    return PeftModel.from_pretrained(base, adapter_dir).eval()


def aggregate_run_metrics(
    *,
    accuracies: list[bool],
    formats: list[float],
    parse_failed: list[bool],
    gen_tokens: list[int],
    clipped: list[bool],
    runtime_sec: float,
) -> dict[str, float]:
    n = len(accuracies) or 1
    tokens_sorted = sorted(gen_tokens)
    mid = len(tokens_sorted) // 2
    strict = [a and f >= 1.0 for a, f in zip(accuracies, formats)]
    return {
        "math_accuracy": sum(float(x) for x in accuracies) / n,
        "format_compliance": sum(formats) / n,
        "strict_success": sum(float(x) for x in strict) / n,
        "parse_failure_rate": sum(float(x) for x in parse_failed) / n,
        "avg_generated_tokens": sum(gen_tokens) / n,
        "median_generated_tokens": float(tokens_sorted[mid]) if tokens_sorted else 0.0,
        "fraction_clipped_at_max": sum(float(x) for x in clipped) / n,
        "runtime_sec": runtime_sec,
        "examples_per_sec": len(accuracies) / runtime_sec if runtime_sec > 0 else 0.0,
    }


def print_base_grpo_table(summary: dict[str, Any], title: str = "Evaluation comparison") -> None:
    b, g = summary["base"], summary["grpo"]

    def pct(x: float) -> str:
        return f"{100.0 * x:.1f}%"

    def num(x: float) -> str:
        return f"{x:.1f}"

    rows = [
        ("Math accuracy", pct(b["math_accuracy"]), pct(g["math_accuracy"])),
        ("Format compliance", pct(b["format_compliance"]), pct(g["format_compliance"])),
        ("Strict correct+format", pct(b["strict_success"]), pct(g["strict_success"])),
        ("Parse failure rate", pct(b["parse_failure_rate"]), pct(g["parse_failure_rate"])),
        ("Avg generated tokens", num(b["avg_generated_tokens"]), num(g["avg_generated_tokens"])),
        ("Median generated tokens", num(b["median_generated_tokens"]), num(g["median_generated_tokens"])),
        ("Clipped at 256", pct(b["fraction_clipped_at_max"]), pct(g["fraction_clipped_at_max"])),
        ("Runtime (sec)", num(b["runtime_sec"]), num(g["runtime_sec"])),
        ("Examples/sec", f"{b['examples_per_sec']:.2f}", f"{g['examples_per_sec']:.2f}"),
    ]
    print(f"\n=== {title} ===")
    print(f"{'Metric':<26} {'Base':>10} {'GRPO':>10}")
    print("-" * 48)
    for label, base_v, grpo_v in rows:
        print(f"{label:<26} {base_v:>10} {grpo_v:>10}")


def timed_generate(
    model,
    tokenizer,
    prompts: list[list[dict]],
    *,
    max_new_tokens: int,
    batch_size: int,
) -> tuple[list[str], list[int], list[bool], float]:
    t0 = time.perf_counter()
    completions, tokens, clipped = batched_generate(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    return completions, tokens, clipped, time.perf_counter() - t0
