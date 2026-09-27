#!/usr/bin/env python3
"""Compare base Qwen2-0.5B-Instruct vs GRPO LoRA on held-out NuminaMath-TIR test examples."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from math_verify import LatexExtractionConfig, parse, verify
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import DATASET_ID, SYSTEM_PROMPT, make_conversation
from rewards import format_reward

DEFAULT_BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"
DEFAULT_ADAPTER_DIR = "checkpoints/Qwen2-0.5B-GRPO"
DEFAULT_OUT_DIR = "evaluation"
_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_EXTRACTION = [LatexExtractionConfig()]


def use_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def load_eval_examples(n: int) -> list[dict]:
    """First n rows from the NuminaMath-TIR test split (disjoint from train[:5%] used in training)."""
    ds = load_dataset(DATASET_ID, split=f"test[:{n}]")
    n = min(n, len(ds))
    rows: list[dict] = []
    for i in range(n):
        ex = ds[i]
        conv = make_conversation(ex)
        rows.append(
            {
                "index": i,
                "problem": ex["problem"],
                "gold_solution": ex["solution"],
                "prompt": conv["prompt"],
            }
        )
    return rows


def _completion_format_score(text: str) -> float:
    return format_reward(completions=[[{"role": "assistant", "content": text}]])[0]


def _answer_text_for_eval(completion: str) -> tuple[str, bool]:
    """Return text to parse and whether an explicit <answer> block was used."""
    match = _ANSWER_BLOCK.search(completion)
    if match:
        return match.group(1).strip(), True
    return completion.strip(), False


def math_accuracy_and_parse(completion: str, gold: str) -> tuple[bool, bool]:
    """
    Eval accuracy with answer-block-first parsing, else full completion.
    Returns (is_correct, parse_failed).
    """
    gold_parsed = parse(
        gold,
        extraction_mode="first_match",
        extraction_config=_EXTRACTION,
    )
    if len(gold_parsed) == 0:
        return True, False

    answer_text, _ = _answer_text_for_eval(completion)
    answer_parsed = parse(
        answer_text,
        extraction_mode="first_match",
        extraction_config=_EXTRACTION,
    )
    if len(answer_parsed) == 0:
        return False, True
    try:
        return bool(verify(answer_parsed, gold_parsed)), False
    except Exception:
        return False, False


@dataclass
class ExampleResult:
    index: int
    problem: str
    gold_solution: str
    base_completion: str
    grpo_completion: str
    base_accuracy: bool
    grpo_accuracy: bool
    base_format: float
    grpo_format: float
    base_strict: bool
    grpo_strict: bool
    base_parse_failed: bool
    grpo_parse_failed: bool
    base_gen_tokens: int
    grpo_gen_tokens: int
    base_clipped: bool
    grpo_clipped: bool


def batched_generate(
    model,
    tokenizer,
    prompts: list[list[dict]],
    *,
    max_new_tokens: int,
    batch_size: int,
) -> tuple[list[str], list[int], list[bool]]:
    """Greedy batched generation; returns completions, token counts, clipped flags."""
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
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for row in range(out.shape[0]):
            new_ids = out[row, prompt_len:]
            # Trim padding artifacts on left-padded batches (new tokens start after prompt width per row)
            # Re-decode from full row: find where prompt ends per sequence
            full = out[row]
            # Count non-pad prompt tokens for this row
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


def load_grpo_model(model_id: str, adapter_dir: str, device: torch.device):
    base = load_base_model(model_id, device)
    return PeftModel.from_pretrained(base, adapter_dir).eval()


def summarize(
    results: list[ExampleResult],
    base_runtime: float,
    grpo_runtime: float,
    *,
    base_model_id: str,
    adapter_dir: str,
    max_new_tokens: int,
) -> dict:
    n = len(results)

    def agg(name: str, vals: list) -> float:
        return sum(vals) / n if n else 0.0

    base_tokens = [r.base_gen_tokens for r in results]
    grpo_tokens = [r.grpo_gen_tokens for r in results]
    base_tokens_sorted = sorted(base_tokens)
    grpo_tokens_sorted = sorted(grpo_tokens)
    mid = n // 2

    total_runtime = base_runtime + grpo_runtime

    return {
        "n_examples": n,
        "base_model": base_model_id,
        "grpo_adapter_dir": adapter_dir,
        "max_new_tokens": max_new_tokens,
        "base": {
            "math_accuracy": agg("acc", [float(r.base_accuracy) for r in results]),
            "format_compliance": agg("fmt", [r.base_format for r in results]),
            "strict_success": agg("strict", [float(r.base_strict) for r in results]),
            "parse_failure_rate": agg("pf", [float(r.base_parse_failed) for r in results]),
            "avg_generated_tokens": agg("tok", base_tokens),
            "median_generated_tokens": base_tokens_sorted[mid] if n else 0,
            "fraction_clipped_at_max": agg("clip", [float(r.base_clipped) for r in results]),
            "runtime_sec": base_runtime,
            "examples_per_sec": n / base_runtime if base_runtime > 0 else 0.0,
        },
        "grpo": {
            "math_accuracy": agg("acc", [float(r.grpo_accuracy) for r in results]),
            "format_compliance": agg("fmt", [r.grpo_format for r in results]),
            "strict_success": agg("strict", [float(r.grpo_strict) for r in results]),
            "parse_failure_rate": agg("pf", [float(r.grpo_parse_failed) for r in results]),
            "avg_generated_tokens": agg("tok", grpo_tokens),
            "median_generated_tokens": grpo_tokens_sorted[mid] if n else 0,
            "fraction_clipped_at_max": agg("clip", [float(r.grpo_clipped) for r in results]),
            "runtime_sec": grpo_runtime,
            "examples_per_sec": n / grpo_runtime if grpo_runtime > 0 else 0.0,
        },
        "total_runtime_sec": total_runtime,
        "system_prompt": SYSTEM_PROMPT,
    }


def print_comparison_table(summary: dict) -> None:
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
    print("\n=== Evaluation comparison ===")
    print(f"{'Metric':<26} {'Base':>10} {'GRPO':>10}")
    print("-" * 48)
    for label, base_v, grpo_v in rows:
        print(f"{label:<26} {base_v:>10} {grpo_v:>10}")
    print(f"\nTotal wall time (both models): {summary['total_runtime_sec']:.1f}s")


def pick_representative(results: list[ExampleResult]) -> dict[str, list[ExampleResult]]:
    grpo_better_acc = [r for r in results if not r.base_accuracy and r.grpo_accuracy]
    grpo_better_fmt = [
        r
        for r in results
        if r.grpo_format > r.base_format and r not in grpo_better_acc
    ]
    grpo_improves = (grpo_better_acc + grpo_better_fmt)[:2]
    both_ok = [r for r in results if r.base_accuracy and r.grpo_accuracy]
    both_bad = [r for r in results if not r.base_accuracy and not r.grpo_accuracy]
    grpo_worse = [r for r in results if r.base_accuracy and not r.grpo_accuracy]

    return {
        "grpo_improves": grpo_improves,
        "both_correct": both_ok[:1],
        "both_wrong": both_bad[:1],
        "grpo_worse": grpo_worse[:1],
    }


def print_representative(reps: dict[str, list[ExampleResult]]) -> None:
    print("\n=== Representative examples ===")

    def show(title: str, r: ExampleResult) -> None:
        print(f"\n--- {title} (index={r.index}) ---")
        print("Problem:", r.problem[:400], "..." if len(r.problem) > 400 else "")
        print(f"Base:  acc={r.base_accuracy} fmt={r.base_format} parse_fail={r.base_parse_failed}")
        print(f"GRPO:  acc={r.grpo_accuracy} fmt={r.grpo_format} parse_fail={r.grpo_parse_failed}")
        print("Base completion (truncated):", r.base_completion[:600].replace("\n", " "))
        print("GRPO completion (truncated):", r.grpo_completion[:600].replace("\n", " "))

    for i, r in enumerate(reps["grpo_improves"], start=1):
        kind = "accuracy" if r.grpo_accuracy and not r.base_accuracy else "format"
        show(f"GRPO improves over base ({kind})", r)
    if len(reps["grpo_improves"]) < 2:
        print(f"\n(Only {len(reps['grpo_improves'])} clear improvement case(s) in this sample.)")
    if reps["both_correct"]:
        show("Both correct", reps["both_correct"][0])
    if reps["both_wrong"]:
        show("Both wrong", reps["both_wrong"][0])
    if reps["grpo_worse"]:
        show("GRPO worse than base", reps["grpo_worse"][0])
    elif not reps["grpo_worse"]:
        print("\n(No case where GRPO is wrong and base is correct.)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-examples", type=int, default=100)
    p.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL)
    p.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required; run on a Perlmutter GPU compute node.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_eval_examples(args.num_examples)
    prompts = [ex["prompt"] for ex in examples]
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Evaluating {len(examples)} held-out test examples on {torch.cuda.get_device_name(0)}")
    print(f"Base: {args.base_model_id}")
    print(f"GRPO adapter: {args.adapter_dir}")

    t0 = time.perf_counter()
    base_model = load_base_model(args.base_model_id, device)
    base_completions, base_tokens, base_clipped = batched_generate(
        base_model,
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    del base_model
    torch.cuda.empty_cache()
    base_runtime = time.perf_counter() - t0

    t1 = time.perf_counter()
    grpo_model = load_grpo_model(args.base_model_id, args.adapter_dir, device)
    grpo_completions, grpo_tokens, grpo_clipped = batched_generate(
        grpo_model,
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    del grpo_model
    torch.cuda.empty_cache()
    grpo_runtime = time.perf_counter() - t1

    results: list[ExampleResult] = []
    for ex, bc, gc, bt, gt, bclip, gclip in zip(
        examples,
        base_completions,
        grpo_completions,
        base_tokens,
        grpo_tokens,
        base_clipped,
        grpo_clipped,
    ):
        b_acc, b_pf = math_accuracy_and_parse(bc, ex["gold_solution"])
        g_acc, g_pf = math_accuracy_and_parse(gc, ex["gold_solution"])
        b_fmt = _completion_format_score(bc)
        g_fmt = _completion_format_score(gc)
        results.append(
            ExampleResult(
                index=ex["index"],
                problem=ex["problem"],
                gold_solution=ex["gold_solution"],
                base_completion=bc,
                grpo_completion=gc,
                base_accuracy=b_acc,
                grpo_accuracy=g_acc,
                base_format=b_fmt,
                grpo_format=g_fmt,
                base_strict=b_acc and b_fmt >= 1.0,
                grpo_strict=g_acc and g_fmt >= 1.0,
                base_parse_failed=b_pf,
                grpo_parse_failed=g_pf,
                base_gen_tokens=bt,
                grpo_gen_tokens=gt,
                base_clipped=bclip,
                grpo_clipped=gclip,
            )
        )

    per_example_path = out_dir / "per_example.jsonl"
    with per_example_path.open("w") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    summary = summarize(
        results,
        base_runtime,
        grpo_runtime,
        base_model_id=args.base_model_id,
        adapter_dir=args.adapter_dir,
        max_new_tokens=args.max_new_tokens,
    )
    summary["base_model_id"] = args.base_model_id
    summary["adapter_dir"] = args.adapter_dir
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print_comparison_table(summary)
    print_representative(pick_representative(results))
    print(f"\nWrote {per_example_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
