#!/usr/bin/env python3
"""GSM8K test-set comparison: base vs final GRPO adapter."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from data import SYSTEM_PROMPT
from eval_utils import (
    aggregate_run_metrics,
    completion_format_score,
    gsm8k_accuracy_and_parse,
    load_adapter_model,
    load_base_model,
    print_base_grpo_table,
    timed_generate,
)

GSM8K_ID = "openai/gsm8k"
DEFAULT_BASE = "Qwen/Qwen2-0.5B-Instruct"
DEFAULT_ADAPTER = "checkpoints/Qwen2-0.5B-GRPO"


def load_gsm8k_examples(n: int) -> list[dict]:
    ds = load_dataset(GSM8K_ID, "main", split=f"test[:{n}]")
    n = min(n, len(ds))
    rows = []
    for i in range(n):
        q = ds[i]["question"]
        rows.append(
            {
                "index": i,
                "question": q,
                "gold_answer": ds[i]["answer"],
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                ],
            }
        )
    return rows


@dataclass
class Gsm8kRow:
    index: int
    question: str
    gold_answer: str
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-examples", type=int, default=500)
    p.add_argument("--base-model-id", default=DEFAULT_BASE)
    p.add_argument("--adapter-dir", default=DEFAULT_ADAPTER)
    p.add_argument("--output-dir", default="evaluation/gsm8k")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    examples = load_gsm8k_examples(args.num_examples)
    prompts = [e["prompt"] for e in examples]
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"GSM8K eval: {len(examples)} test examples on {torch.cuda.get_device_name(0)}")

    base_model = load_base_model(args.base_model_id, device)
    bc, bt, bclip, base_rt = timed_generate(
        base_model, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
    )
    del base_model
    torch.cuda.empty_cache()

    grpo_model = load_adapter_model(args.base_model_id, args.adapter_dir, device)
    gc, gt, gclip, grpo_rt = timed_generate(
        grpo_model, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
    )
    del grpo_model
    torch.cuda.empty_cache()

    rows: list[Gsm8kRow] = []
    for ex, b_comp, g_comp, b_tok, g_tok, b_cl, g_cl in zip(
        examples, bc, gc, bt, gt, bclip, gclip
    ):
        b_acc, b_pf = gsm8k_accuracy_and_parse(b_comp, ex["gold_answer"])
        g_acc, g_pf = gsm8k_accuracy_and_parse(g_comp, ex["gold_answer"])
        b_fmt = completion_format_score(b_comp)
        g_fmt = completion_format_score(g_comp)
        rows.append(
            Gsm8kRow(
                index=ex["index"],
                question=ex["question"],
                gold_answer=ex["gold_answer"],
                base_completion=b_comp,
                grpo_completion=g_comp,
                base_accuracy=b_acc,
                grpo_accuracy=g_acc,
                base_format=b_fmt,
                grpo_format=g_fmt,
                base_strict=b_acc and b_fmt >= 1.0,
                grpo_strict=g_acc and g_fmt >= 1.0,
                base_parse_failed=b_pf,
                grpo_parse_failed=g_pf,
                base_gen_tokens=b_tok,
                grpo_gen_tokens=g_tok,
                base_clipped=b_cl,
                grpo_clipped=g_cl,
            )
        )

    with (out / "per_example.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    summary = {
        "benchmark": "gsm8k",
        "split": "test",
        "n_examples": len(rows),
        "base_model_id": args.base_model_id,
        "adapter_dir": args.adapter_dir,
        "max_new_tokens": args.max_new_tokens,
        "system_prompt": SYSTEM_PROMPT,
        "base": aggregate_run_metrics(
            accuracies=[r.base_accuracy for r in rows],
            formats=[r.base_format for r in rows],
            parse_failed=[r.base_parse_failed for r in rows],
            gen_tokens=[r.base_gen_tokens for r in rows],
            clipped=[r.base_clipped for r in rows],
            runtime_sec=base_rt,
        ),
        "grpo": aggregate_run_metrics(
            accuracies=[r.grpo_accuracy for r in rows],
            formats=[r.grpo_format for r in rows],
            parse_failed=[r.grpo_parse_failed for r in rows],
            gen_tokens=[r.grpo_gen_tokens for r in rows],
            clipped=[r.grpo_clipped for r in rows],
            runtime_sec=grpo_rt,
        ),
        "total_runtime_sec": base_rt + grpo_rt,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print_base_grpo_table(summary, title="GSM8K Base vs GRPO")
    print(f"\nWrote {out / 'per_example.jsonl'}")
    print(f"Wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
