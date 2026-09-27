#!/usr/bin/env python3
"""NuminaMath held-out eval across training checkpoints + learning-curve plot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

from data import SYSTEM_PROMPT
from evaluate import load_eval_examples
from eval_utils import (
    aggregate_run_metrics,
    completion_format_score,
    load_adapter_model,
    load_base_model,
    math_accuracy_and_parse,
    timed_generate,
)

DEFAULT_BASE = "Qwen/Qwen2-0.5B-Instruct"
CHECKPOINT_ROOT = Path("checkpoints/Qwen2-0.5B-GRPO")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-examples", type=int, default=100)
    p.add_argument("--base-model-id", default=DEFAULT_BASE)
    p.add_argument("--checkpoint-root", default=str(CHECKPOINT_ROOT))
    p.add_argument("--steps", default="0,200,500,905")
    p.add_argument("--output-dir", default="evaluation/checkpoints")
    p.add_argument("--plot-path", default="evaluation/plots/checkpoint_learning_curve.png")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def checkpoint_specs(steps: str, root: Path) -> list[tuple[str, int, str | None]]:
    specs: list[tuple[str, int, str | None]] = []
    for part in steps.split(","):
        part = part.strip()
        step = int(part)
        if step == 0:
            specs.append(("base", 0, None))
        else:
            specs.append((f"checkpoint-{step}", step, str(root / f"checkpoint-{step}")))
    return specs


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_path = Path(args.plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    examples = load_eval_examples(args.num_examples)
    prompts = [ex["prompt"] for ex in examples]
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    specs = checkpoint_specs(args.steps, Path(args.checkpoint_root))
    curve_rows: list[dict] = []

    for label, step, adapter_dir in specs:
        print(f"\n--- Evaluating {label} (step={step}) on {len(examples)} NuminaMath examples ---")
        if adapter_dir is None:
            model = load_base_model(args.base_model_id, device)
        else:
            if not Path(adapter_dir).is_dir():
                raise FileNotFoundError(f"Missing checkpoint directory: {adapter_dir}")
            model = load_adapter_model(args.base_model_id, adapter_dir, device)

        completions, tokens, clipped, runtime = timed_generate(
            model,
            tokenizer,
            prompts,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        del model
        torch.cuda.empty_cache()

        accs, fmts, pfs = [], [], []
        for ex, comp in zip(examples, completions):
            a, pf = math_accuracy_and_parse(comp, ex["gold_solution"])
            accs.append(a)
            pfs.append(pf)
            fmts.append(completion_format_score(comp))

        metrics = aggregate_run_metrics(
            accuracies=accs,
            formats=fmts,
            parse_failed=pfs,
            gen_tokens=tokens,
            clipped=clipped,
            runtime_sec=runtime,
        )
        row = {
            "label": label,
            "step": step,
            "adapter_dir": adapter_dir,
            **metrics,
        }
        curve_rows.append(row)

        per_ckpt = out / f"{label}_per_example.jsonl"
        with per_ckpt.open("w") as f:
            for ex, comp, a, fmt, pf, tok, clip in zip(
                examples, completions, accs, fmts, pfs, tokens, clipped
            ):
                f.write(
                    json.dumps(
                        {
                            "index": ex["index"],
                            "step": step,
                            "label": label,
                            "problem": ex["problem"],
                            "gold_solution": ex["gold_solution"],
                            "completion": comp,
                            "math_accuracy": a,
                            "format_compliance": fmt,
                            "strict_success": a and fmt >= 1.0,
                            "parse_failed": pf,
                            "gen_tokens": tok,
                            "clipped": clip,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    summary = {
        "benchmark": "numina_test_checkpoints",
        "n_examples": len(examples),
        "base_model_id": args.base_model_id,
        "checkpoint_root": args.checkpoint_root,
        "max_new_tokens": args.max_new_tokens,
        "system_prompt": SYSTEM_PROMPT,
        "checkpoints": curve_rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    csv_path = out / "learning_curve.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "step",
                "math_accuracy",
                "format_compliance",
                "strict_success",
                "parse_failure_rate",
                "avg_generated_tokens",
                "median_generated_tokens",
                "fraction_clipped_at_max",
                "runtime_sec",
            ],
        )
        writer.writeheader()
        for r in curve_rows:
            writer.writerow({k: r[k] for k in writer.fieldnames if k in r})

    steps = [r["step"] for r in curve_rows]
    fmt = [100 * r["format_compliance"] for r in curve_rows]
    acc = [100 * r["math_accuracy"] for r in curve_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, fmt, marker="o", label="Format compliance (%)", color="#DD8452")
    ax.plot(steps, acc, marker="s", label="Math accuracy (%)", color="#4C72B0")
    ax.set_xlabel("Training step (0 = base)")
    ax.set_ylabel("Rate (%)")
    ax.set_title("NuminaMath held-out: checkpoint learning curve")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote {out / 'summary.json'}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")
    print("\n=== Checkpoint progression (NuminaMath test) ===")
    print(f"{'Step':>6} {'MathAcc':>8} {'Format':>8} {'Strict':>8} {'ParseFail':>10}")
    for r in curve_rows:
        print(
            f"{r['step']:>6} "
            f"{100*r['math_accuracy']:>7.1f}% "
            f"{100*r['format_compliance']:>7.1f}% "
            f"{100*r['strict_success']:>7.1f}% "
            f"{100*r['parse_failure_rate']:>9.1f}%"
        )


if __name__ == "__main__":
    main()
