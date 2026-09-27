#!/usr/bin/env python3
"""Summarize mini-train metrics and sample completions from early vs late checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import load_train_test

MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
MINI_DIR = ROOT / "checkpoints" / "mini"


def load_log_history(mini_dir: Path) -> list[dict]:
    candidates = [mini_dir / "checkpoint-20/trainer_state.json"]
    candidates.extend(sorted(mini_dir.glob("checkpoint-*/trainer_state.json")))
    candidates.append(mini_dir / "trainer_state.json")
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return json.loads(p.read_text()).get("log_history", [])
    return []


def print_metrics(history: list[dict]) -> None:
    keys = [
        "step",
        "rewards/format_reward/mean",
        "rewards/accuracy_reward/mean",
        "reward",
        "reward_std",
        "frac_reward_zero_std",
        "loss",
        "grad_norm",
        "completions/mean_length",
        "completions/clipped_ratio",
    ]
    print("\n=== Per-step metrics ===")
    for row in history:
        if "loss" not in row and "reward" not in row:
            continue
        parts = [f"step={row.get('step', '?')}"]
        for k in keys[1:]:
            if k in row:
                parts.append(f"{k.split('/')[-1]}={row[k]}")
        print("  ".join(parts))


def generate_sample(adapter_dir: Path, label: str) -> None:
    _, test_ds = load_train_test()
    messages = test_ds[0]["prompt"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, device_map="auto")
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    text = tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    print(f"\n=== Sample completion ({label}) from {adapter_dir.name} ===")
    print(text[:2000])


def main() -> None:
    mini_dir = MINI_DIR
    if not mini_dir.exists():
        raise SystemExit(f"Missing {mini_dir}")

    history = load_log_history(mini_dir)
    print_metrics(history)

    early = mini_dir / "checkpoint-1"
    late = mini_dir / "checkpoint-20"
    if not late.exists():
        cps = sorted(mini_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        late = cps[-1] if cps else None
    if early.exists():
        generate_sample(early, "early ~step 1")
    if late and late.exists():
        generate_sample(late, "late ~step 20")


if __name__ == "__main__":
    main()
