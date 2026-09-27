#!/usr/bin/env python3
"""Diagnostic: 4 sampled completions at max_new_tokens 64 / 128 / 256 (no training)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import load_smoke_train
from rewards import accuracy_reward, format_reward

MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
NUM_GENERATIONS = 4
LENGTHS = (64, 128, 256)


def eos_reached(tokenizer, token_ids: torch.Tensor) -> bool:
    eos = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    eos.discard(None)
    return int(token_ids[-1].item()) in eos


def run_length(
    model,
    tokenizer,
    inputs,
    prompt_len: int,
    max_new_tokens: int,
    solution: str,
) -> None:
    gen_cfg = model.generation_config
    gen_cfg.max_new_tokens = max_new_tokens
    gen_cfg.do_sample = True
    gen_cfg.num_return_sequences = NUM_GENERATIONS
    gen_cfg.pad_token_id = tokenizer.pad_token_id

    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg)

    rows = []
    for i in range(NUM_GENERATIONS):
        seq = out[i, prompt_len:]
        text = tokenizer.decode(seq, skip_special_tokens=True)
        rows.append(
            {
                "text": text,
                "token_len": int(seq.shape[0]),
                "eos_reached": eos_reached(tokenizer, seq),
            }
        )

    trl = [[{"role": "assistant", "content": r["text"]}] for r in rows]
    fmt = format_reward(completions=trl)
    acc = accuracy_reward(completions=trl, solution=[solution] * NUM_GENERATIONS)
    clipped = sum(1 for r in rows if not r["eos_reached"]) / len(rows)

    print(f"\n======== max_new_tokens={max_new_tokens} ========")
    print(f"mean token length: {sum(r['token_len'] for r in rows) / len(rows):.1f}")
    print(f"clipped_ratio (no EOS at end): {clipped:.2f}")
    print(f"format rewards: {fmt}")
    print(f"accuracy rewards: {acc}")

    if max_new_tokens == 256:
        print("\n--- four outputs (256 tokens) ---")
        for i, r in enumerate(rows):
            print(f"\n[completion {i}] len={r['token_len']} eos={r['eos_reached']}")
            print(r["text"])


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on GPU compute node inside your allocation.")

    print(f"hostname: {__import__('socket').gethostname()}")
    print(f"gpu: {torch.cuda.get_device_name(0)}")

    ds = load_smoke_train(n=1)
    row = ds[0]
    messages = row["prompt"]
    solution = row["solution"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, device_map={"": 0})
    model.eval()

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    for n in LENGTHS:
        run_length(model, tokenizer, inputs, prompt_len, n, solution)


if __name__ == "__main__":
    main()
