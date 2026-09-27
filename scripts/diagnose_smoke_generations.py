#!/usr/bin/env python3
"""
Generate four completions for one smoke prompt (same settings as train smoke)
and score each with format_reward / accuracy_reward.
"""

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
MAX_NEW_TOKENS = 64
NUM_GENERATIONS = 4


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on a GPU compute node.")

    host = __import__("socket").gethostname()
    print(f"hostname: {host}")
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"cuda_device_count: {torch.cuda.device_count()}")

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

    gen_cfg = model.generation_config
    gen_cfg.max_new_tokens = MAX_NEW_TOKENS
    gen_cfg.do_sample = True
    gen_cfg.num_return_sequences = NUM_GENERATIONS
    gen_cfg.pad_token_id = tokenizer.pad_token_id

    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg)

    completions = []
    for i in range(NUM_GENERATIONS):
        seq = out[i, prompt_len:]
        text = tokenizer.decode(seq, skip_special_tokens=True)
        token_len = int(seq.shape[0])
        hit_max = token_len >= MAX_NEW_TOKENS
        completions.append(
            {
                "index": i,
                "text": text,
                "token_len": token_len,
                "hit_max_completion_length": hit_max,
            }
        )

    trl_fmt = [[{"role": "assistant", "content": c["text"]}] for c in completions]
    fmt_scores = format_reward(completions=trl_fmt)
    acc_scores = accuracy_reward(completions=trl_fmt, solution=[solution] * NUM_GENERATIONS)

    print("\n=== User problem (truncated) ===")
    print(messages[1]["content"][:300])
    print("\n=== Four generated completions ===")
    for c, f, a in zip(completions, fmt_scores, acc_scores):
        print(f"\n--- completion {c['index']} ---")
        print(f"token_len={c['token_len']} hit_max={c['hit_max_completion_length']}")
        print(f"format_reward={f} accuracy_reward={a}")
        print(c["text"][:800] if len(c["text"]) > 800 else c["text"])

    clipped = sum(1 for c in completions if c["hit_max_completion_length"]) / len(completions)
    print(f"\nclipped_ratio (hit max_new_tokens={MAX_NEW_TOKENS}): {clipped}")


if __name__ == "__main__":
    main()
