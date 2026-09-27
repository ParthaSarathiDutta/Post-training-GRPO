#!/usr/bin/env python3
"""Load trained LoRA adapter and generate one held-out test example."""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_train_test

DEFAULT_BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True, help="Path from trainer.save_model()")
    parser.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--test-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, test_dataset = load_train_test()
    example = test_dataset[args.test_index]
    messages = example["prompt"]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if use_bf16() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        dtype=dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("=== User problem ===")
    print(messages[1]["content"])
    print("\n=== Ground-truth solution (truncated) ===")
    sol = example["solution"]
    print(sol[:500], "..." if len(sol) > 500 else "")
    print("\n=== Model completion ===")
    print(completion)


def use_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


if __name__ == "__main__":
    main()
