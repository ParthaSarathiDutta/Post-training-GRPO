#!/usr/bin/env python3
"""GRPO + LoRA post-training for Qwen2-0.5B-Instruct on NuminaMath-TIR."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from data import load_smoke_train, load_train_test
from rewards import accuracy_reward, format_reward

DEFAULT_MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = "checkpoints/Qwen2-0.5B-GRPO"
MINI_OUTPUT_DIR = "checkpoints/mini"
MINI_STEPS = 20


def use_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO reasoning fine-tune with LoRA")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Minimal GRPO path: 2 examples, max_steps=1, grad_accum=1.",
    )
    parser.add_argument(
        "--mini-train",
        action="store_true",
        help=f"Learning validation: train[:5%], max_steps={MINI_STEPS}, max_completion_length=256.",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override num_train_epochs.")
    return parser.parse_args()


def build_grpo_config(
    output_dir: str,
    bf16: bool,
    max_steps: int | None,
    *,
    smoke: bool = False,
    mini: bool = False,
) -> GRPOConfig:
    kwargs: dict = {
        "output_dir": output_dir,
        "learning_rate": 1e-5,
        "num_train_epochs": 1,
        "gradient_accumulation_steps": 16,
        "per_device_train_batch_size": 1,
        "max_completion_length": 256,
        "num_generations": 4,
        "bf16": bf16,
        "fp16": False,
        "remove_unused_columns": False,
        "report_to": ["tensorboard"],
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 50,
        "push_to_hub": False,
    }
    if smoke:
        kwargs["max_completion_length"] = 64
        kwargs["gradient_accumulation_steps"] = 1
        kwargs["logging_steps"] = 1
        kwargs["save_steps"] = 1
        kwargs["per_device_train_batch_size"] = 4

    if mini:
        kwargs["logging_steps"] = 1
        kwargs["save_steps"] = 1
        kwargs["log_completions"] = True

    if max_steps is not None:
        kwargs["max_steps"] = max_steps
        kwargs.pop("num_train_epochs")

    if bf16:
        kwargs["model_init_kwargs"] = {"dtype": torch.bfloat16}

    return GRPOConfig(**kwargs)


def main() -> None:
    args = parse_args()
    bf16 = use_bf16()
    print(f"CUDA available: {torch.cuda.is_available()}, bf16: {bf16}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    smoke = args.smoke_test
    mini = args.mini_train
    if smoke and mini:
        raise SystemExit("Choose only one of --smoke-test or --mini-train.")

    if smoke:
        train_dataset = load_smoke_train(n=2)
        max_steps = 1
        output_dir = Path(args.output_dir or "checkpoints/smoke")
        print(f"Smoke test mode: {len(train_dataset)} rows, max_steps=1, grad_accum=1")
    elif mini:
        train_dataset, _ = load_train_test()
        max_steps = MINI_STEPS
        output_dir = Path(args.output_dir or MINI_OUTPUT_DIR)
        print(f"Mini-train: {len(train_dataset)} rows, max_steps={MINI_STEPS}, max_completion_length=256")
    else:
        train_dataset, _ = load_train_test()
        max_steps = args.max_steps
        output_dir = Path(args.output_dir or DEFAULT_OUTPUT_DIR)
        print(f"Training examples: {len(train_dataset)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    training_args = build_grpo_config(
        str(output_dir),
        bf16=bf16,
        max_steps=max_steps,
        smoke=smoke,
        mini=mini,
    )

    trainer = GRPOTrainer(
        model=args.model_id,
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(str(output_dir))

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"peak_gpu_memory_mb: {peak_mb:.1f}")

    print(f"Saved under {output_dir.resolve()}")


if __name__ == "__main__":
    main()
