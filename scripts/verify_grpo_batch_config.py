#!/usr/bin/env python3
"""Print TRL 1.13.0 derived generation_batch_size for full vs smoke configs."""

from trl import GRPOConfig


def show(label: str, **kwargs) -> None:
    cfg = GRPOConfig(output_dir="/tmp/grpo_cfg_check", **kwargs)
    print(
        f"{label}: per_device_train_batch_size={cfg.per_device_train_batch_size}, "
        f"gradient_accumulation_steps={cfg.gradient_accumulation_steps}, "
        f"num_generations={cfg.num_generations} -> "
        f"generation_batch_size={cfg.generation_batch_size}, "
        f"steps_per_generation={cfg.steps_per_generation}"
    )


def main() -> None:
    show(
        "full_training",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_generations=4,
    )
    show(
        "smoke_current",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_generations=4,
    )
    try:
        show(
            "smoke_batch1_grad1_INVALID",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_generations=4,
        )
    except ValueError as e:
        print(f"smoke_batch1_grad1_INVALID: REJECTED -> {e}")


if __name__ == "__main__":
    main()
