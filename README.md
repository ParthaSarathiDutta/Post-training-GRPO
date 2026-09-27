# GRPO math reasoning (Qwen2-0.5B + LoRA)

Reproducible post-training based on [Post training an LLM for reasoning with GRPO in TRL](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl).

**Project root:** `/global/cfs/cdirs/m3560/Partha/GRPO`

## Environment (Perlmutter)

```bash
cd /global/cfs/cdirs/m3560/Partha/GRPO
bash scripts/setup_env.sh
source .venv/bin/activate
export HF_HOME="${SCRATCH}/hf_cache"
python scripts/check_imports.py
```

Uses `module load pytorch/2.13.0` and a venv with `--system-site-packages` so CUDA torch/transformers come from the module.

## Training (GPU compute node — not on login node)

```bash
cd /global/cfs/cdirs/m3560/Partha/GRPO
module load pytorch/2.13.0
source .venv/bin/activate
export HF_HOME="${SCRATCH}/hf_cache"

python train.py --output-dir checkpoints/Qwen2-0.5B-GRPO
```

Smoke test (8 examples, 1 step):

```bash
python train.py --smoke-test --output-dir checkpoints/smoke
```

## Inference (one test example)

```bash
python inference.py --adapter-dir checkpoints/Qwen2-0.5B-GRPO
```

TensorBoard:

```bash
tensorboard --logdir checkpoints/Qwen2-0.5B-GRPO
```

## Differences from the HF notebook

| Notebook | This repo |
|----------|-----------|
| `get_peft_model()` before trainer | `peft_config` on `GRPOTrainer` |
| `push_to_hub=True` | Checkpoints under `checkpoints/` |
| `bf16=True` always | BF16 only if CUDA supports it |
| `torch_dtype` | `model_init_kwargs={"dtype": torch.bfloat16}` |
| Space-joined prompt at inference | `apply_chat_template` for Qwen Instruct |

## Layout

```
train.py       GRPOTrainer entrypoint
rewards.py     format_reward, accuracy_reward
data.py        NuminaMath-TIR 5% splits + system prompt
inference.py   Single test-sample generation
scripts/       setup_env.sh, check_imports.py
```
