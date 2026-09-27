---
name: compute-efficiency
description: >-
  GPU and Slurm resource efficiency for NERSC Perlmutter GRPO training and
  evaluation. Use before submitting or scripting salloc, sbatch, srun, tmux
  launchers, train.py, evaluate*.py, or any GPU job for this project. Checks
  GPUs reserved vs GPUs used, smoke→mini→full progression, subset evals, HF
  cache, and active vs allocated GPU-hours. Motivated by
  evaluation/compute_usage_summary.md on this project.
---

# Compute efficiency (GRPO / Perlmutter)

Project root (CFS): `/global/cfs/cdirs/m3560/Partha/GRPO`

Reference accounting: `evaluation/compute_usage_summary.md` (and `.json` / `.csv`).

## Core rule

**Never reserve substantially more GPUs than the actual Python workload will use.**

This project’s full GRPO run (job **58848004**) illustrates the failure mode:

- **Reserved:** 8 GPUs (2-node interactive salloc)
- **Actually used:** 1 GPU for `train.py`
- **Active training compute:** ~**1.21 GPU-hours**
- **Allocated compute:** ~**9.91 GPU-hours**
- ~**88%** of the GPU reservation was unused for `train.py`

**Default for single-process / single-GPU training:** request **1 GPU on 1 node**.

Request multiple GPUs **only** when the code actually launches distributed or multi-GPU execution (e.g. `torchrun`, DeepSpeed multi-process, or explicit multi-GPU data parallel in the launch script).

## Pre-flight checklist (training or evaluation)

Before submitting Slurm or starting a long GPU run, answer explicitly:

1. **How many GPUs will the Python process actually use?** (read `train.py`, launcher scripts, `CUDA_VISIBLE_DEVICES`, `srun`/`salloc` lines.)
2. **How many GPUs am I reserving?** (Slurm `gres/gpu`, `--gpus`, `--gpus-per-task`, node count.)
3. **Are those numbers consistent?** If reserved ≫ used, **stop** and propose a 1-GPU (or right-sized) alternative unless the user forbids changing the allocation.
4. **Can I validate on a smaller run first?** (smoke / mini / subset eval.)

Report a short pre-submit estimate:

| Field | Estimate |
|-------|----------|
| GPUs required by code | |
| GPUs / nodes to request | |
| Expected runtime | |
| Approx. **active** GPU-hours | GPUs_used × runtime |
| Approx. **allocated** GPU-hours | GPUs_reserved × wall time |

Flag obviously wasteful configs (e.g. 8-GPU salloc + 1-GPU `python train.py` without user mandate).

## Compute-saving practices

1. **Progression:** smoke test → short mini-training validation → full training, before spending large compute.
2. **Evaluation:** validate scoring and pipeline on **20–100 examples** before **500+** benchmark runs.
3. **Cache:** reuse models/datasets via existing `HF_HOME` (on Perlmutter: `export HF_HOME="${SCRATCH}/hf_cache"`)—avoid redundant downloads and cold loads when unnecessary.
4. **Multi-GPU requests:** verify the launch command creates multiple training processes or otherwise uses **all** requested GPUs.
5. **Reporting:** distinguish **allocated GPU-hours** (Slurm reservation) vs **active GPU-hours** (logged ML runtime × GPUs actually used). See `scripts/build_compute_report.py` and `evaluation/compute_usage_summary.md`.
6. **Pre-submit estimate:** GPUs, nodes, runtime, active vs allocated GPU-hours; call out waste.
7. **Correctness first:** do **not** reduce resources if that makes the code invalid or materially changes the intended experiment.
8. **User-pinned Slurm lines:** if the user gives an exact salloc/sbatch command, **preserve it** when instructed; give compute-saving recommendations **separately**—do not silently change their command.

## Perlmutter patterns for this repo

- **Preferred 1-GPU eval/train dispatch:** `scripts/run_*_gpu.sh` pattern (single `srun`, `--gpus=1`, `--gpus-per-task=1`).
- **Avoid:** login-node GPU for real training (smoke on login wastes wrong resource class and skews accounting).
- **Interactive tmux + salloc:** if user requires a specific multi-GPU salloc, run **1-GPU `srun`** inside it for `train.py` / `evaluate*.py` (as in `scripts/run_full_train_logged.sh`) and document that **billing follows the parent allocation**.

## When to read the compute report

After substantial training/eval phases, or when the user asks about cost/efficiency, read `evaluation/compute_usage_summary.md` or regenerate with:

```bash
cd /global/cfs/cdirs/m3560/Partha/GRPO
python scripts/build_compute_report.py
```

## Optional FLOPs

Do not invent retrospective FLOPs unless defensibly derived. **GPU-hours** (active vs allocated) are the primary metric.
