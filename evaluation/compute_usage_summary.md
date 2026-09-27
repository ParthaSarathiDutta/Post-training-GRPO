# GRPO project — compute usage summary

Project: `/global/cfs/cdirs/m3560/Partha/GRPO`

## Methodology

- **Allocated GPU-hours:** top-level Slurm job `gres/gpu × Elapsed` (avoid counting `.extern` / nested `.0` steps separately).
- **Active GPU-hours:** Python ML time from logs (`train_runtime`, `evaluation/*/summary.json` `total_runtime_sec`, or srun step duration × GPUs actually used).
- **GPU:** Perlmutter **NVIDIA A100** on compute nodes; smoke test log shows **login-node A100-PCIE** (not Slurm-billed).

### Verified full-training reference

- Job **58848004** Slurm elapsed: **01:14:19**
- `train.py` **train_runtime**: **4363 s** (~1.21 h) on **1 GPU**
- Compute node: **nid001176 (from grpo_full_interactive.log)**
- Interactive salloc reserved **8 GPUs × 2 nodes**; `train.py` ran under **1-GPU srun**.

## Phase table

| Phase | Job ID | GPUs alloc | GPUs used | Wall | Active GPU-h | Alloc GPU-h | Source |
|-------|--------|------------|-----------|------|--------------|-------------|--------|
| Smoke GRPO test | N/A (login) | 1 | 1 | 47s est. | 0.013 | 0.000 | SLURM_JOB_ID unset in log; not billed via Slurm. |
| GPU hostname diagnostic | 58810053 | 4 | 1 | 00:00:14 | 0.000 | 0.016 | srun hostname / GPU visibility check. |
| Mini GRPO (20 steps) | 58810184 | 4 | 1 | 00:04:54 | 0.063 | 0.327 | Slurm FAILED after train; wall log 290.0s. |
| Failed salloc / tmux attempts | 58810453,58810561,58847248,58847314,58847502,58847642,58847780 | 8 | 0 | ~51s combined | 0.001 | 0.113 | Short-lived 2-node×8-GPU salloc attempts; minimal Python GPU… |
| Full GRPO training | 58848004 | 8 | 1 | 01:14:19 | 1.212 | 9.909 | Parent reserved 8 GPUs×2 nodes; srun step 58848004.0 used 1 … |
| NuminaMath eval (pilot) | 58903788 | 4 | 1 | 00:01:15 | 0.019 | 0.083 | Early Numina eval run (superseded by 99-example run). |
| NuminaMath eval (99, legacy scorer) | 58903831 | 4 | 1 | 00:02:03 | 0.023 | 0.137 | Original held-out comparison artifact; scorer since correcte… |
| GSM8K eval (500, ×2 runs) | 58918906,58919242 | 4 | 1 | 2× ~9.5 min srun | 0.194 | 1.262 | Two grpo_ext_eval jobs after scorer fix. |
| Numina checkpoint eval (×2 runs) | 58918906,58919242 | 4 | 1 | residual of ext eval steps | 0.119 | 0.000 | Corrected Numina checkpoint curve; not double-counting Slurm… |
| MATH-500 eval (500) | 58966687 | 4 | 1 | 00:07:25 | 0.103 | 0.494 | Final external benchmark. |

## Totals

| Metric | Value |
|--------|------:|
| **Total active GPU-hours** | **1.747** |
| **Total allocated GPU-hours** (Slurm-reserved, phases above) | **12.341** |
| Training active GPU-hours | 1.275 |
| Evaluation active GPU-hours | 0.458 |
| Diagnostic / smoke / failed-salloc active GPU-hours | 0.014 |
| Node-hours (sum over phase rows) | 3.202 |
| CPU-core-hours (sum over phase rows) | 410 |

### Share of **active** GPU-hours

- Full training: **69.4%**
- Mini + smoke + debug: **4.4%**
- Evaluation: **26.2%**

### 8-GPU allocation vs 1-GPU training (job 58848004)

- Allocated GPU-hours: **9.91**
- Active GPU-hours (training): **1.21**
- Approx. **reserved-but-unused** GPU-hours during full training: **8.70** (~7 idle GPUs × allocation duration).

## Exact vs estimated

| Item | Quality |
|------|---------|
| Full training train_runtime & job 58848004 elapsed | **Exact** (logs + sacct) |
| Mini train_runtime | **Exact** (mini_train.log) |
| GSM8K / MATH-500 / Numina 99 eval runtimes | **Exact** (summary.json) |
| Checkpoint eval active time | **Estimated** (ext eval srun step − GSM8K summary, ×2 runs) |
| Smoke test | **Estimated** (login GPU, run.log `elapsed_sec`; **not** in Slurm totals) |
| Failed salloc bundle | **Exact** sacct elapsed × 8 GPUs |

## Optional FLOPs

Not reported — no trustworthy retrospective FLOP counter; **GPU-hours** are the primary metric.
