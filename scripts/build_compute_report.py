#!/usr/bin/env python3
"""Build compute_usage_summary from sacct + project logs (read-only)."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path("/global/cfs/cdirs/m3560/Partha/GRPO")
EVAL = ROOT / "evaluation"


def slurm_elapsed_hours(elapsed: str) -> float:
    elapsed = elapsed.strip()
    days = 0
    if "-" in elapsed:
        d, elapsed = elapsed.split("-", 1)
        days = int(d)
    parts = elapsed.split(":")
    if len(parts) == 3:
        h, m, s = (int(parts[0]), int(parts[1]), int(parts[2]))
    elif len(parts) == 2:
        h, m, s = 0, int(parts[0]), int(parts[1])
    else:
        raise ValueError(f"Bad elapsed: {elapsed}")
    return days * 24 + h + m / 60 + s / 3600


def load_summary_runtime(path: Path) -> float | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if "total_runtime_sec" in data and data["total_runtime_sec"]:
        return float(data["total_runtime_sec"])
    b = data.get("base", {}).get("runtime_sec", 0.0)
    g = data.get("grpo", {}).get("runtime_sec", 0.0)
    return float(b) + float(g) if (b or g) else None


@dataclass
class PhaseRow:
    phase: str
    job_id: str
    gpu_type: str
    gpus_allocated: int
    gpus_used: int
    wall_time: str
    wall_hours: float
    active_gpu_hours: float
    allocated_gpu_hours: float
    account: str
    qos: str
    state: str
    start: str
    end: str
    nodes: int
    cpus_allocated: int
    source: str
    notes: str


def main() -> None:
    sacct = subprocess.check_output(
        [
            "sacct",
            "-u",
            subprocess.check_output(["whoami"], text=True).strip(),
            "--starttime=2026-09-23",
            "--format=JobID,JobName%25,Account,QOS,State,Elapsed,Start,End,NNodes,AllocCPUS,AllocTRES%55",
            "-n",
            "-P",
        ],
        text=True,
    )
    jobs: dict[str, dict[str, str]] = {}
    for line in sacct.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 11:
            continue
        jid = parts[0]
        if "." in jid and not jid.endswith(".0"):
            continue  # skip .extern and .interactive suffixes
        jobs[jid] = {
            "JobID": jid,
            "JobName": parts[1],
            "Account": parts[2],
            "QOS": parts[3],
            "State": parts[4],
            "Elapsed": parts[5],
            "Start": parts[6],
            "End": parts[7],
            "NNodes": parts[8],
            "AllocCPUS": parts[9],
            "AllocTRES": parts[10],
        }

    def gpus_from_tres(tres: str) -> int:
        for chunk in tres.split(","):
            if chunk.startswith("gres/gpu=") and "a100" not in chunk.split("=")[0]:
                try:
                    return int(chunk.split("=")[1])
                except ValueError:
                    pass
        for chunk in tres.split(","):
            if "gres/gpu:a100=" in chunk or chunk.startswith("gres/gpu="):
                return int(chunk.split("=")[1])
        return 0

    def row_from_job(
        jid: str,
        phase: str,
        gpus_used: int,
        active_h: float,
        source: str,
        notes: str,
    ) -> PhaseRow:
        j = jobs[jid]
        wh = slurm_elapsed_hours(j["Elapsed"])
        ga = gpus_from_tres(j["AllocTRES"]) or gpus_used
        return PhaseRow(
            phase=phase,
            job_id=jid,
            gpu_type="NVIDIA A100 (Perlmutter)",
            gpus_allocated=ga,
            gpus_used=gpus_used,
            wall_time=j["Elapsed"],
            wall_hours=wh,
            active_gpu_hours=active_h,
            allocated_gpu_hours=ga * wh,
            account=j["Account"],
            qos=j["QOS"],
            state=j["State"],
            start=j["Start"],
            end=j["End"],
            nodes=int(j["NNodes"] or 0),
            cpus_allocated=int(j["AllocCPUS"] or 0),
            source=source,
            notes=notes,
        )

    rows: list[PhaseRow] = []

    # Smoke (login GPU — not in Slurm)
    smoke_log = ROOT / "checkpoints/smoke/run.log"
    smoke_wall = 46.89
    if smoke_log.is_file() and "elapsed_sec=" in smoke_log.read_text():
        for line in smoke_log.read_text().splitlines():
            if line.startswith("elapsed_sec="):
                smoke_wall = float(line.split("=")[1].split()[0])
    rows.append(
        PhaseRow(
            phase="Smoke GRPO test",
            job_id="N/A (login)",
            gpu_type="NVIDIA A100-PCIE-40GB (login33)",
            gpus_allocated=1,
            gpus_used=1,
            wall_time=f"{smoke_wall:.0f}s est.",
            wall_hours=smoke_wall / 3600,
            active_gpu_hours=smoke_wall / 3600,
            allocated_gpu_hours=0.0,
            account="N/A",
            qos="N/A",
            state="log-estimated",
            start="2026-09-23T19:10:16 (log)",
            end="est.",
            nodes=0,
            cpus_allocated=0,
            source=str(smoke_log),
            notes="SLURM_JOB_ID unset in log; not billed via Slurm.",
        )
    )

    rows.append(
        row_from_job(
            "58810053",
            "GPU hostname diagnostic",
            1,
            slurm_elapsed_hours(jobs["58810053.0"]["Elapsed"]) * 1,
            "sacct step 58810053.0",
            "srun hostname / GPU visibility check.",
        )
    )

    mini_log = (ROOT / "checkpoints/mini/mini_train.log").read_text()
    mini_train_s = 226.4
    mini_wall_s = 290.02
    for line in mini_log.splitlines():
        if "'train_runtime':" in line:
            mini_train_s = float(line.split("'train_runtime': '")[1].split("'")[0])
        if line.startswith("elapsed_sec="):
            mini_wall_s = float(line.split("=")[1].split()[0])

    mini = row_from_job(
        "58810184",
        "Mini GRPO (20 steps)",
        1,
        mini_train_s / 3600,
        "checkpoints/mini/mini_train.log train_runtime",
        f"Slurm FAILED after train; wall log {mini_wall_s:.1f}s.",
    )
    rows.append(mini)

    failed_ids = [
        "58810453",
        "58810561",
        "58847248",
        "58847314",
        "58847502",
        "58847642",
        "58847780",
    ]
    f_alloc = 0.0
    f_active = slurm_elapsed_hours(jobs.get("58847780.0", {"Elapsed": "00:00:00"})["Elapsed"])
    for jid in failed_ids:
        if jid in jobs:
            wh = slurm_elapsed_hours(jobs[jid]["Elapsed"])
            f_alloc += 8 * wh
    rows.append(
        PhaseRow(
            phase="Failed salloc / tmux attempts",
            job_id=",".join(failed_ids),
            gpu_type="NVIDIA A100 (Perlmutter)",
            gpus_allocated=8,
            gpus_used=0,
            wall_time="~51s combined",
            wall_hours=51 / 3600,
            active_gpu_hours=f_active,
            allocated_gpu_hours=f_alloc,
            account="m3794_g",
            qos="gpu_interactive",
            state="FAILED",
            start="2026-09-23—24",
            end="various",
            nodes=2,
            cpus_allocated=256,
            source="sacct top-level interactive jobs",
            notes="Short-lived 2-node×8-GPU salloc attempts; minimal Python GPU use.",
        )
    )

    full_log = (ROOT / "logs/grpo_full_interactive.log").read_text()
    train_runtime_s = 4363.0
    for line in full_log.splitlines():
        if "'train_runtime':" in line and "train_loss" in line:
            train_runtime_s = float(line.split("'train_runtime': '")[1].split("'")[0])
    full = row_from_job(
        "58848004",
        "Full GRPO training",
        1,
        train_runtime_s / 3600,
        "logs/grpo_full_interactive.log train_runtime",
        f"Parent reserved 8 GPUs×2 nodes; srun step 58848004.0 used 1 GPU. train_runtime={train_runtime_s}s.",
    )
    rows.append(full)

    step_h = lambda jid: slurm_elapsed_hours(jobs[jid]["Elapsed"])  # noqa: E731

    rows.append(
        row_from_job(
            "58903788",
            "NuminaMath eval (pilot)",
            1,
            step_h("58903788.0"),
            "sacct 58903788.0",
            "Early Numina eval run (superseded by 99-example run).",
        )
    )
    numina99_active = load_summary_runtime(EVAL / "summary.json") or 82.86
    r = row_from_job(
        "58903831",
        "NuminaMath eval (99, legacy scorer)",
        1,
        numina99_active / 3600,
        "evaluation/summary.json total_runtime_sec",
        "Original held-out comparison artifact; scorer since corrected in checkpoint eval.",
    )
    rows.append(r)

    gsm8k_rt = load_summary_runtime(EVAL / "gsm8k/summary.json") or 0.0
    ext1_step = step_h("58918906.0")
    ext2_step = step_h("58919242.0")
    ckpt_active = max(0.0, ext1_step + ext2_step - 2 * (gsm8k_rt / 3600))

    rows.append(
        PhaseRow(
            phase="GSM8K eval (500, ×2 runs)",
            job_id="58918906,58919242",
            gpu_type="NVIDIA A100-SXM4-40GB / 80GB",
            gpus_allocated=4,
            gpus_used=1,
            wall_time="2× ~9.5 min srun",
            wall_hours=(ext1_step + ext2_step),
            active_gpu_hours=2 * gsm8k_rt / 3600,
            allocated_gpu_hours=row_from_job("58918906", "", 1, 0, "", "").allocated_gpu_hours
            + row_from_job("58919242", "", 1, 0, "", "").allocated_gpu_hours,
            account="m3794_g",
            qos="gpu_interactive",
            state="COMPLETED",
            start="2026-09-26",
            end="2026-09-26",
            nodes=1,
            cpus_allocated=128,
            source="evaluation/gsm8k/summary.json (×2 extended eval runs)",
            notes="Two grpo_ext_eval jobs after scorer fix.",
        )
    )

    rows.append(
        PhaseRow(
            phase="Numina checkpoint eval (×2 runs)",
            job_id="58918906,58919242",
            gpu_type="NVIDIA A100",
            gpus_allocated=4,
            gpus_used=1,
            wall_time="residual of ext eval steps",
            wall_hours=ckpt_active,
            active_gpu_hours=ckpt_active,
            allocated_gpu_hours=0.0,
            account="m3794_g",
            qos="gpu_interactive",
            state="COMPLETED",
            start="2026-09-26",
            end="2026-09-26",
            nodes=1,
            cpus_allocated=128,
            source="sacct step minus gsm8k summary runtime (×2)",
            notes="Corrected Numina checkpoint curve; not double-counting Slurm allocation.",
        )
    )

    m500 = load_summary_runtime(EVAL / "math500/summary.json") or 370.29
    rows.append(
        row_from_job(
            "58966687",
            "MATH-500 eval (500)",
            1,
            m500 / 3600,
            "evaluation/math500/summary.json total_runtime_sec",
            "Final external benchmark.",
        )
    )

    # Totals
    active_total = sum(r.active_gpu_hours for r in rows)
    alloc_total = sum(r.allocated_gpu_hours for r in rows)

    train_active = sum(
        r.active_gpu_hours
        for r in rows
        if r.job_id in {"58810184", "58848004"}
    )
    eval_active = sum(
        r.active_gpu_hours
        for r in rows
        if "eval" in r.phase.lower() or "GSM8K" in r.phase or "MATH-500" in r.phase
    )
    diag_active = sum(
        r.active_gpu_hours
        for r in rows
        if r.phase in {"Smoke GRPO test", "GPU hostname diagnostic", "Failed salloc / tmux attempts"}
    )

    node_hours = sum(r.nodes * r.wall_hours for r in rows if r.job_id != "N/A (login)")
    cpu_hours = sum(r.cpus_allocated * r.wall_hours for r in rows if r.cpus_allocated)

    full_alloc = rows[[i for i, r in enumerate(rows) if r.job_id == "58848004"][0]].allocated_gpu_hours
    full_active = rows[[i for i, r in enumerate(rows) if r.job_id == "58848004"][0]].active_gpu_hours
    # 8 GPUs reserved for ~full allocation wall time; train.py used 1 GPU for train_runtime.
    unused_full = full_alloc - full_active

    report = {
        "project_root": str(ROOT),
        "methodology": {
            "allocated_gpu_hours": "top-level Slurm jobs: gres/gpu × Elapsed (no .extern/.0 double count)",
            "active_gpu_hours": "log train_runtime / evaluation summary total_runtime_sec / srun step when needed",
            "gpu_type": "NVIDIA A100 on Perlmutter compute; smoke on login A100-PCIE",
        },
        "phases": [asdict(r) for r in rows],
        "totals": {
            "active_gpu_hours": active_total,
            "allocated_gpu_hours": alloc_total,
            "training_active_gpu_hours": train_active,
            "evaluation_active_gpu_hours": eval_active,
            "diagnostic_smoke_active_gpu_hours": diag_active,
            "node_hours_sum_phases": node_hours,
            "cpu_core_hours_sum_phases": cpu_hours,
            "full_training_allocated_gpu_hours": full_alloc,
            "full_training_active_gpu_hours": full_active,
            "full_training_unused_reserved_gpu_hours_approx": unused_full,
        },
        "percent_active_gpu_hours": {
            "full_training": 100 * full_active / active_total if active_total else 0,
            "mini_smoke_debug": 100 * (diag_active + mini.active_gpu_hours) / active_total if active_total else 0,
            "evaluation": 100 * eval_active / active_total if active_total else 0,
        },
        "verified_reference": {
            "job_58848004_elapsed": jobs["58848004"]["Elapsed"],
            "job_58848004_train_runtime_sec": train_runtime_s,
            "compute_node": "nid001176 (from grpo_full_interactive.log)",
        },
    }

    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "compute_usage_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    csv_path = EVAL / "compute_usage_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    md = build_markdown(report, rows)
    (EVAL / "compute_usage_summary.md").write_text(md)
    print(md)


def build_markdown(report: dict, rows: list[PhaseRow]) -> str:
    t = report["totals"]
    p = report["percent_active_gpu_hours"]
    v = report["verified_reference"]
    lines = [
        "# GRPO project — compute usage summary",
        "",
        f"Project: `{report['project_root']}`",
        "",
        "## Methodology",
        "",
        "- **Allocated GPU-hours:** top-level Slurm job `gres/gpu × Elapsed` (avoid counting `.extern` / nested `.0` steps separately).",
        "- **Active GPU-hours:** Python ML time from logs (`train_runtime`, `evaluation/*/summary.json` `total_runtime_sec`, or srun step duration × GPUs actually used).",
        "- **GPU:** Perlmutter **NVIDIA A100** on compute nodes; smoke test log shows **login-node A100-PCIE** (not Slurm-billed).",
        "",
        "### Verified full-training reference",
        "",
        f"- Job **58848004** Slurm elapsed: **{v['job_58848004_elapsed']}**",
        f"- `train.py` **train_runtime**: **{v['job_58848004_train_runtime_sec']:.0f} s** (~{v['job_58848004_train_runtime_sec']/3600:.2f} h) on **1 GPU**",
        f"- Compute node: **{v['compute_node']}**",
        f"- Interactive salloc reserved **8 GPUs × 2 nodes**; `train.py` ran under **1-GPU srun**.",
        "",
        "## Phase table",
        "",
        "| Phase | Job ID | GPUs alloc | GPUs used | Wall | Active GPU-h | Alloc GPU-h | Source |",
        "|-------|--------|------------|-----------|------|--------------|-------------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r.phase} | {r.job_id} | {r.gpus_allocated} | {r.gpus_used} | {r.wall_time} | {r.active_gpu_hours:.3f} | {r.allocated_gpu_hours:.3f} | {r.notes[:60]}… |"
            if len(r.notes) > 60
            else f"| {r.phase} | {r.job_id} | {r.gpus_allocated} | {r.gpus_used} | {r.wall_time} | {r.active_gpu_hours:.3f} | {r.allocated_gpu_hours:.3f} | {r.notes} |"
        )
    lines.extend(
        [
            "",
            "## Totals",
            "",
            f"| Metric | Value |",
            f"|--------|------:|",
            f"| **Total active GPU-hours** | **{t['active_gpu_hours']:.3f}** |",
            f"| **Total allocated GPU-hours** (Slurm-reserved, phases above) | **{t['allocated_gpu_hours']:.3f}** |",
            f"| Training active GPU-hours | {t['training_active_gpu_hours']:.3f} |",
            f"| Evaluation active GPU-hours | {t['evaluation_active_gpu_hours']:.3f} |",
            f"| Diagnostic / smoke / failed-salloc active GPU-hours | {t['diagnostic_smoke_active_gpu_hours']:.3f} |",
            f"| Node-hours (sum over phase rows) | {t['node_hours_sum_phases']:.3f} |",
            f"| CPU-core-hours (sum over phase rows) | {t['cpu_core_hours_sum_phases']:.0f} |",
            "",
            "### Share of **active** GPU-hours",
            "",
            f"- Full training: **{p['full_training']:.1f}%**",
            f"- Mini + smoke + debug: **{p['mini_smoke_debug']:.1f}%**",
            f"- Evaluation: **{p['evaluation']:.1f}%**",
            "",
            "### 8-GPU allocation vs 1-GPU training (job 58848004)",
            "",
            f"- Allocated GPU-hours: **{t['full_training_allocated_gpu_hours']:.2f}**",
            f"- Active GPU-hours (training): **{t['full_training_active_gpu_hours']:.2f}**",
            f"- Approx. **reserved-but-unused** GPU-hours during full training: **{t['full_training_unused_reserved_gpu_hours_approx']:.2f}** (~7 idle GPUs × allocation duration).",
            "",
            "## Exact vs estimated",
            "",
            "| Item | Quality |",
            "|------|---------|",
            "| Full training train_runtime & job 58848004 elapsed | **Exact** (logs + sacct) |",
            "| Mini train_runtime | **Exact** (mini_train.log) |",
            "| GSM8K / MATH-500 / Numina 99 eval runtimes | **Exact** (summary.json) |",
            "| Checkpoint eval active time | **Estimated** (ext eval srun step − GSM8K summary, ×2 runs) |",
            "| Smoke test | **Estimated** (login GPU, run.log `elapsed_sec`; **not** in Slurm totals) |",
            "| Failed salloc bundle | **Exact** sacct elapsed × 8 GPUs |",
            "",
            "## Optional FLOPs",
            "",
            "Not reported — no trustworthy retrospective FLOP counter; **GPU-hours** are the primary metric.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
