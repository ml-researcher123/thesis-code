"""One-shot Colab runner for Stage-17: the LLMLingua compression head-to-head.

The paper cites RECOMP/LLMLingua as the compression baseline but never runs one.
This adds LLMLingua-2 as a real policy (chunk_llmlingua), fed the same candidate
pool and the same 160-token budget as the packers, so its answer F1 is directly
comparable to chunk_submod / chunk_focused / chunk_packed on identical inputs.

Runs the lean chunk factorial (packed/focused/submod) + the compression policy on
HotpotQA, budget 160, three seeds, Qwen2.5-3B reader. HotpotQA loads from the HF
Hub, so no dataset upload is needed. LLMLingua-2's compressor (~560M) sits on the
same T4 as the 3B reader; if it OOMs, switch COMPRESSION_MODEL to the bert-base
multilingual variant below.

Usage (from a Colab cell, after cloning this repo into /content/thesis-code):
    %cd /content/thesis-code
    !python control/colab_stage17_compression.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORK_ROOT = Path("/content/ace_rag_work")
RESULT_ROOT = Path("/content/colab_results")

BUDGET = 160
LIMIT = 500
SEEDS = [42, 13, 7]
TOP_K = 5
TOP_K_NODES = 48
MAX_EXPANDED = 5
COMPRESSION_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
# Lighter fallback if the large model OOMs next to the reader:
# COMPRESSION_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"


def log(message: str) -> None:
    print(message, flush=True)


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> int:
    log("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd, cwd=str(cwd) if cwd else None, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log(line.rstrip())
    code = proc.wait()
    if check and code != 0:
        raise RuntimeError(f"command failed with code {code}: {' '.join(cmd)}")
    return code


def prepare_project() -> None:
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    project_zip = REPO_ROOT / "control" / "ace_rag_research_kaggle_ready_v13_analysis.zip"
    if not project_zip.exists():
        raise FileNotFoundError(f"{project_zip} not found -- did the repo clone fully?")
    with zipfile.ZipFile(project_zip) as zf:
        zf.extractall(WORK_ROOT)
    log(f"Extracted {project_zip} to {WORK_ROOT}")
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-cloud.txt"], cwd=WORK_ROOT)
    run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.43.0"], cwd=WORK_ROOT, check=False)
    # The compression baseline: LLMLingua-2.
    run([sys.executable, "-m", "pip", "install", "-q", "llmlingua"], cwd=WORK_ROOT)


def apply_overlay() -> None:
    overlay_root = REPO_ROOT / "control" / "overlay"
    copied: list[str] = []
    for src in sorted(overlay_root.rglob("*.py")):
        rel = src.relative_to(overlay_root)
        dst = WORK_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(rel).replace("\\", "/"))
    log(f"[overlay] applied {len(copied)} file(s): {copied}")


def job_cmd(job_name: str, seed: int) -> tuple[Path, list[str]]:
    out_dir = RESULT_ROOT / job_name
    cmd = [
        sys.executable, "-m", "experiments.run_density_router",
        "--dataset", "hotpotqa",
        "--split", "validation",
        "--limit", str(LIMIT),
        "--seed", str(seed),
        "--ace-retriever", "standard",
        "--top-k", str(TOP_K),
        "--top-k-nodes", str(TOP_K_NODES),
        "--max-expanded-docs", str(MAX_EXPANDED),
        "--embedder", "sentence-transformers",
        "--embedding-model", "BAAI/bge-small-en-v1.5",
        "--embed-device", "cuda",
        "--compressor", "truncate",
        "--compress-dims", "320",
        "--reader-backend", "hf",
        "--reader-model", "Qwen/Qwen2.5-3B-Instruct",
        "--reader-device", "cuda",
        "--reader-batch-size", "2",
        "--mmr-lambda", "0.7",
        "--budget", str(BUDGET),
        "--out-dir", str(out_dir),
        "--lean-policies",
        "--compression-baseline",
        "--compression-model", COMPRESSION_MODEL,
    ]
    return out_dir, cmd


def execute(job_name: str, out_dir: Path, cmd: list[str]) -> bool:
    if out_dir.exists() and any(out_dir.glob("*metrics.csv")):
        log(f"[skip] {job_name} already has metrics")
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    return run(cmd, cwd=WORK_ROOT, check=False) == 0


def publish_results(status: str) -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / "kaggle_results" / f"{timestamp}_colab_stage17"
    latest_dir = REPO_ROOT / "kaggle_results" / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    if RESULT_ROOT.exists():
        shutil.copytree(RESULT_ROOT, run_dir / "colab_results", dirs_exist_ok=True)
    (run_dir / "run_metadata.json").write_text(
        f'{{"status": "{status}", "stage": "stage17-compression-baseline", '
        f'"completed_at": "{time.strftime("%Y-%m-%d %H:%M:%S")}", "runner": "colab"}}\n',
        encoding="utf-8",
    )
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)

    run(["git", "config", "user.email", "colab-runner@example.local"], cwd=REPO_ROOT, check=False)
    run(["git", "config", "user.name", "colab-runner"], cwd=REPO_ROOT, check=False)
    run(["git", "add", "kaggle_results"], cwd=REPO_ROOT, check=False)
    code = run(["git", "commit", "-m", f"Colab Stage-17 compression baseline {timestamp} {status}"], cwd=REPO_ROOT, check=False)
    if code == 0:
        run(["git", "fetch", "origin", "main"], cwd=REPO_ROOT, check=False)
        run(["git", "rebase", "origin/main"], cwd=REPO_ROOT, check=False)
        push_code = run(["git", "push", "origin", "main"], cwd=REPO_ROOT, check=False)
        log("[publish] pushed to origin/main" if push_code == 0
            else "[publish] push failed -- results committed locally, push manually")
    else:
        log("[publish] nothing new to commit")


def main() -> None:
    prepare_project()
    apply_overlay()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}
    for seed in SEEDS:
        job = f"stage17_compression_hotpotqa_qwen3b_budget{BUDGET}_limit{LIMIT}_seed{seed}"
        log(f"--- compression head-to-head seed {seed} ---")
        out_dir, cmd = job_cmd(job, seed)
        try:
            results[f"seed{seed}"] = execute(job, out_dir, cmd)
        except Exception as exc:
            log(f"[error] seed {seed} raised {exc!r}")
            results[f"seed{seed}"] = False
    status = "complete" if all(results.values()) else "failed"
    log(f"[summary] {results}")
    publish_results(status)


if __name__ == "__main__":
    main()
