"""One-shot Colab runner for Stage-15 (semantic answer-in-context), re-run to
validate the NLI hypothesis-framing fix.

Unlike control/main.py (built for Kaggle's unattended polling loop, reading
/kaggle/input mounts), this is a single pass meant to be run from a Colab
notebook cell: extract the frozen project zip that now ships IN this repo
(control/ace_rag_research_kaggle_ready_v13_analysis.zip, so no manual upload
is needed), apply the overlay, run the 3 Stage-15 jobs, then publish results
back to GitHub the same way Kaggle's loop does.

All three Stage-15 datasets (RAGBench/ExpertQA, RAGBench/CovidQA, HotpotQA)
load straight from the Hugging Face Hub -- unlike Stage-13 (MuSiQue) or
Stage-14 (2Wiki), nothing needs to be mounted or gdown'd. That is what makes
this stage the cheap one to migrate first.

Usage (from a Colab cell, after `!git clone` of this repo into /content/thesis-code):
    %cd /content/thesis-code
    !python control/colab_stage15_rerun.py
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
SEEDS = [42, 13]
SEMANTIC_TARGETS = [
    ("expertqa", "ragbench", "expertqa", "test"),
    ("covidqa", "ragbench", "covidqa", "test"),
    ("hotpotqa", "hotpotqa", None, "validation"),
]
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
TOP_K = 5
TOP_K_NODES = 48
MAX_EXPANDED = 5
LIMIT = 500


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


def job_cmd(job_name: str, dataset: str, split: str, ragbench_subset: str | None, seed: int) -> tuple[Path, list[str]]:
    out_dir = RESULT_ROOT / job_name
    cmd = [
        sys.executable, "-m", "experiments.run_density_router",
        "--dataset", dataset,
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
        "--split", split,
        "--semantic-aic", "--nli-model", NLI_MODEL, "--nli-batch-size", "32",
    ]
    if ragbench_subset:
        cmd += ["--ragbench-subset", ragbench_subset]
    return out_dir, cmd


def execute(job_name: str, out_dir: Path, cmd: list[str]) -> bool:
    if out_dir.exists() and any(out_dir.glob("*metrics.csv")):
        log(f"[skip] {job_name} already has metrics")
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    return run(cmd, cwd=WORK_ROOT, check=False) == 0


def publish_results(status: str) -> None:
    """Commit + push RESULT_ROOT into kaggle_results/, same layout Kaggle uses."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / "kaggle_results" / f"{timestamp}_colab_stage15"
    latest_dir = REPO_ROOT / "kaggle_results" / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    if RESULT_ROOT.exists():
        shutil.copytree(RESULT_ROOT, run_dir / "colab_results", dirs_exist_ok=True)
    (run_dir / "run_metadata.json").write_text(
        f'{{"status": "{status}", "stage": "stage15-semantic-aic-rerun", '
        f'"completed_at": "{time.strftime("%Y-%m-%d %H:%M:%S")}", "runner": "colab"}}\n',
        encoding="utf-8",
    )
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)

    run(["git", "config", "user.email", "colab-runner@example.local"], cwd=REPO_ROOT, check=False)
    run(["git", "config", "user.name", "colab-runner"], cwd=REPO_ROOT, check=False)
    run(["git", "add", "kaggle_results"], cwd=REPO_ROOT, check=False)
    code = run(["git", "commit", "-m", f"Colab Stage-15 re-run {timestamp} {status}"], cwd=REPO_ROOT, check=False)
    if code == 0:
        run(["git", "fetch", "origin", "main"], cwd=REPO_ROOT, check=False)
        run(["git", "rebase", "origin/main"], cwd=REPO_ROOT, check=False)
        push_code = run(["git", "push", "origin", "main"], cwd=REPO_ROOT, check=False)
        if push_code == 0:
            log("[publish] pushed to origin/main")
        else:
            log("[publish] push failed -- results are committed locally, push manually")
    else:
        log("[publish] nothing new to commit")


def main() -> None:
    prepare_project()
    apply_overlay()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}
    for label, dataset, subset, split in SEMANTIC_TARGETS:
        for seed in SEEDS:
            job = f"stage15_semanticaic_{label}_qwen3b_budget{BUDGET}_seed{seed}"
            log(f"--- semantic AiC {label} seed {seed} ---")
            out_dir, cmd = job_cmd(job, dataset, split, subset, seed)
            try:
                results[f"{label}_seed{seed}"] = execute(job, out_dir, cmd)
            except Exception as exc:
                log(f"[error] {label} seed {seed} raised {exc!r}")
                results[f"{label}_seed{seed}"] = False
    status = "complete" if all(results.values()) else "failed"
    log(f"[summary] {results}")
    publish_results(status)


if __name__ == "__main__":
    main()
