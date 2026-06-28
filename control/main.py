"""Current Kaggle control job: Stage-4 evidence-density + submodular packing.

This file lives in the GitHub repo under control/main.py. Kaggle's github loop
runs it whenever this file changes.

Design note (how new methodology reaches Kaggle):
  The Kaggle reader does NOT run code straight from GitHub. It extracts a frozen
  project zip (ace_rag_research_kaggle_ready_v13_analysis.zip) that was uploaded
  as a Kaggle input. To ship *new* code (the submodular packer, the density
  diagnostics, the Stage-4 runner) without re-uploading a zip, this control
  script OVERLAYS the files under control/overlay/ (shipped alongside this file
  in the GitHub repo) onto the extracted project before running. The overlay is
  additive: it drops in new modules and the new runner.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
WORK_ROOT = WORKING_ROOT / "ace_rag_research_v13_analysis"
REPO_ROOT = Path(__file__).resolve().parent.parent

JOB_VERSION = "stage4b-mmr-multiseed-hotpotqa-v1"

# Multi-seed robustness for the headline result (chunk_submod beats heuristic
# packers) plus the MMR baseline the factorial was missing. Seed 42 is re-run so
# every seed has the full 2x3 {chunk,ace} x {focused,mmr,submod} table.
SEEDS = [42, 13, 7]


def log(message: str) -> None:
    print(message, flush=True)


def run(cmd: list[str], cwd: Path | None = None, timeout: int | None = None, check: bool = True) -> int:
    log("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert proc.stdout is not None
    start = time.time()
    for line in proc.stdout:
        log(line.rstrip())
        if timeout and time.time() - start > timeout:
            proc.kill()
            raise TimeoutError(f"command timed out after {timeout}s")
    code = proc.wait()
    if check and code != 0:
        raise RuntimeError(f"command failed with code {code}: {' '.join(cmd)}")
    return code


def find_project_zip() -> Path:
    candidates = sorted(
        INPUT_ROOT.rglob("ace_rag_research_kaggle_ready_v13_analysis.zip"),
        key=lambda path: len(str(path)),
    )
    if not candidates:
        candidates = sorted(
            INPUT_ROOT.rglob("ace_rag_research_kaggle_ready*.zip"),
            key=lambda path: (0 if "v13_analysis" in path.name else 1, len(str(path))),
        )
    if not candidates:
        raise FileNotFoundError("Upload an ace_rag_research_kaggle_ready*.zip or extracted project as a Kaggle input.")
    return candidates[0]


def find_extracted_project() -> Path | None:
    candidates = sorted(
        [
            path.parent
            for path in INPUT_ROOT.rglob("requirements-cloud.txt")
            if (path.parent / "experiments" / "run_stage3_router.py").exists()
            and (path.parent / "ace_rag").exists()
        ],
        key=lambda path: (0 if "dataset12" in str(path) else 1, len(str(path))),
    )
    if not candidates:
        return None
    return candidates[0]


def prepare_project() -> None:
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    extracted_project = find_extracted_project()
    if extracted_project is not None:
        shutil.copytree(extracted_project, WORK_ROOT, dirs_exist_ok=True)
        log(f"Copied extracted project {extracted_project} to {WORK_ROOT}")
    else:
        project_zip = find_project_zip()
        with zipfile.ZipFile(project_zip) as zf:
            zf.extractall(WORK_ROOT)
        log(f"Extracted {project_zip} to {WORK_ROOT}")
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-cloud.txt"], cwd=WORK_ROOT, timeout=900)


def apply_overlay() -> None:
    """Drop new/updated source files from the GitHub repo onto the extracted project."""
    overlay_root = REPO_ROOT / "control" / "overlay"
    if not overlay_root.exists():
        log(f"[overlay] no overlay directory at {overlay_root}; running stock project")
        return
    copied: list[str] = []
    for src in sorted(overlay_root.rglob("*.py")):
        rel = src.relative_to(overlay_root)
        dst = WORK_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(rel).replace("\\", "/"))
    log(f"[overlay] applied {len(copied)} file(s) to {WORK_ROOT}: {copied}")


def run_density_hotpotqa(seed: int) -> bool:
    job_name = f"stage4b_density_mmr_hotpotqa_seed{seed}_budget160_limit500_qwen3b"
    out_dir = WORKING_ROOT / "colab_results" / job_name
    if out_dir.exists() and any(out_dir.glob("*metrics.csv")):
        log(f"[skip] {job_name} already has metrics")
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "experiments.run_density_router",
        "--dataset",
        "hotpotqa",
        "--split",
        "validation",
        "--seed",
        str(seed),
        "--limit",
        "500",
        "--embedder",
        "sentence-transformers",
        "--embedding-model",
        "BAAI/bge-small-en-v1.5",
        "--embed-device",
        "cuda",
        "--compressor",
        "truncate",
        "--compress-dims",
        "320",
        "--top-k-nodes",
        "48",
        "--max-expanded-docs",
        "5",
        "--ace-retriever",
        "standard",
        "--reader-backend",
        "hf",
        "--reader-model",
        "Qwen/Qwen2.5-3B-Instruct",
        "--reader-device",
        "cuda",
        "--reader-batch-size",
        "2",
        "--budget",
        "160",
        "--mmr-lambda",
        "0.7",
        "--out-dir",
        str(out_dir),
    ]
    return run(cmd, cwd=WORK_ROOT, timeout=28800, check=False) == 0


def main() -> None:
    log(f"=== control/main.py: Stage-4b MMR + multi-seed density packing ({JOB_VERSION}) ===")
    prepare_project()
    apply_overlay()
    results: dict[int, bool] = {}
    for seed in SEEDS:
        log(f"--- seed {seed} ---")
        results[seed] = run_density_hotpotqa(seed)
    ok = [s for s, good in results.items() if good]
    bad = [s for s, good in results.items() if not good]
    log(f"=== control/main.py done: stage4b mmr multiseed; ok={ok} failed={bad} ===")


if __name__ == "__main__":
    main()
