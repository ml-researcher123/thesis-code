"""Current Kaggle control job.

This file is meant to live in the GitHub repo under control/main.py.
Kaggle's github loop runs it whenever this file changes.
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
WORK_ROOT = WORKING_ROOT / "ace_rag_research_v12_routerfix"
RESULTS = WORKING_ROOT / "colab_results" / "stage3_hotpotqa_seed42_routerfix_limit500_qwen3b"


def log(message: str) -> None:
    print(message, flush=True)


def run(cmd: list[str], cwd: Path | None = None, timeout: int | None = None) -> None:
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
    if code != 0:
        raise RuntimeError(f"command failed with code {code}: {' '.join(cmd)}")


def find_project_zip() -> Path:
    candidates = sorted(
        INPUT_ROOT.rglob("ace_rag_research_kaggle_ready_v12_routerfix.zip"),
        key=lambda path: len(str(path)),
    )
    if not candidates:
        candidates = sorted(
            INPUT_ROOT.rglob("ace_rag_research_kaggle_ready*.zip"),
            key=lambda path: (0 if "v12_routerfix" in path.name else 1, len(str(path))),
        )
    if not candidates:
        raise FileNotFoundError("Upload ace_rag_research_kaggle_ready_v12_routerfix.zip as a Kaggle input dataset.")
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


def main() -> None:
    log("=== control/main.py: HotpotQA seed42 limit500 Qwen3B ===")
    prepare_project()
    RESULTS.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-m",
            "experiments.run_stage3_router",
            "--dataset",
            "hotpotqa",
            "--split",
            "validation",
            "--seed",
            "42",
            "--limit",
            "500",
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
            "--reader-model",
            "Qwen/Qwen2.5-3B-Instruct",
            "--reader-device",
            "cuda",
            "--reader-batch-size",
            "2",
            "--out-dir",
            str(RESULTS),
        ],
        cwd=WORK_ROOT,
        timeout=21600,
    )
    log("=== control/main.py done ===")


if __name__ == "__main__":
    main()
