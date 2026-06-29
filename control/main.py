"""Current Kaggle control job: Stage-11 HotpotQA reader-scale ladder (14B + 7B-4bit bridge).

This file lives in the GitHub repo under control/main.py. Kaggle's github loop
runs it whenever this file changes.

Design note (how new methodology reaches Kaggle):
  The Kaggle reader does NOT run code straight from GitHub. It extracts a frozen
  project zip (ace_rag_research_kaggle_ready_v13_analysis.zip) uploaded as a
  Kaggle input, then this control script OVERLAYS the *.py files under
  control/overlay/ (the submodular packer, density diagnostics, the Stage-4
  runner, and the device_map / load_in_4bit reader loader) onto the extracted
  project before running. The overlay is additive.

Stage-11 extends the reader-scale axis (§6.5) from a single 7B point into a
LADDER: Qwen2.5 at 3B (done, §5), 7B (done fp16, Stage-9), and now 14B, plus a
7B-4bit quantization control. The goal is to trace exactly how the packer's edge
over the focused heuristic is absorbed as the reader grows, and to confirm the
trend is a SCALE effect, not a quantization artifact:

  - 7B-4bit bridge: same model/data/seeds as the Stage-9 7B-fp16 run, changing
    ONLY precision. If the submod-vs-focused contrast looks like 7B-fp16, then
    4-bit quantization (required to fit 14B/32B on the 2xT4 box) is not driving
    the ladder.
  - 14B-4bit: the next rung. 14B nf4 (~8 GB) fits the box; device_map="auto"
    shards if needed.

32B is run separately (Stage-12, likely single-seed) because it is much slower
on Turing T4s (no native int4) and may not finish a full factorial in one
session. Per-seed skip-logic (checks for an existing *metrics.csv) makes every
run resumable across Kaggle's 9-hour session limit.

Everything except the reader matches the §5 HotpotQA factorial exactly
(top-k 5 / nodes 48 / expand 5, budget 160, seeds {42,13}) so the contrasts are
directly comparable across rungs.
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

JOB_VERSION = "stage11-hotpotqa-reader-ladder-14b-v1"

HOTPOT_BUDGET = 160
HOTPOT_LIMIT = 500
HOTPOT_SEEDS = [42, 13]
HOTPOT_TOP_K = 5
HOTPOT_TOP_K_NODES = 48
HOTPOT_MAX_EXPANDED = 5

# Ladder rungs run in this order: the cheaper 7B-4bit bridge first (so the
# quantization control lands even if the session runs short), then 14B-4bit.
# (label, model, reader_batch_size)
LADDER_RUNGS = [
    ("reader7b_4bit", "Qwen/Qwen2.5-7B-Instruct", 2),
    ("reader14b_4bit", "Qwen/Qwen2.5-14B-Instruct", 1),
]


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
    # 4-bit (nf4) loading needs bitsandbytes, which is NOT in the frozen zip's
    # requirements. Install it explicitly; >=0.43 supports Turing (T4, sm_75).
    run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.43.0"], cwd=WORK_ROOT, timeout=600, check=False)


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


def run_ladder_rung(label: str, model: str, batch_size: int, seed: int) -> bool:
    job_name = f"stage11_density_packer_hotpotqa_{label}_budget{HOTPOT_BUDGET}_limit{HOTPOT_LIMIT}_seed{seed}"
    out_dir = WORKING_ROOT / "colab_results" / job_name
    if out_dir.exists() and any(out_dir.glob("*metrics.csv")):
        log(f"[skip] {job_name} already has metrics")
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "experiments.run_density_router",
        "--dataset", "hotpotqa",
        "--split", "validation",
        "--limit", str(HOTPOT_LIMIT),
        "--seed", str(seed),
        "--ace-retriever", "standard",
        "--top-k", str(HOTPOT_TOP_K),
        "--top-k-nodes", str(HOTPOT_TOP_K_NODES),
        "--max-expanded-docs", str(HOTPOT_MAX_EXPANDED),
        "--embedder", "sentence-transformers",
        "--embedding-model", "BAAI/bge-small-en-v1.5",
        "--embed-device", "cuda",
        "--compressor", "truncate",
        "--compress-dims", "320",
        "--reader-backend", "hf",
        "--reader-model", model,
        "--reader-device", "cuda",
        "--reader-load-4bit",            # nf4 + double-quant; auto device_map
        "--reader-batch-size", str(batch_size),
        "--mmr-lambda", "0.7",
        "--budget", str(HOTPOT_BUDGET),
        "--out-dir", str(out_dir),
    ]
    return run(cmd, cwd=WORK_ROOT, timeout=28800, check=False) == 0


def main() -> None:
    log(f"=== control/main.py: Stage-11 HotpotQA reader-scale ladder ({JOB_VERSION}) ===")
    log(f"--- rungs={[r[0] for r in LADDER_RUNGS]} budget={HOTPOT_BUDGET} limit={HOTPOT_LIMIT} seeds={HOTPOT_SEEDS} ---")
    prepare_project()
    apply_overlay()
    results: dict[str, bool] = {}
    for label, model, batch_size in LADDER_RUNGS:
        for seed in HOTPOT_SEEDS:
            log(f"--- ladder {label} ({model}) seed {seed} ---")
            try:
                results[f"{label}_seed{seed}"] = run_ladder_rung(label, model, batch_size, seed)
            except Exception as exc:
                log(f"[error] {label} seed {seed} raised {exc!r}")
                results[f"{label}_seed{seed}"] = False
    ok = [s for s, good in results.items() if good]
    bad = [s for s, good in results.items() if not good]
    log(f"=== control/main.py done: stage11 reader ladder; ok={ok} failed={bad} ===")


if __name__ == "__main__":
    main()
