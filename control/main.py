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

JOB_VERSION = "stage6-budgetcurve-musique-v1"

# Stage-5 (RAGBench) showed the packer win does NOT transfer to single-pass QA,
# consistent with the mechanism (submod wins via complementary multi-hop coverage
# amid distractors). Stage-6 tests the two falsifiable predictions of the scoped
# claim:
#   (a) on HotpotQA the submod advantage GROWS as the budget tightens;
#   (b) on a HARDER multi-hop dataset (MuSiQue) submod wins by a LARGER margin.
# The budget curve is guaranteed; MuSiQue runs only if a mount is found.
BUDGET_CURVE = [96, 128, 224]   # 160 already done in Stage-4b seed 42
BUDGET_SEED = 42
MUSIQUE_BUDGET = 160
MUSIQUE_LIMIT = 500


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


_COMMON_RETRIEVAL_ARGS = [
    "--embedder", "sentence-transformers",
    "--embedding-model", "BAAI/bge-small-en-v1.5",
    "--embed-device", "cuda",
    "--compressor", "truncate",
    "--compress-dims", "320",
    "--top-k-nodes", "48",
    "--max-expanded-docs", "5",
    "--reader-backend", "hf",
    "--reader-model", "Qwen/Qwen2.5-3B-Instruct",
    "--reader-device", "cuda",
    "--reader-batch-size", "2",
    "--mmr-lambda", "0.7",
]


def find_musique_jsonl() -> Path | None:
    """Locate a MuSiQue jsonl mount under /kaggle/input (direct file or inside a zip)."""
    direct = sorted(INPUT_ROOT.rglob("musique.jsonl"), key=lambda path: len(str(path)))
    if direct:
        return direct[0]
    data_root = WORKING_ROOT / "data" / "raw"
    data_root.mkdir(parents=True, exist_ok=True)
    zip_candidates = sorted(
        [path for path in INPUT_ROOT.rglob("*.zip") if "musique" in path.name.lower()],
        key=lambda path: len(str(path)),
    )
    for zip_path in zip_candidates:
        with zipfile.ZipFile(zip_path) as zf:
            jsonl_names = [
                name for name in zf.namelist()
                if name.endswith(".jsonl") and ("dev" in name.lower() or "musique" in name.lower())
            ]
            if not jsonl_names:
                continue
            preferred = sorted(jsonl_names, key=lambda name: (0 if "dev" in name.lower() else 1, len(name)))[0]
            out_path = data_root / "musique.jsonl"
            with zf.open(preferred) as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            log(f"Extracted MuSiQue {preferred} from {zip_path} to {out_path}")
            return out_path
    return None


def run_density_hotpotqa_budget(budget: int) -> bool:
    job_name = f"stage6_density_packer_hotpotqa_seed{BUDGET_SEED}_budget{budget}_limit500_qwen3b"
    out_dir = WORKING_ROOT / "colab_results" / job_name
    if out_dir.exists() and any(out_dir.glob("*metrics.csv")):
        log(f"[skip] {job_name} already has metrics")
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "experiments.run_density_router",
        "--dataset", "hotpotqa",
        "--split", "validation",
        "--seed", str(BUDGET_SEED),
        "--limit", "500",
        "--ace-retriever", "standard",
        *_COMMON_RETRIEVAL_ARGS,
        "--budget", str(budget),
        "--out-dir", str(out_dir),
    ]
    return run(cmd, cwd=WORK_ROOT, timeout=28800, check=False) == 0


def run_density_musique(musique_path: Path) -> bool:
    job_name = f"stage6_density_packer_musique_budget{MUSIQUE_BUDGET}_limit{MUSIQUE_LIMIT}_qwen3b"
    out_dir = WORKING_ROOT / "colab_results" / job_name
    if out_dir.exists() and any(out_dir.glob("*metrics.csv")):
        log(f"[skip] {job_name} already has metrics")
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "experiments.run_density_router",
        "--dataset", "musique_local",
        "--musique-path", str(musique_path),
        "--limit", str(MUSIQUE_LIMIT),
        "--ace-retriever", "standard",
        *_COMMON_RETRIEVAL_ARGS,
        "--budget", str(MUSIQUE_BUDGET),
        "--out-dir", str(out_dir),
    ]
    return run(cmd, cwd=WORK_ROOT, timeout=28800, check=False) == 0


def main() -> None:
    log(f"=== control/main.py: Stage-6 budget curve + MuSiQue ({JOB_VERSION}) ===")
    prepare_project()
    apply_overlay()
    results: dict[str, bool] = {}
    for budget in BUDGET_CURVE:
        log(f"--- hotpotqa budget {budget} ---")
        try:
            results[f"hotpotqa_b{budget}"] = run_density_hotpotqa_budget(budget)
        except Exception as exc:
            log(f"[error] hotpotqa budget {budget} raised {exc!r}")
            results[f"hotpotqa_b{budget}"] = False
    musique_path = find_musique_jsonl()
    if musique_path is not None:
        log(f"--- musique found at {musique_path} ---")
        try:
            results["musique"] = run_density_musique(musique_path)
        except Exception as exc:
            log(f"[error] musique raised {exc!r}")
            results["musique"] = False
    else:
        log("[musique] no MuSiQue mount found under /kaggle/input; skipping (budget curve still ran)")
    ok = [s for s, good in results.items() if good]
    bad = [s for s, good in results.items() if not good]
    log(f"=== control/main.py done: stage6 budgetcurve+musique; ok={ok} failed={bad} ===")


if __name__ == "__main__":
    main()
