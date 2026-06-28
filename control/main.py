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

JOB_VERSION = "stage7-musique-multihop-v1"

# Stage-7: the harder-multi-hop confirmation of the scoped packer claim.
# HotpotQA (Stages 4-6): chunk_submod robustly beats naive packing and beats the
# focused heuristic + MMR at intermediate budgets; the win is mediated by
# answer-in-context and comes from complementary multi-hop coverage amid
# distractors. RAGBench (single-pass) showed no transfer. MuSiQue is a HARDER
# 2-4 hop set, so the mechanism predicts a LARGER submod gap. This job runs the
# 2x3 factorial on MuSiQue at the matched budget 160 (and a tighter 96 to probe
# the curve). If MuSiQue is not mounted under /kaggle/input, it logs and exits
# WITHOUT re-running the finished HotpotQA work.
MUSIQUE_BUDGETS = [160, 96]
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
    """Locate a MuSiQue jsonl mount under /kaggle/input (direct file or inside a zip).

    Matches the exact name ``musique.jsonl`` first, then any ``*.jsonl`` whose name
    looks like a MuSiQue export (e.g. ``musique_ans_v1.0_dev.jsonl``), preferring
    answerable dev splits. Falls back to extracting from a ``*musique*.zip``.
    """
    direct = sorted(INPUT_ROOT.rglob("musique.jsonl"), key=lambda path: len(str(path)))
    if direct:
        return direct[0]

    def _score(p: Path) -> tuple[int, int, int]:
        name = p.name.lower()
        return (
            0 if "musique" in name else 1,
            0 if "dev" in name else (1 if "validation" in name else 2),
            len(str(p)),
        )

    loose = [
        p for p in INPUT_ROOT.rglob("*.jsonl")
        if "musique" in p.name.lower() or "musique" in str(p.parent).lower()
    ]
    # Prefer answerable ('ans') over 'full' when both are present.
    loose.sort(key=lambda p: (0 if "ans" in p.name.lower() else 1, *_score(p)))
    if loose:
        log(f"[musique] using direct jsonl {loose[0]}")
        return loose[0]

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


def run_density_musique(musique_path: Path, budget: int) -> bool:
    job_name = f"stage7_density_packer_musique_budget{budget}_limit{MUSIQUE_LIMIT}_qwen3b"
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
        "--budget", str(budget),
        "--out-dir", str(out_dir),
    ]
    return run(cmd, cwd=WORK_ROOT, timeout=28800, check=False) == 0


def main() -> None:
    log(f"=== control/main.py: Stage-7 MuSiQue multi-hop confirmation ({JOB_VERSION}) ===")
    prepare_project()
    apply_overlay()
    musique_path = find_musique_jsonl()
    if musique_path is None:
        log("[musique] NO MuSiQue mount found under /kaggle/input.")
        log("[musique] Add a Kaggle input containing 'musique.jsonl' or a '*musique*.zip'")
        log("[musique] (with a *dev*.jsonl inside), then re-trigger. Exiting without re-running HotpotQA.")
        log("=== control/main.py done: stage7 musique SKIPPED (not mounted) ===")
        return
    log(f"--- musique found at {musique_path} ---")
    results: dict[str, bool] = {}
    for budget in MUSIQUE_BUDGETS:
        log(f"--- musique budget {budget} ---")
        try:
            results[f"musique_b{budget}"] = run_density_musique(musique_path, budget)
        except Exception as exc:
            log(f"[error] musique budget {budget} raised {exc!r}")
            results[f"musique_b{budget}"] = False
    ok = [s for s, good in results.items() if good]
    bad = [s for s, good in results.items() if not good]
    log(f"=== control/main.py done: stage7 musique; ok={ok} failed={bad} ===")


if __name__ == "__main__":
    main()
