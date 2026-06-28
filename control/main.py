"""Current Kaggle control job: Stage-10a 2WikiMultiHopQA retrieval gate (Run B prereq).

This file lives in the GitHub repo under control/main.py. Kaggle's github loop
runs it whenever this file changes.

Design note (how new methodology reaches Kaggle):
  The Kaggle reader does NOT run code straight from GitHub. It extracts a frozen
  project zip (ace_rag_research_kaggle_ready_v13_analysis.zip) that was uploaded
  as a Kaggle input. To ship *new* code (the submodular packer, the density
  diagnostics, the Stage-4 runner, the device_map reader loader, and now the
  2WikiMultiHopQA loader + retrieval-only gate) without re-uploading a zip, this
  control script OVERLAYS the files under control/overlay/ (shipped alongside
  this file in the GitHub repo) onto the extracted project before running.

Stage-10a is a CHEAP PREREQUISITE GATE for Run B (a second positive multi-hop
dataset). Before paying for a full 2x3 factorial + 3B reader on 2Wiki, we run
ONLY retrieval and measure whether the gold evidence is actually surfaced. The
packer cannot assemble evidence retrieval never returned -- this is exactly the
condition MuSiQue failed (all_gold@5=0.184 -> packer null). If 2Wiki clears the
bar, a follow-up job runs the full factorial; if not, 2Wiki joins the scope map
as another retrieval-bottlenecked boundary. No reader is loaded in this job.
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

JOB_VERSION = "stage10a-2wiki-retrieval-gate-v1"

# Retrieval settings match HotpotQA Stage-4b exactly so all_gold@5 is directly
# comparable across datasets (HotpotQA ~0.76 -> packer wins; MuSiQue 0.184 ->
# packer null). Decision is made OFFLINE after reading chunk all_gold@5:
#   >~ 0.40  -> retrieval surfaces multi-hop evidence; run the full Run B factorial.
#   ~  0.18  -> MuSiQue-like bottleneck; abort Run B, fold 2Wiki into the scope map.
TWO_WIKI_LIMIT = 500
TWO_WIKI_SEED = 42
TWO_WIKI_TOP_K = 5
TWO_WIKI_TOP_K_NODES = 48
TWO_WIKI_MAX_EXPANDED = 5


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


def find_2wiki_file() -> Path:
    """Locate a mounted 2WikiMultiHopQA dev split (.json/.jsonl) under /kaggle/input.

    2Wiki releases name the dev split variously (dev.json inside a 2wiki folder,
    2wiki_dev.json, 2wikimultihopqa_dev.jsonl, ...). We match on a 2wiki/wikimultihop
    token in the path plus a dev/valid token, preferring the shortest path.
    """
    cands: list[Path] = []
    for path in INPUT_ROOT.rglob("*.json*"):
        if not path.is_file() or path.suffix.lower() not in (".json", ".jsonl"):
            continue
        sp = str(path).lower()
        has_2wiki = ("2wiki" in sp) or ("wikimultihop" in sp) or ("2_wiki" in sp)
        has_split = ("dev" in sp) or ("valid" in sp)
        if has_2wiki and has_split:
            cands.append(path)
    cands = sorted(set(cands), key=lambda p: (len(str(p)), str(p)))
    if not cands:
        raise FileNotFoundError(
            "No 2WikiMultiHopQA dev file found. Mount the 2Wiki dev split (.json/.jsonl) "
            "as a Kaggle input; the path must contain '2wiki'/'wikimultihop' and 'dev'/'valid'."
        )
    log(f"[2wiki] using {cands[0]}  ({len(cands)} candidate(s))")
    return cands[0]


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


def run_2wiki_gate(twowiki_file: Path) -> bool:
    job_name = f"stage10a_2wiki_retrieval_gate_limit{TWO_WIKI_LIMIT}_seed{TWO_WIKI_SEED}"
    out_dir = WORKING_ROOT / "colab_results" / job_name
    if out_dir.exists() and any(out_dir.glob("*retrieval_gate*.csv")):
        log(f"[skip] {job_name} already has a gate CSV")
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "experiments.run_density_router",
        "--dataset", "2wiki_local",
        "--twowiki-path", str(twowiki_file),
        "--retrieval-only",
        "--limit", str(TWO_WIKI_LIMIT),
        "--seed", str(TWO_WIKI_SEED),
        "--ace-retriever", "standard",
        "--top-k", str(TWO_WIKI_TOP_K),
        "--top-k-nodes", str(TWO_WIKI_TOP_K_NODES),
        "--max-expanded-docs", str(TWO_WIKI_MAX_EXPANDED),
        "--embedder", "sentence-transformers",
        "--embedding-model", "BAAI/bge-small-en-v1.5",
        "--embed-device", "cuda",
        "--compressor", "truncate",
        "--compress-dims", "320",
        "--out-dir", str(out_dir),
    ]
    return run(cmd, cwd=WORK_ROOT, timeout=3600, check=False) == 0


def main() -> None:
    log(f"=== control/main.py: Stage-10a 2Wiki retrieval gate ({JOB_VERSION}) ===")
    log(f"--- limit={TWO_WIKI_LIMIT} seed={TWO_WIKI_SEED} top-k={TWO_WIKI_TOP_K} "
        f"nodes={TWO_WIKI_TOP_K_NODES} expand={TWO_WIKI_MAX_EXPANDED} (no reader) ---")
    prepare_project()
    apply_overlay()
    twowiki_file = find_2wiki_file()
    ok = run_2wiki_gate(twowiki_file)
    log(f"=== control/main.py done: stage10a 2wiki retrieval gate; ok={ok} ===")
    log("Read chunk all_gold@5 from the gate CSV / [gate] log line to decide on the full Run B factorial.")


if __name__ == "__main__":
    main()
