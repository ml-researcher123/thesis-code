"""Current Kaggle control job: Stage-10b 2WikiMultiHopQA full factorial (Run B).

This file lives in the GitHub repo under control/main.py. Kaggle's github loop
runs it whenever this file changes.

Design note (how new methodology reaches Kaggle):
  The Kaggle reader does NOT run code straight from GitHub. It extracts a frozen
  project zip (ace_rag_research_kaggle_ready_v13_analysis.zip) that was uploaded
  as a Kaggle input. To ship *new* code (the submodular packer, the density
  diagnostics, the Stage-4 runner, and the 2WikiMultiHopQA loader) without
  re-uploading a zip, this control script OVERLAYS the files under
  control/overlay/ (shipped alongside this file in the GitHub repo) onto the
  extracted project before running. The overlay is additive.

Stage-10b is Run B: a SECOND positive multi-hop dataset for the packer claim.
The Stage-10a retrieval gate cleared (2Wiki chunk all_gold@5 = 0.43, well above
MuSiQue's 0.18 bottleneck and clear of the >~0.40 bar), so retrieval surfaces
the multi-hop gold evidence and the packer actually has something to assemble.
This job runs the EXACT HotpotQA Stage-4b 2x3 factorial (chunk/ace x
focused/mmr/submod, plus naive packed) -- same retrieval (top-k 5 / nodes 48 /
expand 5), same budget 160, same Qwen2.5-3B reader, same 3 seeds {42,13,7} --
changing ONLY the dataset to 2Wiki. We deliberately use the 3B reader (not the
Stage-9 7B) because Run B tests whether the win REPLICATES at the scale where it
worked; a 7B run would confound dataset transfer with the Stage-9 reader-scale
null. Outcomes:
  - submod beats best heuristic (sig, 3 seeds) => second positive dataset; the
    HotpotQA win is not a one-dataset artifact; strong paper strengthener.
  - null but submod still beats naive packed  => packing-objective survives,
    edge-over-heuristic is HotpotQA-specific; report as scoped.
  - flat null                                  => 2Wiki joins the scope map.
The paper headline is NOT touched until the user approves the win/lose framing.
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

JOB_VERSION = "stage10b-2wiki-factorial-3b-v1"

# Everything below matches HotpotQA Stage-4b EXACTLY except the dataset, so
# chunk_submod-vs-chunk_focused on 2Wiki is directly comparable to the HotpotQA
# headline. Retrieval (top-k 5 / nodes 48 / expand 5) also matches the Stage-10a
# gate that produced all_gold@5 = 0.43, so the factorial sees the same retrieval
# the gate measured.
READER_MODEL = "Qwen/Qwen2.5-3B-Instruct"
TWO_WIKI_BUDGET = 160
TWO_WIKI_LIMIT = 500
TWO_WIKI_SEEDS = [42, 13, 7]
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


def run_2wiki_factorial(twowiki_file: Path, seed: int) -> bool:
    job_name = f"stage10b_density_packer_2wiki_qwen3b_budget{TWO_WIKI_BUDGET}_limit{TWO_WIKI_LIMIT}_seed{seed}"
    out_dir = WORKING_ROOT / "colab_results" / job_name
    if out_dir.exists() and any(out_dir.glob("*metrics.csv")):
        log(f"[skip] {job_name} already has metrics")
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "experiments.run_density_router",
        "--dataset", "2wiki_local",
        "--twowiki-path", str(twowiki_file),
        "--limit", str(TWO_WIKI_LIMIT),
        "--seed", str(seed),
        "--ace-retriever", "standard",
        "--top-k", str(TWO_WIKI_TOP_K),
        "--top-k-nodes", str(TWO_WIKI_TOP_K_NODES),
        "--max-expanded-docs", str(TWO_WIKI_MAX_EXPANDED),
        "--embedder", "sentence-transformers",
        "--embedding-model", "BAAI/bge-small-en-v1.5",
        "--embed-device", "cuda",
        "--compressor", "truncate",
        "--compress-dims", "320",
        "--reader-backend", "hf",
        "--reader-model", READER_MODEL,
        "--reader-device", "cuda",
        "--reader-batch-size", "2",
        "--mmr-lambda", "0.7",
        "--budget", str(TWO_WIKI_BUDGET),
        "--out-dir", str(out_dir),
    ]
    return run(cmd, cwd=WORK_ROOT, timeout=28800, check=False) == 0


def main() -> None:
    log(f"=== control/main.py: Stage-10b 2Wiki full factorial (Run B) ({JOB_VERSION}) ===")
    log(f"--- reader={READER_MODEL} budget={TWO_WIKI_BUDGET} limit={TWO_WIKI_LIMIT} seeds={TWO_WIKI_SEEDS} ---")
    prepare_project()
    apply_overlay()
    twowiki_file = find_2wiki_file()
    results: dict[str, bool] = {}
    for seed in TWO_WIKI_SEEDS:
        log(f"--- 2wiki factorial 3B seed {seed} ---")
        try:
            results[f"2wiki_seed{seed}"] = run_2wiki_factorial(twowiki_file, seed)
        except Exception as exc:
            log(f"[error] 2wiki factorial seed {seed} raised {exc!r}")
            results[f"2wiki_seed{seed}"] = False
    ok = [s for s, good in results.items() if good]
    bad = [s for s, good in results.items() if not good]
    log(f"=== control/main.py done: stage10b 2wiki factorial; ok={ok} failed={bad} ===")


if __name__ == "__main__":
    main()
