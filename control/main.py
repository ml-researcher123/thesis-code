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
WORK_ROOT = WORKING_ROOT / "ace_rag_research_v13_analysis"

JOB_VERSION = "stage3-cross-ragbench-techqa-v1"
RAGBENCH_SUBSETS = ["techqa", "emanual", "expertqa", "covidqa", "pubmedqa"]


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


def find_musique_jsonl() -> Path | None:
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
                name
                for name in zf.namelist()
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


def ensure_ragbench_stage3_support() -> None:
    stage3_path = WORK_ROOT / "experiments" / "run_stage3_router.py"
    text = stage3_path.read_text(encoding="utf-8")
    updated = text.replace(
        'choices=["toy", "hotpotqa", "musique_local"]',
        'choices=["toy", "hotpotqa", "musique_local", "ragbench"]',
    )
    if 'parser.add_argument("--ragbench-subset", default=None)' not in updated:
        updated = updated.replace(
            'parser.add_argument("--split", default="validation")',
            'parser.add_argument("--split", default="validation")\n    parser.add_argument("--ragbench-subset", default=None)',
        )
    if 'if args.dataset == "ragbench":' not in updated:
        updated = updated.replace(
            '    return load_dataset("hotpotqa", split=args.split, limit=args.limit, seed=args.seed)',
            '    if args.dataset == "ragbench":\n'
            '        return load_dataset("ragbench", split=args.split, subset=args.ragbench_subset, limit=args.limit)\n'
            '    return load_dataset("hotpotqa", split=args.split, limit=args.limit, seed=args.seed)',
        )
    if updated != text:
        stage3_path.write_text(updated, encoding="utf-8")
        log("[patch] enabled RAGBench support in experiments/run_stage3_router.py")


def stage3_base_args(out_dir: Path, budget: str, ace_prefix: str = "ace") -> list[str]:
    candidates = f"chunk_packed_{budget},{ace_prefix}_packed_{budget},{ace_prefix}_focused_{budget}"
    return [
        sys.executable,
        "-m",
        "experiments.run_stage3_router",
        "--limit",
        "200",
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
        "--reader-model",
        "Qwen/Qwen2.5-3B-Instruct",
        "--reader-device",
        "cuda",
        "--reader-batch-size",
        "2",
        "--chunk-budget",
        budget,
        "--ace-packed-budget",
        budget,
        "--ace-focused-budget",
        budget,
        "--router-candidates",
        candidates,
        "--out-dir",
        str(out_dir),
    ]


def run_musique_if_available() -> bool:
    musique_path = find_musique_jsonl()
    if musique_path is None:
        log("[cross-dataset] MuSiQue dataset not found in Kaggle input")
        return False
    job_name = "stage3_cross_musique_bridge_budget160_limit200_qwen3b"
    out_dir = WORKING_ROOT / "colab_results" / job_name
    if out_dir.exists() and any(out_dir.glob("*metrics.csv")):
        log(f"[skip] {job_name} already has metrics")
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = stage3_base_args(out_dir, "160", ace_prefix="ace_bridge")
    cmd.extend(
        [
            "--dataset",
            "musique_local",
            "--musique-path",
            str(musique_path),
            "--ace-retriever",
            "bridge",
        ]
    )
    return run(cmd, cwd=WORK_ROOT, timeout=21600, check=False) == 0


def run_ragbench_fallback() -> bool:
    for subset in RAGBENCH_SUBSETS:
        job_name = f"stage3_cross_ragbench_{subset}_budget160_limit200_qwen3b"
        out_dir = WORKING_ROOT / "colab_results" / job_name
        if out_dir.exists() and any(out_dir.glob("*metrics.csv")):
            log(f"[skip] {job_name} already has metrics")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = stage3_base_args(out_dir, "160")
        cmd.extend(
            [
                "--dataset",
                "ragbench",
                "--split",
                "test",
                "--ragbench-subset",
                subset,
                "--ace-retriever",
                "standard",
            ]
        )
        code = run(cmd, cwd=WORK_ROOT, timeout=21600, check=False)
        if code == 0:
            log(f"[cross-dataset] RAGBench subset succeeded: {subset}")
            return True
        log(f"[cross-dataset] RAGBench subset failed, trying next: {subset}")
    log("[cross-dataset] No MuSiQue input and no RAGBench subset succeeded")
    return False


def main() -> None:
    log(f"=== control/main.py: cross-dataset validation ({JOB_VERSION}) ===")
    prepare_project()
    ensure_ragbench_stage3_support()
    if run_musique_if_available():
        log("=== control/main.py done: musique ===")
        return
    if run_ragbench_fallback():
        log("=== control/main.py done: ragbench ===")
        return
    log("=== control/main.py done: no cross-dataset job available ===")


if __name__ == "__main__":
    main()
