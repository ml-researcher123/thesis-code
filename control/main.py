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

JOBS = [
    {
        "name": "stage3_musique_bridge_routerfix_limit200_qwen3b",
        "dataset": "musique_local",
        "limit": "200",
    },
]


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


def main() -> None:
    log("=== control/main.py: MuSiQue Qwen3B conditional run ===")
    prepare_project()
    for job in JOBS:
        out_dir = WORKING_ROOT / "colab_results" / job["name"]
        if out_dir.exists() and any(out_dir.glob("*metrics.csv")):
            log(f"[skip] {job['name']} already has metrics")
            continue
        musique_path = find_musique_jsonl()
        if musique_path is None:
            log("[skip] MuSiQue dataset not found in Kaggle input. Add musique.jsonl or a MuSiQue zip to run this job.")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"[start] {job['name']}")
        run(
            [
                sys.executable,
                "-m",
                "experiments.run_stage3_router",
                "--dataset",
                "musique_local",
                "--musique-path",
                str(musique_path),
                "--limit",
                job["limit"],
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
                "bridge",
                "--reader-model",
                "Qwen/Qwen2.5-3B-Instruct",
                "--reader-device",
                "cuda",
                "--reader-batch-size",
                "2",
                "--out-dir",
                str(out_dir),
            ],
            cwd=WORK_ROOT,
            timeout=21600,
        )
        log(f"[done] {job['name']}")
    log("=== control/main.py done ===")


if __name__ == "__main__":
    main()
