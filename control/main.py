"""Kaggle control job: post-arXiv strengthening runs (Stages 12-16).

This file lives in the GitHub repo under control/main.py. Kaggle's github loop
runs it whenever this file changes.

Design note (how new methodology reaches Kaggle):
  The Kaggle reader does NOT run code straight from GitHub. It extracts a frozen
  project zip (ace_rag_research_kaggle_ready_v13_analysis.zip) uploaded as a
  Kaggle input, then this control script OVERLAYS the *.py files under
  control/overlay/ onto the extracted project before running. The overlay is
  additive, so new diagnostics (e.g. the semantic answer-in-context variant in
  ace_rag/context_quality.py) ship without re-uploading the zip.

------------------------------------------------------------------------------
HOW TO USE: set STAGE below to one phase, commit, push. The loop picks up the
change and runs it. Every job is skip-guarded on an existing *metrics.csv, so a
stage is resumable across Kaggle's 9-hour session limit -- re-pushing the same
STAGE resumes rather than restarts.
------------------------------------------------------------------------------

Stage plan (ordered by how much each lifts the paper):

  stage12-cross-family   [RUN FIRST] The headline packer win currently rests on
                         a single reader family (Qwen2.5-3B). Replicates the
                         budget-160 HotpotQA factorial on Falcon3-3B and
                         Phi-3.5-mini, 3 seeds each. If +0.022 holds across
                         families, the positive claim stops being "a Qwen
                         quirk" and becomes cross-family.

  stage13-multiseed      Every current positive is 3-seed but the budget sweep
                         (96/128/224) and the MuSiQue runs are single-seed, so
                         the inverted-U rests on single points. Pure compute;
                         removes the easiest reviewer objection in the paper.

  stage14-2wiki-budget   Searches for a SECOND dataset where the packer wins.
                         2Wiki cleared the retrieval bar (all-gold@5=0.43) but
                         was null at budget 160. Sweeps tighter/looser budgets
                         to find a regime where all four scope conditions hold.

  stage15-semantic-aic   The diagnostic is span-based, so it is degenerate on
                         long free-form answers (ExpertQA). Scores the new NLI
                         entailment variant alongside the verbatim one, to show
                         (a) they agree where spans exist and (b) the semantic
                         version reclaims ExpertQA.

  stage16-32b            One more reader-scale rung. The 14B reversal is
                         currently a two-point story; 32B either confirms the
                         monotone "curation stops paying, then costs"
                         trajectory or shows it plateaus.

Everything except the varied axis matches the paper's HotpotQA factorial
exactly (top-k 5 / nodes 48 / expand 5, budget 160) so contrasts stay
comparable to the published numbers.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


# ===========================================================================
# SELECT THE PHASE TO RUN
# ===========================================================================
STAGE = "stage14-2wiki-budget"
# ===========================================================================


INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
WORK_ROOT = WORKING_ROOT / "ace_rag_research_v13_analysis"
REPO_ROOT = Path(__file__).resolve().parent.parent

JOB_VERSION = f"{STAGE}-v1"

# --- shared factorial settings (identical to the published HotpotQA run) ---
BUDGET = 160
LIMIT = 500
TOP_K = 5
TOP_K_NODES = 48
MAX_EXPANDED = 5
SEEDS_3 = [42, 13, 7]

# --- stage 12: cross-family reader replication -----------------------------
# Two non-Qwen families at the same ~3B scale as the headline result, fp16
# (3B-class fits a single T4, so this matches the published 3B fp16 setup
# exactly and no quantization control is needed). Both are ungated (Apache-2.0 /
# open), so no HF_TOKEN or licence acceptance is required -- Falcon3 replaces the
# gated Llama-3.2 to remove that setup friction.
# (label, model, batch_size, gated)
CROSS_FAMILY_READERS = [
    ("falcon3b", "tiiuae/Falcon3-3B-Instruct", 2, False),
    ("phi35mini", "microsoft/Phi-3.5-mini-instruct", 2, False),
]

# --- stage 13: multi-seed the single-seed sweeps ---------------------------
SWEEP_BUDGETS = [96, 128, 224]
SWEEP_SEEDS = [13, 7]          # seed 42 already published
MUSIQUE_SEEDS = [13, 7]        # seed 42 already published

# --- stage 14: 2Wiki win-regime search -------------------------------------
# Budget 160 is already 3-seed and null. Tighter budgets make complementarity
# matter more; 224 is the loose control.
TWO_WIKI_BUDGETS = [96, 128, 224]
TWO_WIKI_SEEDS = [42, 13]
TWO_WIKI_LIMIT = 500

# --- stage 15: semantic answer-in-context ----------------------------------
# ExpertQA is the target (span diagnostic degenerate there); CovidQA and
# HotpotQA give the agreement check where spans DO exist.
# NOTE: RAGBench must use --split test.
SEMANTIC_TARGETS = [
    ("expertqa", "ragbench", "expertqa", "test"),
    ("covidqa", "ragbench", "covidqa", "test"),
    ("hotpotqa", "hotpotqa", None, "validation"),
]
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

# --- stage 16: 32B rung ----------------------------------------------------
# 32B nf4 (~18GB) needs both T4s via device_map=auto and is slow on Turing
# (no native int4), so single-seed by design.
READER_32B = "Qwen/Qwen2.5-32B-Instruct"
SEEDS_32B = [42]


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


def get_secret(name: str) -> str | None:
    """Env var first, then Kaggle secrets (env lets a notebook cell override)."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def setup_hf_auth() -> bool:
    """Export a Hugging Face token if one is available.

    Llama-3.2 is gated: without an accepted licence + token the download 401s.
    Returns whether a token was found so gated models can be skipped cleanly
    rather than burning a session on a guaranteed failure.
    """
    token = get_secret("HF_TOKEN") or get_secret("HUGGINGFACE_TOKEN")
    if not token:
        log("[hf] no HF_TOKEN found; gated models (Llama) will be skipped")
        return False
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    log("[hf] token configured for gated model access")
    return True


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
    """Locate a mounted 2WikiMultiHopQA dev split (.json/.jsonl) under /kaggle/input."""
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


def find_musique_jsonl() -> Path | None:
    """Locate a MuSiQue jsonl mount under /kaggle/input (direct file or inside a zip)."""
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
    # requirements. >=0.43 supports Turing (T4, sm_75).
    run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.43.0"], cwd=WORK_ROOT, timeout=600, check=False)
    # NOTE: two earlier attempts pinned transformers==4.44.2 (to keep
    # DynamicCache.seen_tokens for Phi-3.5's outdated bundled code) then tried
    # to force tokenizers past that pin's declared ceiling (to parse Falcon3's
    # newer tokenizer.json format). Both failed: transformers has a hard
    # *runtime* version guard (dependency_versions_check.py) that refuses to
    # even import if installed tokenizers falls outside its declared range --
    # so pinning old-transformers-plus-new-tokenizers is fundamentally
    # unsatisfiable, not just a pip-resolver inconvenience. Confirmed via
    # web search: seen_tokens was deprecated in 4.41 (replaced by
    # get_seq_length()), and recent transformers (4.48+) requires
    # tokenizers>=0.21 -- i.e. modern transformers and modern tokenizers
    # travel together, they were never in conflict. The actual fix is to NOT
    # pin the environment at all (this also matches the exact combination
    # Stage 8-11 already proved stable for Qwen2.5) and instead patch the one
    # thing that's actually broken -- Phi's removed attribute -- at the point
    # of use. See ace_rag/generator.py's DynamicCache.seen_tokens shim.


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


def factorial_cmd(
    job_name: str,
    *,
    dataset: str,
    reader_model: str,
    budget: int,
    seed: int,
    limit: int = LIMIT,
    split: str = "validation",
    ragbench_subset: str | None = None,
    dataset_path: Path | None = None,
    path_flag: str | None = None,
    batch_size: int = 2,
    load_4bit: bool = False,
    semantic_aic: bool = False,
) -> tuple[Path, list[str]]:
    """Build the standard factorial command; only the varied axis differs."""
    out_dir = WORKING_ROOT / "colab_results" / job_name
    cmd = [
        sys.executable, "-m", "experiments.run_density_router",
        "--dataset", dataset,
        "--limit", str(limit),
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
        "--reader-model", reader_model,
        "--reader-device", "cuda",
        "--reader-batch-size", str(batch_size),
        "--mmr-lambda", "0.7",
        "--budget", str(budget),
        "--out-dir", str(out_dir),
    ]
    if dataset not in ("musique_local", "2wiki_local"):
        cmd += ["--split", split]
    if ragbench_subset:
        cmd += ["--ragbench-subset", ragbench_subset]
    if dataset_path is not None and path_flag:
        cmd += [path_flag, str(dataset_path)]
    if load_4bit:
        cmd += ["--reader-load-4bit"]
    if semantic_aic:
        cmd += ["--semantic-aic", "--nli-model", NLI_MODEL, "--nli-batch-size", "32"]
    return out_dir, cmd


def _published_metrics_dir(job_name: str) -> Path:
    """Where a job's results land after a prior run's publish_results() push.

    REPO_ROOT is the git checkout kaggle_github_loop.py freshly clones/pulls
    on every session start (ensure_repo(), before control/main.py even runs),
    so this path is populated with whatever is currently on GitHub regardless
    of whether this is a continuing session or a brand new one.
    """
    return REPO_ROOT / "kaggle_results" / "latest" / "colab_results" / job_name


def execute(job_name: str, out_dir: Path, cmd: list[str], timeout: int = 28800) -> bool:
    # Ephemeral check: this session already ran it (survives only within one
    # continuous Kaggle kernel -- /kaggle/working is wiped on session restart).
    if out_dir.exists() and any(out_dir.glob("*metrics.csv")):
        log(f"[skip] {job_name} already has metrics (this session)")
        return True
    # Persisted check: a PRIOR session already ran it and pushed the result to
    # GitHub. This survives a session restart, since ensure_repo() re-clones
    # the current GitHub state fresh before control/main.py runs. Without this,
    # a restarted Kaggle session would silently redo every already-successful
    # job in the stage (e.g. Falcon3's 3 seeds while only Phi was fixed).
    published = _published_metrics_dir(job_name)
    if published.exists() and any(published.glob("*metrics.csv")):
        log(f"[skip] {job_name} already has metrics (published to GitHub by a prior run)")
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    return run(cmd, cwd=WORK_ROOT, timeout=timeout, check=False) == 0


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def run_stage12_cross_family() -> dict[str, bool]:
    """Tier 1: replicate the HotpotQA headline on two non-Qwen reader families."""
    has_token = setup_hf_auth()
    results: dict[str, bool] = {}
    for label, model, batch_size, gated in CROSS_FAMILY_READERS:
        if gated and not has_token:
            log(f"[skip] {label} ({model}) is gated and no HF_TOKEN is set")
            results[label] = False
            continue
        for seed in SEEDS_3:
            job = f"stage12_crossfamily_hotpotqa_{label}_budget{BUDGET}_limit{LIMIT}_seed{seed}"
            log(f"--- cross-family {label} ({model}) seed {seed} ---")
            out_dir, cmd = factorial_cmd(
                job, dataset="hotpotqa", reader_model=model,
                budget=BUDGET, seed=seed, batch_size=batch_size,
            )
            try:
                results[f"{label}_seed{seed}"] = execute(job, out_dir, cmd)
            except Exception as exc:
                log(f"[error] {label} seed {seed} raised {exc!r}")
                results[f"{label}_seed{seed}"] = False
    return results


def run_stage13_multiseed() -> dict[str, bool]:
    """Tier 2: bring the budget sweep and MuSiQue up to three seeds."""
    results: dict[str, bool] = {}
    for budget in SWEEP_BUDGETS:
        for seed in SWEEP_SEEDS:
            job = f"stage13_budgetsweep_hotpotqa_qwen3b_budget{budget}_limit{LIMIT}_seed{seed}"
            log(f"--- budget sweep B={budget} seed {seed} ---")
            out_dir, cmd = factorial_cmd(
                job, dataset="hotpotqa", reader_model="Qwen/Qwen2.5-3B-Instruct",
                budget=budget, seed=seed,
            )
            try:
                results[f"budget{budget}_seed{seed}"] = execute(job, out_dir, cmd)
            except Exception as exc:
                log(f"[error] budget {budget} seed {seed} raised {exc!r}")
                results[f"budget{budget}_seed{seed}"] = False

    musique = find_musique_jsonl()
    if musique is None:
        log("[musique] no jsonl mounted; skipping MuSiQue multi-seed")
        return results
    for seed in MUSIQUE_SEEDS:
        job = f"stage13_musique_qwen3b_budget{BUDGET}_limit{LIMIT}_seed{seed}"
        log(f"--- musique seed {seed} ---")
        out_dir, cmd = factorial_cmd(
            job, dataset="musique_local", reader_model="Qwen/Qwen2.5-3B-Instruct",
            budget=BUDGET, seed=seed, dataset_path=musique, path_flag="--musique-path",
        )
        try:
            results[f"musique_seed{seed}"] = execute(job, out_dir, cmd)
        except Exception as exc:
            log(f"[error] musique seed {seed} raised {exc!r}")
            results[f"musique_seed{seed}"] = False
    return results


def run_stage14_2wiki_budget() -> dict[str, bool]:
    """Tier 1: hunt for a second dataset where the packer wins."""
    twowiki = find_2wiki_file()
    results: dict[str, bool] = {}
    for budget in TWO_WIKI_BUDGETS:
        for seed in TWO_WIKI_SEEDS:
            job = f"stage14_2wiki_qwen3b_budget{budget}_limit{TWO_WIKI_LIMIT}_seed{seed}"
            log(f"--- 2wiki B={budget} seed {seed} ---")
            out_dir, cmd = factorial_cmd(
                job, dataset="2wiki_local", reader_model="Qwen/Qwen2.5-3B-Instruct",
                budget=budget, seed=seed, limit=TWO_WIKI_LIMIT,
                dataset_path=twowiki, path_flag="--twowiki-path",
            )
            try:
                results[f"2wiki_b{budget}_seed{seed}"] = execute(job, out_dir, cmd)
            except Exception as exc:
                log(f"[error] 2wiki budget {budget} seed {seed} raised {exc!r}")
                results[f"2wiki_b{budget}_seed{seed}"] = False
    return results


def run_stage15_semantic_aic() -> dict[str, bool]:
    """Tier 1: score the NLI answer-in-context variant, reclaiming ExpertQA."""
    results: dict[str, bool] = {}
    for label, dataset, subset, split in SEMANTIC_TARGETS:
        for seed in SEEDS_3[:2]:  # 42, 13 -- enough for a correlation estimate
            job = f"stage15_semanticaic_{label}_qwen3b_budget{BUDGET}_seed{seed}"
            log(f"--- semantic AiC {label} seed {seed} ---")
            out_dir, cmd = factorial_cmd(
                job, dataset=dataset, reader_model="Qwen/Qwen2.5-3B-Instruct",
                budget=BUDGET, seed=seed, split=split, ragbench_subset=subset,
                semantic_aic=True,
            )
            try:
                results[f"{label}_seed{seed}"] = execute(job, out_dir, cmd)
            except Exception as exc:
                log(f"[error] semantic AiC {label} seed {seed} raised {exc!r}")
                results[f"{label}_seed{seed}"] = False
    return results


def run_stage16_32b() -> dict[str, bool]:
    """Tier 2: one more rung to make the reader-scale trajectory conclusive."""
    results: dict[str, bool] = {}
    for seed in SEEDS_32B:
        job = f"stage16_reader32b_4bit_hotpotqa_budget{BUDGET}_limit{LIMIT}_seed{seed}"
        log(f"--- 32B 4-bit seed {seed} ---")
        out_dir, cmd = factorial_cmd(
            job, dataset="hotpotqa", reader_model=READER_32B,
            budget=BUDGET, seed=seed, batch_size=1, load_4bit=True,
        )
        try:
            results[f"reader32b_seed{seed}"] = execute(job, out_dir, cmd)
        except Exception as exc:
            log(f"[error] 32B seed {seed} raised {exc!r}")
            results[f"reader32b_seed{seed}"] = False
    return results


STAGES = {
    "stage12-cross-family": run_stage12_cross_family,
    "stage13-multiseed": run_stage13_multiseed,
    "stage14-2wiki-budget": run_stage14_2wiki_budget,
    "stage15-semantic-aic": run_stage15_semantic_aic,
    "stage16-32b": run_stage16_32b,
}


def main() -> None:
    if STAGE not in STAGES:
        raise SystemExit(f"Unknown STAGE {STAGE!r}; pick one of {sorted(STAGES)}")
    log(f"=== control/main.py: {STAGE} ({JOB_VERSION}) ===")
    prepare_project()
    apply_overlay()
    results = STAGES[STAGE]()
    ok = [name for name, good in results.items() if good]
    bad = [name for name, good in results.items() if not good]
    log(f"=== control/main.py done: {STAGE}; ok={ok} failed={bad} ===")


if __name__ == "__main__":
    main()
