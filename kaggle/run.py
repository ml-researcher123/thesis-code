"""Experiment runner — the single entry point for one experiment.

Usage:
    python kaggle/run.py --config configs/e1_retrieval_capacity.yaml
    python kaggle/run.py --config configs/e1_retrieval_capacity.yaml --device cuda

Reads a YAML config (which names an ``experiment`` from the registry), runs it, and
writes a self-contained run directory under ``outputs/<name>/``:
    results.json   machine-readable results
    summary.md     human-readable summary (written by the experiment)
    config.yaml    snapshot of the exact config used
    run.log        full log
    *.png          any figures

The run id is the config's ``name`` field (falling back to the config file stem), so a
completed run leaves ``outputs/<name>/results.json`` as an idempotency marker the
auto-runner uses to skip already-finished configs.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback

import yaml

# make repo root importable regardless of CWD
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.common import Logger, RunContext, get_device, set_seed  # noqa: E402
from src.experiments.registry import get_experiment  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--outroot", default=os.path.join(REPO_ROOT, "outputs"))
    ap.add_argument("--force", action="store_true", help="rerun even if results exist")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg.get("name") or os.path.splitext(os.path.basename(args.config))[0]
    outdir = os.path.join(args.outroot, name)
    os.makedirs(outdir, exist_ok=True)

    done_marker = os.path.join(outdir, "results.json")
    if os.path.exists(done_marker) and not args.force:
        print(f"[run] '{name}' already has results.json — skipping (use --force to rerun)")
        return 0

    log = Logger(os.path.join(outdir, "run.log"))
    log(f"=== run '{name}' ===")
    log(f"host={platform.node()} python={platform.python_version()} config={args.config}")

    seed = cfg.get("seed", 0)
    set_seed(seed)
    device = get_device(args.device)
    log(f"device={device} seed={seed}")

    # snapshot config
    with open(os.path.join(outdir, "config.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    ctx = RunContext(cfg=cfg, outdir=outdir, log=log, device=device)
    t0 = time.time()
    try:
        run_fn = get_experiment(cfg["experiment"])
        results, summary = run_fn(ctx)
    except Exception as exc:  # noqa: BLE001
        log(f"!! FAILED: {exc}")
        log(traceback.format_exc())
        with open(os.path.join(outdir, "FAILED.txt"), "w", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())
        return 1

    elapsed = time.time() - t0
    results["_meta"] = {
        "name": name,
        "experiment": cfg["experiment"],
        "seconds": round(elapsed, 2),
        "device": str(device),
        "python": platform.python_version(),
        "host": platform.node(),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(os.path.join(outdir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    with open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")

    log(f"=== done in {elapsed:.1f}s -> {outdir} ===")
    print(f"[run] wrote {done_marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
