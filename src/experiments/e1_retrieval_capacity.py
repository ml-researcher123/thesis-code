"""E1 -- Retrieval capacity curve (reproduce & extend the embedding-dimension wall).

Goal: validate the measurement harness by showing that, for an adversarial relevance
pattern, the *free-embedding* realizability jumps from <1 to 1 only once the embedding
dimension ``d`` exceeds a critical value ``d*`` that grows with corpus size ``n_d``.
That growing ``d*`` is the wall of Weller et al. (arXiv:2508.21038). Establishing it
with our own code is the prerequisite for the compression (E2) and composition (E3)
experiments built on the same primitive.

Output: a results dict (grid of realizability/recall over dim x n_d x seed), a capacity
figure (realizability vs dim, one curve per n_d), and a markdown summary listing the
critical dimension d* per corpus size.
"""
from __future__ import annotations

import json
import os
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..common import RunContext
from ..theory.free_embedding import fit_capacity, result_to_dict
from ..theory.synthetic import build_pattern


def _critical_dim(curve: list[tuple[int, float]], thresh: float) -> int | None:
    """Smallest dim whose mean realizability >= thresh (curve sorted by dim)."""
    for dim, real in sorted(curve):
        if real >= thresh:
            return dim
    return None


def run(ctx: RunContext):
    cfg = ctx.cfg
    log = ctx.log
    p = cfg.get("params", {})

    dims = p.get("dims", [2, 4, 8, 16, 32, 64])
    n_d_list = p.get("n_d_list", [10, 20, 40])
    seeds = p.get("seeds", [0, 1, 2])
    pattern_kind = p.get("pattern", "all_pairs")
    k = p.get("k", 2)
    max_queries = p.get("max_queries", 2000)
    steps = p.get("steps", 600)
    lr = p.get("lr", 0.05)
    margin = p.get("margin", 1.0)
    realize_thresh = p.get("realize_thresh", 0.99)

    log(f"E1 retrieval capacity | dims={dims} n_d={n_d_list} seeds={seeds} "
        f"pattern={pattern_kind} k={k}")

    records: list[dict] = []
    # curve_means[n_d][dim] = mean realizability across seeds
    curve_means: dict[int, dict[int, float]] = {nd: {} for nd in n_d_list}

    for n_d in n_d_list:
        for dim in dims:
            reals, recalls, losses = [], [], []
            for seed in seeds:
                pattern = build_pattern(
                    pattern_kind, n_d=n_d, k=k, max_queries=max_queries,
                    n_q=p.get("n_q", 4 * n_d), seed=seed,
                )
                res = fit_capacity(
                    pattern, dim, steps=steps, lr=lr, margin=margin,
                    seed=seed, device=ctx.device, log=log if cfg.get("verbose") else None,
                )
                rec = result_to_dict(res)
                rec["n_d"] = n_d
                rec["pattern"] = pattern.name
                records.append(rec)
                reals.append(res.realizability)
                recalls.append(res.recall_at_k)
                losses.append(res.final_loss)
            mr = mean(reals)
            curve_means[n_d][dim] = mr
            log(f"  n_d={n_d:3d} dim={dim:3d} | realizability={mr:.3f}"
                f"+-{pstdev(reals):.3f} recall@k={mean(recalls):.3f} loss={mean(losses):.4f}")

    # critical dimension per corpus size
    crit = {}
    for n_d in n_d_list:
        curve = list(curve_means[n_d].items())
        crit[n_d] = _critical_dim(curve, realize_thresh)

    # ---- figure: the wall ----
    fig_path = os.path.join(ctx.outdir, "e1_capacity_wall.png")
    plt.figure(figsize=(6.5, 4.2))
    for n_d in n_d_list:
        xs = sorted(curve_means[n_d])
        ys = [curve_means[n_d][d] for d in xs]
        plt.plot(xs, ys, marker="o", label=f"n_d={n_d}")
    plt.axhline(realize_thresh, ls="--", c="gray", lw=1, label=f"thresh={realize_thresh}")
    plt.xscale("log", base=2)
    plt.xlabel("embedding dimension d (log2)")
    plt.ylabel("realizability (free-embedding best case)")
    plt.title("E1: retrieval capacity wall — d* grows with corpus size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=140)
    plt.close()
    log(f"  saved figure -> {fig_path}")

    results = {
        "experiment": "e1_retrieval_capacity",
        "config_params": p,
        "records": records,
        "curve_means": {str(k_): v for k_, v in curve_means.items()},
        "critical_dim": {str(k_): v for k_, v in crit.items()},
        "realize_thresh": realize_thresh,
        "figure": os.path.basename(fig_path),
    }

    # ---- human summary ----
    lines = [
        "# E1 — Retrieval Capacity Wall",
        "",
        f"Pattern: **{pattern_kind}** (k={k}); dims={dims}; seeds={seeds}.",
        "",
        "Free-embedding realizability is the *best case* for any encoder of a given",
        "dimension. The critical dimension d* (smallest d reaching realizability "
        f">= {realize_thresh}) grows with corpus size — this is the embedding wall.",
        "",
        "| corpus n_d | critical dim d* |",
        "|---|---|",
    ]
    for n_d in n_d_list:
        lines.append(f"| {n_d} | {crit[n_d] if crit[n_d] is not None else '> max dim'} |")
    lines += [
        "",
        "If d* increases with n_d, the harness reproduces Weller et al.'s wall and is",
        "validated for the compression (E2) and composition (E3) stages.",
        "",
        f"![capacity wall]({os.path.basename(fig_path)})",
    ]
    summary = "\n".join(lines)

    # convenience: also drop raw records as csv-ish json for quick inspection
    with open(os.path.join(ctx.outdir, "e1_records.json"), "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)

    return results, summary
