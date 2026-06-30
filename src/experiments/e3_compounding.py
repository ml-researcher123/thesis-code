"""E3 -- Shared-budget compounding (the headline: contribution C2, and C3 falls out).

Thesis: efficient RAG funds TWO fixed-dimensional bottlenecks -- the retrieval embedding
(d_r) and the compression code (D_c) -- from one representational budget B = d_r + D_c.
Either stage given the *full* budget B succeeds; but the pipeline must SPLIT B, so neither
stage gets the whole budget. We show the best split still underperforms either standalone
stage at budget B -- the two bottlenecks compound -- and we read off the optimal allocation.

Method (reuses the two validated primitives):
  - Retrieval wall: free-embedding realizability of a hard relevance pattern vs dim
    (E1 primitive, `fit_capacity`).  ->  recall_R(d)
  - Compression wall: slot-memory associative recall vs total code size (E2 primitive,
    `fit_compression`, D_c = m*d_c).  ->  recall_C(d)
  Both swept on a COMMON dimension grid. Under independent stage errors the pipeline recall
  at split (d_r, D_c) is recall_R(d_r) * recall_C(D_c) -- the *optimistic* baseline
  (correlated errors, E3b, only make it worse). For each budget B we compare the best split
  to min(recall_R(B), recall_C(B)); a positive gap is the compounding cost.

Outputs: the two walls on one axis; the allocation curve (pipeline recall vs split) showing
the interior optimum and the compounding gap; a summary table over budgets B.
"""
from __future__ import annotations

import json
import os
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..common import RunContext
from ..compression.free_slots import fit_compression
from ..theory.free_embedding import fit_capacity
from ..theory.synthetic import build_pattern


def run(ctx: RunContext):
    cfg = ctx.cfg
    log = ctx.log
    p = cfg.get("params", {})

    grid = p.get("grid", [32, 64, 96, 128, 160, 192, 224, 256])
    budgets = p.get("budgets", [128, 192, 256])
    seeds = p.get("seeds", [0, 1, 2])

    # retrieval-stage hardness (E1-style)
    rp = p.get("retrieval", {})
    r_pattern = rp.get("pattern", "random_ksubsets")
    r_n_d = rp.get("n_d", 64)
    r_k = rp.get("k", 8)
    r_n_q = rp.get("n_q", 512)
    r_steps = rp.get("steps", 600)
    r_lr = rp.get("lr", 0.05)
    r_margin = rp.get("margin", 1.0)
    # "recall_at_k" (smooth ramp) or "realizability" (strict all-or-nothing per query)
    r_metric = rp.get("metric", "recall_at_k")

    # compression-stage hardness (E2-style)
    cp = p.get("compression", {})
    c_n_f = cp.get("n_f", 64)
    c_d_c = cp.get("d_c", 16)
    c_V = cp.get("V", 4)
    c_d_key = cp.get("d_key", 128)
    c_P = cp.get("P", 128)
    c_steps = cp.get("steps", 1500)
    c_lr = cp.get("lr", 3e-3)

    log(f"E3 compounding | grid={grid} budgets={budgets} seeds={seeds} device={ctx.device}")

    # ---- retrieval wall on the grid ----
    recall_R: dict[int, float] = {}
    for d in grid:
        vals = []
        for seed in seeds:
            pat = build_pattern(r_pattern, n_d=r_n_d, k=r_k, n_q=r_n_q, max_queries=r_n_q, seed=seed)
            res = fit_capacity(pat, d, steps=r_steps, lr=r_lr, margin=r_margin,
                               seed=seed, device=ctx.device)
            vals.append(res.recall_at_k if r_metric == "recall_at_k" else res.realizability)
        recall_R[d] = mean(vals)
        log(f"  [R] d_r={d:4d} | recall_R={recall_R[d]:.3f}")

    # ---- compression wall on the grid (D_c = m * d_c) ----
    recall_C: dict[int, float] = {}
    for d in grid:
        if d % c_d_c != 0:
            log(f"  [C] D_c={d} not divisible by d_c={c_d_c}; skipping")
            continue
        m = d // c_d_c
        vals = []
        for seed in seeds:
            res = fit_compression(n_f=c_n_f, m=m, d_c=c_d_c, V=c_V, d_key=c_d_key, P=c_P,
                                  steps=c_steps, lr=c_lr, seed=seed, device=ctx.device)
            vals.append(res.recall)
        recall_C[d] = mean(vals)
        log(f"  [C] D_c={d:4d} (m={m:2d}) | recall_C={recall_C[d]:.3f}")

    # ---- compose under independence, per budget ----
    per_budget = {}
    for B in budgets:
        splits = []
        for d_r in grid:
            D_c = B - d_r
            if d_r in recall_R and D_c in recall_C and D_c > 0:
                splits.append((d_r, D_c, recall_R[d_r] * recall_C[D_c]))
        if not splits:
            continue
        best = max(splits, key=lambda t: t[2])
        standalone_R = recall_R.get(B)
        standalone_C = recall_C.get(B)
        min_standalone = None
        gap = None
        if standalone_R is not None and standalone_C is not None:
            min_standalone = min(standalone_R, standalone_C)
            gap = min_standalone - best[2]
        per_budget[B] = {
            "splits": [{"d_r": s[0], "D_c": s[1], "pipeline": s[2]} for s in splits],
            "best_split": {"d_r": best[0], "D_c": best[1], "pipeline": best[2]},
            "standalone_R_at_B": standalone_R,
            "standalone_C_at_B": standalone_C,
            "min_standalone": min_standalone,
            "compounding_gap": gap,
        }
        log(f"  [B={B}] best split d_r={best[0]} D_c={best[1]} pipeline={best[2]:.3f} | "
            f"standalone min={min_standalone} gap={gap}")

    # ---- figure 1: the two walls ----
    fig1 = os.path.join(ctx.outdir, "e3_two_walls.png")
    plt.figure(figsize=(6.5, 4.2))
    xs = sorted(recall_R)
    plt.plot(xs, [recall_R[d] for d in xs], marker="o", label="retrieval recall_R(d_r)")
    xc = sorted(recall_C)
    plt.plot(xc, [recall_C[d] for d in xc], marker="s", label="compression recall_C(D_c)")
    plt.xlabel("dimension (d_r or D_c)")
    plt.ylabel("standalone recall")
    plt.title("E3: the two bottlenecks on a shared budget axis")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig1, dpi=140)
    plt.close()

    # ---- figure 2: allocation curve for the largest budget ----
    fig2 = os.path.join(ctx.outdir, "e3_allocation.png")
    Bmax = max(per_budget) if per_budget else None
    if Bmax is not None:
        info = per_budget[Bmax]
        ds = [s["d_r"] for s in info["splits"]]
        ps = [s["pipeline"] for s in info["splits"]]
        plt.figure(figsize=(6.5, 4.2))
        plt.plot(ds, ps, marker="o", color="purple", label="pipeline (best=split)")
        if info["standalone_R_at_B"] is not None:
            plt.axhline(info["standalone_R_at_B"], ls="--", c="C0", lw=1,
                        label=f"standalone R @B={info['standalone_R_at_B']:.2f}")
        if info["standalone_C_at_B"] is not None:
            plt.axhline(info["standalone_C_at_B"], ls="--", c="C1", lw=1,
                        label=f"standalone C @B={info['standalone_C_at_B']:.2f}")
        bx = info["best_split"]
        plt.scatter([bx["d_r"]], [bx["pipeline"]], color="red", zorder=5,
                    label=f"best split d_r={bx['d_r']}")
        plt.xlabel(f"retrieval budget d_r  (D_c = {Bmax} - d_r)")
        plt.ylabel("pipeline recall")
        plt.title(f"E3: shared-budget allocation at B={Bmax} (gap = compounding cost)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(fig2, dpi=140)
        plt.close()
    log(f"  saved figures -> {os.path.basename(fig1)}, {os.path.basename(fig2)}")

    results = {
        "experiment": "e3_compounding",
        "config_params": p,
        "recall_R": {str(k): v for k, v in recall_R.items()},
        "recall_C": {str(k): v for k, v in recall_C.items()},
        "per_budget": {str(k): v for k, v in per_budget.items()},
        "figures": [os.path.basename(fig1), os.path.basename(fig2)],
    }

    lines = [
        "# E3 — Shared-Budget Compounding (C2 headline + C3 allocation)",
        "",
        "Two bottlenecks share one budget B = d_r + D_c. Pipeline recall (independent-error",
        "baseline) = recall_R(d_r) · recall_C(D_c). A positive gap vs min(standalone at B)",
        "is the compounding cost; the argmax split is the optimal allocation (C3).",
        "",
        "| budget B | standalone R | standalone C | best-split pipeline | optimal d_r:D_c | compounding gap |",
        "|---|---|---|---|---|---|",
    ]
    for B in sorted(per_budget):
        i = per_budget[B]
        b = i["best_split"]
        sr = f"{i['standalone_R_at_B']:.3f}" if i["standalone_R_at_B"] is not None else "—"
        sc = f"{i['standalone_C_at_B']:.3f}" if i["standalone_C_at_B"] is not None else "—"
        gp = f"{i['compounding_gap']:.3f}" if i["compounding_gap"] is not None else "—"
        lines.append(f"| {B} | {sr} | {sc} | {b['pipeline']:.3f} | "
                     f"{b['d_r']}:{b['D_c']} | {gp} |")
    lines += [
        "",
        "If the gap is positive, the pipeline cannot match either stage given the full",
        "budget — the two fixed-dimensional bottlenecks compound. The optimal split is",
        "typically interior; budgeting all of B to one stage (the deployed habit of a big",
        "embedder + a tiny compressor) is off the frontier.",
        "",
        "v1 uses the optimistic independent-error composition; correlated example hardness",
        "(E3b) can only widen the gap. Honest scope note in research-log.",
        "",
        f"![two walls]({os.path.basename(fig1)})",
        "",
        f"![allocation]({os.path.basename(fig2)})",
    ]
    summary = "\n".join(lines)

    return results, summary
