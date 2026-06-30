"""E1c -- Scaling fit for the retrieval wall (turns F3 from INCONCLUSIVE -> a quotable law).

F3 conjectured that sub-critical realizability scales like d/n_d and that the critical dimension
d* grows with corpus size, but on too few seeds/one pattern to fit a curve. E1c nails it down:
a fine (n_d x d) grid over MORE seeds and TWO pattern families (all_pairs k=2 and random k=3
subsets), then it fits two functional forms with R^2:
  (i)  d*(n_d): power law d* = a * n_d^b   vs   logarithmic d* = a*log(n_d) + b;
  (ii) sub-critical realizability vs the ratio r = d/n_d (pooled), to test the ~d/n_d conjecture.
A clean fit here gives the allocation law (C3) a concrete "fund retrieval to d*(n_d)" recipe.
"""
from __future__ import annotations

import json
import os
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..common import RunContext
from ..theory.free_embedding import fit_capacity
from ..theory.synthetic import build_pattern


def _r2(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _fit_dstar(n_ds, dstars):
    """Fit d*(n_d) as power law and logarithmic; return both with R^2 (on valid points)."""
    pts = [(n, d) for n, d in zip(n_ds, dstars) if d is not None]
    if len(pts) < 3:
        return {"power": None, "log": None, "points": pts}
    n = np.array([p[0] for p in pts], float); d = np.array([p[1] for p in pts], float)
    # power law: log d = b log n + log a
    pb, pa = np.polyfit(np.log(n), np.log(d), 1)
    pow_pred = np.exp(pa) * n ** pb
    # logarithmic: d = a log n + b
    la, lb = np.polyfit(np.log(n), d, 1)
    log_pred = la * np.log(n) + lb
    return {
        "power": {"a": float(np.exp(pa)), "b": float(pb), "r2": _r2(d, pow_pred),
                  "form": "d* = a * n_d^b"},
        "log": {"a": float(la), "b": float(lb), "r2": _r2(d, log_pred),
                "form": "d* = a*log(n_d) + b"},
        "points": pts,
    }


def run(ctx: RunContext):
    cfg = ctx.cfg; log = ctx.log; p = cfg.get("params", {})
    dims = p.get("dims", [2, 3, 4, 6, 8, 11, 16, 22, 32])
    n_d_list = p.get("n_d_list", [40, 64, 100, 160, 256, 400])
    seeds = p.get("seeds", [0, 1, 2, 3, 4])
    patterns = p.get("patterns", ["all_pairs", "random_ksubsets"])
    k_for_random = p.get("k_random", 3)
    max_queries = p.get("max_queries", 1500)
    steps = p.get("steps", 500)
    lr = p.get("lr", 0.05)
    margin = p.get("margin", 1.0)
    thresh = p.get("realize_thresh", 0.99)
    sub_lo = p.get("subcritical_floor", 0.02)

    log(f"E1c scaling fit | patterns={patterns} dims={dims} n_d={n_d_list} seeds={seeds}")
    per_pattern = {}
    pooled_ratio, pooled_real = [], []   # for the d/n_d sub-critical fit (across patterns)

    for kind in patterns:
        grid = {nd: {} for nd in n_d_list}          # nd -> d -> mean realizability
        for nd in n_d_list:
            for d in dims:
                reals = []
                for s in seeds:
                    if kind == "random_ksubsets":
                        pat = build_pattern(kind, n_d=nd, k=k_for_random,
                                            n_q=min(max_queries, 4 * nd), seed=s)
                    else:
                        pat = build_pattern(kind, n_d=nd, k=2, max_queries=max_queries, seed=s)
                    res = fit_capacity(pat, d, steps=steps, lr=lr, margin=margin, seed=s,
                                       device=ctx.device, log=None)
                    reals.append(res.realizability)
                mr = mean(reals)
                grid[nd][d] = mr
                if sub_lo < mr < thresh:
                    pooled_ratio.append(d / nd); pooled_real.append(mr)
            dstar = next((d for d in dims if grid[nd][d] >= thresh), None)
            log(f"  [{kind}] n_d={nd:4d} d*={dstar} | "
                + " ".join(f"{d}:{grid[nd][d]:.2f}" for d in dims))
        dstars = [next((d for d in dims if grid[nd][d] >= thresh), None) for nd in n_d_list]
        fit = _fit_dstar(n_d_list, dstars)
        per_pattern[kind] = {"grid": {str(nd): grid[nd] for nd in n_d_list},
                             "dstar": {str(nd): ds for nd, ds in zip(n_d_list, dstars)},
                             "fit": fit}
        if fit["power"]:
            log(f"  [{kind}] d* power: a={fit['power']['a']:.2f} b={fit['power']['b']:.2f} "
                f"R2={fit['power']['r2']:.3f} | log R2={fit['log']['r2']:.3f}")

    # pooled sub-critical fit: realizability ~ slope * (d/n_d)
    ratio_fit = None
    if len(pooled_ratio) >= 4:
        r = np.array(pooled_ratio); y = np.array(pooled_real)
        slope, intercept = np.polyfit(r, y, 1)
        ratio_fit = {"slope": float(slope), "intercept": float(intercept),
                     "r2": _r2(y, slope * r + intercept), "n_points": int(len(r))}
        log(f"  sub-critical: realizability ≈ {slope:.3f}*(d/n_d) + {intercept:.3f} "
            f"(R2={ratio_fit['r2']:.3f}, n={len(r)})")

    # ---- figures ----
    fig1 = os.path.join(ctx.outdir, "e1c_dstar_scaling.png")
    plt.figure(figsize=(6.5, 4.2))
    for kind in patterns:
        pts = per_pattern[kind]["fit"]["points"]
        if pts:
            ns = [x[0] for x in pts]; ds = [x[1] for x in pts]
            plt.plot(ns, ds, "o-", label=f"{kind} d*")
            fp = per_pattern[kind]["fit"]["power"]
            if fp:
                xx = np.linspace(min(ns), max(ns), 50)
                plt.plot(xx, fp["a"] * xx ** fp["b"], "--", lw=1,
                         label=f"{kind} fit n^{fp['b']:.2f} (R²={fp['r2']:.2f})")
    plt.xlabel("corpus size n_d"); plt.ylabel("critical dimension d*")
    plt.title("E1c: how the retrieval wall d* grows with corpus size")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(fig1, dpi=140); plt.close()

    fig2 = os.path.join(ctx.outdir, "e1c_subcritical_ratio.png")
    plt.figure(figsize=(6.5, 4.2))
    plt.scatter(pooled_ratio, pooled_real, s=14, alpha=0.6, label="sub-critical points")
    if ratio_fit:
        xx = np.linspace(min(pooled_ratio), max(pooled_ratio), 50)
        plt.plot(xx, ratio_fit["slope"] * xx + ratio_fit["intercept"], "r--",
                 label=f"fit (R²={ratio_fit['r2']:.2f})")
    plt.xlabel("ratio d / n_d"); plt.ylabel("realizability (sub-critical)")
    plt.title("E1c: sub-critical realizability scales with d/n_d (F3)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(fig2, dpi=140); plt.close()
    log(f"  saved figures -> {os.path.basename(fig1)}, {os.path.basename(fig2)}")

    results = {
        "experiment": "e1c_scaling_fit", "config_params": p,
        "per_pattern": per_pattern, "subcritical_ratio_fit": ratio_fit,
        "figures": [os.path.basename(fig1), os.path.basename(fig2)],
    }

    lines = ["# E1c — Retrieval-Wall Scaling Fit (F3)", "",
             f"Patterns {patterns}; dims={dims}; n_d={n_d_list}; seeds={seeds}.", "",
             "## Critical dimension d*(n_d)", "",
             "| pattern | d* per n_d | power fit d*=a·n_d^b (R²) | log fit (R²) |",
             "|---|---|---|---|"]
    for kind in patterns:
        f = per_pattern[kind]["fit"]
        dline = ", ".join(f"{nd}:{per_pattern[kind]['dstar'][str(nd)]}" for nd in n_d_list)
        if f["power"]:
            pw = f"{f['power']['a']:.2f}·n^{f['power']['b']:.2f} ({f['power']['r2']:.2f})"
            lg = f"{f['log']['r2']:.2f}"
        else:
            pw = lg = "—"
        lines.append(f"| {kind} | {dline} | {pw} | {lg} |")
    lines += ["", "## Sub-critical scaling (F3 conjecture)"]
    if ratio_fit:
        lines += ["",
                  f"Pooled over patterns, sub-critical realizability ≈ "
                  f"**{ratio_fit['slope']:.3f}·(d/n_d) + {ratio_fit['intercept']:.3f}** "
                  f"(R² = {ratio_fit['r2']:.2f}, n = {ratio_fit['n_points']}). A high R² with a "
                  "near-linear dependence on d/n_d upgrades F3 to a quotable scaling and makes the "
                  "allocation law (C3) concrete: fund retrieval to d*(n_d)."]
    else:
        lines += ["", "Too few sub-critical points to fit (widen the grid)."]
    lines += ["", f"![d* scaling]({os.path.basename(fig1)})",
              "", f"![sub-critical ratio]({os.path.basename(fig2)})"]
    return results, "\n".join(lines)
