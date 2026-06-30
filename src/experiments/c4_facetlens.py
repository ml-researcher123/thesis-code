"""C4 -- Facet-lens escape: do specialized routed lenses beat a single vector at equal budget?

Sweeps the total embedding budget d_total and compares three conditions (single full-rank
vector, generic K-view MaxSim, routed F facet-lenses) on a facet-structured relevance
pattern. The claims:
  (1) routed facet-lenses realize the pattern at a far smaller budget than a single vector
      -- an explicit escape from the single-vector wall (E1/Weller);
  (2) routing/specialization beats generic multi-view (MaxSim) at the same budget.
If both hold, the multi-view facet-lens is the paper's principled escape (C4).
"""
from __future__ import annotations

import json
import os
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..common import RunContext
from ..facetlens.lenses import fit_lenses, make_facet_pattern


def run(ctx: RunContext):
    cfg = ctx.cfg; log = ctx.log; p = cfg.get("params", {})
    N = p.get("N", 128)
    Fac = p.get("F", 4)
    G = p.get("G", 4)
    budgets = p.get("budgets", [4, 8, 16, 32, 64])
    modes = p.get("modes", ["single", "multiview", "facetlens"])
    seeds = p.get("seeds", [0, 1, 2])
    steps = p.get("steps", 800)
    lr = p.get("lr", 0.05)
    margin = p.get("margin", 1.0)
    thresh = p.get("realize_thresh", 0.99)

    log(f"C4 facet-lens | N={N} F={Fac} G={G} budgets={budgets} modes={modes} device={ctx.device}")
    A, facet_of_query, _ = make_facet_pattern(N, Fac, G, seed=0)
    log(f"  pattern: {A.shape[0]} queries, group size ~{N//G}")

    curve = {m: {} for m in modes}
    for mode in modes:
        for d in budgets:
            vals = []
            for seed in seeds:
                # regenerate pattern per seed for variance over data as well
                Ai, foq, _ = make_facet_pattern(N, Fac, G, seed=seed)
                r = fit_lenses(Ai, foq, Fac, mode=mode, d_total=d, steps=steps, lr=lr,
                               margin=margin, seed=seed, device=ctx.device,
                               log=log if cfg.get("verbose") else None)
                vals.append(r)
            curve[mode][d] = mean(vals)
            log(f"  {mode:10s} d_total={d:3d} | realizability={mean(vals):.3f}+-{pstdev(vals):.3f}")

    crit = {m: next((d for d in sorted(curve[m]) if curve[m][d] >= thresh), None) for m in modes}

    fig = os.path.join(ctx.outdir, "c4_facetlens_escape.png")
    plt.figure(figsize=(6.5, 4.2))
    marks = {"single": "o", "multiview": "s", "facetlens": "^"}
    for mode in modes:
        xs = sorted(curve[mode])
        plt.plot(xs, [curve[mode][d] for d in xs], marker=marks.get(mode, "o"), label=mode)
    plt.axhline(thresh, ls="--", c="gray", lw=1, label=f"thresh={thresh}")
    plt.xscale("log", base=2)
    plt.xlabel("total embedding budget d_total (log2)")
    plt.ylabel("realizability")
    plt.title(f"C4: facet-lenses escape the single-vector wall (N={N}, F={Fac}, G={G})")
    plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(fig, dpi=140); plt.close()
    log(f"  saved figure -> {os.path.basename(fig)}")

    results = {
        "experiment": "c4_facetlens", "config_params": p,
        "curve": {m: {str(d): curve[m][d] for d in curve[m]} for m in modes},
        "critical_budget": {m: crit[m] for m in modes},
        "realize_thresh": thresh, "figure": os.path.basename(fig),
    }

    lines = [
        f"# C4 — Facet-Lens Escape (N={N}, F={Fac}, G={G})",
        "",
        f"Realizability vs total embedding budget; equal budget across conditions; seeds={seeds}.",
        "",
        "| mode | critical budget d* (realizability ≥ "
        f"{thresh}) |",
        "|---|---|",
    ]
    for mode in modes:
        lines.append(f"| {mode} | {crit[mode] if crit[mode] is not None else '> max'} |")
    lines += [
        "",
        "If routed **facetlens** reaches realizability=1 at a far smaller budget than",
        "**single** (and below generic **multiview**), facet-specialized lenses are a",
        "principled escape from the single-vector capacity wall — the paper's method (C4).",
        "",
        f"![facet-lens escape]({os.path.basename(fig)})",
    ]
    with open(os.path.join(ctx.outdir, "c4_curve.json"), "w", encoding="utf-8") as fh:
        json.dump(results["curve"], fh, indent=2)
    return results, "\n".join(lines)
