"""E3b -- Hardness dependence: when do the two bottlenecks compound super-multiplicatively?

E3 showed budget-splitting compounding under the optimistic *independent* composition.
E3b asks the sharper question: holding the real (E1/E2) marginal recalls fixed at a given
operating point, how does the pipeline recall move as the retrieval<->compression hardness
correlation rho varies from misaligned (rho<0) to aligned (rho>0)?

We study TIGHT operating points (both stages sub-saturated), because that is the only
regime where correlation can matter (if retrieval is saturated, every item passes it and
rho is irrelevant). This is exactly the tight-budget regime where E3's gap was largest.

Headline figure: pipeline recall vs rho for several operating points, with the independent
baseline p_R*p_C and the Frechet envelope. The takeaway is a *conditional* claim --
compounding is super-multiplicative iff hardness is anti-correlated -- and it frames the
sign of rho in real corpora as the next empirical question (real-model experiments).
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
from ..compression.free_slots import fit_compression
from ..pipeline.dependence import frechet_bounds, pipeline_recall
from ..theory.free_embedding import fit_capacity
from ..theory.synthetic import build_pattern


def run(ctx: RunContext):
    cfg = ctx.cfg
    log = ctx.log
    p = cfg.get("params", {})

    seeds = p.get("seeds", [0, 1, 2])
    rhos = p.get("rhos", [-0.9, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.9])
    # tight operating points (d_r, D_c): both stages sub-saturated so rho can bite
    op_points = [tuple(x) for x in p.get("operating_points", [[16, 48], [32, 32], [32, 64]])]

    rp = p.get("retrieval", {})
    cp = p.get("compression", {})

    # marginals needed
    ret_dims = sorted({op[0] for op in op_points})
    comp_dims = sorted({op[1] for op in op_points})

    log(f"E3b dependence | ops={op_points} rhos={rhos} seeds={seeds} device={ctx.device}")

    recall_R = {}
    for d in ret_dims:
        vals = []
        for seed in seeds:
            pat = build_pattern(rp.get("pattern", "random_ksubsets"),
                                n_d=rp.get("n_d", 256), k=rp.get("k", 64),
                                n_q=rp.get("n_q", 256), max_queries=rp.get("n_q", 256), seed=seed)
            res = fit_capacity(pat, d, steps=rp.get("steps", 500), lr=rp.get("lr", 0.05),
                               margin=rp.get("margin", 1.0), seed=seed, device=ctx.device)
            vals.append(res.recall_at_k)
        recall_R[d] = mean(vals)
        log(f"  [R] d_r={d:4d} | p_R={recall_R[d]:.3f}")

    recall_C = {}
    d_c = cp.get("d_c", 16)
    for d in comp_dims:
        vals = []
        for seed in seeds:
            res = fit_compression(n_f=cp.get("n_f", 64), m=d // d_c, d_c=d_c,
                                  V=cp.get("V", 4), d_key=cp.get("d_key", 128),
                                  P=cp.get("P", 128), steps=cp.get("steps", 1500),
                                  lr=cp.get("lr", 3e-3), seed=seed, device=ctx.device)
            vals.append(res.recall)
        recall_C[d] = mean(vals)
        log(f"  [C] D_c={d:4d} (m={d//d_c}) | p_C={recall_C[d]:.3f}")

    # dependence sweep per operating point
    curves = {}
    for (d_r, D_c) in op_points:
        p_R, p_C = recall_R[d_r], recall_C[D_c]
        indep = p_R * p_C
        lo, hi = frechet_bounds(p_R, p_C)
        pts = []
        for rho in rhos:
            pr = mean(pipeline_recall(p_R, p_C, rho, seed=s) for s in range(3))
            pts.append(pr)
        curves[f"{d_r}:{D_c}"] = {
            "d_r": d_r, "D_c": D_c, "p_R": p_R, "p_C": p_C,
            "independent": indep, "frechet_low": lo, "frechet_high": hi,
            "rhos": rhos, "pipeline": pts,
        }
        # super-multiplicative penalty at rho=-0.5 vs independent
        idx_neg = min(range(len(rhos)), key=lambda i: abs(rhos[i] + 0.5))
        log(f"  op {d_r}:{D_c} | p_R={p_R:.3f} p_C={p_C:.3f} indep={indep:.3f} "
            f"pipe(rho=-0.5)={pts[idx_neg]:.3f} pipe(rho=+0.9)={pts[-1]:.3f} "
            f"[frechet {lo:.3f}..{hi:.3f}]")

    # ---- figure: pipeline vs rho ----
    fig = os.path.join(ctx.outdir, "e3b_dependence.png")
    plt.figure(figsize=(7, 4.6))
    for key, c in curves.items():
        line, = plt.plot(c["rhos"], c["pipeline"], marker="o", label=f"op {key} (p_R={c['p_R']:.2f},p_C={c['p_C']:.2f})")
        plt.axhline(c["independent"], ls="--", lw=1, color=line.get_color(), alpha=0.6)
    plt.axvline(0.0, ls=":", c="gray", lw=1)
    plt.xlabel("retrieval↔compression hardness correlation  rho")
    plt.ylabel("pipeline recall")
    plt.title("E3b: compounding is super-multiplicative iff hardness is anti-correlated\n"
              "(dashed = independent baseline p_R·p_C; rho<0 → below it)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig, dpi=140)
    plt.close()
    log(f"  saved figure -> {os.path.basename(fig)}")

    results = {
        "experiment": "e3b_dependence",
        "config_params": p,
        "recall_R": {str(k): v for k, v in recall_R.items()},
        "recall_C": {str(k): v for k, v in recall_C.items()},
        "curves": curves,
        "figure": os.path.basename(fig),
    }

    lines = [
        "# E3b — Hardness Dependence (when does compounding go super-multiplicative?)",
        "",
        "At tight operating points (both stages sub-saturated), pipeline recall vs the",
        "retrieval↔compression hardness correlation rho. Independent baseline = p_R·p_C.",
        "",
        "| operating point d_r:D_c | p_R | p_C | independent (rho=0) | pipeline rho=-0.5 | pipeline rho=+0.9 | Fréchet [lo,hi] |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, c in curves.items():
        idx_neg = min(range(len(c["rhos"])), key=lambda i: abs(c["rhos"][i] + 0.5))
        lines.append(
            f"| {key} | {c['p_R']:.3f} | {c['p_C']:.3f} | {c['independent']:.3f} | "
            f"{c['pipeline'][idx_neg]:.3f} | {c['pipeline'][-1]:.3f} | "
            f"[{c['frechet_low']:.3f}, {c['frechet_high']:.3f}] |"
        )
    lines += [
        "",
        "Reading: rho < 0 (retrieval-easy items are compression-hard and vice versa) pushes",
        "the pipeline *below* the independent product — super-multiplicative compounding. rho",
        "> 0 (same items hard for both) makes failures redundant, *above* the product. So the",
        "compounding sign is set by hardness alignment, an empirical property of the corpus —",
        "the next step is to measure rho for real retrievers+compressors.",
        "",
        f"![dependence]({os.path.basename(fig)})",
    ]
    summary = "\n".join(lines)

    return results, summary
