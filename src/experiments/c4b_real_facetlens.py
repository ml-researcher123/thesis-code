"""C4b -- Facet-lens escape on REAL encoder embeddings (the real-model version of C4).

Same three conditions as C4 (single / generic multiview / routed facetlens) at equal doc-side
budget, but the substrate is a frozen real sentence encoder's output, not free vectors. We build a
facet-structured natural-language corpus (entities x F semantic facets), encode docs+queries once,
then learn low-rank lenses over those frozen embeddings and sweep the budget d_total. Headline
metric is mAP (smooth; realizability=1 is too stringent on real embeddings). If routed facetlens
mAP-dominates single (and generic multiview) at small budgets, the escape is a property of the
representation geometry -- it holds for real encoders, fulfilling C4's real-model claim.
"""
from __future__ import annotations

import json
import os
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..common import RunContext, ensure_deps
from ..facetlens.real_lenses import FACETS, fit_real_lenses, make_real_facet_corpus


def run(ctx: RunContext):
    cfg = ctx.cfg; log = ctx.log; p = cfg.get("params", {})
    ensure_deps({"sentence_transformers": "sentence-transformers"}, log)
    from sentence_transformers import SentenceTransformer

    model_name = p.get("model", "mixedbread-ai/mxbai-embed-large-v1")
    N = p.get("N", 240)
    facet_names = p.get("facets", list(FACETS.keys()))        # default all 6 facets
    budgets = p.get("budgets", [8, 16, 32, 64, 128])
    modes = p.get("modes", ["single", "multiview", "facetlens"])
    seeds = p.get("seeds", [0, 1, 2])
    steps = p.get("steps", 600)
    lr = p.get("lr", 0.01)
    margin = p.get("margin", 0.5)
    target_map = p.get("target_map", 0.9)
    F = len(facet_names)

    log(f"C4b real facet-lens | model={model_name} N={N} F={F} budgets={budgets} device={ctx.device}")
    model = SentenceTransformer(model_name, trust_remote_code=p.get("trust_remote_code", False))
    if ctx.device.type == "cuda":
        model = model.to(ctx.device)

    # encode once per seed-0 corpus shape; embeddings depend on text, so re-encode per seed
    # (cheap: a few hundred short strings). Cache by seed to avoid re-encoding across budgets.
    emb_cache: dict[int, tuple] = {}
    def get_embeddings(seed):
        if seed not in emb_cache:
            docs, qtexts, attrs, A, foq = make_real_facet_corpus(N, facet_names, seed)
            de = model.encode(docs, batch_size=256, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
            qe = model.encode(qtexts, batch_size=256, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
            emb_cache[seed] = (de, qe, A, foq)
            log(f"  [seed {seed}] encoded {de.shape[0]} docs, {qe.shape[0]} queries, dim={de.shape[1]}")
        return emb_cache[seed]

    curve = {m: {} for m in modes}        # mode -> d -> mean mAP
    curve_real = {m: {} for m in modes}   # mode -> d -> mean realizability
    for mode in modes:
        for d in budgets:
            maps, reals = [], []
            for seed in seeds:
                de, qe, A, foq = get_embeddings(seed)
                real, ap = fit_real_lenses(de, qe, A, foq, F, mode=mode, d_total=d, steps=steps,
                                           lr=lr, margin=margin, seed=seed, device=ctx.device,
                                           log=log if cfg.get("verbose") else None)
                maps.append(ap); reals.append(real)
            curve[mode][d] = mean(maps); curve_real[mode][d] = mean(reals)
            log(f"  {mode:10s} d={d:4d} | mAP={mean(maps):.3f}+-{pstdev(maps):.3f} "
                f"real={mean(reals):.3f}")

    # budget to reach target mAP per mode (the "escape" number)
    crit = {m: next((d for d in sorted(curve[m]) if curve[m][d] >= target_map), None) for m in modes}

    fig = os.path.join(ctx.outdir, "c4b_real_facetlens.png")
    plt.figure(figsize=(6.5, 4.2))
    marks = {"single": "o", "multiview": "s", "facetlens": "^"}
    for mode in modes:
        xs = sorted(curve[mode])
        plt.plot(xs, [curve[mode][d] for d in xs], marker=marks.get(mode, "o"), label=mode)
    plt.axhline(target_map, ls="--", c="gray", lw=1, label=f"target mAP={target_map}")
    plt.xscale("log", base=2)
    plt.xlabel("total doc-side budget d_total (log2)")
    plt.ylabel("mAP (real frozen encoder)")
    plt.title(f"C4b: facet-lenses escape on real embeddings ({model_name.split('/')[-1]}, F={F})")
    plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(fig, dpi=140); plt.close()
    log(f"  saved figure -> {os.path.basename(fig)}")

    results = {
        "experiment": "c4b_real_facetlens", "config_params": p, "model": model_name,
        "N": N, "F": F, "facets": facet_names,
        "map_curve": {m: {str(d): curve[m][d] for d in curve[m]} for m in modes},
        "real_curve": {m: {str(d): curve_real[m][d] for d in curve_real[m]} for m in modes},
        "critical_budget_for_target": {m: crit[m] for m in modes}, "target_map": target_map,
        "figure": os.path.basename(fig),
    }

    lines = [
        f"# C4b — Real-Encoder Facet-Lens Escape ({model_name.split('/')[-1]}, F={F})",
        "",
        f"Frozen `{model_name}` embeddings of a facet-structured corpus (N={N}, facets="
        f"{facet_names}); learned low-rank lenses at equal doc-side budget; seeds={seeds}.",
        "",
        "| budget d_total | " + " | ".join(f"{m} mAP" for m in modes) + " |",
        "|---|" + "|".join("---" for _ in modes) + "|",
    ]
    for d in budgets:
        lines.append(f"| {d} | " + " | ".join(f"{curve[m][d]:.3f}" for m in modes) + " |")
    lines += [
        "",
        f"Budget to reach mAP ≥ {target_map}: "
        + ", ".join(f"**{m}** {crit[m] if crit[m] is not None else '> max'}" for m in modes) + ".",
        "",
        "If routed **facetlens** reaches the target at a smaller budget than **single** (and",
        "beats generic **multiview**), the C4 escape holds on real semantic embeddings, not just",
        "free vectors — the representation-geometry claim survives a real frozen encoder.",
        "",
        f"![real facet-lens escape]({os.path.basename(fig)})",
    ]
    return results, "\n".join(lines)
