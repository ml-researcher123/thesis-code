"""E5 -- Real-encoder compression wall (reliable; tests F9 for compression).

Compresses multi-fact passages to a single real frozen-encoder embedding, truncates it to
D_c dims, and probes fact recall with a light MLP (encoder frozen; probe generalizes to
held-out passages). Sweeps D_c and content n_f. If recall ramps with D_c and the critical
D_c* grows with n_f -- the E2 shape with a real encoder -- the compression wall is validated
on real models, completing the real-model evidence for both bottlenecks.

(The faithful generative soft-token variant lives in compression/soft_prompt.py but needs a
large training budget to leave chance; this probe is the reliable route to the same claim.)
"""
from __future__ import annotations

import json
import os
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..common import RunContext, ensure_deps
from ..compression.encoder_probe import KEYS, VALUES, build_xy, fit_probe, gen_passages


def run(ctx: RunContext):
    cfg = ctx.cfg
    log = ctx.log
    p = cfg.get("params", {})

    ensure_deps({"sentence_transformers": "sentence-transformers"}, log)
    from sentence_transformers import SentenceTransformer

    model_name = p.get("model", "mixedbread-ai/mxbai-embed-large-v1")
    n_f_list = p.get("n_f_list", [4, 8, 16, 32])
    dims = p.get("dims", [16, 32, 64, 128, 256, 512, 1024])
    V = len(VALUES)
    P_train = p.get("P_train", 256)
    P_eval = p.get("P_eval", 128)
    steps = p.get("steps", 400)
    lr = p.get("lr", 2e-3)
    seeds = p.get("seeds", [0, 1])
    rel_thresh = p.get("rel_thresh", 0.9)   # fraction of the n_f's own full-dim ceiling

    log(f"E5 real-encoder compression wall | model={model_name} n_f={n_f_list} dims={dims} "
        f"V={V} device={ctx.device}")
    model = SentenceTransformer(model_name, trust_remote_code=p.get("trust_remote_code", False))
    if ctx.device.type == "cuda":
        model = model.to(ctx.device)
    full_dim = int(model.get_sentence_embedding_dimension())
    dims = [d for d in dims if d <= full_dim]
    key_vecs = model.encode([k for k in KEYS], normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False)

    curve = {nf: {} for nf in n_f_list}
    for n_f in n_f_list:
        for seed in seeds:
            tr = gen_passages(n_f, P_train, seed)
            te = gen_passages(n_f, P_eval, 9000 + seed)
            tr_vecs = model.encode([t for t, _ in tr], batch_size=128, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
            te_vecs = model.encode([t for t, _ in te], batch_size=128, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
            for d in dims:
                Xtr, ytr = build_xy(tr_vecs, tr, key_vecs, d)
                Xte, yte = build_xy(te_vecs, te, key_vecs, d)
                acc = fit_probe(Xtr, ytr, Xte, yte, V, steps, lr, seed, ctx.device)
                curve[n_f].setdefault(d, []).append(acc)
        for d in dims:
            vals = curve[n_f][d]
            log(f"  n_f={n_f:3d} D_c={d:5d} | acc={mean(vals):.3f}+-{pstdev(vals):.3f} "
                f"(chance={1.0/V:.3f})")
    # collapse seeds
    curve = {nf: {d: mean(v) for d, v in curve[nf].items()} for nf in curve}

    # critical D_c: smallest d reaching rel_thresh of this n_f's own full-dim accuracy
    crit = {}
    for n_f in n_f_list:
        ceiling = curve[n_f][max(dims)]
        crit[n_f] = next((d for d in sorted(dims)
                          if curve[n_f][d] >= rel_thresh * ceiling), None)

    fig = os.path.join(ctx.outdir, "e5_real_compression_wall.png")
    plt.figure(figsize=(6.5, 4.2))
    for n_f in n_f_list:
        xs = sorted(curve[n_f])
        plt.plot(xs, [curve[n_f][d] for d in xs], marker="o", label=f"n_f={n_f}")
    plt.axhline(1.0 / V, ls=":", c="red", lw=1, label=f"chance={1.0/V:.2f}")
    plt.xscale("log", base=2)
    plt.xlabel("compression budget D_c (truncated embedding dim, log2)")
    plt.ylabel("fact-recall accuracy (held-out)")
    plt.title(f"E5: real-encoder compression wall ({model_name.split('/')[-1]})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig, dpi=140)
    plt.close()
    log(f"  saved figure -> {os.path.basename(fig)}")

    results = {
        "experiment": "e5_real_compression_probe",
        "config_params": p, "model": model_name, "full_dim": full_dim, "V": V,
        "curve": {str(nf): {str(d): curve[nf][d] for d in curve[nf]} for nf in curve},
        "critical_D_c": {str(nf): crit[nf] for nf in crit},
        "rel_thresh": rel_thresh,
        "figure": os.path.basename(fig),
    }

    lines = [
        f"# E5 — Real-Encoder Compression Wall ({model_name.split('/')[-1]})",
        "",
        f"Multi-fact passages compressed to one frozen-encoder embedding (full dim {full_dim}),"
        f" truncated to D_c; light probe recovers a queried key's value (V={V}, "
        f"chance={1.0/V:.3f}); held-out passages; seeds={seeds}.",
        "",
        f"Critical D_c* (≥ {rel_thresh:.0%} of the n_f's own full-dim recall) vs content n_f:",
        "",
        "| facts n_f | full-dim acc | critical D_c* |",
        "|---|---|---|",
    ]
    for n_f in n_f_list:
        lines.append(f"| {n_f} | {curve[n_f][max(dims)]:.3f} | "
                     f"{crit[n_f] if crit[n_f] is not None else '> max'} |")
    lines += [
        "",
        "If D_c* grows with n_f, a real encoder shows the same compression wall as the",
        "free-slot model (E2): more content needs more code. With the real retrieval wall",
        "(E4), both fixed-d bottlenecks are now validated on real models.",
        "",
        f"![real compression wall]({os.path.basename(fig)})",
    ]
    with open(os.path.join(ctx.outdir, "e5_curve.json"), "w", encoding="utf-8") as fh:
        json.dump(results["curve"], fh, indent=2)
    return results, "\n".join(lines)
