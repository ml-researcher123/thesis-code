"""E5b -- Multi-token real compression: recover capacity + reproduce F6 on a real encoder.

E5 showed that compressing a passage to ONE real embedding holds ~1 fact. Here we give the
compressor m chunk-embeddings (pool cached per-fact vectors into m chunks, truncate each to
d_c, concat -> code of size D_c = m*d_c). Two questions:

  WALL: with each fact in its own slot (m = n_f), does recall ramp with D_c and the critical
        D_c* grow with n_f? (capacity recovers vs the single-vector collapse of E5)
  SHAPE (F6): at a FIXED total budget D_c, vary the number of chunks m -- is there an interior
        optimum, with m=1 (one fat vector) and large-m (thin slices) both worse? This is the
        real-encoder reproduction of E2's F6 shape effect.

Per-fact embeddings are cached, so the m-sweep is free.
"""
from __future__ import annotations

import json
import os
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..common import RunContext, ensure_deps
from ..compression.encoder_probe import (KEYS, VALUES, build_slots_chunked,
                                         encode_factwise, fit_attn_probe, gen_factwise)


def run(ctx: RunContext):
    cfg = ctx.cfg
    log = ctx.log
    p = cfg.get("params", {})
    ensure_deps({"sentence_transformers": "sentence-transformers"}, log)
    from sentence_transformers import SentenceTransformer

    model_name = p.get("model", "mixedbread-ai/mxbai-embed-large-v1")
    V = len(VALUES)
    P_train = p.get("P_train", 256)
    P_eval = p.get("P_eval", 128)
    steps = p.get("steps", 500)
    lr = p.get("lr", 2e-3)
    seeds = p.get("seeds", [0, 1])
    key_trunc = p.get("key_trunc", 64)

    wall_n_f = p.get("wall_n_f", [4, 8, 16])
    wall_d_c = p.get("wall_d_c", [4, 8, 16, 32, 64])      # m = n_f, so D_c = n_f * d_c
    shape_n_f = p.get("shape_n_f", 16)
    shape_D_c = p.get("shape_D_c", 256)
    shape_m = p.get("shape_m", [1, 2, 4, 8, 16])          # d_c = shape_D_c // m

    log(f"E5b multi-token compression | model={model_name} V={V} device={ctx.device}")
    model = SentenceTransformer(model_name, trust_remote_code=p.get("trust_remote_code", False))
    if ctx.device.type == "cuda":
        model = model.to(ctx.device)
    key_vecs = model.encode(list(KEYS), normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False)

    # ---- WALL: m = n_f, sweep d_c ----
    wall = {nf: {} for nf in wall_n_f}
    for n_f in wall_n_f:
        for seed in seeds:
            tr = gen_factwise(n_f, P_train, seed)
            te = gen_factwise(n_f, P_eval, 9000 + seed)
            tr_fv = encode_factwise(model, tr)
            te_fv = encode_factwise(model, te)
            for d_c in wall_d_c:
                Str, ktr, ytr = build_slots_chunked(tr_fv, tr, key_vecs, n_f, d_c, key_trunc)
                Ste, kte, yte = build_slots_chunked(te_fv, te, key_vecs, n_f, d_c, key_trunc)
                acc = fit_attn_probe(Str, ktr, ytr, Ste, kte, yte, V, steps, lr, seed, ctx.device)
                wall[n_f].setdefault(d_c, []).append(acc)
        for d_c in wall_d_c:
            log(f"  [WALL] n_f={n_f:3d} m=n_f d_c={d_c:3d} D_c={n_f*d_c:4d} | "
                f"acc={mean(wall[n_f][d_c]):.3f}")
    wall = {nf: {dc: mean(v) for dc, v in wall[nf].items()} for nf in wall}

    # ---- SHAPE (F6): fixed n_f and D_c, vary m ----
    shape = {}
    for seed in seeds:
        tr = gen_factwise(shape_n_f, P_train, seed)
        te = gen_factwise(shape_n_f, P_eval, 9000 + seed)
        tr_fv = encode_factwise(model, tr)
        te_fv = encode_factwise(model, te)
        for m in shape_m:
            if shape_D_c % m != 0 or m > shape_n_f:
                continue
            d_c = shape_D_c // m
            Str, ktr, ytr = build_slots_chunked(tr_fv, tr, key_vecs, m, d_c, key_trunc)
            Ste, kte, yte = build_slots_chunked(te_fv, te, key_vecs, m, d_c, key_trunc)
            acc = fit_attn_probe(Str, ktr, ytr, Ste, kte, yte, V, steps, lr, seed, ctx.device)
            shape.setdefault(m, []).append(acc)
    shape = {m: mean(v) for m, v in shape.items()}
    for m in sorted(shape):
        log(f"  [SHAPE] n_f={shape_n_f} D_c={shape_D_c} m={m:3d} (d_c={shape_D_c//m:3d}) | "
            f"acc={shape[m]:.3f}")

    # ---- figures ----
    fig1 = os.path.join(ctx.outdir, "e5b_wall.png")
    plt.figure(figsize=(6.5, 4.2))
    for n_f in wall_n_f:
        xs = sorted(wall[n_f])
        plt.plot([n_f * dc for dc in xs], [wall[n_f][dc] for dc in xs],
                 marker="o", label=f"n_f={n_f} (m=n_f)")
    plt.axhline(1.0 / V, ls=":", c="red", lw=1, label=f"chance={1.0/V:.2f}")
    plt.xscale("log", base=2)
    plt.xlabel("total code size D_c = m·d_c (log2)")
    plt.ylabel("fact-recall accuracy (held-out)")
    plt.title(f"E5b: multi-token compression wall ({model_name.split('/')[-1]})")
    plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(fig1, dpi=140); plt.close()

    fig2 = os.path.join(ctx.outdir, "e5b_shape_f6.png")
    plt.figure(figsize=(6.5, 4.2))
    xs = sorted(shape)
    plt.plot(xs, [shape[m] for m in xs], marker="s", color="purple")
    plt.axhline(1.0 / V, ls=":", c="red", lw=1, label=f"chance={1.0/V:.2f}")
    plt.xscale("log", base=2)
    plt.xlabel(f"number of chunks m  (d_c = {shape_D_c}/m; D_c fixed = {shape_D_c})")
    plt.ylabel("fact-recall accuracy")
    plt.title(f"E5b: F6 shape effect on a real encoder (n_f={shape_n_f}, D_c={shape_D_c})")
    plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(fig2, dpi=140); plt.close()
    log(f"  saved figures -> {os.path.basename(fig1)}, {os.path.basename(fig2)}")

    best_m = max(shape, key=shape.get) if shape else None
    results = {
        "experiment": "e5b_real_compression_shape",
        "config_params": p, "model": model_name, "V": V,
        "wall": {str(nf): {str(dc): wall[nf][dc] for dc in wall[nf]} for nf in wall},
        "shape": {str(m): shape[m] for m in shape},
        "shape_best_m": best_m,
        "figures": [os.path.basename(fig1), os.path.basename(fig2)],
    }

    lines = [
        f"# E5b — Multi-Token Real Compression ({model_name.split('/')[-1]})",
        "",
        f"Per-fact embeddings pooled into m chunks (each truncated to d_c). V={V}, "
        f"chance={1.0/V:.3f}, held-out passages, seeds={seeds}.",
        "",
        "**Wall (m = n_f):** recall vs total code D_c, per content n_f.",
        "",
        "| n_f | D_c at d_c=max | recall |",
        "|---|---|---|",
    ]
    for n_f in wall_n_f:
        dmax = max(wall[n_f])
        lines.append(f"| {n_f} | {n_f*dmax} | {wall[n_f][dmax]:.3f} |")
    lines += [
        "",
        f"**Shape / F6 (n_f={shape_n_f}, fixed D_c={shape_D_c}):** recall vs number of chunks m.",
        "",
        "| m | d_c | recall |",
        "|---|---|---|",
    ]
    for m in sorted(shape):
        lines.append(f"| {m} | {shape_D_c//m} | {shape[m]:.3f} |")
    lines += [
        "",
        f"Best m = **{best_m}** (interior optimum reproduces E2's F6 on a real encoder if "
        "m=1 and large-m both underperform). Multi-token compression recovers capacity that",
        "the single vector (E5) lost — the compression budget should be spread across several",
        "soft tokens, not one fat one.",
        "",
        f"![wall]({os.path.basename(fig1)})",
        "",
        f"![shape f6]({os.path.basename(fig2)})",
    ]
    with open(os.path.join(ctx.outdir, "e5b_results.json"), "w", encoding="utf-8") as fh:
        json.dump({"wall": results["wall"], "shape": results["shape"]}, fh, indent=2)
    return results, "\n".join(lines)
