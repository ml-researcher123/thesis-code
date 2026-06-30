"""E2 -- Compression capacity wall (contribution C1, the first novel experiment).

Two questions:
  (1) MAIN: does soft-token compression have a sharp capacity wall in total code size
      D_c = m*d_c, and does the critical D_c* grow with content complexity n_f -- i.e.
      the same geometric form as the retrieval wall in E1?
  (2) SPLIT: at fixed D_c, does only the *product* m*d_c matter, or does the shape
      (number of slots m vs slot width d_c) matter? This previews the allocation law (C3).

Outputs: a results dict, a wall figure (recall vs D_c, one curve per n_f), a split-
invariance figure (recall vs m at fixed D_c), and a markdown summary with critical D_c*.
"""
from __future__ import annotations

import json
import os
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..common import RunContext
from ..compression.free_slots import fit_compression, result_to_dict


def _critical_dc(curve: list[tuple[int, float]], thresh: float):
    for dc, val in sorted(curve):
        if val >= thresh:
            return dc
    return None


def run(ctx: RunContext):
    cfg = ctx.cfg
    log = ctx.log
    p = cfg.get("params", {})

    n_f_list = p.get("n_f_list", [16, 32, 64, 128])
    m_list = p.get("m_list", [1, 2, 4, 8, 16, 32])
    d_c = p.get("d_c", 16)
    V = p.get("V", 4)
    d_key = p.get("d_key", 128)
    P = p.get("P", 128)
    hidden = p.get("hidden", 64)
    steps = p.get("steps", 1500)
    lr = p.get("lr", 3e-3)
    seeds = p.get("seeds", [0, 1, 2])
    recall_thresh = p.get("recall_thresh", 0.95)

    # split-invariance probe: at this fixed D_c, try several (m, d_c) factorizations
    split_D_c = p.get("split_D_c", 128)
    split_m_list = p.get("split_m_list", [1, 2, 4, 8, 16, 32, 64, 128])
    split_n_f = p.get("split_n_f", 64)

    log(f"E2 compression capacity | n_f={n_f_list} m={m_list} d_c={d_c} V={V} "
        f"P={P} seeds={seeds} device={ctx.device}")

    records: list[dict] = []
    # curve_means[n_f][D_c] = mean recall across seeds
    curve_means: dict[int, dict[int, float]] = {nf: {} for nf in n_f_list}

    for n_f in n_f_list:
        for m in m_list:
            recs, perfs, losses = [], [], []
            for seed in seeds:
                r = fit_compression(
                    n_f=n_f, m=m, d_c=d_c, V=V, d_key=d_key, P=P, hidden=hidden,
                    steps=steps, lr=lr, seed=seed, device=ctx.device,
                    log=log if cfg.get("verbose") else None,
                )
                rec = result_to_dict(r)
                records.append(rec)
                recs.append(r.recall)
                perfs.append(r.perfect_rate)
                losses.append(r.final_loss)
            D_c = m * d_c
            curve_means[n_f][D_c] = mean(recs)
            log(f"  n_f={n_f:4d} D_c={D_c:4d} (m={m:3d}) | recall={mean(recs):.3f}"
                f"+-{pstdev(recs):.3f} perfect={mean(perfs):.3f} loss={mean(losses):.4f}")

    crit = {nf: _critical_dc(list(curve_means[nf].items()), recall_thresh) for nf in n_f_list}

    # ---- split-invariance probe ----
    log(f"E2 split-invariance | fixed D_c={split_D_c} n_f={split_n_f} m in {split_m_list}")
    split_records = []
    split_curve = {}
    for m in split_m_list:
        if split_D_c % m != 0:
            continue
        dc = split_D_c // m
        recs = []
        for seed in seeds:
            r = fit_compression(
                n_f=split_n_f, m=m, d_c=dc, V=V, d_key=d_key, P=P, hidden=hidden,
                steps=steps, lr=lr, seed=seed, device=ctx.device,
            )
            split_records.append(result_to_dict(r))
            recs.append(r.recall)
        split_curve[m] = mean(recs)
        log(f"  split m={m:4d} d_c={dc:4d} (D_c={split_D_c}) | recall={mean(recs):.3f}")

    # ---- figure 1: the compression wall ----
    fig1 = os.path.join(ctx.outdir, "e2_compression_wall.png")
    plt.figure(figsize=(6.5, 4.2))
    for n_f in n_f_list:
        xs = sorted(curve_means[n_f])
        ys = [curve_means[n_f][d] for d in xs]
        plt.plot(xs, ys, marker="o", label=f"n_f={n_f}")
    plt.axhline(recall_thresh, ls="--", c="gray", lw=1, label=f"thresh={recall_thresh}")
    plt.axhline(1.0 / V, ls=":", c="red", lw=1, label=f"chance={1.0/V:.2f}")
    plt.xscale("log", base=2)
    plt.xlabel("total code dimension D_c = m·d_c (log2)")
    plt.ylabel("associative recall (best case)")
    plt.title("E2: compression capacity wall — D_c* grows with content n_f")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig1, dpi=140)
    plt.close()

    # ---- figure 2: split invariance ----
    fig2 = os.path.join(ctx.outdir, "e2_split_invariance.png")
    plt.figure(figsize=(6.5, 4.2))
    xs = sorted(split_curve)
    ys = [split_curve[m] for m in xs]
    plt.plot(xs, ys, marker="s", color="purple")
    plt.xscale("log", base=2)
    plt.xlabel(f"number of slots m  (d_c = {split_D_c}/m, so D_c fixed = {split_D_c})")
    plt.ylabel("associative recall")
    plt.title(f"E2: split invariance at fixed D_c={split_D_c}, n_f={split_n_f}")
    plt.tight_layout()
    plt.savefig(fig2, dpi=140)
    plt.close()
    log(f"  saved figures -> {os.path.basename(fig1)}, {os.path.basename(fig2)}")

    results = {
        "experiment": "e2_compression_capacity",
        "config_params": p,
        "records": records,
        "curve_means": {str(k): v for k, v in curve_means.items()},
        "critical_D_c": {str(k): v for k, v in crit.items()},
        "recall_thresh": recall_thresh,
        "split_invariance": {
            "fixed_D_c": split_D_c, "n_f": split_n_f,
            "recall_by_m": {str(k): v for k, v in split_curve.items()},
            "records": split_records,
        },
        "figures": [os.path.basename(fig1), os.path.basename(fig2)],
    }

    lines = [
        "# E2 — Compression Capacity Wall (C1)",
        "",
        f"Slot-memory associative recall; d_c={d_c}, V={V} (chance={1.0/V:.2f}), "
        f"P={P} passages, seeds={seeds}.",
        "",
        "Critical code size D_c* (smallest D_c reaching recall "
        f">= {recall_thresh}) vs content complexity n_f:",
        "",
        "| facts n_f | critical D_c* |",
        "|---|---|",
    ]
    for n_f in n_f_list:
        lines.append(f"| {n_f} | {crit[n_f] if crit[n_f] is not None else '> max D_c'} |")
    lines += [
        "",
        "If D_c* grows with n_f, soft-token compression has a dimension wall of the same",
        "geometric form as retrieval (E1) — i.e. a genuine *second* bottleneck. The split-",
        "invariance probe (fig 2) tests whether only the product m·d_c matters at fixed D_c.",
        "",
        f"![compression wall]({os.path.basename(fig1)})",
        "",
        f"![split invariance]({os.path.basename(fig2)})",
    ]
    summary = "\n".join(lines)

    with open(os.path.join(ctx.outdir, "e2_records.json"), "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)

    return results, summary
