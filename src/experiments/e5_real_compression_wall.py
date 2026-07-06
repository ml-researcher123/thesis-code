"""E5 -- Real-model compression wall (tests F9 for compression; mirrors E2 with a real LLM).

Sweeps the number of soft tokens m and the content complexity n_f, measuring closed-set
associative-recall accuracy when a FROZEN LLM reads m soft tokens produced by a trained
write-projector. If accuracy ramps with m and the critical m* grows with n_f -- the same
shape as E2's free-slot wall -- then soft-token compression has a real capacity wall, and
both bottlenecks of the thesis are validated on real models.
"""
from __future__ import annotations

import json
import os
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..common import RunContext, ensure_deps
from ..compression.soft_prompt import fit_real_compression, result_to_dict


def run(ctx: RunContext):
    cfg = ctx.cfg
    log = ctx.log
    p = cfg.get("params", {})

    use_lora = p.get("use_lora", True)
    deps = {"transformers": "transformers"}
    if use_lora:
        deps.update({"peft": "peft", "accelerate": "accelerate"})
    ensure_deps(deps, log)

    model_name = p.get("model", "Qwen/Qwen2.5-0.5B")
    n_f_list = p.get("n_f_list", [4, 8, 16, 32])
    m_list = p.get("m_list", [1, 2, 4, 8, 16, 32])
    V = p.get("V", 16)
    K = p.get("K", 64)
    steps = p.get("steps", 300)
    batch = p.get("batch", 32)
    eval_batches = p.get("eval_batches", 8)
    lr = p.get("lr", 1e-3)
    lora_lr = p.get("lora_lr", None)
    seeds = p.get("seeds", [0, 1])
    acc_thresh = p.get("acc_thresh", 0.9)
    lora_r = p.get("lora_r", 16)
    cache_dir = p.get("cache_dir", None)

    log(f"E5 real compression wall | model={model_name} n_f={n_f_list} m={m_list} "
        f"V={V} seeds={seeds} device={ctx.device}")

    records = []
    curve = {nf: {} for nf in n_f_list}
    for n_f in n_f_list:
        for m in m_list:
            if m > 2 * n_f:  # can't pool 2*n_f passage tokens into more than 2*n_f chunks
                continue
            accs, losses = [], []
            for seed in seeds:
                r = fit_real_compression(
                    model_name=model_name, n_f=n_f, m=m, V=V, K=K, steps=steps,
                    batch=batch, eval_batches=eval_batches, lr=lr, lora_lr=lora_lr,
                    seed=seed, device=ctx.device, log=log if cfg.get("verbose") else None,
                    use_lora=use_lora, lora_r=lora_r, cache_dir=cache_dir,
                )
                records.append(result_to_dict(r))
                accs.append(r.accuracy)
                losses.append(r.final_loss)
            curve[n_f][m] = mean(accs)
            log(f"  n_f={n_f:3d} m={m:3d} | acc={mean(accs):.3f}+-{pstdev(accs):.3f} "
                f"loss={mean(losses):.4f} (chance={1.0/V:.3f})")

    crit = {}
    for n_f in n_f_list:
        crit[n_f] = next((m for m in sorted(curve[n_f]) if curve[n_f][m] >= acc_thresh), None)

    fig = os.path.join(ctx.outdir, "e5_real_compression_wall.png")
    plt.figure(figsize=(6.5, 4.2))
    for n_f in n_f_list:
        xs = sorted(curve[n_f])
        if xs:
            plt.plot(xs, [curve[n_f][m] for m in xs], marker="o", label=f"n_f={n_f}")
    plt.axhline(1.0 / V, ls=":", c="red", lw=1, label=f"chance={1.0/V:.2f}")
    plt.axhline(acc_thresh, ls="--", c="gray", lw=1, label=f"thresh={acc_thresh}")
    plt.xscale("log", base=2)
    plt.xlabel("soft tokens m (log2)")
    plt.ylabel("associative recall accuracy")
    plt.title(f"E5: real-model compression wall ({model_name.split('/')[-1]})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig, dpi=140)
    plt.close()
    log(f"  saved figure -> {os.path.basename(fig)}")

    results = {
        "experiment": "e5_real_compression_wall",
        "config_params": p, "model": model_name,
        "records": records,
        "curve": {str(nf): {str(m): v for m, v in curve[nf].items()} for nf in curve},
        "critical_m": {str(nf): crit[nf] for nf in crit},
        "acc_thresh": acc_thresh,
        "figure": os.path.basename(fig),
    }

    lines = [
        f"# E5 — Real-Model Compression Wall ({model_name.split('/')[-1]})",
        "",
        f"Frozen LLM reads m soft tokens (trained write-projector + read-head); closed-set "
        f"recall over V={V} values (chance={1.0/V:.3f}); seeds={seeds}.",
        "",
        "Critical soft-token count m* (acc ≥ "
        f"{acc_thresh}) vs content complexity n_f:",
        "",
        "| facts n_f | critical m* |",
        "|---|---|",
    ]
    for n_f in n_f_list:
        lines.append(f"| {n_f} | {crit[n_f] if crit[n_f] is not None else '> max m'} |")
    lines += [
        "",
        "If m* grows with n_f, a real frozen LLM shows the same compression capacity wall as",
        "the free-slot model (E2) — validating F9 on the compression side and completing the",
        "real-model evidence that both bottlenecks are genuine.",
        "",
        f"![real compression wall]({os.path.basename(fig)})",
    ]
    return results, "\n".join(lines)
