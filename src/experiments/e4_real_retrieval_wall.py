"""E4 -- Real-model retrieval wall (first real-model experiment; tests F9 for retrieval).

Does the free-vector retrieval wall (E1) appear in a *real* embedder? We embed a real
retrieval benchmark (BEIR) with a Matryoshka embedder, then truncate the embedding to
dimension d (Matryoshka-style, renormalized) and measure recall@k vs d. A wall -- recall
ramping then saturating as d grows -- means the geometric capacity limit is not an artifact
of free vectors but a property real encoders inherit. The saturation dimension is the
real-model analog of E1's critical d*, and (per the allocation law, C3) any embedding
dimension beyond it is wasted budget that should go to compression.

Deps (transformers/sentence-transformers/datasets) are typically preinstalled on Kaggle;
imported lazily so unrelated experiments don't pay for them.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..common import RunContext
from ..eval.metrics import ndcg_at_k, recall_at_k
from ..retrieval.beir_eval import (load_beir, load_limit, pca_project_normalize, rank_scores,
                                   truncate_normalize)


def run(ctx: RunContext):
    cfg = ctx.cfg
    log = ctx.log
    p = cfg.get("params", {})

    model_name = p.get("model", "mixedbread-ai/mxbai-embed-large-v1")
    dataset = p.get("dataset", "scifact")
    split = p.get("split", "test")
    dims = p.get("dims", [16, 32, 64, 128, 256, 512, 1024])
    k = p.get("k", 10)
    max_docs = p.get("max_docs", 0)
    query_prompt = p.get("query_prompt", "")  # some models want a query prefix
    truncation = p.get("truncation", "matryoshka")  # "matryoshka" (prefix) or "pca" (best linear)

    from ..common import ensure_deps
    ensure_deps({"sentence_transformers": "sentence-transformers", "datasets": "datasets"}, log)
    from sentence_transformers import SentenceTransformer

    log(f"E4 real retrieval wall | model={model_name} dataset={dataset} dims={dims} k={k} "
        f"trunc={truncation}")
    if dataset.startswith("limit"):
        corpus, queries, qrels = load_limit("small" if "small" in dataset else "full", split)
    else:
        corpus, queries, qrels = load_beir(dataset, split, max_docs=max_docs)
    doc_ids = list(corpus)
    q_ids = list(queries)
    log(f"  corpus={len(doc_ids)} queries={len(q_ids)}")

    model = SentenceTransformer(model_name, trust_remote_code=p.get("trust_remote_code", False))
    if ctx.device.type == "cuda":
        model = model.to(ctx.device)

    full_dim = int(model.get_sentence_embedding_dimension())
    dims = [d for d in dims if d <= full_dim]
    log(f"  model full dim={full_dim}; eval dims={dims}")

    d_emb_full = model.encode([corpus[i] for i in doc_ids], batch_size=128,
                              normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False)
    q_emb_full = model.encode([query_prompt + queries[i] for i in q_ids], batch_size=128,
                              normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False)

    gold = [qrels[q] for q in q_ids]
    curve = {}
    for d in dims:
        if truncation == "pca":
            de, qe = pca_project_normalize(d_emb_full, d_emb_full, q_emb_full, d)
        else:
            de = truncate_normalize(d_emb_full, d)
            qe = truncate_normalize(q_emb_full, d)
        ranked = rank_scores(qe, de, k)
        recalls, ndcgs = [], []
        for i in range(len(q_ids)):
            ranked_ids = [doc_ids[j] for j in ranked[i]]
            recalls.append(recall_at_k(ranked_ids, gold[i], k))
            ndcgs.append(ndcg_at_k(ranked_ids, gold[i], k))
        curve[d] = {"recall_at_k": float(np.mean(recalls)),
                    "ndcg_at_k": float(np.mean(ndcgs))}
        log(f"  d={d:5d} | recall@{k}={curve[d]['recall_at_k']:.3f} "
            f"ndcg@{k}={curve[d]['ndcg_at_k']:.3f}")

    # saturation dim: smallest d reaching >= frac of full-dim recall
    frac = p.get("saturation_frac", 0.98)
    full_recall = curve[max(dims)]["recall_at_k"]
    sat = next((d for d in sorted(dims)
                if curve[d]["recall_at_k"] >= frac * full_recall), None)

    fig = os.path.join(ctx.outdir, "e4_real_retrieval_wall.png")
    plt.figure(figsize=(6.5, 4.2))
    xs = sorted(curve)
    plt.plot(xs, [curve[d]["recall_at_k"] for d in xs], marker="o", label=f"recall@{k}")
    plt.plot(xs, [curve[d]["ndcg_at_k"] for d in xs], marker="s", label=f"ndcg@{k}")
    if sat:
        plt.axvline(sat, ls="--", c="red", lw=1, label=f"saturation d≈{sat}")
    plt.xscale("log", base=2)
    plt.xlabel("truncated embedding dimension d (log2)")
    plt.ylabel("retrieval quality")
    plt.title(f"E4: real-model retrieval wall ({model_name.split('/')[-1]}, {dataset})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig, dpi=140)
    plt.close()
    log(f"  saved figure -> {os.path.basename(fig)}  (saturation d≈{sat})")

    results = {
        "experiment": "e4_real_retrieval_wall",
        "config_params": p,
        "model": model_name, "full_dim": full_dim,
        "dataset": dataset, "n_docs": len(doc_ids), "n_queries": len(q_ids),
        "curve": {str(d): curve[d] for d in curve},
        "saturation_dim": sat, "saturation_frac": frac,
        "figure": os.path.basename(fig),
    }

    lines = [
        f"# E4 — Real-Model Retrieval Wall ({dataset})",
        "",
        f"Embedder: `{model_name}` (full dim {full_dim}); corpus {len(doc_ids)} docs, "
        f"{len(q_ids)} queries; Matryoshka truncation + renormalize.",
        "",
        f"| dim d | recall@{k} | ndcg@{k} |",
        "|---|---|---|",
    ]
    for d in sorted(curve):
        lines.append(f"| {d} | {curve[d]['recall_at_k']:.3f} | {curve[d]['ndcg_at_k']:.3f} |")
    lines += [
        "",
        f"Saturation dimension (>= {frac:.0%} of full-dim recall): **d ≈ {sat}**. Embedding",
        "dimension beyond this adds little retrieval quality — the real-model echo of E1's",
        "wall, and (per C3) wasted budget that the allocation law would move to compression.",
        "",
        f"![real retrieval wall]({os.path.basename(fig)})",
    ]
    summary = "\n".join(lines)
    with open(os.path.join(ctx.outdir, "e4_curve.json"), "w", encoding="utf-8") as fh:
        json.dump(results["curve"], fh, indent=2)
    return results, summary
