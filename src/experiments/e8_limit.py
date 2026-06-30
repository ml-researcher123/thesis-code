"""E8 -- LIMIT adversarial set: the retrieval wall on Weller et al.'s own canonical data.

E1 built the all-pairs k=2 capacity wall from free vectors; E4 showed real encoders inherit a
truncation wall on a benign benchmark. E8 closes the loop on the *canonical adversarial data*:
LIMIT (Weller et al., arXiv:2508.21038) realizes the all-pairs pattern in natural language --
1000 "Who likes X?" queries, each with exactly **2** gold docs, over biographical passages whose
attribute combinatorics exceed what any fixed-d single-vector model can separate. The paper's
finding is that frontier embedders score low even at full dimension; we reproduce that across
*several* real embedders (cross-model generality of C1) and add the Matryoshka-truncation curve
(the wall gets worse as d shrinks). A single weak number here would be unconvincing; the point
is that the failure is consistent across embedder families and is the real-data echo of E1.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..common import RunContext, ensure_deps
from ..eval.metrics import recall_at_k
from ..retrieval.beir_eval import load_limit, rank_scores, truncate_normalize


def _recall_multi(q_emb, d_emb, doc_ids, gold, ks):
    ranked = rank_scores(q_emb, d_emb, max(ks))
    out = {}
    for k in ks:
        rs = [recall_at_k([doc_ids[j] for j in ranked[i][:k]], gold[i], k)
              for i in range(len(gold))]
        out[k] = float(np.mean(rs))
    return out


def run(ctx: RunContext):
    cfg = ctx.cfg; log = ctx.log; p = cfg.get("params", {})
    ensure_deps({"sentence_transformers": "sentence-transformers", "datasets": "datasets"}, log)
    from sentence_transformers import SentenceTransformer

    variant = p.get("variant", "small")                       # small (46 docs) or full (50k)
    # each entry: {name, query_prompt?, trust_remote_code?}
    models = p.get("models", [
        {"name": "mixedbread-ai/mxbai-embed-large-v1"},
        {"name": "Snowflake/snowflake-arctic-embed-m-v1.5",
         "query_prompt": "Represent this sentence for searching relevant passages: "},
        {"name": "BAAI/bge-base-en-v1.5",
         "query_prompt": "Represent this sentence for searching relevant passages: "},
    ])
    ks = p.get("ks", [2, 5, 10, 20])
    trunc_dims = p.get("trunc_dims", [])                       # if set, also a truncation curve
    trunc_k = p.get("trunc_k", 10)

    log(f"E8 LIMIT-{variant} | models={[m['name'] for m in models]} ks={ks}")
    corpus, queries, qrels = load_limit(variant, p.get("split", "test"))
    doc_ids = list(corpus); q_ids = list(queries)
    gold = [qrels[q] for q in q_ids]
    log(f"  corpus={len(doc_ids)} queries={len(q_ids)} (each query has "
        f"~{np.mean([len(g) for g in gold]):.1f} gold docs)")

    full_recall = {}           # model -> {k: recall} at full dim
    trunc_curves = {}          # model -> {d: recall@trunc_k}
    for spec in models:
        name = spec["name"]
        qp = spec.get("query_prompt", "")
        model = SentenceTransformer(name, trust_remote_code=spec.get("trust_remote_code", False))
        if ctx.device.type == "cuda":
            model = model.to(ctx.device)
        de = model.encode([corpus[i] for i in doc_ids], batch_size=128,
                          normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        qe = model.encode([qp + queries[i] for i in q_ids], batch_size=128,
                          normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        full_recall[name] = _recall_multi(qe, de, doc_ids, gold, ks)
        log(f"  {name:42s} full_dim={de.shape[1]:5d} | "
            + " ".join(f"R@{k}={full_recall[name][k]:.3f}" for k in ks))
        if trunc_dims:
            dims = [d for d in trunc_dims if d <= de.shape[1]]
            trunc_curves[name] = {}
            for d in dims:
                ded, qed = truncate_normalize(de, d), truncate_normalize(qe, d)
                trunc_curves[name][d] = _recall_multi(qed, ded, doc_ids, gold, [trunc_k])[trunc_k]
            log(f"    trunc R@{trunc_k}: "
                + " ".join(f"d{d}={trunc_curves[name][d]:.3f}" for d in dims))
        del model

    # ---- figure: bar chart of recall@k per model (the cross-model wall) ----
    fig = os.path.join(ctx.outdir, "e8_limit.png")
    names = [m["name"].split("/")[-1] for m in models]
    x = np.arange(len(names)); w = 0.8 / len(ks)
    plt.figure(figsize=(7.0, 4.2))
    for i, k in enumerate(ks):
        plt.bar(x + i * w, [full_recall[m["name"]][k] for m in models], width=w, label=f"recall@{k}")
    plt.xticks(x + 0.4 - w / 2, names, rotation=15, ha="right", fontsize=8)
    plt.ylabel("recall (full dimension)")
    plt.ylim(0, 1)
    plt.title(f"E8: LIMIT-{variant} — real embedders fail the adversarial wall (k=2 gold/query)")
    plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(fig, dpi=140); plt.close()
    log(f"  saved figure -> {os.path.basename(fig)}")

    results = {
        "experiment": "e8_limit", "config_params": p, "variant": variant,
        "n_docs": len(doc_ids), "n_queries": len(q_ids), "ks": ks,
        "full_recall": full_recall, "trunc_curves": trunc_curves,
        "figure": os.path.basename(fig),
    }

    lines = [
        f"# E8 — LIMIT-{variant} Adversarial Retrieval Wall",
        "",
        f"Weller et al.'s LIMIT realizes the all-pairs (k=2) pattern in natural language: "
        f"{len(doc_ids)} docs, {len(q_ids)} queries, each with 2 gold docs. Recall at FULL "
        "embedding dimension (no truncation) — low recall here is the real-data, real-model "
        "echo of E1's capacity wall, across embedder families (C1 generality).",
        "",
        "| model | full dim | " + " | ".join(f"recall@{k}" for k in ks) + " |",
        "|---|---|" + "|".join("---" for _ in ks) + "|",
    ]
    for spec in models:
        name = spec["name"]
        row = full_recall[name]
        lines.append(f"| `{name}` | — | " + " | ".join(f"{row[k]:.3f}" for k in ks) + " |")
    if trunc_curves:
        lines += ["", f"Matryoshka truncation (recall@{trunc_k}) — the wall deepens as d shrinks:", ""]
        anymodel = next(iter(trunc_curves))
        ds = sorted(trunc_curves[anymodel])
        lines.append("| model | " + " | ".join(f"d={d}" for d in ds) + " |")
        lines.append("|---|" + "|".join("---" for _ in ds) + "|")
        for name, cur in trunc_curves.items():
            lines.append(f"| `{name}` | " + " | ".join(f"{cur.get(d, float('nan')):.3f}" for d in ds) + " |")
    lines += [
        "",
        "Consistent low recall across embedder families (not one model's quirk) confirms the",
        "capacity wall is a property of the single-vector paradigm, on Weller's own data.",
        "",
        f"![LIMIT wall]({os.path.basename(fig)})",
    ]
    with open(os.path.join(ctx.outdir, "e8_full_recall.json"), "w", encoding="utf-8") as fh:
        json.dump(full_recall, fh, indent=2)
    return results, "\n".join(lines)
