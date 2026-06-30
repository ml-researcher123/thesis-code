"""Load BEIR-style retrieval datasets from the HF Hub and score truncated embeddings.

Used by E4 (the first real-model experiment) to test whether the free-vector retrieval
wall (E1) shows up in a *real* embedder when its embedding dimension is truncated
Matryoshka-style. Kept dependency-light: `datasets` for loading, numpy for scoring.
"""
from __future__ import annotations

import numpy as np


def load_beir(name: str = "scifact", split: str = "test", max_docs: int = 0):
    """Return (corpus: id->text, queries: id->text, qrels: qid->set(docid)).

    Uses the standard BEIR-on-HF layout: `BeIR/<name>` with 'corpus'/'queries' configs and
    `BeIR/<name>-qrels` with the relevance split. If ``max_docs`` > 0 the corpus is capped
    (gold docs for the kept queries are always retained so recall stays well-defined).
    """
    from datasets import load_dataset

    corpus_ds = load_dataset(f"BeIR/{name}", "corpus")["corpus"]
    queries_ds = load_dataset(f"BeIR/{name}", "queries")["queries"]
    qrels_ds = load_dataset(f"BeIR/{name}-qrels")[split]

    qrels: dict[str, set[str]] = {}
    for r in qrels_ds:
        if int(r["score"]) > 0:
            qrels.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))

    corpus = {}
    for r in corpus_ds:
        text = ((r.get("title") or "") + " " + (r.get("text") or "")).strip()
        corpus[str(r["_id"])] = text
    queries = {str(r["_id"]): r["text"] for r in queries_ds}

    # keep only queries whose gold docs exist in the corpus
    queries = {q: t for q, t in queries.items() if q in qrels and (qrels[q] & corpus.keys())}
    qrels = {q: (qrels[q] & corpus.keys()) for q in queries}

    if max_docs and len(corpus) > max_docs:
        gold = set().union(*qrels.values()) if qrels else set()
        others = [d for d in corpus if d not in gold]
        rng = np.random.default_rng(0)
        keep = set(gold) | set(rng.choice(others, size=max(0, max_docs - len(gold)),
                                          replace=False).tolist())
        corpus = {d: corpus[d] for d in corpus if d in keep}

    return corpus, queries, qrels


def truncate_normalize(emb: np.ndarray, d: int) -> np.ndarray:
    """Matryoshka truncation: keep the first d dims and L2-renormalize."""
    e = emb[:, :d].astype(np.float32)
    n = np.linalg.norm(e, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return e / n


def rank_scores(q_emb: np.ndarray, d_emb: np.ndarray, k: int) -> np.ndarray:
    """Return top-k document indices for each query (Q x k), by cosine (emb are unit)."""
    scores = q_emb @ d_emb.T                      # (Q, N)
    k = min(k, scores.shape[1])
    # argpartition for top-k then sort those
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-part, axis=1)
    return np.take_along_axis(idx, order, axis=1)
