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


def load_limit(variant: str = "small", split: str = "test"):
    """Load the Weller et al. LIMIT adversarial retrieval set (arXiv:2508.21038, ICLR'26).

    LIMIT is the all-pairs k=2 pattern realized in natural language: 1000 "Who likes X?"
    queries over biographical docs, each query with **exactly 2** gold docs — the canonical
    real-data instance of E1's wall (frontier models score low even at full dim). ``variant``
    is "small" (46 docs, `orionweller/LIMIT-small`) or "full" (50k docs, `orionweller/LIMIT`).
    HF layout: config `corpus` (split default), `queries` (split default), `default` (qrels,
    split test; columns corpus-id/query-id/score). Returns (corpus, queries, qrels) like load_beir.
    """
    from datasets import load_dataset

    repo = "orionweller/LIMIT-small" if variant == "small" else "orionweller/LIMIT"

    def _first(ds):  # take the single split inside a one-config DatasetDict
        return ds[list(ds.keys())[0]]

    corpus_ds = _first(load_dataset(repo, "corpus"))
    queries_ds = _first(load_dataset(repo, "queries"))
    qrels_dd = load_dataset(repo, "default")
    qrels_ds = qrels_dd[split] if split in qrels_dd else _first(qrels_dd)

    corpus = {str(r["_id"]): ((r.get("title") or "") + " " + (r.get("text") or "")).strip()
              for r in corpus_ds}
    queries = {str(r["_id"]): r["text"] for r in queries_ds}
    qrels: dict[str, set[str]] = {}
    for r in qrels_ds:
        if int(r["score"]) > 0:
            qrels.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))

    queries = {q: t for q, t in queries.items() if q in qrels and (qrels[q] & corpus.keys())}
    qrels = {q: (qrels[q] & corpus.keys()) for q in queries}
    return corpus, queries, qrels


def truncate_normalize(emb: np.ndarray, d: int) -> np.ndarray:
    """Matryoshka truncation: keep the first d dims and L2-renormalize."""
    e = emb[:, :d].astype(np.float32)
    n = np.linalg.norm(e, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return e / n


def pca_project_normalize(fit_emb: np.ndarray, doc_emb: np.ndarray, q_emb: np.ndarray,
                          d: int):
    """Best d-dim *linear* code: PCA fit on docs, project docs+queries, L2-renormalize.

    For non-Matryoshka encoders, raw prefix truncation is unfair (dims aren't importance-
    ordered). PCA gives the optimal d-dim linear subspace of the doc distribution, so a wall
    that survives PCA truncation is a genuine capacity limit, not an MRL artifact. Both sides
    use the same projection (fit on the corpus) so inner products stay meaningful.
    """
    mean = fit_emb.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(fit_emb - mean, full_matrices=False)
    comps = vt[:d]                                   # (d, D)
    de = (doc_emb - mean) @ comps.T
    qe = (q_emb - mean) @ comps.T
    dn = np.linalg.norm(de, axis=1, keepdims=True); dn[dn == 0] = 1.0
    qn = np.linalg.norm(qe, axis=1, keepdims=True); qn[qn == 0] = 1.0
    return (de / dn).astype(np.float32), (qe / qn).astype(np.float32)


def rank_scores(q_emb: np.ndarray, d_emb: np.ndarray, k: int) -> np.ndarray:
    """Return top-k document indices for each query (Q x k), by cosine (emb are unit)."""
    scores = q_emb @ d_emb.T                      # (Q, N)
    k = min(k, scores.shape[1])
    # argpartition for top-k then sort those
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-part, axis=1)
    return np.take_along_axis(idx, order, axis=1)
