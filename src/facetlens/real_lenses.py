"""Real-encoder facet-lens (C4 on real embeddings): learned low-rank lenses over a FROZEN encoder.

The free-vector C4 (lenses.py) showed routed facet-lenses escape the single-vector wall when the
embeddings are free parameters. The reviewer question is whether that survives on *real* semantic
embeddings, which already cram many facets into a fixed-d space. Here every doc/query is real
natural language about an entity with F categorical facets; we encode them ONCE with a frozen
sentence encoder (dim D), then learn three things at EQUAL doc-side budget d_total:

  single     : one projection W: R^D -> R^{d_total}; score = <W_q q, W_d d>.
  multiview  : K projections of width d_total/K; score = max_k <W_q^k q, W_d^k d> (ColBERT-like, no routing).
  facetlens  : F projections of width d_total/F, ROUTED by the query's facet; the query of facet f
               is scored only through lens f.

A single low-rank projection of a real embedding must keep all F facets separable at once; a routed
lens only needs to keep its one facet. If facetlens beats single (and generic multiview) at small
d_total in retrieval quality (mAP), the escape is a property of the representation geometry, not of
free vectors -- exactly the claim C4 needs on real models.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F_

from ..common import set_seed

# Realistic semantic facets (entities/attributes/relations) -> natural-language docs & queries.
FACETS: dict[str, dict] = {
    "profession": {"values": ["doctor", "teacher", "engineer", "artist"],
                   "doc": "works as a {v}", "q": "Find people who work as a {v}."},
    "city":       {"values": ["Paris", "Tokyo", "Cairo", "Lima"],
                   "doc": "lives in {v}", "q": "Find people who live in {v}."},
    "hobby":      {"values": ["chess", "hiking", "painting", "cooking"],
                   "doc": "enjoys {v}", "q": "Find people who enjoy {v}."},
    "pet":        {"values": ["a dog", "a cat", "a parrot", "a turtle"],
                   "doc": "owns {v}", "q": "Find people who own {v}."},
    "vehicle":    {"values": ["a bicycle", "a sedan", "a scooter", "a truck"],
                   "doc": "drives {v}", "q": "Find people who drive {v}."},
    "instrument": {"values": ["the piano", "the guitar", "the violin", "the drums"],
                   "doc": "plays {v}", "q": "Find people who play {v}."},
    "color":      {"values": ["red", "green", "blue", "yellow"],
                   "doc": "prefers the color {v}", "q": "Find people who prefer the color {v}."},
    "sport":      {"values": ["tennis", "soccer", "swimming", "cycling"],
                   "doc": "practices {v}", "q": "Find people who practice {v}."},
    "drink":      {"values": ["coffee", "tea", "juice", "water"],
                   "doc": "drinks {v}", "q": "Find people who drink {v}."},
    "language":   {"values": ["French", "Spanish", "German", "Arabic"],
                   "doc": "speaks {v}", "q": "Find people who speak {v}."},
}
NAMES = ("Alex Sam Jordan Casey Riley Morgan Taylor Jamie Avery Quinn Drew Reese Skyler "
         "Rowan Emerson Finley Harper Kendall Logan Parker Sawyer Tatum Blake Cameron").split()


def make_real_facet_corpus(N: int, facet_names: list[str], seed: int):
    """N entities, each with one value per facet. Returns doc texts, query specs, attrs, A, foq.

    A (M,N) relevance: query (facet f, value g) -> all docs with attrs[:,f]==g. facet_of_query (M,).
    """
    rng = np.random.default_rng(seed)
    facets = [(fn, FACETS[fn]) for fn in facet_names]
    F = len(facets)
    G = min(len(spec["values"]) for _, spec in facets)
    attrs = np.zeros((N, F), dtype=np.int64)
    for fi, (_, spec) in enumerate(facets):
        col = np.tile(np.arange(len(spec["values"])), N // len(spec["values"]) + 1)[:N]
        rng.shuffle(col)
        attrs[:, fi] = col

    doc_texts = []
    for i in range(N):
        name = NAMES[i % len(NAMES)] + (f" {i // len(NAMES)}" if i >= len(NAMES) else "")
        clauses = [spec["doc"].format(v=spec["values"][attrs[i, fi]]) for fi, (_, spec) in enumerate(facets)]
        doc_texts.append(f"{name} " + ", ".join(clauses) + ".")

    A, facet_of_query, query_texts = [], [], []
    for fi, (_, spec) in enumerate(facets):
        for g in range(G):
            A.append(attrs[:, fi] == g)
            facet_of_query.append(fi)
            query_texts.append(spec["q"].format(v=spec["values"][g]))
    A = np.stack(A).astype(np.float32)                 # (M, N)
    return doc_texts, query_texts, attrs, A, np.asarray(facet_of_query, dtype=np.int64)


def _average_precision(scores_row: torch.Tensor, rel_row: torch.Tensor) -> torch.Tensor:
    """Mean average precision contribution for one query (scores, binary rel over N docs)."""
    order = torch.argsort(scores_row, descending=True)
    rel_sorted = rel_row[order]
    csum = torch.cumsum(rel_sorted, dim=0)
    ranks = torch.arange(1, rel_sorted.numel() + 1, device=scores_row.device, dtype=scores_row.dtype)
    prec_at = csum / ranks
    denom = rel_sorted.sum().clamp(min=1)
    return (prec_at * rel_sorted).sum() / denom


def _metrics(S: torch.Tensor, A: torch.Tensor, margin: float):
    """S,A: (M,N). Returns (rank_loss, realizability, mAP)."""
    neg_inf = torch.finfo(S.dtype).min
    rel = torch.where(A > 0, S, torch.full_like(S, float("inf")))
    nonrel = torch.where(A > 0, torch.full_like(S, neg_inf), S)
    min_rel = rel.min(dim=1).values
    max_non = nonrel.max(dim=1).values
    loss = F_.softplus(margin - (min_rel - max_non)).mean()
    real = (min_rel > max_non).float().mean()
    ap = torch.stack([_average_precision(S[i], A[i]) for i in range(S.shape[0])])
    return loss, float(real.detach()), float(ap.mean().detach())


def fit_real_lenses(doc_emb, query_emb, A, facet_of_query, F, *, mode, d_total,
                    steps=600, lr=0.01, margin=0.5, seed=0, device="cpu", log=None):
    """Learn projections of FROZEN encoder embeddings. Returns (realizability, mAP)."""
    set_seed(seed)
    device = torch.device(device)
    D = doc_emb.shape[1]
    Demb = torch.tensor(doc_emb, dtype=torch.float32, device=device)      # (N, D)
    Qemb = torch.tensor(query_emb, dtype=torch.float32, device=device)    # (M, D)
    A = torch.tensor(A, dtype=torch.float32, device=device)
    foq = torch.tensor(facet_of_query, dtype=torch.long, device=device)
    M = A.shape[0]

    K = 1 if mode == "single" else F
    dk = max(1, d_total // K)
    # one (doc-proj, query-proj) pair per lens; doc-side width dk is the budget that "costs".
    Wd = torch.nn.Parameter(torch.randn(K, D, dk, device=device) * (D ** -0.5))
    Wq = torch.nn.Parameter(torch.randn(K, D, dk, device=device) * (D ** -0.5))
    opt = torch.optim.Adam([Wd, Wq], lr=lr)

    def scores():
        if mode == "single":
            return (Qemb @ Wq[0]) @ (Demb @ Wd[0]).t()
        if mode == "multiview":
            per = torch.stack([(Qemb @ Wq[k]) @ (Demb @ Wd[k]).t() for k in range(K)], dim=0)
            return per.max(dim=0).values
        # facetlens: route each query through its facet's lens
        Dp = torch.stack([Demb @ Wd[k] for k in range(K)], dim=0)         # (K, N, dk)
        Qp = torch.stack([Qemb @ Wq[k] for k in range(K)], dim=0)         # (K, M, dk)
        Qr = Qp[foq, torch.arange(M, device=device)]                      # (M, dk)
        Dr = Dp[foq]                                                      # (M, N, dk)
        return torch.einsum("md,mnd->mn", Qr, Dr)

    real = ap = 0.0
    for step in range(steps):
        S = scores()
        loss, real, ap = _metrics(S, A, margin)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if log and (step % max(1, steps // 4) == 0 or step == steps - 1):
            log(f"    {mode} d={d_total} seed={seed} step={step:4d} "
                f"loss={float(loss):.4f} real={real:.3f} mAP={ap:.3f}")
    return real, ap
