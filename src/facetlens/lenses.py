"""Facet-lens multi-view retrieval (C4): escape the single-vector wall via specialization.

Premise (E1/Weller): a single d-dim embedding cannot realize arbitrary relevance patterns —
there is a capacity wall. We test whether splitting the SAME total budget into K
facet-specialized low-rank "lenses", combined per query, escapes that wall.

Three conditions at EQUAL total doc-embedding budget (N x d_total params):
  single     : one d_total embedding; score = <q, x>.
  multiview  : K views of width d_total/K, no routing; score = max_k <q_k, x_k> (ColBERT-like).
  facetlens  : F lenses of width d_total/F, ROUTED by the query's facet; score = <q, x_{f(q)}>.

Pattern has explicit facet structure: each doc has F categorical facets; a query targets one
(facet, value) and its relevant set is all docs sharing that value. A single embedding must
cram all F facets into d_total and they interfere; routed facet-lenses let each query use only
the lens that matters, so each lens needs to separate just one facet. We measure realizability
(every relevant doc outranks every non-relevant doc) vs budget.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F_

from ..common import set_seed


def make_facet_pattern(N: int, F: int, G: int, seed: int):
    """N docs with F categorical facets (G values each). Queries = every (facet, value).

    Returns A (M,N) relevance, facet_of_query (M,), and per-query group sizes.
    """
    rng = np.random.default_rng(seed)
    # balanced assignment: each facet's values split the N docs as evenly as possible
    attrs = np.zeros((N, F), dtype=np.int64)
    for f in range(F):
        col = np.tile(np.arange(G), N // G + 1)[:N]
        rng.shuffle(col)
        attrs[:, f] = col
    A, facet_of_query = [], []
    for f in range(F):
        for g in range(G):
            A.append(attrs[:, f] == g)
            facet_of_query.append(f)
    A = np.stack(A).astype(np.float32)              # (M, N)
    return torch.tensor(A), torch.tensor(facet_of_query, dtype=torch.long), attrs


def _rank_loss_and_real(S: torch.Tensor, A: torch.Tensor, margin: float):
    """S,A: (M,N). Every relevant doc should outrank every non-relevant doc, per query."""
    neg_inf = torch.finfo(S.dtype).min
    rel = torch.where(A > 0, S, torch.full_like(S, float("inf")))
    nonrel = torch.where(A > 0, torch.full_like(S, neg_inf), S)
    min_rel = rel.min(dim=1).values                  # (M,)
    max_non = nonrel.max(dim=1).values               # (M,)
    loss = F_.softplus(margin - (min_rel - max_non)).mean()
    real = (min_rel > max_non).float().mean()
    return loss, real


def fit_lenses(A, facet_of_query, F, *, mode, d_total, steps=800, lr=0.05, margin=1.0,
               seed=0, device="cpu", log=None):
    set_seed(seed)
    device = torch.device(device)
    A = A.to(device); facet_of_query = facet_of_query.to(device)
    M, N = A.shape

    if mode == "single":
        K, dk = 1, d_total
    elif mode == "multiview":
        K = F                       # match lens count for a fair comparison
        dk = max(1, d_total // K)
    elif mode == "facetlens":
        K = F
        dk = max(1, d_total // K)
    else:
        raise ValueError(mode)

    X = torch.nn.Parameter(torch.randn(K, N, dk, device=device) * 0.1)
    Q = torch.nn.Parameter(torch.randn(K, M, dk, device=device) * 0.1)
    opt = torch.optim.Adam([X, Q], lr=lr)

    def scores():
        if mode == "single":
            return Q[0] @ X[0].t()
        if mode == "multiview":
            per = torch.stack([Q[k] @ X[k].t() for k in range(K)], dim=0)  # (K,M,N)
            return per.max(dim=0).values
        # facetlens: each query uses the lens (doc bank + query vec) of its facet
        Xq = X[facet_of_query]                        # (M, N, dk)
        Qrouted = Q[facet_of_query, torch.arange(M, device=device)]  # (M, dk)
        return torch.einsum("md,mnd->mn", Qrouted, Xq)

    real = 0.0
    for step in range(steps):
        S = scores()
        loss, real_t = _rank_loss_and_real(S, A, margin)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        real = float(real_t.detach().cpu())
        if log and (step % max(1, steps // 4) == 0 or step == steps - 1):
            log(f"    {mode} d={d_total} seed={seed} step={step:4d} loss={float(loss):.4f} real={real:.3f}")
    return real
