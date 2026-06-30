"""Free-embedding capacity estimator.

Given a target relevance pattern, we directly optimize the document and query
vectors (no encoder, no text -- the embeddings are free parameters) to realize the
pattern under inner-product top-k retrieval. Because the vectors are unconstrained,
this measures the *best case* for any embedding model of a given dimension: if even
free vectors of dimension ``d`` cannot realize the pattern, then no real encoder can.
This is the methodology behind Weller et al.'s retrieval wall, reimplemented here as
a reusable measurement primitive.

Key outputs per (pattern, dim, seed):
- ``realizability``: fraction of queries whose every gold doc outranks every non-gold
  doc (i.e., the pattern is perfectly realized for that query).
- ``recall_at_k``: mean recall@k (k = gold-set size) -- a softer, standard metric.
- ``final_loss``: the residual pairwise loss (0 => perfectly separable).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from ..common import set_seed
from .synthetic import RelevancePattern


def complement_indices(pos_idx: torch.Tensor, n_d: int) -> torch.Tensor:
    """Return (n_q, n_d-k) negative (non-gold) indices for each query.

    Relies on every row having exactly k positives, so the boolean complement has a
    constant per-row count and reshapes cleanly.
    """
    n_q, k = pos_idx.shape
    # build the mask on pos_idx's device so scatter_ with a CUDA index works on GPU
    mask = torch.ones(n_q, n_d, dtype=torch.bool, device=pos_idx.device)
    mask.scatter_(1, pos_idx, False)
    neg = mask.nonzero(as_tuple=False)[:, 1].reshape(n_q, n_d - k)
    return neg


@dataclass
class CapacityResult:
    dim: int
    n_d: int
    n_q: int
    k: int
    seed: int
    realizability: float
    recall_at_k: float
    final_loss: float
    steps: int


def fit_capacity(
    pattern: RelevancePattern,
    dim: int,
    *,
    steps: int = 600,
    lr: float = 0.05,
    margin: float = 1.0,
    init_scale: float = 0.1,
    seed: int = 0,
    device: torch.device | str = "cpu",
    log=None,
) -> CapacityResult:
    """Optimize free doc/query embeddings of size ``dim`` to realize ``pattern``."""
    set_seed(seed)
    device = torch.device(device)
    pos_idx = pattern.pos_idx.to(device)
    n_d, k = pattern.n_d, pattern.k
    n_q = pattern.n_q
    neg_idx = complement_indices(pos_idx, n_d).to(device)

    X = torch.nn.Parameter(torch.randn(n_d, dim, device=device) * init_scale)
    Y = torch.nn.Parameter(torch.randn(n_q, dim, device=device) * init_scale)
    opt = torch.optim.Adam([X, Y], lr=lr)

    final_loss = float("nan")
    for step in range(steps):
        scores = Y @ X.t()                       # (n_q, n_d)
        pos = scores.gather(1, pos_idx)          # (n_q, k)
        neg = scores.gather(1, neg_idx)          # (n_q, n_d-k)
        # Every positive should beat every negative by `margin`.
        diff = pos.unsqueeze(2) - neg.unsqueeze(1)   # (n_q, k, n_d-k)
        loss = F.softplus(margin - diff).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu())
        if log and (step % max(1, steps // 4) == 0 or step == steps - 1):
            log(f"    dim={dim} seed={seed} step={step:4d} loss={final_loss:.4f}")

    with torch.no_grad():
        scores = Y @ X.t()
        pos = scores.gather(1, pos_idx)
        neg = scores.gather(1, neg_idx)
        realized = (pos.min(dim=1).values > neg.max(dim=1).values)
        realizability = float(realized.float().mean().cpu())
        topk = scores.topk(k, dim=1).indices                       # (n_q, k)
        hit = (pos_idx.unsqueeze(2) == topk.unsqueeze(1)).any(dim=2)  # (n_q, k)
        recall_at_k = float(hit.float().mean().cpu())

    return CapacityResult(
        dim=dim, n_d=n_d, n_q=n_q, k=k, seed=seed,
        realizability=realizability, recall_at_k=recall_at_k,
        final_loss=final_loss, steps=steps,
    )


def result_to_dict(r: CapacityResult) -> dict:
    return asdict(r)
