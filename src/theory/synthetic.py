"""Synthetic relevance-pattern generators with controllable combinatorial complexity.

A relevance pattern is represented compactly by ``pos_idx`` of shape (n_q, k):
each of the ``n_q`` queries is relevant to exactly ``k`` of the ``n_d`` documents.
Keeping ``k`` fixed per pattern makes the downstream capacity optimizer fully
vectorizable.

The "all_pairs" pattern (k=2, every document pair is some query's gold set) is the
clean adversarial construction whose realizable dimension provably grows with the
corpus size -- this is what exposes the embedding-dimension wall.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RelevancePattern:
    pos_idx: torch.Tensor  # (n_q, k) long
    n_d: int
    k: int
    name: str

    @property
    def n_q(self) -> int:
        return int(self.pos_idx.shape[0])


def _subset_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def all_pairs(n_d: int, max_queries: int | None = None, seed: int = 0) -> RelevancePattern:
    """Every 2-subset of documents is the gold set of one query (k=2).

    The number of distinct 2-subsets is C(n_d, 2); if ``max_queries`` is set and
    smaller, a random subset of pairs is sampled (without replacement).
    """
    pairs = list(itertools.combinations(range(n_d), 2))
    if max_queries is not None and len(pairs) > max_queries:
        rng = _subset_rng(seed)
        idx = rng.choice(len(pairs), size=max_queries, replace=False)
        pairs = [pairs[i] for i in idx]
    pos = torch.tensor(pairs, dtype=torch.long)
    return RelevancePattern(pos_idx=pos, n_d=n_d, k=2, name=f"all_pairs(n_d={n_d})")


def all_ksubsets(n_d: int, k: int, max_queries: int | None = None, seed: int = 0) -> RelevancePattern:
    """Every k-subset is a query's gold set (sampled if there are too many)."""
    combos = itertools.combinations(range(n_d), k)
    if max_queries is None:
        sel = list(combos)
    else:
        # Reservoir-free: count then sample indices to avoid materializing huge lists.
        total = 1
        for i in range(k):
            total = total * (n_d - i) // (i + 1)
        if total <= max_queries:
            sel = list(combos)
        else:
            rng = _subset_rng(seed)
            wanted = set(rng.choice(total, size=max_queries, replace=False).tolist())
            sel = [c for i, c in enumerate(combos) if i in wanted]
    pos = torch.tensor(sel, dtype=torch.long)
    return RelevancePattern(pos_idx=pos, n_d=n_d, k=k, name=f"all_{k}subsets(n_d={n_d})")


def random_ksubsets(n_d: int, k: int, n_q: int, seed: int = 0) -> RelevancePattern:
    """``n_q`` random distinct k-subsets. Complexity is controlled by ``n_q``."""
    rng = _subset_rng(seed)
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    guard = 0
    while len(out) < n_q and guard < 50 * n_q:
        guard += 1
        s = tuple(sorted(rng.choice(n_d, size=k, replace=False).tolist()))
        if s not in seen:
            seen.add(s)
            out.append(s)
    pos = torch.tensor(out, dtype=torch.long)
    return RelevancePattern(pos_idx=pos, n_d=n_d, k=k, name=f"random_{k}subsets(n_d={n_d},n_q={len(out)})")


def build_pattern(kind: str, **kw) -> RelevancePattern:
    """Dispatch by name for config-driven construction."""
    if kind == "all_pairs":
        return all_pairs(kw["n_d"], kw.get("max_queries"), kw.get("seed", 0))
    if kind == "all_ksubsets":
        return all_ksubsets(kw["n_d"], kw["k"], kw.get("max_queries"), kw.get("seed", 0))
    if kind == "random_ksubsets":
        return random_ksubsets(kw["n_d"], kw["k"], kw["n_q"], kw.get("seed", 0))
    raise ValueError(f"unknown pattern kind: {kind}")
