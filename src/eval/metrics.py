"""Retrieval / ranking metrics used across experiments.

Kept dependency-light (numpy/torch) so it runs on the Kaggle CPU/GPU image without
extra installs. Retrieval-from-text and answer-EM/F1 metrics will be added when the
compression and pipeline stages land (E2+).
"""
from __future__ import annotations

import numpy as np


def recall_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    if not gold:
        return 0.0
    topk = set(ranked[:k])
    return len(topk & gold) / len(gold)


def dcg(rels: list[float]) -> float:
    return float(sum(r / np.log2(i + 2) for i, r in enumerate(rels)))


def ndcg_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    gains = [1.0 if d in gold else 0.0 for d in ranked[:k]]
    ideal = [1.0] * min(len(gold), k)
    denom = dcg(ideal)
    return dcg(gains) / denom if denom > 0 else 0.0


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)
