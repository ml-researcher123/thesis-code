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


# ---- span-answer QA metrics (SQuAD/HotpotQA style) : used by E9 real-QA ----

import re
import string


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (SQuAD normalization)."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    """SQuAD token-level F1 between a predicted and a gold answer string."""
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)  # both empty -> 1, one empty -> 0
    common: dict[str, int] = {}
    for t in pred_toks:
        if t in gold_toks:
            common[t] = min(pred_toks.count(t), gold_toks.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def answer_in_context(answer: str, context: str) -> float:
    """1 if the normalized gold answer is a contiguous token subsequence of the context.

    The 'answer-in-context' diagnostic: does the answer survive into what the reader sees?
    """
    a = normalize_answer(answer)
    c = normalize_answer(context)
    if not a:
        return 0.0
    return float(f" {a} " in f" {c} " or c.startswith(a + " ") or c.endswith(" " + a) or c == a)
