"""Reader-context evidence-density diagnostics.

These metrics measure the quantity that the budget-constrained RAG thesis cares
about but that document ``recall@k`` does *not* capture: how much of the reader's
limited context is actually answer-bearing, and whether the answer is even
present in the context the reader sees.

They operate on a *materialized reader context* (a :class:`RetrievalRun` whose
hits are the packed snippets shown to the reader), not on the upstream retrieval
set. This is the key distinction from ``ace_rag.metrics.evaluate_retrieval``,
which scores the retrieved document ids before packing.

Definitions (all per-question, averaged for a policy):

``ans_in_context``
    1.0 if a gold answer string appears as a contiguous token sub-sequence of
    the packed context. This is the necessary condition for an extractive
    reader to be correct, and the mediator we expect to explain why higher
    document recall does not imply higher answer quality under a budget.

``gold_token_density``
    Fraction of packed-context tokens that come from gold documents. The
    "signal-to-noise" of the reader context: distractor tokens crowd out the
    answer under a fixed budget.

``gold_doc_reader_cov`` / ``all_gold_reader``
    Fraction of gold documents that contribute >=1 snippet to the reader
    context, and whether *all* of them do. This is reader-level coverage, which
    can differ sharply from retrieval recall once packing throws evidence away.

``distractor_docs``
    Number of distinct non-gold documents contributing snippets.
"""

from __future__ import annotations

from typing import Any

from .schema import Question, RetrievalRun
from .text import normalize_answer, tokenize


def _context_tokens(run: RetrievalRun) -> int:
    return sum(len(tokenize(hit.text)) for hit in run.hits)


def _seq_contains(haystack: list[str], needle: list[str]) -> bool:
    n = len(needle)
    if n == 0:
        return False
    limit = len(haystack) - n
    for i in range(limit + 1):
        if haystack[i : i + n] == needle:
            return True
    return False


def answer_in_context(run: RetrievalRun, answers: list[str]) -> float:
    """1.0 if any normalized gold answer is a contiguous token run in the context."""
    context_tokens = normalize_answer(" ".join(hit.text for hit in run.hits)).split()
    if not context_tokens:
        return 0.0
    for answer in answers:
        answer_tokens = normalize_answer(answer).split()
        if answer_tokens and _seq_contains(context_tokens, answer_tokens):
            return 1.0
    return 0.0


def gold_token_density(run: RetrievalRun, gold_doc_ids: set[str]) -> float:
    if not gold_doc_ids:
        return 0.0
    total = 0
    gold = 0
    for hit in run.hits:
        n = len(tokenize(hit.text))
        total += n
        if hit.source_doc_id in gold_doc_ids:
            gold += n
    return gold / total if total else 0.0


def gold_doc_reader_coverage(run: RetrievalRun, gold_doc_ids: set[str]) -> tuple[float, float]:
    if not gold_doc_ids:
        return 0.0, 0.0
    docs = {hit.source_doc_id for hit in run.hits if hit.source_doc_id}
    covered = len(docs & gold_doc_ids)
    all_covered = float(gold_doc_ids.issubset(docs))
    return covered / len(gold_doc_ids), all_covered


def context_quality(run: RetrievalRun, question: Question) -> dict[str, float]:
    cov, all_cov = gold_doc_reader_coverage(run, question.gold_doc_ids)
    docs = {hit.source_doc_id for hit in run.hits if hit.source_doc_id}
    return {
        "ans_in_context": answer_in_context(run, question.answers),
        "gold_token_density": gold_token_density(run, question.gold_doc_ids),
        "gold_doc_reader_cov": cov,
        "all_gold_reader": all_cov,
        "distractor_docs": float(len(docs - question.gold_doc_ids)),
        "context_tokens": float(_context_tokens(run)),
        "n_snippets": float(len(run.hits)),
    }


_KEYS = (
    "ans_in_context",
    "gold_token_density",
    "gold_doc_reader_cov",
    "all_gold_reader",
    "distractor_docs",
    "context_tokens",
    "n_snippets",
)


def aggregate_context_quality(
    runs: list[RetrievalRun], questions: list[Question]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Return (policy-level means with ``mean_`` prefix, per-question records)."""
    by_qid = {q.qid: q for q in questions}
    totals = {key: 0.0 for key in _KEYS}
    per_question: list[dict[str, Any]] = []
    n = 0
    for run in runs:
        question = by_qid.get(run.qid)
        if question is None:
            continue
        cq = context_quality(run, question)
        for key in _KEYS:
            totals[key] += cq[key]
        n += 1
        per_question.append({"qid": run.qid, **cq})
    denom = max(1, n)
    aggregate = {f"mean_{key}": round(totals[key] / denom, 4) for key in _KEYS}
    return aggregate, per_question
