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

``ans_in_context_soft``
    Graded (0-1) variant of ``ans_in_context``: the fraction of gold-answer
    tokens present in the packed context, maximized over gold answers. The
    verbatim-span check is binary and degenerate for long free-form answers
    (RAGBench ExpertQA), where the answer never appears as a contiguous run.
    This soft variant is free to compute and stays informative there, so the
    diagnostic extends past extractive QA.

For a stronger semantic variant (entailment rather than lexical overlap) see
:func:`score_nli_entailment`, which is opt-in because it loads a cross-encoder.
"""

from __future__ import annotations

import re
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


def answer_in_context_soft(run: RetrievalRun, answers: list[str]) -> float:
    """Best *local* gold-answer token recall in the context (max over answers).

    Graded counterpart to :func:`answer_in_context`. Recall is measured inside a
    sliding window rather than over the whole context, which matters: a plain
    bag-of-words recall scores "New Delhi is large. York is a city." a perfect
    1.0 for the answer "New York City" because every token appears *somewhere*.
    Requiring the tokens to co-occur within ``2 * len(answer)`` tokens
    approximates the adjacency the binary check demands without insisting on an
    exact contiguous run, so scattered tokens no longer game the metric.

    Equals 1.0 whenever the verbatim check fires (a contiguous run always fits
    inside its own 2x window), so the two agree by construction on short spans;
    unlike the binary check it stays informative when the answer is long and
    free-form (RAGBench ExpertQA), where no contiguous run ever occurs.
    """
    context_tokens = normalize_answer(" ".join(hit.text for hit in run.hits)).split()
    if not context_tokens:
        return 0.0
    best = 0.0
    for answer in answers:
        answer_tokens = normalize_answer(answer).split()
        if not answer_tokens:
            continue
        wanted = set(answer_tokens)
        window = max(2 * len(answer_tokens), 5)
        if window >= len(context_tokens):
            spans = [context_tokens]
        else:
            spans = [context_tokens[i : i + window] for i in range(len(context_tokens) - window + 1)]
        for span in spans:
            present = len(wanted & set(span))
            best = max(best, present / len(wanted))
            if best >= 1.0:
                return 1.0
    return best


def context_text(run: RetrievalRun) -> str:
    """The packed context exactly as the reader sees it (NLI premise)."""
    return "\n".join(hit.text for hit in run.hits)


def split_sentences(text: str, max_sentences: int = 40) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    return parts[:max_sentences]


def score_nli_entailment(
    pairs: list[tuple[str, str]],
    model_name: str = "cross-encoder/nli-deberta-v3-small",
    batch_size: int = 32,
    device: str = "cuda",
) -> list[float]:
    """Max entailment probability for each (premise, hypothesis) pair.

    Used for the semantic ℓanswer-in-context variant: premise is the packed
    reader context, hypothesis is the gold answer rendered as a claim. Loaded
    lazily and only when requested, so the default pipeline stays model-free.
    Returns 0.0 for every pair if the model cannot be loaded (kept non-fatal so
    a long Kaggle run never dies on an optional diagnostic).
    """
    if not pairs:
        return []
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment guard
        print(f"[nli] transformers unavailable ({exc!r}); returning zeros", flush=True)
        return [0.0] * len(pairs)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        if device.startswith("cuda") and torch.cuda.is_available():
            model.to(device)
        else:
            device = "cpu"
    except Exception as exc:  # pragma: no cover - network/gating guard
        print(f"[nli] could not load {model_name} ({exc!r}); returning zeros", flush=True)
        return [0.0] * len(pairs)

    # Locate the entailment logit by label name; MNLI heads are not ordered
    # consistently across checkpoints, so never hardcode the index.
    label_map = {str(v).lower(): int(k) for k, v in (model.config.id2label or {}).items()}
    entail_idx = next((i for name, i in label_map.items() if "entail" in name), model.config.num_labels - 1)

    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = tokenizer(
                [premise for premise, _ in batch],
                [hypothesis for _, hypothesis in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[:, entail_idx]
            scores.extend(probs.detach().float().cpu().tolist())
            print(f"[nli] scored {min(start + len(batch), len(pairs))}/{len(pairs)}", flush=True)

    del model
    try:
        import torch as _torch

        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception:
        pass
    return scores


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
        "ans_in_context_soft": round(answer_in_context_soft(run, question.answers), 4),
        "gold_token_density": gold_token_density(run, question.gold_doc_ids),
        "gold_doc_reader_cov": cov,
        "all_gold_reader": all_cov,
        "distractor_docs": float(len(docs - question.gold_doc_ids)),
        "context_tokens": float(_context_tokens(run)),
        "n_snippets": float(len(run.hits)),
    }


_KEYS = (
    "ans_in_context",
    "ans_in_context_soft",
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
