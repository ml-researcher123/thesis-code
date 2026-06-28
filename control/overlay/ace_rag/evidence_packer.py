"""Budget-constrained submodular evidence packing for ACE-RAG.

This module is a *principled* replacement for the heuristic snippet packers in
``experiments/run_qwen_eval.py`` (``_packed_snippet_context_run`` and
``_focused_packed_context_run``). Those packers greedily add snippets by a
hand-tuned lexical score and only check the token budget after the fact. They
never normalize gain by snippet length, so they do not actually optimize
*evidence value per reader token* under the budget.

Here we frame reader-context construction as **budgeted monotone submodular
maximization**. Given retrieved evidence and a hard token budget ``B``, we
select a subset ``S`` of source-grounded snippets that maximizes::

    F(S) = w_rel   * Relevance(S)             # per-snippet query/evidence relevance
         + w_query * QueryTermCoverage(S)     # answer-bearing query-term coverage
         + w_cover * Representativeness(S)     # saturated central-evidence cover
         + w_div   * SourceDiversity(S)        # concave spread across documents

All four terms are monotone and submodular (relevance is modular; the coverage
terms are submodular set-cover functions; diversity is concave-over-groups), and
are normalized to a comparable [0, 1] scale. Relevance is led most strongly so
the packer prefers answer-bearing snippets, with the other terms acting as
coverage/redundancy regularizers. Selection uses **cost-scaled (per-token)
greedy** maximization,
combined with the single-best-feasible-element fallback of Lin & Bilmes (2011),
which gives a constant-factor approximation for budgeted monotone submodular
maximization. This is the standard, citable algorithm; the contribution here is
applying it to *reader-context evidence packing* and tying the objective to the
answer-density quantity measured in :mod:`ace_rag.context_quality`.

The candidate generation deliberately mirrors ``_focused_packed_context_run`` so
that the only thing that changes between the heuristic baseline and this packer
is the *selection objective and algorithm*, not the candidate features. This
makes the packer comparison a clean controlled experiment.

Reference:
    Hui Lin and Jeff Bilmes (2011). "A Class of Submodular Functions for
    Document Summarization." ACL-HLT 2011.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from .schema import CorpusDataset, RetrievalHit, RetrievalRun
from .text import cosine_from_counters, lexical_overlap, split_sentences, tokenize

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "being", "what", "which", "who",
    "whom", "whose", "where", "when", "why", "how", "did", "does", "do", "that",
    "this", "these", "those", "as", "at", "from", "it", "its", "their", "his",
    "her", "he", "she", "they", "them", "same",
}


def _content_terms(text: str) -> set[str]:
    return {tok for tok in tokenize(text) if len(tok) > 2 and tok not in _STOPWORDS}


def _clip_text_tokens(text: str, max_tokens: int) -> str:
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text.strip()
    return " ".join(tokens[:max_tokens]).strip()


def _candidate_doc_ids(run: RetrievalRun) -> list[str]:
    """Replicates run_qwen_eval._candidate_doc_ids so candidates match exactly."""
    candidate_doc_ids: list[str] = []
    for hit in run.hits:
        for doc_id in hit.expanded_doc_ids:
            if doc_id not in candidate_doc_ids:
                candidate_doc_ids.append(doc_id)
        if hit.source_doc_id and hit.source_doc_id not in candidate_doc_ids:
            candidate_doc_ids.append(hit.source_doc_id)
    for doc_id in run.retrieved_doc_ids:
        if doc_id not in candidate_doc_ids:
            candidate_doc_ids.append(doc_id)
    return candidate_doc_ids


@dataclass
class SnippetCandidate:
    doc_id: str
    title: str
    text: str
    tokens: int
    relevance: float
    terms: frozenset[str]
    counter: Counter = field(repr=False)


def build_candidates(
    dataset: CorpusDataset,
    run: RetrievalRun,
    snippet_window: int = 1,
    max_snippet_tokens: int = 80,
    max_candidates: int = 160,
) -> list[SnippetCandidate]:
    """Build windowed sentence snippets with the same features the focused packer uses.

    The per-snippet ``relevance`` is identical to ``_focused_packed_context_run``'s
    ``base_score`` so that the heuristic and submodular packers see the same
    candidate set and the same singleton scores.
    """

    candidate_doc_ids = _candidate_doc_ids(run)
    doc_rank = {doc_id: idx for idx, doc_id in enumerate(candidate_doc_ids)}
    candidates: list[SnippetCandidate] = []
    seen_texts: set[str] = set()

    for doc_id in candidate_doc_ids:
        doc = dataset.documents.get(doc_id)
        if doc is None:
            continue
        sentences = split_sentences(doc.text)
        if not sentences:
            continue
        related_hits = [
            hit for hit in run.hits
            if hit.source_doc_id == doc_id or doc_id in hit.expanded_doc_ids
        ] or run.hits
        for idx, (_sentence, _span) in enumerate(sentences):
            left = max(0, idx - snippet_window)
            right = min(len(sentences), idx + snippet_window + 1)
            snippet = " ".join(sent for sent, _ in sentences[left:right]).strip()
            snippet = _clip_text_tokens(snippet, max_snippet_tokens)
            if not snippet:
                continue
            key = " ".join(snippet.lower().split())
            if key in seen_texts:
                continue
            seen_texts.add(key)
            q_overlap = lexical_overlap(snippet, run.query)
            hit_overlap = max((lexical_overlap(snippet, hit.text) for hit in related_hits), default=0.0)
            title_overlap = lexical_overlap(doc.title, run.query) if doc.title else 0.0
            rank_bonus = 0.05 / (1 + doc_rank.get(doc_id, 0))
            relevance = 1.15 * q_overlap + 0.65 * hit_overlap + 0.25 * title_overlap + rank_bonus
            if relevance <= 0:
                continue
            candidates.append(
                SnippetCandidate(
                    doc_id=doc_id,
                    title=doc.title,
                    text=snippet,
                    tokens=max(1, len(tokenize(snippet))),
                    relevance=relevance,
                    terms=frozenset(_content_terms(snippet)),
                    counter=Counter(tokenize(snippet)),
                )
            )

    # Cap to the most relevant candidates to bound the O(n^2) similarity matrix.
    candidates.sort(key=lambda c: c.relevance, reverse=True)
    return candidates[:max_candidates]


def submodular_select(
    candidates: list[SnippetCandidate],
    query: str,
    token_budget: int,
    max_snippets: int = 8,
    w_rel: float = 1.0,
    w_query: float = 0.5,
    w_cover: float = 0.4,
    w_div: float = 0.3,
    sat_alpha: float = 0.3,
    cost_power: float = 1.0,
) -> list[SnippetCandidate]:
    """Cost-scaled greedy maximization of the submodular packing objective.

    Returns the selected candidates in selection order. Always returns at least
    one candidate when the input is non-empty (the first pick is allowed to
    exceed the budget, matching the heuristic packers' behaviour).
    """

    n = len(candidates)
    if n == 0:
        return []

    query_terms = frozenset(_content_terms(query))
    qsize = max(1, len(query_terms))

    # Pairwise lexical-cosine similarity (symmetric, unit diagonal).
    sims: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        ci = candidates[i].counter
        sims[i][i] = 1.0
        for j in range(i + 1, n):
            s = cosine_from_counters(ci, candidates[j].counter)
            sims[i][j] = s
            sims[j][i] = s
    deg = [sum(sims[i]) for i in range(n)]
    caps = [sat_alpha * deg[i] for i in range(n)]
    caps_total = sum(caps) or 1.0
    total_rel = sum(c.relevance for c in candidates) or 1.0
    rel_norm = math.sqrt(total_rel)

    selected: list[int] = []
    selected_set: set[int] = set()
    used = 0
    cov_partial = [0.0] * n          # sum_{j in S} sim(i, j)
    group_sum: dict[str, float] = {}  # doc_id -> sum relevance in S
    covered_q: set[str] = set()

    def marginal_gain(idx: int) -> float:
        cand = candidates[idx]
        row = sims[idx]
        cov_gain = 0.0
        for i in range(n):
            cap = caps[i]
            old = cov_partial[i] if cov_partial[i] < cap else cap
            cand_new = cov_partial[i] + row[i]
            new = cand_new if cand_new < cap else cap
            cov_gain += new - old
        cov_gain /= caps_total
        gs = group_sum.get(cand.doc_id, 0.0)
        div_gain = (math.sqrt(gs + cand.relevance) - math.sqrt(gs)) / rel_norm
        new_q = (cand.terms & query_terms) - covered_q
        q_gain = len(new_q) / qsize
        rel_gain = cand.relevance / total_rel
        return w_rel * rel_gain + w_query * q_gain + w_cover * cov_gain + w_div * div_gain

    def commit(idx: int) -> None:
        nonlocal used
        cand = candidates[idx]
        row = sims[idx]
        for i in range(n):
            cov_partial[i] += row[i]
        group_sum[cand.doc_id] = group_sum.get(cand.doc_id, 0.0) + cand.relevance
        covered_q.update(cand.terms & query_terms)
        selected.append(idx)
        selected_set.add(idx)
        used += cand.tokens

    while len(selected) < max_snippets:
        best_idx = -1
        best_ratio = 0.0
        for idx in range(n):
            if idx in selected_set:
                continue
            cand = candidates[idx]
            if selected and used + cand.tokens > token_budget:
                continue
            gain = marginal_gain(idx)
            if gain <= 0.0:
                continue
            ratio = gain / (cand.tokens ** cost_power)
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = idx
        if best_idx < 0:
            break
        commit(best_idx)
        if used >= token_budget:
            break

    # Lin & Bilmes single-element fallback: if the best single feasible snippet
    # scores higher than the greedy set, return it instead. This is what makes
    # the cost-scaled greedy a constant-factor approximation under the budget.
    greedy_obj = _objective_value(selected, candidates, sims, caps, caps_total, total_rel, rel_norm, query_terms, qsize, w_rel, w_query, w_cover, w_div)
    best_single = -1
    best_single_obj = greedy_obj
    for idx in range(n):
        if selected and candidates[idx].tokens > token_budget:
            continue
        obj = _objective_value([idx], candidates, sims, caps, caps_total, total_rel, rel_norm, query_terms, qsize, w_rel, w_query, w_cover, w_div)
        if obj > best_single_obj:
            best_single_obj = obj
            best_single = idx
    if best_single >= 0:
        return [candidates[best_single]]
    return [candidates[i] for i in selected]


def _objective_value(
    idxs: list[int],
    candidates: list[SnippetCandidate],
    sims: list[list[float]],
    caps: list[float],
    caps_total: float,
    total_rel: float,
    rel_norm: float,
    query_terms: frozenset[str],
    qsize: int,
    w_rel: float,
    w_query: float,
    w_cover: float,
    w_div: float,
) -> float:
    if not idxs:
        return 0.0
    n = len(candidates)
    cov = 0.0
    for i in range(n):
        s = sum(sims[i][j] for j in idxs)
        cov += min(s, caps[i])
    cov /= caps_total
    group_sum: dict[str, float] = {}
    covered_q: set[str] = set()
    rel = 0.0
    for j in idxs:
        cand = candidates[j]
        rel += cand.relevance
        group_sum[cand.doc_id] = group_sum.get(cand.doc_id, 0.0) + cand.relevance
        covered_q.update(cand.terms & query_terms)
    rel /= total_rel
    div = sum(math.sqrt(v) for v in group_sum.values()) / rel_norm
    q = len(covered_q & query_terms) / qsize
    return w_rel * rel + w_query * q + w_cover * cov + w_div * div


def mmr_select(
    candidates: list[SnippetCandidate],
    token_budget: int,
    max_snippets: int = 8,
    lam: float = 0.7,
) -> list[SnippetCandidate]:
    """Maximal Marginal Relevance selection (Carbonell & Goldstein 1998).

    The canonical redundancy-aware reranker and the natural baseline for the
    submodular packer: at each step pick the candidate maximizing
    ``lam * rel(i) - (1 - lam) * max_{j in S} sim(i, j)`` under the token budget.
    Relevance is min-max normalized so it is comparable to the [0, 1] cosine
    redundancy term. Candidates are identical to those the submodular and focused
    packers see, so MMR-vs-submod isolates the *selection rule*.
    """

    n = len(candidates)
    if n == 0:
        return []
    rels = [c.relevance for c in candidates]
    lo, hi = min(rels), max(rels)
    span = (hi - lo) or 1.0
    norm_rel = [(r - lo) / span for r in rels]

    sim_cache: dict[tuple[int, int], float] = {}

    def sim(i: int, j: int) -> float:
        key = (i, j) if i < j else (j, i)
        cached = sim_cache.get(key)
        if cached is None:
            cached = cosine_from_counters(candidates[i].counter, candidates[j].counter)
            sim_cache[key] = cached
        return cached

    selected: list[int] = []
    selected_set: set[int] = set()
    used = 0
    while len(selected) < max_snippets:
        best_idx = -1
        best_score = -1e18
        for i in range(n):
            if i in selected_set:
                continue
            if selected and used + candidates[i].tokens > token_budget:
                continue
            redundancy = max((sim(i, j) for j in selected_set), default=0.0)
            score = lam * norm_rel[i] - (1.0 - lam) * redundancy
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            break
        selected.append(best_idx)
        selected_set.add(best_idx)
        used += candidates[best_idx].tokens
        if used >= token_budget:
            break
    return [candidates[i] for i in selected]


def _hits_from_candidates(chosen: list[SnippetCandidate], tag: str) -> list[RetrievalHit]:
    hits: list[RetrievalHit] = []
    for i, cand in enumerate(chosen):
        hits.append(
            RetrievalHit(
                node_id=f"reader_{tag}::{cand.doc_id}::{i}",
                node_type=f"{tag}_snippet",
                text=cand.text,
                score=float(cand.relevance),
                source_doc_id=cand.doc_id,
                expanded_doc_ids=[cand.doc_id],
                metadata={"title": cand.title},
            )
        )
    return hits


def pack_mmr_run(
    dataset: CorpusDataset,
    run: RetrievalRun,
    snippet_window: int = 1,
    max_snippets: int = 8,
    max_snippet_tokens: int = 80,
    token_budget: int = 160,
    mmr_lambda: float = 0.7,
    max_candidates: int = 160,
) -> RetrievalRun:
    """Materialize a reader context for ``run`` using MMR evidence packing."""

    candidates = build_candidates(
        dataset,
        run,
        snippet_window=snippet_window,
        max_snippet_tokens=max_snippet_tokens,
        max_candidates=max_candidates,
    )
    chosen = mmr_select(candidates, token_budget=token_budget, max_snippets=max_snippets, lam=mmr_lambda)
    hits = _hits_from_candidates(chosen, "mmr")
    return RetrievalRun(
        qid=run.qid,
        query=run.query,
        hits=hits,
        retrieved_doc_ids=list(dict.fromkeys(h.source_doc_id or "" for h in hits if h.source_doc_id)),
        diagnostics={
            **run.diagnostics,
            "reader_context": "mmr_packed",
            "packed_token_budget": token_budget,
            "mmr_lambda": mmr_lambda,
        },
    )


def pack_submodular_run(
    dataset: CorpusDataset,
    run: RetrievalRun,
    snippet_window: int = 1,
    max_snippets: int = 8,
    max_snippet_tokens: int = 80,
    token_budget: int = 160,
    w_rel: float = 1.0,
    w_query: float = 0.5,
    w_cover: float = 0.4,
    w_div: float = 0.3,
    sat_alpha: float = 0.3,
    cost_power: float = 1.0,
    max_candidates: int = 160,
) -> RetrievalRun:
    """Materialize a reader context for ``run`` using submodular evidence packing."""

    candidates = build_candidates(
        dataset,
        run,
        snippet_window=snippet_window,
        max_snippet_tokens=max_snippet_tokens,
        max_candidates=max_candidates,
    )
    chosen = submodular_select(
        candidates,
        run.query,
        token_budget=token_budget,
        max_snippets=max_snippets,
        w_rel=w_rel,
        w_query=w_query,
        w_cover=w_cover,
        w_div=w_div,
        sat_alpha=sat_alpha,
        cost_power=cost_power,
    )
    hits = _hits_from_candidates(chosen, "submod")
    return RetrievalRun(
        qid=run.qid,
        query=run.query,
        hits=hits,
        retrieved_doc_ids=list(dict.fromkeys(h.source_doc_id or "" for h in hits if h.source_doc_id)),
        diagnostics={
            **run.diagnostics,
            "reader_context": "submodular_packed",
            "packed_token_budget": token_budget,
        },
    )
