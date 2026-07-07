"""Load a knowledge-intensive QA benchmark as a real RAG task for E9.

Multi-hop (HotpotQA, 2WikiMultihopQA): each example ships a question, a short gold answer, and ~10
candidate paragraphs (2 supporting + distractors) with sentence-level structure. Single-hop (SQuAD):
each example ships a question, one gold Wikipedia paragraph, and a short answer span. In both cases
we pool every question's paragraph(s) into ONE shared retrieval corpus, so retrieval must find its
gold among the whole pool -- a genuine retrieval bottleneck on real data (not a synthetic one).
Sentences are kept so the compression stage can do query-conditioned selection under a token budget.
Multi-hop matches gold by supporting-paragraph title; single-hop SQuAD has paragraph-specific gold
(many paragraphs share an article title), so we assign each paragraph a UNIQUE title and match on
that. This lets E9 test whether the compounding + interior optimum are a multi-hop artifact.
"""
from __future__ import annotations

import re

import numpy as np

# dataset key -> list of (repo_id, config, kwargs) attempts, newest-compatible first.
_DATASET_SOURCES = {
    "hotpotqa": [
        ("hotpotqa/hotpot_qa", "distractor", {}),
        ("hotpot_qa", "distractor", {}),
        ("hotpot_qa", "distractor", {"trust_remote_code": True}),
    ],
    "2wiki": [
        ("xanhho/2WikiMultihopQA", None, {}),
        ("framolfese/2WikiMultihopQA", None, {}),
    ],
    "squad": [
        ("rajpurkar/squad", None, {}),
        ("squad", None, {}),
        ("squad", None, {"trust_remote_code": True}),
    ],
}


def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence split (SQuAD contexts are plain paragraphs, no sentence structure)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s and s.strip()]


def _build_squad(ds, n_q: int, seed: int, max_paragraphs: int):
    """SQuAD (single-hop): dedupe contexts into a pooled corpus with paragraph-unique titles.

    Each question's gold is exactly ONE paragraph (the one it was written from), so retrieval is a
    clean single-gold problem among the whole pool -- the single-hop analogue of the multi-hop
    distractor setting.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds)).tolist()

    # Phase 1: build the pooled corpus of unique paragraphs (up to max_paragraphs), so retrieval
    # is among a large real pool -- comparable to the multi-hop setting -- not just the questions'
    # own golds. Phase 2: collect n_q questions whose gold paragraph is in that corpus.
    corpus: dict[str, dict] = {}
    ctx_to_pid: dict[str, str] = {}
    for j in order:
        if len(corpus) >= max_paragraphs:
            break
        ctx = ds[j]["context"]
        if ctx in ctx_to_pid:
            continue
        idx = len(corpus)
        pid = f"sq{idx}"
        # paragraph-unique title so single-hop gold matches at PARAGRAPH granularity
        title = f"{ds[j].get('title', 'doc')} [p{idx}]"
        corpus[pid] = {"title": title, "text": ctx.strip(), "sentences": _split_sentences(ctx)}
        ctx_to_pid[ctx] = pid

    questions: list[dict] = []
    for j in order:
        if len(questions) >= n_q:
            break
        ex = ds[j]
        pid = ctx_to_pid.get(ex["context"])
        if pid is None:
            continue  # gold paragraph didn't make the corpus cap
        answers = ex["answers"]["text"] if isinstance(ex["answers"], dict) else ex["answers"]
        if not answers:
            continue
        questions.append({
            "qid": str(ex["id"]),
            "question": ex["question"],
            "answer": answers[0],
            "gold_titles": {corpus[pid]["title"]},
            "own_pids": [pid],
            "type": "single-hop",
        })
    return corpus, questions


def _load_split(split: str, dataset: str = "hotpotqa"):
    from datasets import load_dataset
    # Newer `datasets` releases require a canonical "namespace/name" repo id and reject bare legacy
    # names (HfUriError), so try the namespaced ids first; a couple of mirrors give resilience.
    errors = []
    for repo_id, config, kwargs in _DATASET_SOURCES[dataset]:
        try:
            ds = load_dataset(repo_id, config, **kwargs) if config else load_dataset(repo_id, **kwargs)
            return ds[split] if split in ds else ds[list(ds.keys())[0]]
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{repo_id}/{config} {kwargs}: {exc}")
    raise RuntimeError(f"all {dataset} load attempts failed:\n" + "\n".join(errors))


def load_hotpotqa(split: str = "validation", n_q: int = 300, seed: int = 0,
                  dataset: str = "hotpotqa", max_paragraphs: int = 1500):
    """Return (corpus, questions).

    corpus:   pid(str) -> {"title", "text", "sentences"}  (pooled across the sampled questions)
    questions: list of {"qid","question","answer","gold_titles"(set),"own_pids"(list),"type"}
    """
    ds = _load_split(split, dataset)
    if dataset == "squad":
        return _build_squad(ds, n_q, seed, max_paragraphs)
    n = len(ds)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n_q, n), replace=False)

    corpus: dict[str, dict] = {}
    questions: list[dict] = []
    for j in idx.tolist():
        ex = ds[j]
        ctx = ex["context"]
        # both HotpotQA and 2Wiki use {title:[...], sentences:[[...]]}; some 2Wiki mirrors ship
        # context as a list of [title, [sentences]] pairs instead -- handle both shapes.
        if isinstance(ctx, dict):
            titles = ctx["title"]; sent_lists = ctx["sentences"]
        else:
            titles = [c[0] for c in ctx]; sent_lists = [c[1] for c in ctx]
        sf = ex["supporting_facts"]
        gold_titles = set(sf["title"] if isinstance(sf, dict) else [s[0] for s in sf])
        own_pids = []
        for pi, (title, sents) in enumerate(zip(titles, sent_lists)):
            pid = f"{ex['id']}__{pi}"
            sents = [s.strip() for s in sents if s and s.strip()]
            text = (title + ". " + " ".join(sents)).strip()
            corpus[pid] = {"title": title, "text": text, "sentences": sents}
            own_pids.append(pid)
        questions.append({
            "qid": str(ex["id"]),
            "question": ex["question"],
            "answer": ex["answer"],
            "gold_titles": gold_titles,
            "own_pids": own_pids,
            "type": ex.get("type", ""),
        })
    return corpus, questions
