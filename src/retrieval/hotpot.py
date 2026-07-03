"""Load HotpotQA (distractor) as a real multi-hop RAG task for E9.

Each HotpotQA example ships a question, a short gold answer, and 10 candidate paragraphs
(2 supporting + 8 distractors) with sentence-level structure. We pool every question's
paragraphs into ONE shared retrieval corpus, so retrieval for a question must find its 2 gold
paragraphs among all N*10 -- a genuine retrieval bottleneck on real data (not a synthetic one).
Gold is matched by supporting-paragraph title (the standard HotpotQA notion). Sentences are kept
so the compression stage can do query-conditioned sentence selection under a token budget.
"""
from __future__ import annotations

import numpy as np


def _load_split(split: str):
    from datasets import load_dataset
    try:
        return load_dataset("hotpot_qa", "distractor")[split]
    except Exception:
        # older loader scripts need trust_remote_code
        return load_dataset("hotpot_qa", "distractor", trust_remote_code=True)[split]


def load_hotpotqa(split: str = "validation", n_q: int = 300, seed: int = 0):
    """Return (corpus, questions).

    corpus:   pid(str) -> {"title", "text", "sentences"}  (pooled across the sampled questions)
    questions: list of {"qid","question","answer","gold_titles"(set),"own_pids"(list),"type"}
    """
    ds = _load_split(split)
    n = len(ds)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n_q, n), replace=False)

    corpus: dict[str, dict] = {}
    questions: list[dict] = []
    for j in idx.tolist():
        ex = ds[j]
        ctx = ex["context"]
        titles = ctx["title"]
        sent_lists = ctx["sentences"]
        gold_titles = set(ex["supporting_facts"]["title"])
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
