"""Load a multi-hop QA benchmark (HotpotQA or 2WikiMultihopQA) as a real RAG task for E9.

Each example ships a question, a short gold answer, and ~10 candidate paragraphs (2 supporting +
distractors) with sentence-level structure. We pool every question's paragraphs into ONE shared
retrieval corpus, so retrieval for a question must find its gold paragraphs among all N*~10 -- a
genuine retrieval bottleneck on real data (not a synthetic one). Gold is matched by
supporting-paragraph title (the standard notion). Sentences are kept so the compression stage can
do query-conditioned sentence selection under a token budget. 2WikiMultihopQA is arranged exactly
like HotpotQA (same context.title/sentences + supporting_facts.title schema), so the same pooling
pipeline works for both -- which is what lets E9 test real-QA compounding on a SECOND dataset.
"""
from __future__ import annotations

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
}


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
                  dataset: str = "hotpotqa"):
    """Return (corpus, questions).

    corpus:   pid(str) -> {"title", "text", "sentences"}  (pooled across the sampled questions)
    questions: list of {"qid","question","answer","gold_titles"(set),"own_pids"(list),"type"}
    """
    ds = _load_split(split, dataset)
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
