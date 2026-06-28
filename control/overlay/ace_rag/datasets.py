"""Dataset loaders for ACE-RAG.

The toy loader is fully offline. External loaders import `datasets` lazily so
the package can still run smoke tests without network access or heavy installs.
"""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path
from typing import Any

from .schema import CorpusDataset, Document, Question


def load_toy() -> CorpusDataset:
    documents = {
        "curie_bio": Document(
            "curie_bio",
            "Marie Curie was a physicist and chemist. She won the Nobel Prize in Physics in 1903. "
            "Marie Curie won the Nobel Prize in Chemistry in 1911.",
            "Marie Curie",
        ),
        "curie_family": Document(
            "curie_family",
            "Irene Joliot-Curie was the daughter of Marie Curie. Irene Joliot-Curie won the Nobel Prize "
            "in Chemistry in 1935.",
            "Irene Joliot-Curie",
        ),
        "paris": Document(
            "paris",
            "The Eiffel Tower is located in Paris. Paris is the capital city of France.",
            "Paris",
        ),
        "france": Document(
            "france",
            "France is a country in Western Europe. Its capital is Paris.",
            "France",
        ),
        "conflict_a": Document(
            "conflict_a",
            "The fictional Acme wind sensor reports gusts above 45 knots during storm alerts.",
            "Acme Sensor A",
        ),
        "conflict_b": Document(
            "conflict_b",
            "The fictional Acme wind sensor does not report gusts above 45 knots during storm alerts.",
            "Acme Sensor B",
        ),
    }
    questions = [
        Question(
            "toy_q1",
            "Which city contains the Eiffel Tower and is the capital of France?",
            ["Paris"],
            {"paris", "france"},
        ),
        Question(
            "toy_q2",
            "In what year did Marie Curie win the Nobel Prize in Chemistry?",
            ["1911"],
            {"curie_bio"},
        ),
        Question(
            "toy_q3",
            "Who was Marie Curie's daughter that won a Nobel Prize in Chemistry?",
            ["Irene Joliot-Curie"],
            {"curie_family", "curie_bio"},
        ),
        Question(
            "toy_q4",
            "Does the fictional Acme wind sensor report gusts above 45 knots during storm alerts?",
            ["conflicting evidence"],
            {"conflict_a", "conflict_b"},
        ),
    ]
    return CorpusDataset("toy", documents, questions)


def load_hotpotqa(split: str = "validation", limit: int = 300, seed: int = 0) -> CorpusDataset:
    try:
        from datasets import load_dataset

        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
        if seed:
            ds = ds.shuffle(seed=seed)
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        rows = list(ds)
        source = "hf"
    except Exception as exc:
        print(f"[hotpotqa] Hugging Face loader failed; using raw JSON fallback: {exc}", flush=True)
        rows = _load_hotpotqa_raw_rows(split)
        if seed:
            random.Random(seed).shuffle(rows)
        if limit:
            rows = rows[:limit]
        source = "raw"

    documents: dict[str, Document] = {}
    questions: list[Question] = []
    for ex in rows:
        titles, sentence_lists = _hotpot_context(ex)
        for title, sentences in zip(titles, sentence_lists):
            doc_id = str(title)
            documents[doc_id] = Document(
                doc_id=doc_id,
                title=str(title),
                text=" ".join(s.strip() for s in sentences).strip(),
                metadata={"dataset": "hotpotqa"},
            )
        gold = _hotpot_supporting_titles(ex)
        questions.append(
            Question(
                qid=str(ex.get("id", ex.get("_id", len(questions)))),
                text=str(ex["question"]),
                answers=[str(ex["answer"])],
                gold_doc_ids=gold,
            )
        )
    return CorpusDataset(f"hotpotqa[{source}:{split}:{len(questions)}]", documents, questions)


def _load_hotpotqa_raw_rows(split: str) -> list[dict[str, Any]]:
    split_name = split.lower()
    if split_name in {"validation", "dev", "dev_distractor"}:
        filename = "hotpot_dev_distractor_v1.json"
        url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
    elif split_name in {"train", "training"}:
        filename = "hotpot_train_v1.1.json"
        url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
    else:
        raise ValueError(f"unsupported HotpotQA split for raw fallback: {split}")

    cache_dir = Path.home() / ".cache" / "ace_rag"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / filename
    if not path.exists():
        print(f"[hotpotqa] downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    return json.loads(path.read_text(encoding="utf-8"))


def _hotpot_context(ex: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    context = ex["context"]
    if isinstance(context, dict):
        return list(context["title"]), [list(sentences) for sentences in context["sentences"]]
    titles: list[str] = []
    sentence_lists: list[list[str]] = []
    for title, sentences in context:
        titles.append(str(title))
        sentence_lists.append([str(sentence) for sentence in sentences])
    return titles, sentence_lists


def _hotpot_supporting_titles(ex: dict[str, Any]) -> set[str]:
    supporting = ex["supporting_facts"]
    if isinstance(supporting, dict):
        return {str(title) for title in supporting["title"]}
    return {str(item[0]) for item in supporting}


def load_musique_local(path: str | Path, limit: int = 300) -> CorpusDataset:
    """Load MuSiQue-like JSON/JSONL from a local file.

    This accepts common fields used by MuSiQue exports:
    `question`, `answer`/`answers`, and `paragraphs`/`contexts`.
    """

    path = Path(path)
    rows: list[dict[str, Any]]
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        obj = json.loads(path.read_text(encoding="utf-8"))
        rows = obj if isinstance(obj, list) else obj.get("data", obj.get("examples", []))
    if limit:
        rows = rows[:limit]

    documents: dict[str, Document] = {}
    questions: list[Question] = []
    for i, ex in enumerate(rows):
        qid = str(ex.get("id") or ex.get("qid") or f"musique_{i}")
        paragraphs = ex.get("paragraphs") or ex.get("contexts") or ex.get("context") or []
        gold: set[str] = set()
        for j, paragraph in enumerate(paragraphs):
            if isinstance(paragraph, dict):
                title = str(paragraph.get("title") or paragraph.get("idx") or f"{qid}_p{j}")
                text = str(paragraph.get("paragraph_text") or paragraph.get("text") or paragraph.get("content") or "")
                is_support = bool(paragraph.get("is_supporting") or paragraph.get("supporting"))
            else:
                title = f"{qid}_p{j}"
                text = str(paragraph)
                is_support = True
            doc_id = f"{qid}::{title}"
            documents[doc_id] = Document(doc_id=doc_id, title=title, text=text, metadata={"dataset": "musique"})
            if is_support:
                gold.add(doc_id)
        answers = ex.get("answers") or [ex.get("answer", "")]
        if isinstance(answers, str):
            answers = [answers]
        questions.append(
            Question(qid=qid, text=str(ex.get("question", "")), answers=[str(a) for a in answers], gold_doc_ids=gold)
        )
    return CorpusDataset(f"musique_local[{len(questions)}]", documents, questions)


def load_2wiki_local(path: str | Path, limit: int = 300, seed: int = 0) -> CorpusDataset:
    """Load 2WikiMultiHopQA from a local JSON/JSONL file.

    2Wiki's native schema is identical to HotpotQA's distractor format
    (``context`` = list of [title, sentences]; ``supporting_facts`` =
    list of [title, sent_id]), so we reuse the Hotpot context/supporting
    helpers. Seed-shuffle + limit mirror :func:`load_hotpotqa` for a
    HotpotQA-comparable 500-question slice.
    """

    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        obj = json.loads(path.read_text(encoding="utf-8"))
        rows = obj if isinstance(obj, list) else obj.get("data", obj.get("examples", []))
    if seed:
        random.Random(seed).shuffle(rows)
    if limit:
        rows = rows[:limit]

    documents: dict[str, Document] = {}
    questions: list[Question] = []
    for ex in rows:
        titles, sentence_lists = _hotpot_context(ex)
        for title, sentences in zip(titles, sentence_lists):
            doc_id = str(title)
            documents[doc_id] = Document(
                doc_id=doc_id,
                title=str(title),
                text=" ".join(s.strip() for s in sentences).strip(),
                metadata={"dataset": "2wiki"},
            )
        gold = _hotpot_supporting_titles(ex)
        questions.append(
            Question(
                qid=str(ex.get("_id", ex.get("id", len(questions)))),
                text=str(ex["question"]),
                answers=[str(ex["answer"])],
                gold_doc_ids=gold,
            )
        )
    return CorpusDataset(f"2wiki_local[{len(questions)}]", documents, questions)


def load_ragbench(split: str = "test", subset: str | None = None, limit: int = 300) -> CorpusDataset:
    from datasets import load_dataset

    ds = load_dataset("rungalileo/ragbench", subset, split=split) if subset else load_dataset("rungalileo/ragbench", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    documents: dict[str, Document] = {}
    questions: list[Question] = []
    for i, ex in enumerate(ds):
        qid = str(ex.get("id") or ex.get("qid") or i)
        question = str(ex.get("question") or ex.get("query") or ex.get("prompt") or "")
        answer = str(ex.get("answer") or ex.get("response") or ex.get("generated_answer") or "")
        contexts = ex.get("documents") or ex.get("contexts") or ex.get("passages") or ex.get("context") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        gold: set[str] = set()
        for j, context in enumerate(contexts):
            if isinstance(context, dict):
                text = str(context.get("text") or context.get("content") or context.get("document") or "")
                title = str(context.get("title") or f"{qid}_ctx{j}")
            else:
                text = str(context)
                title = f"{qid}_ctx{j}"
            doc_id = f"{qid}::ctx{j}"
            documents[doc_id] = Document(doc_id=doc_id, title=title, text=text, metadata={"dataset": "ragbench"})
            gold.add(doc_id)
        questions.append(Question(qid=qid, text=question, answers=[answer] if answer else [], gold_doc_ids=gold))
    return CorpusDataset(f"ragbench[{split}:{len(questions)}]", documents, questions)


def load_dataset(name: str, **kwargs: Any) -> CorpusDataset:
    if name == "toy":
        return load_toy()
    if name == "hotpotqa":
        return load_hotpotqa(**kwargs)
    if name == "musique_local":
        return load_musique_local(**kwargs)
    if name == "2wiki_local":
        return load_2wiki_local(**kwargs)
    if name == "ragbench":
        return load_ragbench(**kwargs)
    raise ValueError(f"unknown dataset '{name}'")
