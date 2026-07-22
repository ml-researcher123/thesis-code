"""Re-slice existing per-question results by QUESTION TYPE (bridge vs comparison).

Motivation (the "2Wiki null" hunt): the submodular packer beats the focused
heuristic on HotpotQA but is null on 2WikiMultiHopQA *in aggregate*. The scope
map predicts *why*: the packer only helps when the extra gold it forces into
context actually CONTAINS the answer (raises answer-in-context). On 2Wiki the
answer usually sits in the doc the heuristic already ranks first, so the extra
gold is bridging/reasoning scaffolding -- coverage up, answer-in-context flat.

That mechanism predicts a testable split: the win should concentrate in
*bridge / compositional* questions (answer downstream) and vanish or reverse on
*comparison* questions (answer derivable from the top doc). Both HotpotQA and
2Wiki tag every question with a `type`, and our per-question CSVs already carry
the native question id (2Wiki `_id`, HotpotQA `id`), so we can test this
*without re-running anything* -- just join the CSVs to the type field.

This needs NO GPU. It reads per-question CSVs (already on GitHub) and a
type-source (the 2Wiki dev.json you mount/download, or HotpotQA via the HF Hub).

Usage:
    # 2Wiki: point --type-source at the dev.json (Kaggle mount, gdown, or HF)
    python analysis/analyze_by_question_type.py \\
        --per-question kaggle_results/**/stage14_2wiki_*_per_question.csv \\
        --type-source /path/to/2wiki/dev.json

    # HotpotQA: load the type field straight from the HF Hub
    python analysis/analyze_by_question_type.py \\
        --per-question kaggle_results/**/stage12_*hotpotqa*_per_question.csv \\
        --type-source hotpotqa-hf

A positive `delta` on bridge/compositional types with a ~0/negative delta on
comparison types is the signal that the win concentrates in bridge questions --
i.e. a *sub-population* second win hiding inside the aggregate 2Wiki null.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_type_map(source: str) -> dict[str, str]:
    """Map native question id -> question type.

    Accepts a 2Wiki-style dev.json (list of objects with `_id`/`id` and `type`)
    or the literal "hotpotqa-hf" to pull HotpotQA's type field from the HF Hub.
    """
    if source == "hotpotqa-hf":
        from datasets import load_dataset

        ds = load_dataset("hotpotqa/hotpotqa", "distractor", split="validation")
        return {str(ex["id"]): str(ex.get("type", "unknown")) for ex in ds}

    path = Path(source)
    if not path.exists():
        raise SystemExit(f"--type-source {source!r} is not 'hotpotqa-hf' and not an existing file")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):  # some dumps wrap the list under a key
        data = data.get("data") or next((v for v in data.values() if isinstance(v, list)), [])
    type_map: dict[str, str] = {}
    for ex in data:
        qid = str(ex.get("_id") or ex.get("id") or "")
        if qid:
            type_map[qid] = str(ex.get("type", "unknown"))
    if not type_map:
        raise SystemExit(f"No (id, type) pairs parsed from {source!r}")
    return type_map


def collect_per_question(
    patterns: list[str], prefix_a: str, prefix_b: str
) -> list[tuple[str, float, float]]:
    """Return (qid, f1_a, f1_b) for every question where BOTH policies are present.

    Pools across all matched files (seeds/budgets); a question appearing in
    several files contributes one row per file, which is what we want for a
    pooled per-type mean.
    """
    paths: list[str] = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat, recursive=True)))
    if not paths:
        raise SystemExit(f"No per-question CSVs matched: {patterns}")

    rows: list[tuple[str, float, float]] = []
    for p in paths:
        # group this file's rows by qid: {qid: {'a': f1, 'b': f1}}
        by_qid: dict[str, dict[str, float]] = defaultdict(dict)
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                pol = r.get("policy", "")
                try:
                    f1 = float(r["f1"])
                except (KeyError, ValueError):
                    continue
                if pol.startswith(prefix_a):
                    by_qid[r["qid"]]["a"] = f1
                elif pol.startswith(prefix_b):
                    by_qid[r["qid"]]["b"] = f1
        for qid, d in by_qid.items():
            if "a" in d and "b" in d:
                rows.append((qid, d["a"], d["b"]))
    print(f"[collect] {len(paths)} file(s), {len(rows)} paired question rows")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-question", nargs="+", required=True,
                    help="One or more paths / globs to *_per_question.csv (use ** with the shell or quote for recursive).")
    ap.add_argument("--type-source", required=True,
                    help="Path to a 2Wiki-style dev.json, or 'hotpotqa-hf' to load HotpotQA types from the HF Hub.")
    ap.add_argument("--policy-a", default="chunk_submod", help="Prefix of policy A (the packer). Budget suffix is ignored.")
    ap.add_argument("--policy-b", default="chunk_focused", help="Prefix of policy B (the heuristic baseline).")
    args = ap.parse_args()

    type_map = load_type_map(args.type_source)
    print(f"[types] {len(type_map)} ids; type histogram: "
          + ", ".join(f"{t}={n}" for t, n in sorted(
              {t: sum(1 for v in type_map.values() if v == t) for t in set(type_map.values())}.items())))

    rows = collect_per_question(args.per_question, args.policy_a, args.policy_b)

    # group deltas by question type
    by_type: dict[str, list[tuple[float, float]]] = defaultdict(list)
    unmatched = 0
    for qid, fa, fb in rows:
        t = type_map.get(qid)
        if t is None:
            unmatched += 1
            continue
        by_type[t].append((fa, fb))
    if unmatched:
        print(f"[warn] {unmatched} question rows had no type match (qid not in type-source) -- excluded")

    print()
    print(f"{'question type':22} {'n':>5} {args.policy_a[:10]:>10} {args.policy_b[:10]:>10} {'delta':>9}")
    print("-" * 62)
    overall_a: list[float] = []
    overall_b: list[float] = []
    for t in sorted(by_type, key=lambda k: -len(by_type[k])):
        a = [x[0] for x in by_type[t]]
        b = [x[1] for x in by_type[t]]
        overall_a += a
        overall_b += b
        ma, mb = statistics.mean(a), statistics.mean(b)
        print(f"{t:22} {len(a):>5} {ma:>10.4f} {mb:>10.4f} {ma-mb:>+9.4f}")
    print("-" * 62)
    if overall_a:
        print(f"{'ALL':22} {len(overall_a):>5} {statistics.mean(overall_a):>10.4f} "
              f"{statistics.mean(overall_b):>10.4f} {statistics.mean(overall_a)-statistics.mean(overall_b):>+9.4f}")
    print()
    print("Read: a clearly POSITIVE delta on bridge/compositional types with a ~0 or")
    print("negative delta on comparison types = the packer win concentrates in bridge")
    print("questions (a sub-population second win inside the aggregate 2Wiki null).")


if __name__ == "__main__":
    main()
