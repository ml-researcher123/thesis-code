"""Stage-4: evidence-density evaluation and principled budgeted packing.

This runner tests the next research hypothesis after the fixed/oracle routers:

    Document recall is the wrong objective for budget-constrained RAG. What
    matters is answer-bearing evidence *density* in the reader context, and a
    packer that directly optimizes that density beats the heuristic packers.

It runs a clean 2x2 factorial over {chunk, ACE} representations x
{focused (heuristic), submodular (principled)} packers at a single shared token
budget, plus the standard ``chunk_packed`` anchor and an oracle upper bound.

For every policy it reports, alongside the usual answer metrics, the
reader-context density diagnostics from :mod:`ace_rag.context_quality`
(answer-in-context rate, gold-token density, reader-level gold coverage). It also
writes a per-question CSV so the mediation claim -- density predicts answer
quality better than recall@k -- can be checked offline.

The principled packer is :func:`ace_rag.evidence_packer.pack_submodular_run`.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

from ace_rag.context_quality import aggregate_context_quality, context_quality
from ace_rag.datasets import load_dataset
from ace_rag.evidence_packer import pack_mmr_run, pack_submodular_run
from ace_rag.generator import ExtractiveGenerator, HuggingFaceGenerator
from ace_rag.metrics import (
    all_gold_retrieved_at_k,
    evaluate_retrieval,
    exact_match,
    retrieval_recall_at_k,
    token_f1,
)
from ace_rag.schema import CorpusDataset, RetrievalRun
from ace_rag.text import tokenize
from experiments.run_qwen_eval import (
    generation_metrics,
    materialize_reader_context,
    retrieve_ace,
    retrieve_chunk,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="hotpotqa", choices=["toy", "hotpotqa", "musique_local", "2wiki_local", "ragbench"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--ragbench-subset", default=None)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--musique-path", default=None)
    parser.add_argument("--twowiki-path", default=None)
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Run only chunk+ACE retrieval, report recall@5/all_gold@5, and exit before "
                             "loading the reader. A cheap prerequisite gate for a new dataset.")
    parser.add_argument("--embedder", default="sentence-transformers", choices=["lexical", "sentence-transformers"])
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--embed-device", default="cuda")
    parser.add_argument("--compressor", default="truncate", choices=["identity", "truncate", "pca", "binary"])
    parser.add_argument("--compress-dims", type=int, default=320)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-k-nodes", type=int, default=48)
    parser.add_argument("--max-expanded-docs", type=int, default=5)
    parser.add_argument("--ace-retriever", default="standard", choices=["standard", "bridge"])
    parser.add_argument("--bridge-seed-nodes", type=int, default=12)
    parser.add_argument("--bridge-terms", type=int, default=10)
    parser.add_argument("--bridge-weight", type=float, default=0.35)
    parser.add_argument("--hybrid-alpha", type=float, default=0.5)
    parser.add_argument("--reader-backend", default="hf", choices=["hf", "extractive"])
    parser.add_argument("--reader-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--reader-device", default="cuda")
    parser.add_argument("--reader-batch-size", type=int, default=2)
    parser.add_argument("--reader-device-map", default=None,
                        help="transformers device_map (e.g. 'auto') to shard a large reader across visible GPUs.")
    parser.add_argument("--reader-load-4bit", action="store_true",
                        help="Load the reader in 4-bit (bitsandbytes) to fit a large model on one GPU.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--snippet-window", type=int, default=1)
    parser.add_argument("--max-snippets", type=int, default=8)
    parser.add_argument("--max-snippet-tokens", type=int, default=80)
    parser.add_argument("--budget", type=int, default=160, help="Shared reader token budget for the 2x2 factorial.")
    parser.add_argument("--mmr-lambda", type=float, default=0.7, help="MMR relevance/redundancy trade-off.")
    parser.add_argument("--submod-w-rel", type=float, default=1.0)
    parser.add_argument("--submod-w-query", type=float, default=0.5)
    parser.add_argument("--submod-w-cover", type=float, default=0.4)
    parser.add_argument("--submod-w-div", type=float, default=0.3)
    parser.add_argument("--submod-sat-alpha", type=float, default=0.3)
    parser.add_argument("--submod-cost-power", type=float, default=1.0)
    parser.add_argument("--submod-max-candidates", type=int, default=160)
    parser.add_argument("--out-dir", default="cloud_results")
    return parser.parse_args()


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_from_args(args: argparse.Namespace) -> CorpusDataset:
    if args.dataset == "toy":
        return load_dataset("toy")
    if args.dataset == "musique_local":
        if not args.musique_path:
            raise SystemExit("--musique-path is required for musique_local")
        return load_dataset("musique_local", path=args.musique_path, limit=args.limit)
    if args.dataset == "2wiki_local":
        if not args.twowiki_path:
            raise SystemExit("--twowiki-path is required for 2wiki_local")
        return load_dataset("2wiki_local", path=args.twowiki_path, limit=args.limit, seed=args.seed)
    if args.dataset == "ragbench":
        return load_dataset("ragbench", split=args.split, subset=args.ragbench_subset, limit=args.limit)
    return load_dataset("hotpotqa", split=args.split, limit=args.limit, seed=args.seed)


def build_policies(args: argparse.Namespace) -> list[dict[str, Any]]:
    b = args.budget
    ace_prefix = "ace_bridge" if args.ace_retriever == "bridge" else "ace"
    return [
        {"name": f"chunk_packed_{b}", "base": "chunk", "representation": "chunk", "packer": "packed", "mode": "packed_snippets"},
        {"name": f"chunk_focused_{b}", "base": "chunk", "representation": "chunk", "packer": "focused", "mode": "focused_packed"},
        {"name": f"chunk_mmr_{b}", "base": "chunk", "representation": "chunk", "packer": "mmr", "mode": "mmr"},
        {"name": f"chunk_submod_{b}", "base": "chunk", "representation": "chunk", "packer": "submod", "mode": "submodular"},
        {"name": f"{ace_prefix}_focused_{b}", "base": "ace", "representation": "ace", "packer": "focused", "mode": "focused_packed"},
        {"name": f"{ace_prefix}_mmr_{b}", "base": "ace", "representation": "ace", "packer": "mmr", "mode": "mmr"},
        {"name": f"{ace_prefix}_submod_{b}", "base": "ace", "representation": "ace", "packer": "submod", "mode": "submodular"},
    ]


def materialize_policy(
    dataset: CorpusDataset, base_runs: list[RetrievalRun], policy: dict[str, Any], args: argparse.Namespace
) -> list[RetrievalRun]:
    if policy["mode"] == "submodular":
        return [
            pack_submodular_run(
                dataset,
                run,
                snippet_window=args.snippet_window,
                max_snippets=args.max_snippets,
                max_snippet_tokens=args.max_snippet_tokens,
                token_budget=args.budget,
                w_rel=args.submod_w_rel,
                w_query=args.submod_w_query,
                w_cover=args.submod_w_cover,
                w_div=args.submod_w_div,
                sat_alpha=args.submod_sat_alpha,
                cost_power=args.submod_cost_power,
                max_candidates=args.submod_max_candidates,
            )
            for run in base_runs
        ]
    if policy["mode"] == "mmr":
        return [
            pack_mmr_run(
                dataset,
                run,
                snippet_window=args.snippet_window,
                max_snippets=args.max_snippets,
                max_snippet_tokens=args.max_snippet_tokens,
                token_budget=args.budget,
                mmr_lambda=args.mmr_lambda,
                max_candidates=args.submod_max_candidates,
            )
            for run in base_runs
        ]
    return materialize_reader_context(
        dataset,
        base_runs,
        policy["mode"],
        snippet_window=args.snippet_window,
        max_snippets=args.max_snippets,
        max_snippet_tokens=args.max_snippet_tokens,
        packed_token_budget=args.budget,
    )


def generate(reader: Any, backend: str, reader_runs: list[RetrievalRun]) -> list[str]:
    if backend == "hf":
        return reader.answer_many([(run.query, run) for run in reader_runs])
    return [reader.answer(run.query, run) for run in reader_runs]


def base_question_metrics(runs: list[RetrievalRun], dataset: CorpusDataset) -> dict[str, tuple[float, float]]:
    by_qid = {q.qid: q for q in dataset.questions}
    out: dict[str, tuple[float, float]] = {}
    for run in runs:
        q = by_qid[run.qid]
        out[run.qid] = (
            retrieval_recall_at_k(run.retrieved_doc_ids, q.gold_doc_ids, 5),
            all_gold_retrieved_at_k(run.retrieved_doc_ids, q.gold_doc_ids, 5),
        )
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_args(args)
    print(dataset.summary(), flush=True)
    by_qid = {q.qid: q for q in dataset.questions}

    print("[density] retrieving chunk baseline", flush=True)
    chunk_runs = retrieve_chunk(dataset, args)
    cleanup_cuda()
    print("[density] retrieving ACE graph", flush=True)
    ace_runs = retrieve_ace(dataset, args)
    cleanup_cuda()

    base_runs_by_name = {"chunk": chunk_runs, "ace": ace_runs}
    base_metrics_by_rep = {
        "chunk": base_question_metrics(chunk_runs, dataset),
        "ace": base_question_metrics(ace_runs, dataset),
    }

    if args.retrieval_only:
        # Cheap prerequisite gate: report whether retrieval surfaces the gold
        # evidence (the binding precondition for the packer to help), then exit
        # before paying for the reader. all_gold@5 is the decisive number: the
        # packer cannot assemble evidence retrieval never surfaced.
        gate_rows: list[dict[str, Any]] = []
        for rep, runs in (("chunk", chunk_runs), ("ace", ace_runs)):
            ret = evaluate_retrieval(runs, dataset.questions, k_values=(1, 2, 5))
            row = {"dataset": args.dataset, "representation": rep, "n": len(dataset.questions), **ret}
            gate_rows.append(row)
            print(f"[gate] {rep}: recall@5={ret.get('recall@5')} all_gold@5={ret.get('all_gold@5')}", flush=True)
        gate_path = out_dir / f"{args.dataset}_retrieval_gate_limit{args.limit}_seed{args.seed}.csv"
        write_csv(gate_path, gate_rows)
        print(f"wrote {gate_path}", flush=True)
        chunk_ag = next((r.get("all_gold@5", 0.0) for r in gate_rows if r["representation"] == "chunk"), 0.0)
        print(f"[gate] DECISION INPUT: chunk all_gold@5={chunk_ag} "
              f"(MuSiQue was 0.184 -> packer null; HotpotQA ~0.76 -> packer win)", flush=True)
        return

    if args.reader_backend == "hf":
        print("[density] loading reader once", flush=True)
        reader: Any = HuggingFaceGenerator(
            model_name=args.reader_model,
            device=args.reader_device,
            batch_size=args.reader_batch_size,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            device_map=args.reader_device_map,
            load_in_4bit=args.reader_load_4bit,
        )
        reader_name = args.reader_model
    else:
        reader = ExtractiveGenerator()
        reader_name = "extractive"

    policies = build_policies(args)
    candidate_outputs: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    per_question_rows: list[dict[str, Any]] = []

    for policy in policies:
        name = policy["name"]
        print(f"[density] materializing policy={name}", flush=True)
        base_runs = base_runs_by_name[policy["base"]]
        reader_runs = materialize_policy(dataset, base_runs, policy, args)
        print(f"[density] generating policy={name}", flush=True)
        predictions = generate(reader, args.reader_backend, reader_runs)
        candidate_outputs[name] = {
            "policy": policy,
            "reader_by_qid": {run.qid: run for run in reader_runs},
            "prediction_by_qid": {run.qid: pred for run, pred in zip(reader_runs, predictions)},
        }

        row: dict[str, Any] = {
            "policy": name,
            "router_type": "fixed",
            "representation": policy["representation"],
            "packer": policy["packer"],
            "reader_model": reader_name,
            "packed_token_budget": args.budget,
        }
        row.update(evaluate_retrieval(base_runs, dataset.questions, k_values=(1, 2, 5)))
        row.update(generation_metrics(dataset, reader_runs, predictions))
        agg, _ = aggregate_context_quality(reader_runs, dataset.questions)
        row.update(agg)
        rows.append(row)
        print("[density] " + ", ".join(f"{k}={v}" for k, v in row.items()), flush=True)

        rep = policy["representation"]
        for run, pred in zip(reader_runs, predictions):
            q = by_qid[run.qid]
            cq = context_quality(run, q)
            base_recall, base_all_gold = base_metrics_by_rep[rep].get(run.qid, (0.0, 0.0))
            per_question_rows.append(
                {
                    "qid": run.qid,
                    "policy": name,
                    "representation": rep,
                    "packer": policy["packer"],
                    "em": exact_match(pred, q.answers),
                    "f1": round(token_f1(pred, q.answers), 4),
                    "ans_in_context": cq["ans_in_context"],
                    "gold_token_density": round(cq["gold_token_density"], 4),
                    "gold_doc_reader_cov": round(cq["gold_doc_reader_cov"], 4),
                    "context_tokens": cq["context_tokens"],
                    "base_recall@5": round(base_recall, 4),
                    "base_all_gold@5": round(base_all_gold, 4),
                }
            )
        cleanup_cuda()

    # Oracle upper bound over the 2x2 packing policies (exclude the anchor).
    oracle_candidates = [p["name"] for p in policies if p["packer"] != "packed"]
    oracle_predictions: list[str] = []
    oracle_runs: list[RetrievalRun] = []
    oracle_choices: list[str] = []
    for q in dataset.questions:
        best_name = oracle_candidates[0]
        best_key = (-1.0, -1.0, 0)
        for name in oracle_candidates:
            pred = candidate_outputs[name]["prediction_by_qid"][q.qid]
            run = candidate_outputs[name]["reader_by_qid"][q.qid]
            key = (token_f1(pred, q.answers), exact_match(pred, q.answers), -sum(len(tokenize(h.text)) for h in run.hits))
            if key > best_key:
                best_key = key
                best_name = name
        oracle_choices.append(best_name)
        oracle_runs.append(candidate_outputs[best_name]["reader_by_qid"][q.qid])
        oracle_predictions.append(candidate_outputs[best_name]["prediction_by_qid"][q.qid])

    from collections import Counter

    oracle_row: dict[str, Any] = {
        "policy": "router_oracle_packer",
        "router_type": "oracle",
        "representation": "mixed",
        "packer": "oracle",
        "reader_model": reader_name,
        "packed_token_budget": args.budget,
        "selection_counts": json.dumps(dict(sorted(Counter(oracle_choices).items()))),
    }
    oracle_row.update(generation_metrics(dataset, oracle_runs, oracle_predictions))
    agg, _ = aggregate_context_quality(oracle_runs, dataset.questions)
    oracle_row.update(agg)
    rows.append(oracle_row)
    print("[density] " + ", ".join(f"{k}={v}" for k, v in oracle_row.items()), flush=True)

    tag = f"{args.dataset}_density_{args.ace_retriever}_budget{args.budget}_limit{args.limit}"
    metrics_path = out_dir / f"{tag}_metrics.csv"
    perq_path = out_dir / f"{tag}_per_question.csv"
    write_csv(metrics_path, rows)
    write_csv(perq_path, per_question_rows)
    print(f"wrote {metrics_path}", flush=True)
    print(f"wrote {perq_path}", flush=True)


if __name__ == "__main__":
    main()
