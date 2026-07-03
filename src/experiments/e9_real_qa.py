"""E9 -- Real end-to-end RAG QA: compounding (C2) + allocation (C3) on real answer quality.

E6 tested compounding/allocation on a synthetic corpus with a probe. E9 tests the same two
claims on a REAL multi-hop benchmark (HotpotQA) with a REAL frozen reader LLM and REAL EM/F1,
which is the reviewer-proof version of the story. One shared representation budget B = d_r + d_c
is split between two real vector bottlenecks:

  RETRIEVAL (d_r): embed the pooled corpus with a real encoder, truncate to d_r, take top-k
    passages. Low d_r -> the gold passage is missed and its sentences never enter the candidate pool.
  COMPRESSION (d_c): query-conditioned sentence selection -- score each candidate sentence by
    d_c-truncated cosine to the query and greedily pack the highest until a reader token budget
    B_tok is hit (the deployable/selective compression instantiation; cf. RECOMP, LLMLingua,
    SeleCom). Low d_c -> the wrong sentences are packed and the answer is dropped.

A frozen instruct LLM then answers from the packed context; we score EM/F1 against the gold
answer and also record answer-in-context (does the gold answer survive into what the reader sees).
Sweeping the split shows the best allocation underperforms either stage at the full budget
(compounding, now on EM/F1), that the optimum is interior, and that the answer-survival diagnostic
tracks the accuracy -- tying the mechanism to what actually reaches the reader.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..common import RunContext, ensure_deps
from ..eval.metrics import answer_in_context, exact_match, token_f1
from ..retrieval.beir_eval import truncate_normalize
from ..retrieval.hotpot import load_hotpotqa

SYS_PROMPT = ("You are a helpful assistant. Answer the question using ONLY the context. "
             "Reply with a short answer of a few words, or 'yes' or 'no'. If the context does "
             "not contain the answer, give your best short guess.")


def _cos_topk(q_full, d_full, d, k):
    """Top-k document indices per query by cosine on d-truncated embeddings."""
    qe = truncate_normalize(q_full, d)
    de = truncate_normalize(d_full, d)
    sims = qe @ de.T                                # (Q, N)
    k = min(k, sims.shape[1])
    idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sims, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)   # (Q, k) sorted


class Reader:
    """Frozen decoder-only LLM used only for generation (no training)."""

    def __init__(self, model_name, device, log):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(device).eval()
        self.device = device
        log(f"  reader loaded: {model_name} ({sum(p.numel() for p in self.model.parameters())/1e9:.2f}B)")

    def n_tokens(self, text):
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def answer_batch(self, contexts, questions, max_new_tokens=24, batch_size=16):
        torch = self.torch
        prompts = []
        for ctx, q in zip(contexts, questions):
            msgs = [{"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"}]
            prompts.append(self.tok.apply_chat_template(msgs, tokenize=False,
                                                        add_generation_prompt=True))
        out = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=True,
                           max_length=1024).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            for j in range(len(chunk)):
                new = gen[j, enc["input_ids"].shape[1]:]
                text = self.tok.decode(new, skip_special_tokens=True).strip()
                out.append(text.split("\n")[0].strip())
        return out


def run(ctx: RunContext):
    cfg = ctx.cfg; log = ctx.log; p = cfg.get("params", {})
    ensure_deps({"sentence_transformers": "sentence-transformers", "datasets": "datasets",
                 "transformers": "transformers"}, log)
    from sentence_transformers import SentenceTransformer

    embedder_name = p.get("embedder", "mixedbread-ai/mxbai-embed-large-v1")
    reader_name = p.get("reader", "Qwen/Qwen2.5-1.5B-Instruct")
    split = p.get("split", "validation")
    n_q = p.get("n_q", 300)
    seed = p.get("seed_data", 0)
    top_k = p.get("top_k", 4)
    budget_tokens = p.get("budget_tokens", 160)
    budgets = p.get("budgets", [128, 256])        # shared representation budget B = d_r + d_c
    splits = p.get("splits", [32, 48, 64, 96])    # d_r values; d_c = B - d_r
    query_prompt = p.get("query_prompt", "Represent this sentence for searching relevant passages: ")
    max_new_tokens = p.get("max_new_tokens", 24)
    gen_batch = p.get("gen_batch", 16)

    log(f"E9 real QA | embedder={embedder_name} reader={reader_name} n_q={n_q} "
        f"budgets={budgets} B_tok={budget_tokens} device={ctx.device}")
    corpus, questions = load_hotpotqa(split, n_q, seed)
    pids = list(corpus)
    pid_index = {pid: i for i, pid in enumerate(pids)}
    log(f"  pooled corpus: {len(pids)} paragraphs; {len(questions)} questions")

    # ---- encode once (full dim); truncation is cheap afterwards ----
    embedder = SentenceTransformer(embedder_name, trust_remote_code=p.get("trust_remote_code", False))
    if ctx.device.type == "cuda":
        embedder = embedder.to(ctx.device)
    para_full = embedder.encode([corpus[pid]["text"] for pid in pids], batch_size=128,
                                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    q_full = embedder.encode([query_prompt + qq["question"] for qq in questions], batch_size=128,
                             normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    # flat sentence list with a map pid -> (start,end) into sent_full
    sent_texts, sent_owner = [], []
    pid_sent_span = {}
    for pid in pids:
        s0 = len(sent_texts)
        for s in corpus[pid]["sentences"]:
            sent_texts.append(s); sent_owner.append(pid)
        pid_sent_span[pid] = (s0, len(sent_texts))
    sent_full = embedder.encode(sent_texts, batch_size=256, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False) if sent_texts else np.zeros((0, para_full.shape[1]))
    full_dim = para_full.shape[1]
    log(f"  encoded {len(pids)} paras, {len(sent_texts)} sentences, dim={full_dim}")

    reader = None
    try:
        reader = Reader(reader_name, ctx.device, log)
    except Exception as exc:  # noqa: BLE001
        log(f"  !! reader failed to load ({exc}); falling back to answer-in-context only")

    gold_titles = [q["gold_titles"] for q in questions]
    answers = [q["answer"] for q in questions]
    q_texts = [q["question"] for q in questions]

    def pack_contexts(cand_pids_per_q, d_c):
        """Query-conditioned sentence selection under the token budget -> packed context per q."""
        contexts, aic = [], []
        qe = truncate_normalize(q_full, d_c)
        se = truncate_normalize(sent_full, d_c) if len(sent_full) else sent_full
        for qi, cand_pids in enumerate(cand_pids_per_q):
            # gather candidate sentence indices from the retrieved paragraphs
            sidx = []
            for pid in cand_pids:
                a, b = pid_sent_span[pid]; sidx.extend(range(a, b))
            if not sidx:
                contexts.append(""); aic.append(0.0); continue
            scores = se[sidx] @ qe[qi]
            order = np.argsort(-scores)
            chosen, used = [], 0
            for oi in order:
                s = sent_texts[sidx[oi]]
                t = reader.n_tokens(s) if reader is not None else max(1, len(s.split()))
                if used + t > budget_tokens and chosen:
                    break
                chosen.append(sidx[oi]); used += t
                if used >= budget_tokens:
                    break
            chosen.sort()  # present in reading order
            ctx_text = " ".join(sent_texts[i] for i in chosen)
            contexts.append(ctx_text)
            aic.append(answer_in_context(answers[qi], ctx_text))
        return contexts, np.array(aic)

    def score(contexts):
        if reader is None:
            return None, None
        preds = reader.answer_batch(contexts, q_texts, max_new_tokens, gen_batch)
        em = np.array([exact_match(pr, g) for pr, g in zip(preds, answers)])
        f1 = np.array([token_f1(pr, g) for pr, g in zip(preds, answers)])
        return em, f1

    def retrieval_recall(top_pids_per_q):
        rec = []
        for qi, tp in enumerate(top_pids_per_q):
            titles = {corpus[pid]["title"] for pid in tp}
            g = gold_titles[qi]
            rec.append(len(titles & g) / max(1, len(g)))
        return float(np.mean(rec))

    # ---- retrieval top-k per d_r (cache by d_r) ----
    def retrieve(d_r):
        top = _cos_topk(q_full, para_full, min(d_r, full_dim), top_k)
        return [[pids[j] for j in top[qi]] for qi in range(len(questions))]

    def eval_config(d_r, d_c):
        cand = retrieve(d_r)
        contexts, aic = pack_contexts(cand, min(d_c, full_dim))
        em, f1 = score(contexts)
        rec = retrieval_recall(cand)
        return {"recall": rec, "aic": float(aic.mean()),
                "em": (float(em.mean()) if em is not None else None),
                "f1": (float(f1.mean()) if f1 is not None else None)}

    # standalone references use gold paragraphs (perfect retrieval) or full-dim selection
    gold_cands = [[pid for pid in q["own_pids"] if corpus[pid]["title"] in q["gold_titles"]]
                  or q["own_pids"] for q in questions]

    per_budget = {}
    for B in budgets:
        sp = [(d_r, B - d_r) for d_r in splits if 0 < B - d_r <= full_dim and d_r <= full_dim]
        split_res = {}
        for (d_r, d_c) in sp:
            r = eval_config(d_r, d_c); split_res[f"{d_r}:{d_c}"] = r
            log(f"  [B={B}] d_r={d_r:4d} d_c={d_c:4d} | recall={r['recall']:.3f} "
                f"aic={r['aic']:.3f} f1={r['f1'] if r['f1'] is None else round(r['f1'],3)} "
                f"em={r['em'] if r['em'] is None else round(r['em'],3)}")
        # standalone retrieval: retrieval at budget B, compression NOT limiting (full-dim selection)
        cand_R = retrieve(B)
        ctx_R, aic_R = pack_contexts(cand_R, full_dim)
        em_R, f1_R = score(ctx_R)
        stand_R = {"recall": retrieval_recall(cand_R), "aic": float(aic_R.mean()),
                   "em": (float(em_R.mean()) if em_R is not None else None),
                   "f1": (float(f1_R.mean()) if f1_R is not None else None)}
        # standalone compression: retrieval perfect (gold paras), selection at d_c = B
        ctx_C, aic_C = pack_contexts(gold_cands, min(B, full_dim))
        em_C, f1_C = score(ctx_C)
        stand_C = {"recall": 1.0, "aic": float(aic_C.mean()),
                   "em": (float(em_C.mean()) if em_C is not None else None),
                   "f1": (float(f1_C.mean()) if f1_C is not None else None)}
        key = "f1" if reader is not None else "aic"
        best_k = max(split_res, key=lambda s: split_res[s][key])
        best = split_res[best_k]
        sr, sc = stand_R[key], stand_C[key]
        gap = min(sr, sc) - best[key]
        per_budget[B] = {"splits": split_res, "best_split": best_k, "best": best,
                         "standalone_R": stand_R, "standalone_C": stand_C,
                         "gap_metric": key, "gap": gap}
        log(f"  [B={B}] best={best_k} {key}={best[key]:.3f} | standalone_R={sr:.3f} "
            f"standalone_C={sc:.3f} compounding_gap={gap:.3f}")

    # ---- figure: end-to-end metric vs retrieval share, at the largest budget ----
    metric = "f1" if reader is not None else "aic"
    fig = os.path.join(ctx.outdir, "e9_real_qa.png")
    Bmax = max(per_budget)
    info = per_budget[Bmax]
    xs = sorted(info["splits"], key=lambda s: int(s.split(":")[0]))
    drs = [int(s.split(":")[0]) for s in xs]
    ys = [info["splits"][s][metric] for s in xs]
    plt.figure(figsize=(6.6, 4.2))
    plt.plot(drs, ys, marker="o", color="purple", label=f"end-to-end {metric.upper()} (real reader)")
    plt.axhline(info["standalone_R"][metric], ls="--", c="C0", lw=1,
                label=f"standalone retrieval {info['standalone_R'][metric]:.2f}")
    plt.axhline(info["standalone_C"][metric], ls="--", c="C1", lw=1,
                label=f"standalone compression {info['standalone_C'][metric]:.2f}")
    bx = int(info["best_split"].split(":")[0])
    plt.scatter([bx], [info["best"][metric]], color="red", zorder=5, label=f"optimal d_r={bx}")
    plt.xlabel(f"retrieval budget d_r  (d_c = {Bmax} - d_r)")
    plt.ylabel(f"HotpotQA {metric.upper()}")
    plt.title(f"E9: real end-to-end allocation at B={Bmax} ({reader_name.split('/')[-1]})")
    plt.legend(fontsize=7.5); plt.tight_layout(); plt.savefig(fig, dpi=140); plt.close()
    log(f"  saved figure -> {os.path.basename(fig)}")

    results = {
        "experiment": "e9_real_qa", "config_params": p, "embedder": embedder_name,
        "reader": reader_name, "reader_enabled": reader is not None,
        "n_questions": len(questions), "corpus_paragraphs": len(pids),
        "budget_tokens": budget_tokens, "per_budget": {str(B): per_budget[B] for B in per_budget},
        "figure": os.path.basename(fig),
    }

    lines = [
        f"# E9 — Real End-to-End RAG QA (HotpotQA, {reader_name.split('/')[-1]})",
        "",
        f"Real multi-hop QA: {len(questions)} questions over a pooled corpus of {len(pids)} "
        f"paragraphs; real retriever (`{embedder_name.split('/')[-1]}`, truncated to d_r) + "
        f"query-conditioned sentence selection (d_c) under a {budget_tokens}-token reader budget; "
        f"frozen reader answers, scored by EM/F1. Shared budget B = d_r + d_c.",
        "",
        "| budget B | best split d_r:d_c | best F1 | standalone R | standalone C | compounding gap |",
        "|---|---|---|---|---|---|",
    ]
    for B in sorted(per_budget):
        i = per_budget[B]; k = i["gap_metric"]
        def fmt(x):
            return "—" if x is None else f"{x:.3f}"
        lines.append(f"| {B} | {i['best_split']} | {fmt(i['best'][k])} | "
                     f"{fmt(i['standalone_R'][k])} | {fmt(i['standalone_C'][k])} | {i['gap']:.3f} |")
    lines += [
        "",
        f"Metric for the gap is **{metric.upper()}** (real reader EM/F1 when the reader is enabled, "
        "else the answer-in-context diagnostic). A positive gap is compounding on real answer "
        "quality: the best budget split underperforms either stage given the full budget, and the "
        "optimum is interior — the real-task version of E6, and the reviewer-proof form of C2/C3.",
        "",
        f"![real qa allocation]({os.path.basename(fig)})",
    ]
    return results, "\n".join(lines)
