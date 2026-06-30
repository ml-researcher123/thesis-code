# CLAUDE.md — Autonomous Research Project Specification

> **You (Claude) are the principal researcher on this project.** This file is your
> standing brief. Read it at the start of every session. It defines the scientific
> goal, the theory you are proving, the experiments you must run, how the autonomous
> Kaggle execution loop works, and the conventions you must follow so that work
> compounds across sessions instead of restarting each time.

---

## 0. TL;DR

We are writing one **A\*-conference-grade paper** on a single, sharp thesis:

> **Modern "efficient RAG" stacks two fixed-dimensional vector bottlenecks in series —
> dense retrieval (embedding dimension `d_r`) and soft-context compression (`m` tokens
> of dimension `d_c`) — and these bottlenecks _compound_: the end-to-end system loses
> information that *either stage alone would have preserved*. This compounding is
> predictable from the geometry of the two representations, it implies an optimal
> budget allocation that deployed systems violate, and it can be escaped with a
> multi-view ("facet-lens") multi-agent representation.**

Working title: **"The Compounding Bottleneck: A Unified Capacity Theory of
Retrieve-then-Compress RAG, and a Multi-View Escape."**

Target venues (in priority order): **NeurIPS, ICML, ICLR, ACL/EMNLP** (main track).

---

## 1. Why this is novel (and defensible)

Three foundational 2025–2026 results each describe a *capacity ceiling* of fixed-dimensional
vectors, but **in isolation**:

| Work | What it shows | What it ignores |
|------|---------------|-----------------|
| Weller et al., *On the Theoretical Limitations of Embedding-Based Retrieval* (arXiv:2508.21038, DeepMind) | For embedding dim `d_r`, there exist relevant-document combinations that **no** single-vector model can return as top-k for any query. Introduces the **LIMIT** dataset; frontier models get <20% recall@100. | The *compression* stage that follows retrieval in efficient RAG. |
| *Detecting Overflow in Compressed Token Representations for RAG* (arXiv:2602.12235) | Soft-compression tokens "overflow" — a finite capacity past which passages cannot be encoded. | Only *detects* overflow empirically; never ties it to the retrieval bound or gives a capacity law. |
| *Scaling Laws for Embedding Dimension in IR* (arXiv:2602.05062); *SeleCom / Query-Conditioned Soft Compression* (arXiv:2602.15856) | Embedding-dim scaling for retrieval; info-theoretic argument for query-conditioned compression. | SeleCom **explicitly declines** to give a capacity bound or connect to the embedding-dimension wall; leaves "optimal compression ratios across tasks" and "multi-hop synthesis" open. |

**The unclaimed gap:** nobody has unified the *retrieval* bottleneck and the *compression*
bottleneck into one theory, shown that **composing** them is strictly worse than either
(a *compounding* failure, not an additive one), derived the **budget-allocation law**
between them, or proposed a representation that provably escapes both at once.

That is our paper. It is non-generic: the headline is a **surprising, theory-predicted
empirical finding** (failure-transfer / compounding), not "yet another RAG pipeline."

---

## 2. Contributions (the paper's spine)

The paper makes four contributions. Each maps to an experiment block in §5 and a
"definition of done" in §9.

**C1 — Unified capacity theory.**
Formalize soft-token compression as a *second* thresholded-separation (sign-rank–type)
problem of the **same geometric form** as Weller's retrieval bound. Prove the composed
retrieve-then-compress system has effective capacity
`c_eff ≤ min(c_retrieve, c_compress)`, with **strict** inequality generically, because
the retriever's discriminative subspace and the compressor's reconstructive subspace are
**misaligned**. (Submultiplicativity / subspace-intersection argument.)

**C2 — The compounding / failure-transfer finding (HEADLINE).**
Empirically exhibit and characterize inputs where (a) retrieval alone succeeds (gold docs
in top-k) **and** (b) compression of the *gold* docs alone succeeds (generator answers
from compressed gold), **yet** (c) the composed pipeline fails — and show these failures
are **predicted by the geometric capacity overlap**, not by either stage's standalone
metric. This breaks the field's implicit assumption that retrieval error and compression
error are independent/additive.

**C3 — Capacity-allocation law.**
Under a fixed representational budget `B` (e.g. FLOP- or parameter-weighted combination
of `d_r` and `m·d_c`), derive and empirically validate the optimal split. Show deployed
configurations (large `d_r ≈ 768–1024`, tiny `m ≈ 1–8`) sit **far from the Pareto
frontier**, and that re-allocating budget yields free accuracy at equal cost.

**C4 — Multi-View "Facet-Lens" escape (the method + multi-agent angle).**
A set of `K` low-rank **lens agents**, each specialized to a semantic facet
(entities, numerics, negation, relations, …), each performing its **own** retrieval and
compression, composed by a lightweight **router/coordinator**. Prove the union of `K`
thresholded low-rank subspaces realizes a strictly larger family of relevance/answer
patterns than a single `d_r`-subspace of equal total budget (effective-rank increase),
with a bound on the `K` needed for a target pattern family. Show empirical Pareto-dominance.

> A defensible paper can ship with **C1+C2+C4** as the core and **C3** as a strong
> section. C2 is the part reviewers will remember; protect it.

---

## 3. Formalism & notation (the theory you are building)

Keep this consistent across code, notes, and the paper.

- Corpus `D = {d_1, …, d_N}`; queries `q`; ground-truth relevance matrix
  `A ∈ {0,1}^{|Q|×N}` (row = query, 1 = relevant).
- **Retrieval stage:** doc encoder `φ: d ↦ x ∈ ℝ^{d_r}`, query encoder `ψ: q ↦ y ∈ ℝ^{d_r}`,
  score `s(q,d) = ⟨y, x⟩`, top-k by score. Feasibility of realizing `A` by thresholded
  `YXᵀ` is controlled by the **sign-rank** of `(2A − 1)` → this is Weller's wall.
- **Compression stage:** passage `P ↦ Z ∈ ℝ^{m×d_c}` (m soft tokens), generator `g`
  (frozen attention) answers from `Z`. Cast "can `g` recover the answer-relevant fact
  from `Z` among distractors" as a **second** thresholded-separation problem with its own
  capacity `c_compress`, governed by an analogous rank quantity on the *answer-selection*
  matrix.
- **Composition:** the second stage operates on the *output* of the first, so the
  realizable set is the **intersection** of two subspace-constrained families ⇒
  `c_eff ≤ min(·)`, strict under generic misalignment. Define the **failure-transfer set**
  `T = {x : retrieve(x)=✓ ∧ compress_gold(x)=✓ ∧ pipeline(x)=✗}` and predict `|T|` from
  the principal angles between the retriever and compressor subspaces.
- **Budget:** `B = α·d_r + β·(m·d_c)` (α, β from FLOP/param accounting in §5/E4).
- **Facet-lens:** `K` encoders `φ_1..φ_K` of rank `r = d_r/K` each; union feasibility via
  sign-rank of block/stacked construction; router `R: q ↦ Δ(K)` weights.

> If a clean closed-form bound for the compression stage proves too hard, the fallback is
> a **constructive + empirical** capacity characterization (measure the realizable-pattern
> count directly on synthetic corpora, §5/E2). The empirical capacity curves are
> sufficient to support C2–C4 even if C1 lands as a *characterization* rather than a
> closed-form theorem. **Do not let a stuck proof block the empirical program.**

---

## 4. Related work to track (keep `notes/related-work.md` current)

Retrieval limits: Weller et al. 2508.21038; Scaling-laws 2602.05062; granularity dilemma
/ CapRetrieval 2506.08592; negation/semantic-collapse 2603.17580.
Compression: overflow 2602.12235; SeleCom 2602.15856; AttnComp 2509.17486; CORE-RAG
2508.19282; ArcAligner 2601.05038; mean-pooling/multi-ratio 2510.20797; GRC 2605.09100;
xRAG, GIST, ICAE, 500xCompressor (classic soft-compression baselines).
Agentic/multi-agent RAG: MA-RAG 2505.20096; MARAG-R1 2510.27569; Search-R1; ReSearch;
RAG-Reasoning survey 2507.09477.
Multi-vector: ColBERT / ColBERTv2 (the known partial escape — **must** position against).

**Positioning rule:** every time a new arXiv paper appears that touches "compression
capacity," "embedding dimension limits," or "multi-vector escape," log it and write one
sentence on how we differ. The single biggest threat is someone publishing C1 or C2 first
— monitor weekly.

---

## 5. Experimental program

All experiments must be **cheap enough for Kaggle** (see §6), **seeded**, **multi-seed for
headline claims**, and report **compute (FLOPs/GPU-hours)** alongside accuracy.

**Models (small, frozen-or-LoRA):**
- Embedders: `e5-small/base`, `bge-small`, `gte-small`, `Qwen3-Embedding-0.6B`. Plus
  **trainable tiny linear/MLP encoders** on synthetic data for direct capacity probing.
- Soft compressors: re-implement xRAG-style (1-token), GIST-style, ICAE-style on a small
  decoder (`Qwen2.5-1.5B`, `Llama-3.2-1B`, or `Pythia-1.4B`) with **LoRA + trained
  projector, frozen generator**.
- Generators: `Qwen2.5-1.5B-Instruct` / `Llama-3.2-1B-Instruct` (scale to 3B only if quota
  allows).

**Datasets:**
- **LIMIT** (Weller) — adversarial retrieval capacity probe (small, ideal).
- **Synthetic controlled corpora** — *our own generator* that varies combinatorial
  complexity so we can measure capacity directly and validate the bound cleanly. This is
  the backbone of C1–C3.
- **CapRetrieval** — granularity stress.
- **BEIR** subsets (NQ, HotpotQA, FiQA, SciFact) — standard retrieval.
- **Multi-hop** (HotpotQA, 2WikiMultiHop, MuSiQue) — where compounding bites hardest.

**Experiment blocks:**
- **E1 — Retrieval capacity curve** vs `d_r` on synthetic + LIMIT (reproduce/extend Weller).
- **E2 — Compression capacity curve** vs `m·d_c`; locate the overflow point on controlled
  passages → validates the *new* compression capacity bound.
- **E3 — Composition / compounding (HEADLINE):** measure `c_eff`, demonstrate
  `c_eff < min(c_retrieve, c_compress)`, and exhibit + characterize the **failure-transfer
  set `T`**; predict `|T|` from subspace principal angles. This experiment is the paper.
- **E4 — Allocation sweep** over `(d_r, m, d_c)` at fixed budget `B`; map the empirical
  Pareto frontier; show deployed configs are off-frontier.
- **E5 — Facet-Lens method** vs single-view at equal total budget; Pareto curves; ablate
  `K`, facet definitions, router; analyze which lens fires when.
- **E6 — Generalization:** do trends hold across embedder families and 0.5B–3B generators?

**Metrics:** recall@k, nDCG@10, answer EM/F1, realizable-pattern count (capacity),
compression ratio, FLOPs, latency, principal-angle / subspace-overlap diagnostics.

**A\* bar (enforce on every headline result):** strong baselines (incl. ColBERT-style
multi-vector and SeleCom-style query-conditioned compression), ≥3 seeds with variance,
significance tests, full reproducibility (configs + seeds committed), honest compute
reporting, and at least one result that is *surprising given prior work* (that's C2).

---

## 6. Kaggle execution model (how runs actually happen)

**The loop (your hands are the keyboard; Kaggle is the GPU):**
1. You (Claude) write/modify code + an experiment **config** (`configs/*.yaml`) and
   **commit & push** to the GitHub repo (branch `research`, PRs into `main`).
2. A Kaggle notebook/kernel runs an **infinite watch loop**: it polls the repo, and on a
   new commit it `git pull`s, runs the **runner** (`kaggle/run.py`) against the queued
   config(s), then **writes results + logs back** to the repo's output area and pushes.
3. Next session, you **read `outputs/` and `notes/`**, interpret results, decide the next
   experiment, and repeat.

**Hard constraints to respect (verify exact current numbers, don't assume):**
- Free GPU quota is **limited per week** (≈30 GPU-hours; T4×2 or P100 16 GB). **Budget it.**
- A single session has a **wall-clock cap** (≈9–12 h). Every run must **checkpoint** and be
  **resumable**; never design a job that *must* exceed one session.
- **No persistent local disk** across sessions except (a) what's committed to the repo and
  (b) Kaggle Datasets / Hugging Face Hub. Large artifacts (model weights, indices,
  embeddings) go to **HF Hub or a Kaggle Dataset**, *not* git. Git holds **code, configs,
  small result JSON/CSV, figures, logs, notes.**
- Internet may need to be explicitly enabled in the kernel for `git`/`pip`/HF downloads.

**Design rules that follow from the above:**
- Every experiment = one self-contained, seeded, resumable job with a declarative config.
- Prefer **eval-only / frozen-model** experiments and **tiny LoRA** training. Avoid any
  full fine-tune that can't finish in one session.
- Cache embeddings/indices to HF/Kaggle Datasets and reuse across runs.
- Each run writes a machine-readable `results.json` **and** a human-readable `summary.md`.
- Make failure loud and logged; a crashed run must leave a diagnosis in `outputs/.../log`.

---

## 7. Repository structure

```
/CLAUDE.md                # this file — the standing brief
/README.md                # public-facing summary
/paper/                   # LaTeX source, figures, bibliography, drafts
/src/
  /theory/                # sign-rank/capacity estimators, synthetic-corpus generators
  /retrieval/             # embedders, indexing, retrieval eval
  /compression/           # xRAG/GIST/ICAE-style soft compressors + projectors
  /pipeline/              # retrieve-then-compress composition + diagnostics (T set, angles)
  /facetlens/             # multi-view lenses + router (the C4 method)
  /eval/                  # metrics, benchmark adapters, significance tests
/configs/                 # YAML experiment specs — the QUEUE the Kaggle loop consumes
/experiments/             # one dir per experiment run: config snapshot + results
/kaggle/                  # run.py (runner), watch-loop notebook, env/requirements
/data/                    # download/prepare scripts ONLY (no large data in git)
/outputs/                 # committed results.json, summary.md, logs, figures
/notes/                   # research-log.md, findings.md, related-work.md, decisions.md
```

---

## 8. Autonomous working conventions (read every session)

**State & continuity**
- `notes/research-log.md` — **append-only, dated** ledger. Every session: what you ran,
  what you found, what you decided next. This is how you remember across context resets.
- `notes/findings.md` — every empirical/theoretical claim with status:
  `HYPOTHESIS → SUPPORTED / REFUTED / INCONCLUSIVE`, with the evidence pointer.
- `notes/decisions.md` — irreversible or load-bearing choices and *why*.
- `notes/related-work.md` — the threat-monitor list from §4.

**Doing experiments**
- New experiment ⇒ new `configs/<id>.yaml` (seed, model, dataset, budget, hypothesis).
- Never overwrite results; new run ⇒ new `experiments/<id>/` and `outputs/<id>/`.
- A claim is only "supported" with **≥3 seeds + variance + a baseline**.
- **Record negative results.** A clean refutation of one of our hypotheses is valuable and
  must be logged, not buried. Scientific honesty is non-negotiable — never fabricate or
  smooth over numbers; report what the run actually produced, including failures.

**Git**
- Work on branch `research`; open PRs into `main`. Conventional, descriptive commits.
- Commit code + configs + small results + notes. **Never** commit large weights/indices.
- End commit bodies with the project co-author trailer as configured for this repo.

**Scope discipline**
- Protect the headline (C2). Don't drift into a generic "better RAG pipeline" paper.
- If blocked on the proof (C1), switch to the empirical capacity characterization and keep
  the experimental program moving (see §3 fallback).

---

## 9. Definition of done (per contribution)

- **C1 done:** either a proved `c_eff ≤ min(·)` submultiplicativity theorem **or** a
  rigorous empirical capacity characterization with the misalignment mechanism identified.
- **C2 done:** a reproducible, multi-seed demonstration of the failure-transfer set `T`
  on ≥2 datasets, with `|T|` correlated to the predicted subspace-overlap diagnostic
  (report the correlation + significance).
- **C3 done:** an allocation Pareto frontier on ≥2 datasets showing an interior optimum
  and quantifying how far standard configs sit from it.
- **C4 done:** Facet-Lens Pareto-dominates single-view at equal budget on ≥2 datasets,
  with `K`/facet/router ablations and a stated `K`-bound result.

**Paper done:** C1+C2+C4 solid (C3 strong), all claims reproducible from committed
configs, baselines include ColBERT-style multi-vector and SeleCom-style compression,
related-work threat-list clear, draft in `/paper`.

---

## 10. Immediate next steps (start here)

1. Scaffold the repo per §7 (empty modules + `kaggle/run.py` runner + watch-loop notebook
   + `requirements.txt`); commit on `research`.
2. Implement `src/theory/` synthetic-corpus generator and the capacity/sign-rank estimator.
3. Run **E1** (retrieval capacity curve) on synthetic + LIMIT to reproduce Weller's wall —
   this validates the measurement harness before we build on it.
4. Implement a minimal soft compressor (xRAG-style, 1 token) and run **E2**.
5. Compose and chase **E3** (the headline). Everything else follows.

> First action this session if asked to begin building: create the scaffold and the
> theory/synthetic harness, then verify the harness by reproducing the retrieval wall (E1).
