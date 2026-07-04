# Research Log — The Compounding Bottleneck

Plain-English narrative of the **significant** stages and findings. Not a play-by-play —
just moments worth remembering, what they mean, and where they point. Newest entries at
the bottom. (Detailed claim tracking lives in `findings.md`.)

---

## 2026-06-30 — Project kickoff & topic locked

Settled the thesis after a literature sweep: efficient RAG stacks **two** fixed-dimensional
vector bottlenecks in series — dense retrieval (`d_r`) and soft-context compression
(`m·d_c`) — and we suspect they *compound* rather than add. The retrieval-limit camp
(Weller et al.) and the compression-overflow camp have never been connected; that bridge,
plus a budget-allocation law and a multi-view ("facet-lens") escape, is the paper. Full
spec in `CLAUDE.md`. Headline target is contribution **C2** (a compounding / failure-transfer
finding), which must be protected from scope creep.

## 2026-06-30 — Harness built and the retrieval wall reproduced (E1 / E1b)

**What we did.** Built the measurement primitive everything else depends on: a
*free-embedding* capacity optimizer. For a target relevance pattern, it directly optimizes
the document and query vectors (no encoder, no text) to realize the pattern under
inner-product top-k retrieval. Because the vectors are unconstrained, the result is the
**best case for any encoder** of that dimension — if free vectors of dimension `d` can't
realize the pattern, no real model can. We used the adversarial "all-pairs" pattern (every
document pair is some query's gold set, k=2).

**What we found (real numbers, CPU, 2–3 seeds).**
- There is a **sharp capacity phase transition**, not a gentle slope. For a 40-doc corpus,
  realizability is ~0.38 at d=4 and snaps to 1.000 by d=8.
- The **critical dimension d\* grows with corpus size** — the wall. At d=8 realizability
  decays `1.000 → 0.993 → 0.978 → 0.974` as the corpus grows `40 → 80 → 160 → 320`,
  cracking below our 0.99 bar around n_d≈160. So d\* moves 8 → 16 as the corpus grows.
- In the **sub-critical regime** (d below the wall) realizability decays predictably:
  at d=4 it roughly **halves every time the corpus doubles** (`0.38 → 0.15 → 0.07 → 0.04`),
  i.e. capacity scales about like `d / n_d`.

**Why it matters.** Two things. (1) The harness is validated — it independently reproduces
Weller et al.'s embedding wall with our own code, so we can trust it for the stages that
have no published baseline. (2) The clean, near-quantitative scaling (sharp threshold +
`~d/n_d` sub-critical decay) is encouraging for getting an actual *law* out of the
compression stage, not just a vibe.

**Probable next steps.**
1. **E2 — compression capacity.** Reuse the exact same primitive on the *compression* side:
   treat "can a generator recover the answer-relevant fact from `m` soft tokens of dim
   `d_c`, among distractors" as a second realizability problem, and measure its wall. The
   bet is that it has the *same geometric form* as retrieval (that's contribution C1).
2. Keep E2 first as a **pure free-vector** version (no LLM) to isolate the geometry, then
   swap in a real small soft-compressor (xRAG/GIST-style) to confirm the effect survives
   contact with a trained model.
3. Only after E1+E2 are both clean do we compose them for **E3** (the compounding headline).

## 2026-06-30 — Compression has its own wall, and a surprising shape effect (E2)

First experiment run by the **autonomous loop end-to-end**: I pushed code+config only,
Kaggle ran it on GPU (device=cuda, ~6 min), and pushed results back — the pipeline works.

**What we did.** Built the compression analog of the E1 primitive: model soft tokens as
attention-read slot memory (m slots × d_c dims, total code `D_c = m·d_c`), store n_f
key→value facts per "passage", and measure best-case associative recall. This is the C1
test: does compression have a capacity wall of the *same form* as retrieval?

**Finding 1 — yes, compression has a wall (C1 supported).** Critical code size `D_c*`
(recall ≥ 0.95) grows ~linearly with content: `{n_f=16 → 32, 32 → 64, 64 → 128}` — a clean
doubling, i.e. `D_c* ≈ 2·n_f` (n_f=128 is borderline at the 0.95 bar; by strict
all-facts-perfect recall it needs ~256). 3 seeds, low variance. So compression is a genuine
*second* fixed-dimensional bottleneck with the same geometry as retrieval. One nuance worth
keeping: the compression transition is **softer/more graded** than retrieval's sharp
phase-snap in E1 — same scaling law, gentler knee.

**Finding 2 — capacity is NOT shape-invariant (surprising, and useful).** At a *fixed*
budget D_c=128, recall depends strongly on how you split it into (slots m × width d_c). It
peaks at an interior point (m≈4–16, recall ~1.0) and **collapses to near-chance at both
extremes**: one fat token (m=1, d_c=128 → 0.31) and many thin tokens (m=128, d_c=1 → 0.36)
both fail. Mechanistically, m=1 gives attention nothing to select among (softmax over one
slot is a no-op), so a single soft token can't support content-based read-out of many
facts; thin slots can't be told apart. **Implication:** popular single-token compressors
(xRAG-style) sit at the bad end of this curve for multi-fact passages — this is a concrete,
publishable observation and the empirical seed of the allocation law (C3).

**Honest caveat.** The m=1 collapse is probably *amplified* by our toy reader, whose
attention is over slots only. In a real LLM, soft tokens are attended jointly with the
query tokens, so m=1 is less degenerate there. The interior optimum and the thin-slot
collapse are the robust parts; the exact low-m behaviour must be re-checked with a real
compressor later. (Logged so we don't over-claim.)

**Probable next steps.**
1. **E3 — the headline (C2), designed as a shared-budget composition.** Give retrieval and
   compression a *single* total budget `B`. Standalone retrieval with dim=B succeeds;
   standalone compression with D_c=B succeeds; but the pipeline must *split* B into
   `d_r + D_c`, and if the retrieval-relevant and answer-relevant features are misaligned,
   neither stage gets enough → the pipeline fails on the *union* of each stage's hard items.
   Compounding = `best_split recall < min(standalone_R, standalone_C)`. This makes C2 (the
   compounding finding) and C3 (the optimal split) fall out of one experiment.
2. Carry the E2 split-invariance forward: the interior optimum is the C3 frontier in miniature.

## 2026-06-30 — The loop caught a GPU-only bug (process note)

E3's first Kaggle run failed fast: a latent device bug in the retrieval primitive (a mask
built on CPU while indices were on CUDA). It hid because E1 was run *locally on CPU* and E2
uses different code, so E3 was the first time `fit_capacity` touched a real GPU. Worth
noting for two reasons: (1) the autonomous loop behaved exactly as designed — it pushed a
`FAILED.txt` with the traceback instead of silently stalling, so I could diagnose it next
turn; (2) the lesson — I can't run CUDA locally, so device-correctness must be by
construction. Audited every tensor-creation site; this was the only offender. Fixed,
hardened the runner to clear stale failure markers, re-queued E3.

## 2026-06-30 — The headline holds: budgets compound, and an allocation law (E3)

The composed retrieve-then-compress pipeline under a shared budget B = d_r + D_c, run on
GPU. Numbers verified by hand against the E1/E2 curves (retrieval wall at d=48, compression
ramps slowly to ~1.0 by D_c~160).

**Finding — compounding is real and largest when you most want efficiency.** At a tight
budget B=64, each stage *alone* given the full B succeeds (retrieval 1.00, compression
0.95), yet the best budget-split pipeline reaches only **0.559** — a **0.39 compounding
gap**. The gap then shrinks monotonically as the budget loosens: 0.39 (B=64) → 0.11 (96) →
0.03 (128) → 0.008 (160) → 0.001 (192). So the two fixed-d bottlenecks compound precisely
in the tight-budget regime that compression exists to serve; with ample budget they stop
interfering. This is the C2 statement (independent-error version).

**Finding — a crisp allocation law (C3).** The optimal split is not 50/50 and not
"everything to the embedder." It is **fund retrieval exactly to its wall, pour the rest
into compression**: d_r* = 48 (= the retrieval wall) for every B ≥ 96, with D_c* = B − 48.
Below the wall (B=64) it's forced to 32:32 and both stages starve. This directly indicts
the deployed habit (embedding dim ~768–1024 — far past its wall, wasted — plus a 1–8 token
compressor — starved): real systems sit on the wrong side of this frontier.

**Honest limit of v1.** This uses the *optimistic* independent-error composition
(pipeline = recall_R · recall_C). It already shows compounding, but a reviewer can call
budget-splitting "expected." The genuinely surprising claim — that misaligned bottlenecks
compound **super-multiplicatively** (pipeline < recall_R · recall_C) — needs correlated
example hardness. That's the next experiment.

**Next — E3b (the non-obvious core).** Faithful two-bottleneck architecture, but items have
*anti-correlated* hardness: some are retrieval-easy/answer-hard, others
retrieval-hard/answer-easy. Each stage's marginal recall stays high (each handles its easy
half), but the pipeline needs BOTH, so it fails on the union → end-to-end falls *below*
recall_R · recall_C. If that holds, the compounding is a real interference effect, not just
budget division.

## 2026-06-30 — Compounding has a sign, and the corpus sets it (E3b)

Held the real E1/E2 marginals fixed at tight operating points and swept the
retrieval↔compression hardness correlation rho via a Gaussian copula (copula math checked
analytically: rho=0 reproduces the product; rho=−1 hits the Fréchet floor).

**Finding — the compounding sign is conditional.** At a balanced, sub-saturated point
(p_R=0.78, p_C=0.72), anti-correlated hardness (rho=−0.5) drives the pipeline *below* the
independent product (0.559 → 0.515; full anti-correlation floor 0.497) — super-multiplicative.
Aligned hardness (rho>0) lifts it *above* the product (→0.687) — failures become redundant.
The effect is largest when both stages are balanced and sub-saturated, and vanishes when
either saturates (32:64, p_C=0.95, barely moves). So we do **not** claim "always worse":
whether the two bottlenecks compound super- or sub-multiplicatively is set by the sign of
hardness correlation — a property of the corpus + the chosen retriever/compressor. The
sharper, honest version of C2, and it raises a concrete empirical question: **what is rho
for real systems?**

**Decision — pivot to real models.** The free-vector arc (E1 wall, E2 wall + shape, E3
budget compounding + allocation, E3b dependence) is complete and internally consistent — a
clean best-case-geometry theory, but a toy on its own. Highest-value next step: show the
walls and compounding survive **trained models**, and measure rho in the wild. Start with
the cheapest real-model test — the retrieval wall with a real (Matryoshka) embedder,
truncated across dimensions on a hard retrieval set. No training needed.

## 2026-06-30 — The wall is real: a production embedder wastes half its dimensions (E4)

First real-model experiment, on GPU. Embedded full SciFact (5,183 docs, 300 queries) with
**mxbai-embed-large-v1 (1024-d, Matryoshka)** and truncated across dimensions.

**Finding — real embedders inherit the wall (F9, retrieval side).** recall@10 ramps
0.09 (d=8) → 0.57 (32) → 0.80 (128) → 0.83 (256) → **0.872 (512) → 0.872 (1024)**. The curve
saturates: the top half of the embedding (dims 512–1024) adds *exactly nothing* on this
corpus (0.872 = 0.872). So the free-vector wall of E1 is not a toy artifact — a strong
production embedder shows the same ramp-then-plateau, and here ~half its representational
budget is dead weight. Per the allocation law (C3), that budget belongs in compression.
This is a clean, quotable real-model anchor for the paper.

**Caveat / next.** One model, one corpus; need ≥1 more embedder and a harder set (LIMIT,
Weller's adversarial set, where the wall should bite far sooner) to generalize. But the
direction is unambiguous.

**Next — E5, the harder half: a real compression wall.** Mirror E2 with an actual soft-
prompt compressor on a small LLM (mean-pool the passage hidden states into m soft tokens +
a trained projector, frozen-ish decoder), and measure answer accuracy vs m. This is the
first *training* experiment, so it will lean on the Kaggle GPU properly. If a real
compressor shows the same D_c wall, both bottlenecks are validated on real models and the
empirical core of the paper is in place.

## 2026-06-30 — A 1024-d embedding holds ~one fact: the real compression wall (E5)

Two parts. **(1) A genuine negative result.** The faithful generative soft-token compressor
(soft_prompt.py) would not leave chance: a frozen LLM can't cheaply learn to read novel soft
tokens — LoRA, identity-init, and higher lr all left it at ~chance in a feasible budget. That
is *why* GIST/ICAE/xRAG train extensively; brute-forcing it on Kaggle wasn't worth the quota.
Kept as documented future work, config not queued. **(2) The reliable route (E5 probe).**
Compress a multi-fact passage to one frozen mxbai-large embedding, truncate to D_c, recover a
queried key's value with a light probe on held-out passages.

**Finding — single-vector compression capacity is tiny, and collapses with content.** Full
(1024-d) held-out recall: **0.98 (1 fact) → 0.53 (2) → 0.32 (4) → 0.24 (8) → 0.20 (16)**, with
chance 0.125. A SOTA 1024-dimensional embedding retrievably stores barely more than a single
fact; by ~16 facts it is near chance. The D_c ramp is clean where there is signal (n_f=1:
critical D_c=8). This is the real-model echo of E2 **and** a direct confirmation of F6 (a
single fat token collapses) and the granularity dilemma — now with a production encoder.

**Scope note.** This is deliberately the m=1 (one vector) regime — F6's worst case. A
multi-vector real compressor (m>1) should recover far more and, per F6, exhibit an interior
optimum; building that (and composing E4+E5 for real-model compounding) is the natural next
real-model step.

**Milestone.** Both fixed-d bottlenecks are now validated on real models: retrieval (E4, a
production embedder wastes half its dims) and compression (E5, one vector ≈ one fact). On top
of the free-vector theory (E1–E3b) the empirical core of the paper is in place. The remaining
big build is the proposed *method* — the multi-view "facet-lens" escape (C4).

## 2026-06-30 — Multi-token compression recovers capacity; F6's interior optimum is mechanism-specific (E5b)

Extended the real compressor to m chunk-embeddings read by an **attention** probe (a plain
concat-MLP failed at slot lookup — the read mechanism matters as much as the budget).

**Finding 1 — capacity recovers, strongly.** Giving each fact its own slot (m=n_f) with an
adequate per-slot width reaches near-perfect held-out recall: **0.96 (n_f=4) → 0.98 (8) →
0.99 (16) → 0.99 (32)** on mxbai. The single vector (E5) held ~1 fact (0.20 at n_f=16);
m tokens hold them all. The wall is clean: usable code ≈ n_f × ~64 dims/fact. So the fix for
single-vector collapse is to spread the budget across several soft tokens — but each needs a
width floor.

**Finding 2 — an honest boundary on F6.** At a *tight* fixed budget (D_c=64, n_f=32), recall
is **monotonic in m**: m=1 best (0.21), decaying to chance by m=32 (d_c=2). Only F6's
thin-slot collapse reproduces; the "one fat token collapses" half does **not** for real-
encoder truncation. So F6's interior optimum is **specific to E2's attention-read *learned*
soft tokens** (where a lone slot is a degenerate attention target), not a universal law of
compression. (The experiment's auto-summary mislabels best-m=1 as an "interior optimum" — it
is not; recording the correct reading here.) Net refinement for C3: the m-vs-d_c split has a
per-slot width floor; buy more slots only while each stays above it.

**Status.** Real-model evidence is now rich: retrieval wall (E4), single-vector collapse
(E5), multi-token recovery + width floor (E5b). Next: **E6**, a real end-to-end allocation —
retrieve the gold passage among many (budget d_r) then compress it (budget D_c) on one
synthetic corpus with unique per-passage topics, and show the compounding gap + optimal split
on real components. Then the C4 facet-lens method.

## 2026-06-30 — Capstone: compounding + allocation on a REAL end-to-end pipeline (E6)

One corpus (N=2000 passages, similar topics, n_f=8 facts), real mxbai retrieval (paraphrase
queries, truncated topic embeddings) + real multi-token compression sharing B = d_r + D_c,
composed per query.

**Finding — the compounding gap is large and real, the optimal split interior.** At B=128,
each stage alone scores 0.63 (retrieval) / 0.89 (compression), but the best budget split
yields only **0.234** end-to-end — a **0.39 compounding gap**; the optimum is 64:64, not all
to one stage. At B=256 the gap is 0.137 (best 128:128). Retrieval is the binding bottleneck
here — matching paraphrases against 2000 near-duplicate topics is genuinely hard, so starving
it under a tight budget hurts most. This is E3's synthetic result reproduced on real
embeddings and a single task: the deployed big-embedder/tiny-compressor habit is off the
frontier.

**Milestone — empirical core complete.** C1 (walls, both stages: E1/E2 synthetic, E4/E5/E5b
real), C2 (compounding: E3/E3b synthetic, E6 real), C3 (allocation: E3 synthetic, E6 real) are
all validated, synthetic→real. Paper outline in `paper/outline.md`. Remaining is optional
elevation (C4 facet-lens as an explicit method — E5b already partially demos the multi-view
escape) and write-up. Minor: E6 standalone at B=384 is blank only because 384 isn't on the
measurement grid; cosmetic.

## 2026-06-30 — The escape, and what actually causes it (C4)

Built the proposed method: split a fixed embedding budget into K facet-specialized low-rank
lenses, route each query to its facet's lens. Compared against a single full-rank vector and
a generic K-view MaxSim (ColBERT-like, no routing) at EQUAL budget, on a facet-structured
relevance pattern.

**Finding 1 — facet-lenses escape the single-vector wall.** Routed facet-lenses realize the
pattern at critical budget **d*=12 vs single d*=24** — half the budget. So the wall (E1/E4)
is not fundamental to the task, only to the single-vector *form*; specialization escapes it.

**Finding 2 (the sharp one) — it's routing/specialization, not multi-vector per se.** Generic
MaxSim multi-view is **worse** than a single vector (d*=48 vs 24): naively splitting into K
views and max-combining doesn't help and can hurt (the non-smooth max is hard to optimize and
nothing tells a query which view to trust). Only when each lens *specializes* on a facet and
the query is *routed* to it does the escape appear. This distinguishes C4 from "multi-vector
beats single" (known): the lever is specialization + routing.

**Status — all four contributions demonstrated.** C1 (walls), C2 (compounding), C3
(allocation), C4 (escape), each on controlled + (for C1–C3) real models. The paper is
complete in skeleton: theory/diagnosis (a problem), an actionable allocation (what to do
now), and a principled escape (how to do better). Remaining is write-up + generality
(2nd embedder, LIMIT) + optionally a real-encoder facet-lens.

## 2026-07-01 — Closing the generality gaps: 4 new experiments queued for Kaggle

Goal this session: finish ALL remaining experimental execution so the paper is write-ready.
Built four new experiment families, validated each locally (CPU + a real small encoder,
all-MiniLM-L6-v2), and queued the canonical/strong-encoder versions for the Kaggle GPU loop.

**E8 — LIMIT adversarial wall (canonical real data for C1).** Weller et al.'s LIMIT realizes
our all-pairs k=2 pattern in natural language (46 docs, 1000 queries, exactly 2 gold/query).
Smoke (MiniLM, LIMIT-small): **recall@10 = 0.48** — i.e. even allowed to return 10 of just 46
docs (>20% of the corpus), it recovers under half the gold. Truncation makes it worse
(R@10 0.39→0.32 as d 128→32). This is the real-data, real-model echo of E1, on the field's own
adversarial benchmark. Kaggle will run 3 embedder families × {small, full 50k} for the headline.

**E4 generality — 2nd embedder + a 2nd dataset + PCA truncation.** Queued arctic-embed-m-v1.5
(a 2nd MRL embedder) on SciFact, mxbai on FiQA (domain shift), and bge-large under *PCA*
truncation (optimal linear projection, not Matryoshka — preempts "your wall is an MRL
artifact"). If the truncation wall replicates across all, C1 is not mxbai/SciFact-specific.

**E7 — measured hardness correlation ρ (closes F10).** On E6's shared real pipeline, recorded
per-query retrieval margin and compression logit margin. Smoke (MiniLM): **mean ρ_phi ≈ −0.01 ≈ 0**;
the E3b copula predicts the observed pipeline recall within ~0.006 at every operating point.
So a real retriever+compressor defaults to the **≈multiplicative (independent)** regime — NOT
super-multiplicative. This is good news for C3 (clean p_R·p_C allocation) and frames the sharp
open question: can adversarial inputs push ρ negative? Kaggle re-runs on mxbai for the paper number.

**C4b — real-encoder facet-lens (the honest scope of C4).** Ported the facet-lens to learned
low-rank projections over a FROZEN real encoder. Key result (MiniLM, 10 facets): a single
*optimal learned* projection WINS at low/moderate budgets (it can unmix linearly-separable
facets) but **plateaus at a sign-rank ceiling** (~0.90 mAP, barely moving d 80→160), while routed
facetlens keeps climbing and overtakes only at high budget (0.916 vs 0.899 at d=160). Generic
multiview is consistently worst (replicates C4's free-vector finding). **Honest takeaway:** the
free-vector escape (C4: d*=12 vs 24) is real but its *useful-budget* advantage does NOT transfer
to strong real encoders — multi-view routing is not a free lunch. This scopes C4 precisely and
fits the paper's C1+C2+C3 spine (C4 as a "when does the escape help?" section). Kaggle runs mxbai.

Also: E1c (F3 scaling fit) smoke gives a clean **d\* ∝ √n_d** (power-law exponent b=0.50,
R²=0.81) on free vectors — a quotable law that makes the C3 allocation recipe concrete.

Net: pushing 8 configs (c4b, e1c, e4b/c/d, e7, e8 small/full). After Kaggle returns results,
finalize findings F15–F18 and the paper is write-ready.

## 2026-07-01 (later) — All 8 Kaggle runs back: the generality program is complete

Every queued experiment returned. Headline outcomes, with the honest negatives kept:

**Generality of the retrieval wall (F15) — SUPPORTED on three independent axes.**
- 2nd embedder: arctic-embed-m (768d) on SciFact ramps recall@10 0.04→0.82, ~97% of full by d=256.
- 2nd dataset: mxbai on FiQA (domain shift) ramps 0.03→0.65, 92% of full by d=256.
- 2nd truncation method: bge-large under **PCA** (optimal linear projection, not Matryoshka)
  saturates at **d=256** (0.847 = 98.5% of full; the top 768 dims add 0.013). This is the
  cleanest wall and it kills the "your wall is just an MRL ordering artifact" objection dead.

**Canonical adversarial data (F16) — SUPPORTED, 3 embedder families.** On Weller et al.'s full
LIMIT (50k docs, 2 gold/query), recall@100 = mxbai 2.8%, arctic 8.2%, bge 4.5% — all
catastrophic, matching the paper's "<20%" for frontier models (our smaller models do worse).
That three different families fail consistently shows it is the single-vector *paradigm's*
limit, not a model quirk. This is our strongest single piece of C1 evidence: real models, real
adversarial data, the field's own benchmark.

**Real ρ (F17, closes F10) — SUPPORTED.** mxbai retriever+compressor on the shared corpus:
mean ρ_phi = +0.009 ≈ 0. The E3b copula reproduces observed pipeline recall to ~0.001
(observed {0.234, 0.443, 0.555} vs product {0.227, 0.439, 0.555}). So real RAG defaults to the
**multiplicative** regime — compounding is real and equals p_R·p_C, which is exactly the case
the allocation law (C3) is built on. Super-multiplicative compounding needs anti-correlated
hardness, which we do not see by default; whether adversarial inputs induce it is left open.

**Real-encoder facet-lens (F18) — an honest NEGATIVE that sharpens the paper.** On mxbai (10
facets), a single *learned* low-rank projection ties routed facet-lenses (both reach mAP≥0.9 at
d=40) and actually *leads* at tight budgets (d=20: 0.735 vs 0.581); generic multiview is far
worse (d*=320). The free-vector escape (C4: d*=12 vs 24) is a worst-case sign-rank phenomenon —
it does **not** transfer to strong real encoders, because a powerful pretrained encoder has
already spent capacity linearizing the facets, so one projection extracts them and routing is
redundant. Takeaway for the paper: the practical lever is **allocation (C3)**, not multi-view.
C4 becomes a scoped "when does the escape help?" section; the spine is C1+C2+C3.

**Scaling fit (F3) — refuted; F2 reconfirmed.** The d/n_d sub-critical law does not hold
(R²=0.18). But d* does grow with corpus size (8→11 as n_d 40→400, both pattern families),
slowly (sub-log; b≈0.16–0.18) and damped by the fixed 1500-query cap thinning constraints.
We drop the closed-form claim and keep the qualitative one, which is all C3 needs.

**Status: the experimental program is COMPLETE.** C1 (walls) is now validated on free vectors,
three real embedder families, two datasets, PCA truncation, AND the canonical LIMIT set. C2
(compounding) and C3 (allocation) hold synthetically and end-to-end on real models, with ρ
measured (≈0, multiplicative). C4 (escape) is demonstrated in the worst case and honestly scoped
out of the real-encoder regime. The research is ready to be written.

## 2026-07-01 (later still) — E9: real HotpotQA RAG confirms compounding, exposes a grid gap

Closed the biggest gap flagged when comparing against the "What Survives Into Context" diagnostic
paper: ran C2/C3 on a REAL multi-hop benchmark with a REAL frozen reader and REAL EM/F1, not just
the synthetic E6 corpus. Built E9: HotpotQA questions pooled into one shared corpus (a genuine
retrieval problem, not per-question isolated), real embedder retrieval truncated to d_r, query-
conditioned sentence selection under a token budget (d_c) as the compression stage, Qwen2.5-1.5B
answers, scored by SQuAD EM/F1 + answer-in-context.

Hit two infra snags mid-run: (1) local git push started failing with 403 "permission denied" —
turned out to be a stale cached Windows credential (GitHub for Visual Studio entry) holding an
invalid token; user refreshed it and the push went through. (2) Worried the same stale-token issue
was silently killing the Kaggle watch loop's push-back (a real risk — the loop's `except: continue`
would hide it), but the console log the user pasted showed the run actively progressing (correct
HotpotQA sizes, correct embedding dims) — it was just slow first-run downloads, not a failure.

**Result (E9 pilot, 1 seed, n_q=300):** at B=128, best split 96:32 reaches F1=0.337 vs standalone
retrieval 0.387 / standalone compression 0.564 — a real **compounding gap of +0.050 F1**, i.e. C2
holds on real answer quality. At B=256 the gap is −0.011 (≈null), consistent with the theory that
compounding shrinks as budget grows (matches E3/E6's own direction). But: F1 climbs monotonically
to d_r=96 — the largest d_r in the tested grid — at BOTH budgets, so we have not actually located
where retrieval budget stops paying off. C3's "interior optimum" claim, demonstrated cleanly on
synthetic E6, is NOT yet demonstrated on real QA — an honest gap, not a result to paper over.

**Decision:** widen the d_r grid to 224 (well past the point where it might turn over) and add
seeds {0,1,2} for real variance, since the pilot took only 351s on a T4 — three widened runs cost
under 30 minutes of GPU time, trivial against the weekly budget. Queued as e9b_real_qa_seed{0,1,2}.
Recorded the pilot honestly as F19 (PILOT status) rather than waiting to report anything.

Next: once E9b lands, finalize F19, fold the real EM/F1 numbers into the paper's compounding (§5)
and allocation (§6) sections, and reassess whether the real-QA gap fully closes the comparison
against the diagnostic paper's rigor bar.

## 2026-07-01 (E9b seed0) — the interior allocation optimum DOES appear on real QA

After a Kaggle kernel stall on the Qwen weight-load (user restarted; idempotency meant a clean
re-pick-up), E9b seed0 (widened d_r grid to 224) landed and resolved the open question from the E9
pilot. On real HotpotQA EM/F1 the allocation curve (F1 vs d_r at fixed B) is now clearly
single-peaked with an INTERIOR maximum, not monotone: at B=256 F1 rises to 0.391 at d_r=96 and then
declines steadily along a long tail to 0.326 at d_r=224 — pouring more budget into retrieval past
the optimum HURTS, because it starves the compression stage. At B=128 the peak is d_r=80 (F1=0.345).
So C3 (interior budget optimum) now holds on real answer quality, not just the synthetic E6.
Compounding gap: +0.042 F1 at B=128 (best split below the weaker standalone), ~null (−0.011) at
B=256 — the same shrink-with-budget pattern as E3/E6 and consistent with the measured rho~0 (E7).

Holding the findings/paper finalization for the full 3-seed aggregate (seeds 1,2 running) to hit
the >=3-seed rigor bar; seed0 alone already answers the qualitative question (optimum is interior).

## 2026-07-01 (E9c) — real-QA compounding is reader-robust (3B reader), + infra fixes

Ran the reader-scale robustness check I flagged as the top reviewer question: same E9b pipeline,
stronger Qwen2.5-3B reader (vs 1.5B). Two infra snags first: (1) switched to Colab briefly (Kaggle
issues), which exposed that newer `datasets` versions reject the bare "hotpot_qa" repo id (HfUriError)
— fixed the loader to try the namespaced "hotpotqa/hotpot_qa" first (robust across both platforms);
(2) added an unbuffered-output fix to both watch loops so long model downloads stop *looking* hung.
Kaggle came back and ran E9c after picking up the fix.

**Result (E9c seed0, 3B reader):** the compounding gap PERSISTS and is if anything slightly larger:
+0.053 F1 at B=128 (vs 1.5B's +0.048±0.007) and +0.024 at B=256 (vs +0.009). The allocation curve
is still single-peaked (interior optimum). So the effect is reader-robust — it's an INFORMATION
bottleneck (evidence discarded at the two stages can't be recovered by a better reader), not a
reasoning one. Honest wrinkle: the 3B reader's *absolute* F1 is lower than 1.5B's (0.275 vs 0.335 at
B=128 best), but recall is IDENTICAL at every d_r (retrieval is reader-independent), so the pipeline
is unchanged up to the reader — the lower F1 is a span-F1-vs-verbosity artifact of the larger instruct
model, and it cancels out of the gap and the curve shape (both same-reader relative quantities).

Queued E9c seeds 1,2 for the 3-seed reader-scale claim to match E9b's rigor. Once in, add a
reader-scale robustness row to the paper (the gap survives 1.5B->3B) and finalize.

## 2026-07-01 (E9c 3-seed) — reader-robustness confirmed and finalized

All 3 E9c seeds (3B reader) in. The compounding gap PERSISTS and slightly GROWS with reader scale:
B=128 gap +0.058±0.005 F1 (vs 1.5B +0.048±0.007), B=256 +0.036±0.018 (vs 1.5B's null +0.009), with
the interior optimum at d_r=80 in all 3 seeds (rock-stable). So the effect strengthens, not weakens,
with a better reader — the clean signature of an information bottleneck (a stronger reader extracts
more from good context, making the penalty for evidence lost at the two stages MORE visible). This
is the opposite of a context-cleanup heuristic whose benefit a strong reader would absorb, which is a
nice distinguishing property. Added F20 (SUPPORTED, 2 reader scales × 3 seeds) and a compact
reader-scale sentence to the paper's section 6. The real-QA arc (E9/E9b/E9c) is now complete and
rigorous: compounding + interior optimum on real HotpotQA EM/F1, robust across 1.5B and 3B readers.

## 2026-07-01 (E9d 3-seed) — real-QA compounding is cross-dataset general (2Wiki)

Added a second multi-hop dataset (2WikiMultihopQA) to test whether the real-QA compounding is a
HotpotQA artifact. First hit an infra snag: E9d failed with CUDA "no kernel image is available for
the device" — NOT my code (2Wiki loaded fine on Kaggle, 3000 paragraphs). Root cause: the bootstrap's
`pip install -r requirements.txt` pinned torch>=2.1, which upgraded Kaggle's GPU-matched torch to a
generic PyPI wheel lacking kernels for the session's GPU. Fixed requirements to never install torch
(preinstalled + GPU-matched on Kaggle/Colab); user restarted the kernel and it ran clean.

**Result (E9d, 2Wiki, 3 seeds, 1.5B):** the compounding + interior optimum REPLICATE. B=128 gap
+0.048±0.014 F1 — matching HotpotQA's +0.048±0.007 essentially exactly — with a single-peaked
interior optimum (peak d_r≈96, declining tail), null gap at B=256. So the real-QA validation now
spans 2 datasets × 2 reader scales (HotpotQA 1.5B/3B + 2Wiki 1.5B), 3 seeds each, all with a
~+0.05 F1 tight-budget compounding gap and an interior allocation optimum. Added F21 (SUPPORTED),
a compact cross-dataset clause + 2 appendix rows (E9c/E9d) to the paper. Main text still 8pp.

The real-QA program (E9/E9b/E9c/E9d) is now comprehensively validated and, I think, closes the
gap vs the diagnostic paper's rigor bar: real benchmarks, real readers, EM/F1, multi-seed,
cross-dataset, cross-reader-scale, with the mechanism (information vs reasoning bottleneck) tested.

## 2026-07-04 (E9e 7B) — reader-scale trend complete: 1.5B→3B→7B, gap flat, no reversal

Added the third reader-scale point (Qwen2.5-7B in 4-bit NF4, since 7B fp16 ~15GB won't fit fp16 on
a single 16GB GPU; the gap is a same-reader relative quantity so quantization only shifts absolute
F1). Canary seed0 confirmed it fits + the gap persists; ran all 3 seeds. Full reader-scale trend,
compounding gap at the tight budget B=128 (3 seeds each): 1.5B +0.048±0.007, 3B +0.058±0.005, 7B
+0.058±0.017 — FLAT across a 5× reader-size range, no reversal, interior optimum at every scale.
This is the clean information-bottleneck signature and a meaningful contrast with the diagnostic
paper (whose packing edge REVERSES with reader scale). At B=256 the gap grows with scale
(+0.009→+0.036→+0.027): a stronger reader makes the discarded-evidence penalty more visible. 7B's
absolute F1 is highest (0.40–0.45), confirming the earlier 3B<1.5B absolute dip was a
span-F1-vs-verbosity artifact (recall identical at every d_r). Finalized F20, paper §6 (3-point
trend) + appendix. Real-QA validation is now: 2 datasets × 3 reader scales, 3 seeds each. This
comprehensively closes the reader-scale question; not doing 14B/MuSiQue (diminishing returns).
