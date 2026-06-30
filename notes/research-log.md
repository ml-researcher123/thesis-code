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
