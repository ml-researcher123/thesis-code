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
