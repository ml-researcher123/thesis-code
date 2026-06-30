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
