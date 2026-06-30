# Paper Outline — The Compounding Bottleneck

Working title: **"The Compounding Bottleneck: A Unified Capacity Theory of
Retrieve-then-Compress RAG, and a Budget-Optimal Allocation."**

Target: NeurIPS / ICML / ICLR / ACL main track. Status: empirical spine complete (E1–E6);
this outline maps results → sections and flags what still needs doing.

---

## Abstract (claim)
Efficient RAG funds two fixed-dimensional vector bottlenecks from one budget — the dense
retrieval embedding (d_r) and the soft-context compression code (D_c). We give a unified
capacity account showing each has a sharp wall whose critical dimension grows with content,
prove/measure that *composing* them **compounds** (the best budget split underperforms either
stage given the full budget), derive the **budget-optimal allocation**, and show real systems
(big embedder + tiny compressor) sit far off the frontier. Validated from controlled
free-vector geometry up to production encoders.

## 1. Introduction
- Efficient RAG = retrieve (embedding) → compress (soft tokens) → generate. Two fixed-d codes.
- Prior work bounds retrieval (Weller et al.) **or** studies compression overflow — never the
  composition. We unify them and quantify the compounding + the allocation.
- Contributions: **C1** unified capacity wall (both stages, same geometry); **C2** compounding;
  **C3** budget-optimal allocation; real-model validation end-to-end.

## 2. Related work
Retrieval limits (Weller 2508.21038; scaling-laws 2602.05062; granularity 2506.08592);
compression (overflow 2602.12235; SeleCom 2602.15856; GIST/ICAE/xRAG); multi-vector (ColBERT).
Positioning: nobody bounds the *composition* or derives the cross-stage allocation. (`notes/related-work.md`)

## 3. Setup & capacity primitive
Free-embedding optimization as a best-case capacity measure (an upper bound for any encoder);
sign-rank / thresholded-separation framing. (§3 of CLAUDE.md; `src/theory/`.)

## 4. The two walls (C1)
- **4.1 Retrieval wall.** E1: free-embedding realizability has a sharp phase transition; d*
  grows with corpus (8→16 as n_d 40→320). E4 (real): mxbai-large on SciFact plateaus by
  d≈512 — the top half of a 1024-d production embedding is wasted.
- **4.2 Compression wall.** E2: slot-memory recall has the same-form wall; D_c* ≈ 2·n_f. E5
  (real): a single 1024-d embedding holds ~1 fact (0.98→0.20, n_f 1→16). E5b: multi-token
  compression recovers capacity (→0.99) with a per-slot width floor.
- **4.3 Shape effect (F6) and its scope.** E2 shows an interior-optimal slot count (one fat
  token and many thin tokens both collapse) — but it is *mechanism-specific* (attention-read
  learned soft tokens); real-encoder truncation shows only the thin-slot collapse (E5b). Honest.

## 5. Compounding (C2)
- **5.1 Theory / independent baseline.** Under independent stage errors, pipeline =
  recall_R(d_r)·recall_C(D_c); composing two sub-full recalls underperforms either at full B.
- **5.2 Dependence (E3b).** Sign of the deviation from the product is set by the
  retrieval↔compression hardness correlation: anti-correlated → super-multiplicative;
  aligned → redundant. Bounded by Fréchet. Frames an empirical question (ρ in real corpora).
- **5.3 Real end-to-end (E6).** One corpus, real retrieval + real compression sharing B:
  large compounding gap at tight budget (≈0.35 at B=128 in smoke), shrinking as B grows.

## 6. Budget-optimal allocation (C3)
- E3: optimal split = fund retrieval to its wall, pour the rest into compression; d_r* ≈ wall.
- E4+E5/E5b real anchors: production embedders waste dims past their wall while compression
  starves — deployed configs are off the frontier. E6: interior-optimal split on real models.
- Actionable recipe: measure each wall, allocate B accordingly; respect the per-slot width floor.

## 7. Discussion / limitations
- Free-vector = best case; real encoders do worse. F6 scope. Single-task synthetic corpora for
  controlled compounding; ρ-in-the-wild left open. Generative soft-token compression (F11) needs
  a large training budget — we use a probe to isolate capacity.

## 8. Reproducibility
All experiments are config-driven (`configs/`), seeded, run via `kaggle/run.py`; results in
`outputs/`. Autonomous Kaggle loop in `kaggle/`.

---

## Results table (fill final mxbai numbers from outputs/)
| Claim | Synthetic | Real model |
|---|---|---|
| Retrieval wall (C1) | E1 | E4 (mxbai/SciFact) |
| Compression wall (C1) | E2 | E5 / E5b (mxbai) |
| Shape effect F6 (+scope) | E2 | E5b (boundary) |
| Compounding (C2) | E3, E3b | E6 |
| Allocation (C3) | E3 | E6 |

## Headline numbers (final, from outputs/)
- Retrieval wall: mxbai/SciFact recall@10 0.09→0.87, **d=512 ≡ d=1024** (top half wasted).
- Compression: single 1024-d vector holds ~1 fact (0.98→0.20); **multi-token recovers to 0.99**.
- Compounding (real, E6): **B=128 gap 0.392** (stages 0.63/0.89 alone → pipeline 0.234, opt 64:64).

## TODO before submission
- [x] E6 mxbai numbers in. Next: make the 4 headline figures publication-grade.
- [ ] A second embedder + a harder retrieval set (LIMIT) for the retrieval wall (generality).
- [ ] Formalize C1 bound (or present as rigorous empirical characterization + mechanism).
- [ ] (Stretch) C4 facet-lens method as an explicit *escape*; E5b is already a partial demo.
- [ ] (Stretch) measure ρ for a real retriever+compressor on a shared benchmark (F10).
- [ ] Write LaTeX in `paper/`; port figures from `outputs/`.
