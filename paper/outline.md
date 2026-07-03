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
  grows with corpus (E1c: 8→11 across n_d 40→400, sub-log). E4 (real): mxbai-large on SciFact
  plateaus by d≈512 — the top half of a 1024-d production embedding is wasted. **Generality
  (E4b/c/d):** the wall replicates on a 2nd MRL embedder (arctic, 97% of full recall by d=256),
  a 2nd dataset (FiQA, 92% by d=256), and under **PCA** truncation (bge, saturates at d=256 =
  98.5%) — so it is not a Matryoshka-ordering artifact. **Canonical (E8/LIMIT):** on Weller et
  al.'s 50k adversarial set, three embedder families all collapse to recall@100 ≈ 3–8% — the
  paradigm's wall on the field's own benchmark, the strongest single piece of C1 evidence.
- **4.2 Compression wall.** E2: slot-memory recall has the same-form wall; D_c* ≈ 2·n_f. E5
  (real): a single 1024-d embedding holds ~1 fact (0.98→0.20, n_f 1→16). E5b: multi-token
  compression recovers capacity (→0.99) with a per-slot width floor.
- **4.3 Shape effect (F6) and its scope.** E2 shows an interior-optimal slot count (one fat
  token and many thin tokens both collapse) — but it is *mechanism-specific* (attention-read
  learned soft tokens); real-encoder truncation shows only the thin-slot collapse (E5b). Honest.

## 5. Compounding (C2)
- **5.1 Theory / independent baseline.** Under independent stage errors, pipeline =
  recall_R(d_r)·recall_C(D_c); composing two sub-full recalls underperforms either at full B.
- **5.2 Dependence (E3b) + measured ρ (E7).** Sign of the deviation from the product is set by
  the retrieval↔compression hardness correlation: anti-correlated → super-multiplicative;
  aligned → redundant; bounded by Fréchet. **E7 measures it on a real pipeline: ρ_phi ≈ +0.009
  ≈ 0**, and the copula reproduces observed end-to-end recall to ~0.001 (product law). So real
  RAG defaults to the **multiplicative** regime — compounding is real and equals p_R·p_C; the
  super-multiplicative regime needs adversarial anti-correlation (left as the open question).
- **5.3 Real end-to-end (E6).** One corpus, real retrieval + real compression sharing B:
  compounding gap **0.392 at B=128** (stages 0.63/0.89 alone → pipeline 0.234), shrinking as B grows.

## 6. Budget-optimal allocation (C3)
- E3: optimal split = fund retrieval to its wall, pour the rest into compression; d_r* ≈ wall.
- E4+E5/E5b real anchors: production embedders waste dims past their wall while compression
  starves — deployed configs are off the frontier. E6: interior-optimal split on real models.
- Actionable recipe: measure each wall, allocate B accordingly; respect the per-slot width floor.

## 6b. The escape and its scope (C4 / C4b)
- **Free-vector (C4):** routed facet-specialized lenses escape the single-vector wall at half
  the budget (d*=12 vs 24); crucially it is *specialization + routing*, not multi-vector per se —
  generic MaxSim multiview is worse than a single vector (d*=48).
- **Real encoders (C4b) — honest negative:** the escape does NOT transfer. On mxbai (10 facets)
  a single *learned* low-rank projection ties routed lenses (both d*=40) and leads at tight
  budgets; generic multiview is far worse (d*=320). Mechanism: a strong pretrained encoder has
  already spent capacity linearizing the facets, so one projection extracts them and routing is
  redundant. **Takeaway: on real systems the lever is allocation (C3), not multi-view.** This
  scopes the method honestly and keeps the paper's spine C1+C2+C3.

## 7. Discussion / limitations
- Free-vector = best case; real encoders do worse. F6 scope. C4's escape is worst-case-only
  (C4b: no real-encoder transfer). Single-task synthetic corpora for controlled compounding;
  E7 measures ρ≈0 on a benign corpus but ρ under *adversarial* pressure is left open. Generative
  soft-token compression (F11) needs a large training budget — we use a probe to isolate capacity.

## 8. Reproducibility
All experiments are config-driven (`configs/`), seeded, run via `kaggle/run.py`; results in
`outputs/`. Autonomous Kaggle loop in `kaggle/`.

---

## Results table (all numbers final, from outputs/)
| Claim | Synthetic | Real model |
|---|---|---|
| Retrieval wall (C1) | E1 | E4 mxbai/SciFact; **E4b arctic/SciFact; E4c mxbai/FiQA; E4d bge/PCA**; **E8 LIMIT ×3 embedders** |
| Compression wall (C1) | E2 | E5 / E5b (mxbai) |
| Shape effect F6 (+scope) | E2 | E5b (boundary) |
| Compounding (C2) | E3, E3b | E6; E7 (ρ≈0 measured + copula validated); **E9/E9b real HotpotQA EM/F1 (+0.048±0.007 F1 gap); E9c reader-robust 3B (+0.058); E9d cross-dataset 2Wiki (+0.048)** |
| Allocation (C3) | E3 | E6; **E9b real HotpotQA interior optimum (3 seeds); replicated E9c/E9d** |
| Facet-lens escape (C4) | C4 (d*=12 vs 24; routing > generic multiview) | **C4b: does NOT transfer — single ties routed lenses (d*=40) on mxbai (scoped negative)** |

## Headline numbers (final, from outputs/)
- Retrieval wall: mxbai/SciFact recall@10 0.09→0.87, **d=512 ≡ d=1024** (top half wasted).
- **Generality:** arctic/SciFact 0.04→0.82 (~97% by d=256); mxbai/FiQA 0.03→0.65 (92% by d=256);
  **bge under PCA saturates at d=256 (98.5% of full)** — the wall is not an MRL-ordering artifact.
- **LIMIT (canonical adversarial, 50k docs, 2 gold/query):** recall@100 = **mxbai 2.8% / arctic
  8.2% / bge 4.5%** — three embedder families fail; the paradigm's wall on the field's own benchmark.
- Compression: single 1024-d vector holds ~1 fact (0.98→0.20); **multi-token recovers to 0.99**.
- Compounding (real, E6): **B=128 gap 0.392** (stages 0.63/0.89 alone → pipeline 0.234, opt 64:64).
- **Measured ρ (E7, mxbai): ρ_phi ≈ +0.009 ≈ 0**; observed pipeline = product law to ~0.001 →
  real RAG is **multiplicative**, so C3's allocation applies cleanly.
- **Facet-lens (C4b, mxbai): single & routed lenses tie at d*=40**; the free-vector escape does
  not transfer to strong real encoders (honest scope; the practical lever is allocation, not multi-view).
- **Real-QA (E9/E9b/E9c/E9d, HotpotQA+2Wiki):** compounding gap **~+0.05 F1 at tight budget** and an
  **interior allocation optimum** on real EM/F1 — robust across 2 datasets, 2 reader scales (1.5B/3B),
  3 seeds each; gap *grows* with reader scale (information-bottleneck signature, not reasoning).

## Status: experimental program COMPLETE. Paper drafted + compiles (paper/latex/iclr2026/main.tex,
## 8pp main text, 47 refs). Remaining = write-up refinement + submission prep (ICLR 2027, ~Sept 2026).
- [x] All four contributions validated on real models (C1 walls+generality+LIMIT; C2 compounding
      synthetic+ρ+real-QA; C3 allocation synthetic+real-QA; C4 escape + honest real-encoder scope).
- [x] Real-QA end-to-end EM/F1 validation (E9 family) — closes the "synthetic-only" gap.
- [x] LaTeX draft written, compiles clean (main.tex, 8pp), 47 verified references, figures ported.
- [ ] (polish) Publication-grade pass on the ported result figures; port the 2027 style files when
      ICLR releases them (currently on 2026 style — cosmetic year swap).
- [ ] (optional stretch) 3rd real-QA dataset (MuSiQue) or ρ-under-adversarial-pressure — diminishing
      returns; the program is already thorough. Present C1 as empirical characterization (no closed form).
