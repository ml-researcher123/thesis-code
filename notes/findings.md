# Findings Ledger

Every load-bearing claim with a status: `HYPOTHESIS → SUPPORTED / REFUTED / INCONCLUSIVE`,
plus the evidence pointer. A claim is only `SUPPORTED` with ≥3 seeds + variance + a baseline
(per the A* bar in `CLAUDE.md`). Negative results are logged here too — they are valuable.

| ID | Claim | Status | Evidence |
|----|-------|--------|----------|
| F1 | Free-embedding realizability of the all-pairs pattern has a **sharp** phase transition in `d`. | SUPPORTED | E1 (`outputs/e1_retrieval_capacity`): n_d=40 jumps 0.38→1.0 between d=4 and d=8, 3 seeds, ~0 variance. |
| F2 | The critical dimension `d*` **grows with corpus size** (the embedding wall). | SUPPORTED | E1b (`outputs/e1b_retrieval_capacity_extended`): d=8 realizability decays 1.000→0.974 as n_d 40→320; d* moves 8→16 by n_d≈160. |
| F3 | Sub-critical realizability scales ≈ `d / n_d` (halves when corpus doubles at fixed d). | INCONCLUSIVE | E1b d=4: 0.38→0.15→0.07→0.04 over n_d 40→320 is consistent, but only 2 seeds and one pattern; needs more seeds + a second pattern (random k-subsets) and a fitted curve. |
| F4 | Soft-token compression has a capacity wall of the **same geometric form** as retrieval; critical code `D_c* ≈ 2·n_f` (grows ~linearly with content). | SUPPORTED | E2 (`outputs/e2_compression_capacity`): D_c* = {16→32, 32→64, 64→128}, 3 seeds, low variance, GPU. Transition softer than E1's. |
| F6 | At fixed code budget D_c, capacity is **not shape-invariant**: an interior-optimal slot count exists; one fat token (m=1) and many thin tokens both collapse to ~chance. | SUPPORTED in E2 (attention-read learned soft tokens); does NOT generalize | E2 split probe: 0.31 (m=1) → 0.999 (m=4) → 0.36 (m=128). But E5b (real-encoder truncation) shows only the thin-slot collapse; m=1 stays best at tight budget → interior optimum is mechanism-specific (see F12). |
| F12 | Multi-token real compression recovers the capacity a single vector loses, with a per-slot width floor (thin slots collapse). | SUPPORTED | E5b (`outputs/e5b_real_compression_shape`): m=n_f reaches 0.96–0.99 recall for n_f=4–32 (vs single-vector 0.20); but at tight D_c=64/n_f=32, recall is monotonic in m (m=1 best, m=32→d_c=2 near chance). |
| F5 | Under a fixed shared budget B, the pipeline's best split underperforms either stage given the full B (compounding). | SUPPORTED (independent-composition) | E3 (`outputs/e3_compounding`, GPU, 3 seeds): B=64 gap=0.389 (each stage ≥0.95 alone, pipeline 0.559); gap →0.001 by B=192. Numbers verified vs E1/E2 curves. |
| F7 | Optimal allocation = fund retrieval to its wall, rest to compression: d_r* = retrieval wall (≈48), D_c* = B − d_r*. Deployed big-embedder/tiny-compressor is off-frontier. | SUPPORTED | E3: optimal split = 48:(B−48) for all B≥96; 32:32 only when B below 2× the wall. |
| F8 | Compounding sign is **conditional on hardness correlation**: anti-correlated → super-multiplicative (pipeline < p_R·p_C); aligned → redundant (> p_R·p_C); bounded by Fréchet. Largest when both stages balanced + sub-saturated. | SUPPORTED (model-conditional) | E3b (`outputs/e3b_dependence`): at 32:32, ρ=−0.5→0.515 vs indep 0.559 (floor 0.497); ρ=+0.9→0.687. Copula verified analytically. Marginals from real E1/E2 walls. |
| F9 | The free-vector walls (E1/E2) persist with **real models**, both sides. | SUPPORTED (both sides) | E4: mxbai-large on SciFact plateaus by d=512 (top half of dims wasted). E5 (`outputs/e5_real_compression_probe`): single mxbai-1024d embedding holds ~1 fact — full-dim recall 0.98→0.53→0.32→0.24→0.20 for n_f=1,2,4,8,16 (chance 0.125). |
| F11 | The faithful generative soft-token compressor does **not** leave chance without a large training budget (frozen LLM can't cheaply learn to read novel soft tokens). | SUPPORTED (negative) | E5 soft-prompt smokes: LoRA + identity-init + lr=2e-3 all ≈chance at ≤200 steps. Kept as future work; not the same as F9 (which the probe established). |
| F10 | Real RAG corpora have a measurable hardness-correlation ρ that determines their compounding regime. | OPEN | Needs real retriever+compressor per-item success on a shared benchmark. |

| F13 | Compounding (C2) + interior-optimal allocation (C3) hold **end-to-end on real models, one task**. | SUPPORTED | E6 (`outputs/e6_real_allocation`, mxbai, N=2000): B=128 gap=0.392 (stages alone 0.63/0.89, pipeline 0.234, optimal 64:64); B=256 gap=0.137. |
| F14 | Routed facet-specialized lenses **escape** the single-vector wall at equal budget; and it is specialization+routing, NOT multi-vector per se, that does it. | SUPPORTED | C4 (`outputs/c4_facetlens`): facetlens d*=12 vs single d*=24 (2x); generic MaxSim multiview d*=48 (worse than single). |

## Open threads / risks
- F3 wants a clean functional fit; if it's really `~d/n_d`, that's a quotable scaling and a
  strong setup for the allocation law (C3).
- Watch for anyone publishing F4/F5 first — see `related-work.md`.
