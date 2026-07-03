# Related Work & Threat Monitor

Positioning + a watch-list. The biggest risk to this paper is someone publishing our C1
(unified capacity bound) or C2 (compounding finding) first. Re-scan arXiv weekly for
"compression capacity", "embedding dimension limits", "multi-vector escape".

## Foundations we build on
- **Weller et al., On the Theoretical Limitations of Embedding-Based Retrieval**
  (arXiv:2508.21038, DeepMind). The retrieval wall; LIMIT dataset. We reproduce it (E1/E1b)
  and extend the methodology to the compression stage.
- **Detecting Overflow in Compressed Token Representations for RAG** (arXiv:2602.12235).
  Soft-token "overflow" — but empirical detection only, no capacity law, no tie to retrieval.
- **Granularity dilemma / CapRetrieval** (arXiv:2506.08592). Fine-grained matching failures.

## Nearest neighbors (must position against)
- **Scaling Laws for Embedding Dimension in IR** (arXiv:2602.05062). Retrieval dim scaling
  only; no compression; no cross-stage allocation. We add the second bottleneck + allocation.
- **SeleCom / Query-Conditioned Soft Compression** (arXiv:2602.15856). Info-theoretic MI
  argument; *explicitly declines* a capacity bound and the link to the embedding wall.
  Leaves "optimal compression ratio across tasks" + "multi-hop synthesis" open — our lane.
- **ColBERT / ColBERTv2** (multi-vector). The known *partial* escape from the single-vector
  bound. Our facet-lens (C4) must be benchmarked against it and differentiated (facet
  specialization + joint retrieve-and-compress, not token-level late interaction).

## Method/context to track
- Multi-agent RAG: MA-RAG (2505.20096), MARAG-R1 (2510.27569).
- Soft-compression baselines to reimplement: xRAG, GIST, ICAE, 500xCompressor.

## Differentiation one-liner (keep sharp)
> Prior work bounds retrieval *or* studies compression; nobody bounds **the composition**.
> We show the two fixed-`d` bottlenecks compound, derive the allocation between them, and
> escape both with a multi-view representation.

## Threat monitor — 2026-07-01 scan
Weekly arXiv scan for anyone scooping C1 (unified capacity wall) or C2 (compounding). No direct
scoop found; the unified retrieval+compression capacity account and the cross-stage allocation
still appear unclaimed. Neighbors to cite/position against:
- **Fixed RAG Compression Collapses Measured Reader Scaling** (arXiv:2606.21807, Panthi &
  Abdelfattah). Compression×reader-scale: "compression gain decreases with reader baseline
  (9/10 settings p<0.05); rescues weak readers by removing noise, harms strong readers by
  dropping details." NOT compounding and NOT cross-stage allocation, so no threat to C1-C3 — a
  clean related-work cite for us. NB: it strongly overlaps the *diagnostic paper's* reader-ladder
  (condition 4: edge absorbed by 7B, reverses by 14B) — worth flagging for that paper's positioning.
- **DAST** (2502.11493) — dynamic allocation of *soft tokens* (compression-only budget), adjacent
  to C3 but single-stage. **ElasticMem** (2605.30690) — elastic latent budget over retrieved chunks.
- **QRAG** (embedding PCA 8-bit + residual 4-bit, 3-5x at 0.95 recall) and adaptive context
  compression (2507.22931) — compression *methods*, not capacity theory; cite as motivation for
  the budget being scarce, not as competitors to the bound.
