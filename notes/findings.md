# Findings Ledger

Every load-bearing claim with a status: `HYPOTHESIS → SUPPORTED / REFUTED / INCONCLUSIVE`,
plus the evidence pointer. A claim is only `SUPPORTED` with ≥3 seeds + variance + a baseline
(per the A* bar in `CLAUDE.md`). Negative results are logged here too — they are valuable.

| ID | Claim | Status | Evidence |
|----|-------|--------|----------|
| F1 | Free-embedding realizability of the all-pairs pattern has a **sharp** phase transition in `d`. | SUPPORTED | E1 (`outputs/e1_retrieval_capacity`): n_d=40 jumps 0.38→1.0 between d=4 and d=8, 3 seeds, ~0 variance. |
| F2 | The critical dimension `d*` **grows with corpus size** (the embedding wall). | SUPPORTED | E1b (`outputs/e1b_retrieval_capacity_extended`): d=8 realizability decays 1.000→0.974 as n_d 40→320; d* moves 8→16 by n_d≈160. |
| F3 | Sub-critical realizability scales ≈ `d / n_d` (halves when corpus doubles at fixed d). | INCONCLUSIVE | E1b d=4: 0.38→0.15→0.07→0.04 over n_d 40→320 is consistent, but only 2 seeds and one pattern; needs more seeds + a second pattern (random k-subsets) and a fitted curve. |
| F4 | Soft-token compression has a realizability wall of the **same geometric form** as retrieval. | HYPOTHESIS | E2 (not yet run). This is contribution C1. |
| F5 | Retrieve-then-compress fails on inputs where each stage alone succeeds (compounding). | HYPOTHESIS | E3 (not yet run). This is the headline, C2. |

## Open threads / risks
- F3 wants a clean functional fit; if it's really `~d/n_d`, that's a quotable scaling and a
  strong setup for the allocation law (C3).
- Watch for anyone publishing F4/F5 first — see `related-work.md`.
