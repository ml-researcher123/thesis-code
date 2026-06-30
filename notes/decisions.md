# Decisions Log

Load-bearing choices and why, so future sessions don't re-litigate them.

## D1 — Free-embedding optimization as the capacity primitive (2026-06-30)
Measure capacity by directly optimizing unconstrained doc/query vectors to fit a relevance
pattern, rather than training encoders on text. **Why:** isolates the *geometric* limit
(an upper bound for any encoder) from confounds of data/model/training. Reused across
retrieval (E1) and compression (E2). Validated by reproducing Weller's wall.

## D2 — "all-pairs" (k=2) as the primary adversarial pattern (2026-06-30)
**Why:** it is the densest, cleanest combinatorial construction whose realizable dimension
provably grows with corpus size, giving a sharp, low-variance phase transition. Random
k-subsets kept as a secondary pattern for robustness checks (F3).

## D3 — Runner / loop separation of duties (2026-06-30)
The Kaggle runner writes only `outputs/`; Claude writes only code, `configs/`, `notes/`.
**Why:** disjoint git paths ⇒ effectively no merge conflicts in the autonomous loop. The
research-log interpretation is Claude's job, never the runner's.

## D4 — Idempotency by results.json marker (2026-06-30)
A config is "done" iff `outputs/<name>/results.json` exists; the loop skips done configs.
**Why:** simplest robust queue; re-run = delete marker or bump config `name:`. Config
filename and `name:` are kept equal by convention.

## D5 — Scope guard (2026-06-30)
Ship C1 + C2 + C4 as the core, C3 as a strong section. Protect C2 (the compounding finding)
from drifting into a generic "better RAG pipeline" paper. If the C1 proof stalls, fall back
to an empirical capacity *characterization* and keep the experiments moving.
