# E1 — Retrieval Capacity Wall

Pattern: **all_pairs** (k=2); dims=[2, 4, 8, 16, 32, 64]; seeds=[0, 1, 2].

Free-embedding realizability is the *best case* for any encoder of a given
dimension. The critical dimension d* (smallest d reaching realizability >= 0.99) grows with corpus size — this is the embedding wall.

| corpus n_d | critical dim d* |
|---|---|
| 10 | 8 |
| 20 | 8 |
| 40 | 8 |

If d* increases with n_d, the harness reproduces Weller et al.'s wall and is
validated for the compression (E2) and composition (E3) stages.

![capacity wall](e1_capacity_wall.png)
