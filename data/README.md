# data/

Download/prepare **scripts only** — no large data in git. Raw data and caches go under
`data/raw/` and `data/cache/` which are gitignored; on Kaggle, prefer attaching a Kaggle
Dataset or pulling from Hugging Face Hub at runtime.

Planned adapters (added with the stages that need them):
- **LIMIT** (Weller et al.) — retrieval capacity probe (E1 real-encoder variant).
- **CapRetrieval** — fine-grained granularity stress.
- **BEIR** subsets (NQ, HotpotQA, FiQA, SciFact) — standard retrieval.
- **Multi-hop** (HotpotQA, 2WikiMultiHop, MuSiQue) — where compounding (E3) bites hardest.
