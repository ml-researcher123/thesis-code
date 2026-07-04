# E9 — Real End-to-End RAG QA (HotpotQA, Qwen2.5-7B-Instruct)

Real multi-hop QA: 300 questions over a pooled corpus of 2983 paragraphs; real retriever (`mxbai-embed-large-v1`, truncated to d_r) + query-conditioned sentence selection (d_c) under a 160-token reader budget; frozen reader answers, scored by EM/F1. Shared budget B = d_r + d_c.

| budget B | best split d_r:d_c | best F1 | standalone R | standalone C | compounding gap |
|---|---|---|---|---|---|
| 128 | 112:16 | 0.412 | 0.466 | 0.691 | 0.054 |
| 256 | 128:128 | 0.454 | 0.509 | 0.706 | 0.054 |

Metric for the gap is **F1** (real reader EM/F1 when the reader is enabled, else the answer-in-context diagnostic). A positive gap is compounding on real answer quality: the best budget split underperforms either stage given the full budget, and the optimum is interior — the real-task version of E6, and the reviewer-proof form of C2/C3.

![real qa allocation](e9_real_qa.png)
