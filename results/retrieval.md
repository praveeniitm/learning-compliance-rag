# Retrieval: one search, top 5 chunks

| Configuration | P@5 | R@5 | MRR | ΔR@5 vs dense only [95% CI] |
|---|---|---|---|---|
| dense only | 0.292 | 0.791 | 0.751 | – |
| BM25 only | 0.236 | 0.688 | 0.692 | -0.103 [-0.252, +0.051] |
| hybrid: BM25 + dense, RRF fusion | 0.287 | 0.799 | 0.796 | +0.009 [-0.073, +0.103] |
| hybrid + LLM re-rank (used by the agent) | 0.344 | 0.94 | 0.987 | +0.150 [+0.047, +0.256] |
|   chunking: without contextual header | 0.297 | 0.812 | 0.815 | +0.021 [-0.128, +0.171] |
|   chunking: fixed 320-token windows | 0.313 | 0.897 | 0.831 | +0.107 [-0.000, +0.222] |
|   embedding: text-embedding-3-large, dense only | 0.323 | 0.859 | 0.88 | +0.068 [-0.004, +0.162] |

Test split, 39 questions with gold passages. P@5 is capped near 0.4–0.6 because most questions have only 1–3 gold passages. The 95% interval is a paired bootstrap over questions.
