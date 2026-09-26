# Retrieval (test split, k=5)

| Configuration | P@5 | R@5 | Hit@5 | MRR | dev R@5 | ΔR@5 vs dense [95% CI] |
|---|---|---|---|---|---|---|
| dense | 0.292 | 0.791 | 0.923 | 0.751 | 0.803 | - |
| BM25 | 0.236 | 0.688 | 0.872 | 0.692 | 0.795 | (-0.103, -0.252, 0.051) |
| hybrid RRF | 0.287 | 0.799 | 0.949 | 0.796 | 0.886 | (0.009, -0.073, 0.103) |
| hybrid RRF + LLM re-rank (default) | 0.344 | 0.944 | 1.0 | 0.957 | 0.909 | (0.154, 0.056, 0.261) |
| chunking: no contextual header, dense | 0.262 | 0.697 | 0.821 | 0.675 | 0.659 | (-0.094, -0.201, -0.009) |
| chunking: no contextual header, full | 0.292 | 0.821 | 0.923 | 0.841 | 0.894 | (0.03, -0.115, 0.171) |
| chunking: fixed 320 tokens, dense | 0.277 | 0.735 | 0.897 | 0.71 | 0.75 | (-0.056, -0.141, 0.021) |
| chunking: fixed 320 tokens, full | 0.303 | 0.885 | 1.0 | 0.835 | 0.871 | (0.094, -0.013, 0.205) |
| embedding: 3-large, dense | 0.323 | 0.859 | 0.974 | 0.893 | 0.826 | (0.068, -0.004, 0.162) |
| embedding: 3-large, full | 0.359 | 0.962 | 1.0 | 0.949 | 0.924 | (0.171, 0.077, 0.278) |

Recall@5 by question type (default, dev+test): conflict 0.78 (n=3), identifier 1.00 (n=8), law_vs_policy 0.88 (n=7), multi_doc 0.92 (n=14), policy_fact 0.94 (n=8), reg_fact 1.00 (n=16), version 0.80 (n=5)
