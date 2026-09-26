# Retrieval (test split, k=5)

| Configuration | P@5 | R@5 | Hit@5 | MRR | dev R@5 | ΔR@5 vs dense [95% CI] |
|---|---|---|---|---|---|---|
| dense | 0.263 | 0.781 | 0.938 | 0.693 | 0.852 | - |
| BM25 | 0.225 | 0.646 | 0.812 | 0.635 | 0.722 | (-0.135, -0.365, 0.115) |
| hybrid RRF | 0.275 | 0.792 | 0.938 | 0.742 | 0.907 | (0.01, -0.104, 0.135) |
| hybrid RRF + LLM re-rank (default) | 0.313 | 0.906 | 1.0 | 0.969 | 0.889 | (0.125, -0.042, 0.302) |
| hybrid RRF + re-rank with gpt-4.1-nano | 0.275 | 0.844 | 1.0 | 0.869 | 0.722 | (0.062, -0.115, 0.281) |
| chunking: no contextual header, dense | 0.263 | 0.75 | 0.875 | 0.641 | 0.685 | (-0.031, -0.188, 0.094) |
| chunking: no contextual header, full | 0.3 | 0.802 | 0.875 | 0.755 | 0.852 | (0.021, -0.229, 0.271) |
| chunking: fixed 320 tokens, dense | 0.288 | 0.802 | 0.938 | 0.703 | 0.852 | (0.021, -0.073, 0.115) |
| chunking: fixed 320 tokens, full | 0.325 | 0.938 | 1.0 | 0.776 | 0.833 | (0.156, 0.01, 0.333) |
| embedding: 3-large, dense | 0.325 | 0.896 | 1.0 | 0.844 | 0.87 | (0.115, 0.0, 0.271) |
| embedding: 3-large, full | 0.338 | 0.958 | 1.0 | 0.885 | 0.926 | (0.177, 0.042, 0.354) |

Recall@5 by question type (default, dev+test): conflict 0.67 (n=3), identifier 1.00 (n=4), law_vs_policy 1.00 (n=4), multi_doc 0.92 (n=4), policy_fact 1.00 (n=4), reg_fact 1.00 (n=3), version 0.61 (n=3)
