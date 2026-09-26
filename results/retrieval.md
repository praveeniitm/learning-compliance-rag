# Retrieval results

Test split unless noted; questions with gold passages only (ambiguous / insufficient / out-of-scope questions have none). Latency is the median per query on an Apple M-series laptop (MPS), excluding model load.

## retriever

| Configuration | P@5 | R@5 | R@10 | Hit@5 | MRR | nDCG@5 | dev R@5 | p50 ms |
|---|---|---|---|---|---|---|---|---|
| dense (bge-small) | 0.200 | 0.598 | 0.769 | 0.795 | 0.693 | 0.574 | 0.644 | 10 |
| sparse (BM25) | 0.246 | 0.765 | 0.863 | 0.897 | 0.719 | 0.681 | 0.826 | 1 |
| hybrid RRF | 0.246 | 0.765 | 0.855 | 0.897 | 0.719 | 0.664 | 0.811 | 15 |
| hybrid convex a=0.5 | 0.236 | 0.714 | 0.863 | 0.872 | 0.752 | 0.661 | 0.826 | 16 |
| hybrid convex a=0.7 | 0.236 | 0.692 | 0.791 | 0.872 | 0.727 | 0.642 | 0.705 | 11 |
| dense + rerank | 0.272 | 0.786 | 0.833 | 0.872 | 0.776 | 0.745 | 0.758 | 128 |
| hybrid RRF + rerank (default) | 0.282 | 0.850 | 0.897 | 0.949 | 0.817 | 0.785 | 0.871 | 127 |

Paired bootstrap (test R@5, 2,000 resamples) vs. **hybrid RRF + rerank (default)**: dense (bge-small): Δ=-0.252 [-0.355, -0.158]; sparse (BM25): Δ=-0.085 [-0.175, -0.013]; hybrid RRF: Δ=-0.085 [-0.162, -0.017]; hybrid convex a=0.5: Δ=-0.137 [-0.226, -0.056]; hybrid convex a=0.7: Δ=-0.158 [-0.256, -0.068]; dense + rerank: Δ=-0.064 [-0.141, +0.004]

## reranker

| Configuration | P@5 | R@5 | R@10 | Hit@5 | MRR | nDCG@5 | dev R@5 | p50 ms |
|---|---|---|---|---|---|---|---|---|
| ms-marco-MiniLM-L-6 (22M) | 0.282 | 0.850 | 0.897 | 0.949 | 0.817 | 0.785 | 0.871 | 126 |
| bge-reranker-base (278M) | 0.282 | 0.829 | 0.910 | 0.923 | 0.789 | 0.752 | 0.886 | 629 |

Paired bootstrap (test R@5, 2,000 resamples) vs. **ms-marco-MiniLM-L-6 (22M)**: bge-reranker-base (278M): Δ=-0.021 [-0.098, +0.047]

## ablation

| Configuration | P@5 | R@5 | R@10 | Hit@5 | MRR | nDCG@5 | dev R@5 | p50 ms |
|---|---|---|---|---|---|---|---|---|
| default (all components) | 0.282 | 0.850 | 0.897 | 0.949 | 0.817 | 0.785 | 0.871 | 129 |
| - query expansion | 0.297 | 0.846 | 0.927 | 0.949 | 0.775 | 0.757 | 0.871 | 126 |
| + legal-source slot | 0.282 | 0.850 | 0.897 | 0.949 | 0.817 | 0.785 | 0.871 | 129 |
| + legal-source slot, bge-reranker-base | 0.282 | 0.829 | 0.910 | 0.923 | 0.789 | 0.752 | 0.909 | 629 |
| - superseded demotion | 0.277 | 0.842 | 0.897 | 0.949 | 0.752 | 0.737 | 0.856 | 129 |
| - contextual headers | 0.231 | 0.662 | 0.803 | 0.795 | 0.654 | 0.590 | 0.697 | 137 |
| - expansion, demotion, headers | 0.215 | 0.611 | 0.799 | 0.795 | 0.616 | 0.538 | 0.644 | 135 |

Paired bootstrap (test R@5, 2,000 resamples) vs. **default (all components)**: - query expansion: Δ=-0.004 [-0.038, +0.026]; + legal-source slot: Δ=+0.000 [+0.000, +0.000]; + legal-source slot, bge-reranker-base: Δ=-0.021 [-0.098, +0.047]; - superseded demotion: Δ=-0.009 [-0.026, +0.000]; - contextual headers: Δ=-0.188 [-0.316, -0.077]; - expansion, demotion, headers: Δ=-0.239 [-0.363, -0.124]

## chunking

| Configuration | P@5 | R@5 | R@10 | Hit@5 | MRR | nDCG@5 | dev R@5 | p50 ms |
|---|---|---|---|---|---|---|---|---|
| structure350 / dense | 0.200 | 0.598 | 0.769 | 0.795 | 0.693 | 0.574 | 0.644 | 10 |
| structure350 / hybrid_rerank | 0.282 | 0.850 | 0.897 | 0.949 | 0.817 | 0.785 | 0.871 | 130 |
| structure200 / dense | 0.195 | 0.543 | 0.709 | 0.718 | 0.645 | 0.542 | 0.636 | 11 |
| structure200 / hybrid_rerank | 0.292 | 0.821 | 0.880 | 0.923 | 0.788 | 0.767 | 0.864 | 84 |
| structure350-nohdr / dense | 0.190 | 0.607 | 0.756 | 0.769 | 0.603 | 0.528 | 0.606 | 12 |
| structure350-nohdr / hybrid_rerank | 0.231 | 0.662 | 0.803 | 0.795 | 0.654 | 0.590 | 0.697 | 140 |
| fixed256o48 / dense | 0.262 | 0.650 | 0.795 | 0.872 | 0.645 | 0.556 | 0.651 | 10 |
| fixed256o48 / hybrid_rerank | 0.292 | 0.756 | 0.910 | 0.897 | 0.697 | 0.637 | 0.735 | 106 |
| fixed512o64 / dense | 0.277 | 0.752 | 0.786 | 0.923 | 0.750 | 0.682 | 0.636 | 10 |
| fixed512o64 / hybrid_rerank | 0.303 | 0.803 | 0.880 | 0.949 | 0.724 | 0.676 | 0.735 | 190 |

Paired bootstrap (test R@5, 2,000 resamples) vs. **structure350 / hybrid_rerank**: structure350 / dense: Δ=-0.252 [-0.355, -0.158]; structure200 / dense: Δ=-0.308 [-0.419, -0.197]; structure200 / hybrid_rerank: Δ=-0.030 [-0.107, +0.038]; structure350-nohdr / dense: Δ=-0.244 [-0.355, -0.132]; structure350-nohdr / hybrid_rerank: Δ=-0.188 [-0.316, -0.077]; fixed256o48 / dense: Δ=-0.201 [-0.329, -0.073]; fixed256o48 / hybrid_rerank: Δ=-0.094 [-0.201, +0.000]; fixed512o64 / dense: Δ=-0.098 [-0.218, +0.017]; fixed512o64 / hybrid_rerank: Δ=-0.047 [-0.141, +0.051]

## embedding

| Configuration | P@5 | R@5 | R@10 | Hit@5 | MRR | nDCG@5 | dev R@5 | p50 ms |
|---|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 / dense | 0.210 | 0.611 | 0.761 | 0.769 | 0.561 | 0.527 | 0.621 | 8 |
| all-MiniLM-L6-v2 / hybrid_rerank | 0.287 | 0.855 | 0.914 | 0.949 | 0.830 | 0.798 | 0.871 | 128 |
| bge-small-en-v1.5 / dense | 0.200 | 0.598 | 0.769 | 0.795 | 0.693 | 0.574 | 0.644 | 11 |
| bge-small-en-v1.5 / hybrid_rerank | 0.282 | 0.850 | 0.897 | 0.949 | 0.817 | 0.785 | 0.871 | 132 |
| bge-base-en-v1.5 / dense | 0.251 | 0.718 | 0.778 | 0.846 | 0.717 | 0.665 | 0.735 | 19 |
| bge-base-en-v1.5 / hybrid_rerank | 0.277 | 0.842 | 0.889 | 0.949 | 0.829 | 0.788 | 0.856 | 128 |
| e5-base-v2 / dense | 0.262 | 0.744 | 0.855 | 0.872 | 0.722 | 0.659 | 0.674 | 12 |
| e5-base-v2 / hybrid_rerank | 0.287 | 0.846 | 0.902 | 0.949 | 0.815 | 0.782 | 0.856 | 127 |

Paired bootstrap (test R@5, 2,000 resamples) vs. **bge-small-en-v1.5 / hybrid_rerank**: all-MiniLM-L6-v2 / dense: Δ=-0.239 [-0.372, -0.124]; all-MiniLM-L6-v2 / hybrid_rerank: Δ=+0.004 [-0.030, +0.043]; bge-small-en-v1.5 / dense: Δ=-0.252 [-0.355, -0.158]; bge-base-en-v1.5 / dense: Δ=-0.132 [-0.244, -0.026]; bge-base-en-v1.5 / hybrid_rerank: Δ=-0.009 [-0.026, +0.000]; e5-base-v2 / dense: Δ=-0.107 [-0.218, -0.004]; e5-base-v2 / hybrid_rerank: Δ=-0.004 [-0.038, +0.026]

## Recall@5 by question type (default configuration, test)

| Type | R@5 | MRR |
|---|---|---|
| conflict | 0.833 | 0.667 |
| identifier | 1.000 | 0.767 |
| law_vs_policy | 0.767 | 0.840 |
| multi_doc | 0.722 | 0.720 |
| policy_fact | 0.900 | 1.000 |
| reg_fact | 0.917 | 0.850 |
| version | 0.833 | 0.833 |
