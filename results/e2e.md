# End-to-end (generator gpt-4.1-nano, judge gpt-4.1-mini)

| Metric | dev | test |
|---|---|---|
| n | 7 | 13 |
| status accuracy | 1.0 | 0.769 |
| key-fact accuracy (answerable) | 1.0 | 1.0 |
| faithfulness (judge, share of supported statements) | 0.911 | 1.0 |
| answers revised by grader loop | 0.0 | 0.077 |

| id | type | expected | got | key facts | faithfulness |
|---|---|---|---|---|---|
| q57 (test) | ambiguous | needs_clarification | answered | None | 1.0 |
| q54 (test) | conflict | answered | unverified | True | 1.0 |
| q46 (dev) | identifier | answered | answered | True | 1.0 |
| q62 (test) | insufficient | insufficient_context | insufficient_context | None | - |
| q24 (dev) | law_vs_policy | answered | answered | True | 1.0 |
| q28 (test) | multi_doc | answered | answered | True | 1.0 |
| q66 (dev) | out_of_scope | out_of_scope | blocked | None | - |
| q74 (test) | policy_fact | answered | answered | True | 1.0 |
| q12 (test) | reg_fact | answered | answered | True | 1.0 |
| q53 (test) | version | answered | answered | True | 1.0 |
| q59 (test) | ambiguous | needs_clarification | answered | None | 1.0 |
| q55 (dev) | conflict | answered | answered | True | 0.8 |
| q47 (test) | identifier | answered | answered | True | 1.0 |
| q65 (test) | insufficient | insufficient_context | blocked | None | - |
| q22 (test) | law_vs_policy | answered | answered | True | 1.0 |
| q35 (dev) | multi_doc | answered | answered | True | 0.67 |
| q68 (test) | out_of_scope | out_of_scope | blocked | None | - |
| q05 (dev) | policy_fact | answered | answered | True | 1.0 |
| q18 (dev) | reg_fact | answered | answered | True | 1.0 |
| q49 (test) | version | answered | answered | True | 1.0 |
