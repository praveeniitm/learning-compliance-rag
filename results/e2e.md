# End-to-end (generator gpt-4.1-mini, judge gpt-4.1)

| Metric | dev | test |
|---|---|---|
| n | 27 | 47 |
| status accuracy | 0.963 | 0.957 |
| key-fact accuracy (answerable) | 0.955 | 1.0 |
| faithfulness (judge, share of supported statements) | 0.982 | 1.0 |
| answers revised by grader loop | 0.0 | 0.021 |

| id | type | expected | got | key facts | faithfulness |
|---|---|---|---|---|---|
| q01 (dev) | reg_fact | answered | answered | True | 1.0 |
| q02 (test) | policy_fact | answered | answered | True | 1.0 |
| q03 (dev) | law_vs_policy | answered | answered | True | 1.0 |
| q04 (test) | policy_fact | answered | answered | True | 1.0 |
| q05 (dev) | policy_fact | answered | answered | True | 1.0 |
| q06 (test) | reg_fact | answered | answered | True | 1.0 |
| q07 (test) | reg_fact | answered | answered | True | 1.0 |
| q08 (test) | reg_fact | answered | answered | True | 1.0 |
| q09 (test) | reg_fact | answered | answered | True | 1.0 |
| q10 (dev) | reg_fact | answered | answered | True | 1.0 |
| q11 (dev) | reg_fact | answered | answered | True | 1.0 |
| q12 (test) | reg_fact | answered | answered | True | 1.0 |
| q13 (test) | reg_fact | answered | answered | True | 1.0 |
| q14 (dev) | reg_fact | answered | answered | True | 1.0 |
| q15 (test) | reg_fact | answered | answered | True | 1.0 |
| q16 (test) | reg_fact | answered | answered | True | 1.0 |
| q17 (test) | reg_fact | answered | answered | True | 1.0 |
| q18 (dev) | reg_fact | answered | answered | True | 1.0 |
| q19 (test) | reg_fact | answered | answered | True | 1.0 |
| q20 (dev) | reg_fact | answered | answered | False | 1.0 |
| q21 (test) | law_vs_policy | answered | answered | True | 1.0 |
| q22 (test) | law_vs_policy | answered | answered | True | 1.0 |
| q23 (test) | law_vs_policy | answered | answered | True | 1.0 |
| q24 (dev) | law_vs_policy | answered | answered | True | 1.0 |
| q25 (test) | law_vs_policy | answered | answered | True | 1.0 |
| q26 (test) | law_vs_policy | answered | answered | True | 1.0 |
| q27 (test) | multi_doc | answered | answered | True | 1.0 |
| q28 (test) | multi_doc | answered | answered | True | 1.0 |
| q29 (test) | multi_doc | answered | answered | True | 1.0 |
| q30 (test) | multi_doc | answered | answered | True | 1.0 |
| q31 (dev) | multi_doc | answered | answered | True | 1.0 |
| q32 (test) | multi_doc | answered | answered | True | 1.0 |
| q33 (test) | multi_doc | answered | answered | True | 1.0 |
| q34 (dev) | multi_doc | answered | answered | True | 1.0 |
| q35 (dev) | multi_doc | answered | answered | True | 1.0 |
| q36 (test) | multi_doc | answered | answered | True | 1.0 |
| q37 (test) | multi_doc | answered | answered | True | 1.0 |
| q38 (test) | multi_doc | answered | answered | True | 1.0 |
| q39 (dev) | multi_doc | answered | answered | True | 1.0 |
| q40 (dev) | multi_doc | answered | answered | True | 0.83 |
| q41 (test) | identifier | answered | answered | True | 1.0 |
| q42 (dev) | identifier | answered | answered | True | 1.0 |
| q43 (test) | identifier | answered | answered | True | 1.0 |
| q44 (test) | identifier | answered | answered | True | 1.0 |
| q45 (test) | identifier | answered | answered | True | 1.0 |
| q46 (dev) | identifier | answered | answered | True | 1.0 |
| q47 (test) | identifier | answered | answered | True | 1.0 |
| q48 (dev) | identifier | answered | answered | True | 1.0 |
| q49 (test) | version | answered | answered | True | 1.0 |
| q50 (dev) | version | answered | answered | True | 1.0 |
| q51 (test) | version | answered | answered | True | 1.0 |
| q52 (dev) | version | answered | answered | True | 1.0 |
| q53 (test) | version | answered | answered | True | 1.0 |
| q54 (test) | conflict | answered | answered | True | 1.0 |
| q55 (dev) | conflict | answered | answered | True | 0.75 |
| q56 (test) | conflict | answered | answered | True | 1.0 |
| q57 (test) | ambiguous | needs_clarification | answered | None | 1.0 |
| q58 (dev) | ambiguous | needs_clarification | answered | None | 1.0 |
| q59 (test) | ambiguous | needs_clarification | answered | None | 1.0 |
| q60 (dev) | policy_fact | answered | answered | True | 1.0 |
| q61 (dev) | insufficient | insufficient_context | insufficient_context | None | - |
| q62 (test) | insufficient | insufficient_context | insufficient_context | None | - |
| q63 (dev) | insufficient | insufficient_context | insufficient_context | None | - |
| q64 (test) | insufficient | insufficient_context | insufficient_context | None | - |
| q65 (test) | insufficient | insufficient_context | insufficient_context | None | - |
| q66 (dev) | out_of_scope | out_of_scope | insufficient_context | None | - |
| q67 (test) | out_of_scope | out_of_scope | insufficient_context | None | - |
| q68 (test) | out_of_scope | out_of_scope | insufficient_context | None | - |
| q69 (dev) | out_of_scope | out_of_scope | insufficient_context | None | - |
| q70 (test) | out_of_scope | out_of_scope | insufficient_context | None | - |
| q71 (dev) | policy_fact | answered | answered | True | 1.0 |
| q72 (test) | policy_fact | answered | answered | True | 1.0 |
| q73 (test) | policy_fact | answered | answered | True | 1.0 |
| q74 (test) | policy_fact | answered | answered | True | 1.0 |
