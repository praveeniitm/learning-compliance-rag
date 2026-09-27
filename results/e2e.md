# End to end: the agent

| Metric | dev | test |
|---|---|---|
| Status accuracy (answer / clarify / decline) | 0.889 | 0.936 |
| Key-fact accuracy (answerable questions) | 0.909 | 0.974 |
| Faithfulness, all answers (share of statements supported by sources) | 0.95 | 0.982 |
| Faithfulness, answers that passed the grounding check | 1.0 | 0.994 |
| Answers flagged 'unverified' to the user | 0.037 | 0.043 |
| Evidence recall (gold passages found by the agent's searches) | 0.97 | 0.932 |
| Searches per question | 1.889 | 1.702 |
| Answers revised after the grounding check | 0.111 | 0.043 |
| Latency per question (s) | 8.46 | 8.05 |
| Cost per question (USD) | 0.008 | 0.007 |

74 questions (27 dev / 47 test). Generator gpt-4.1-nano; planner, re-ranker, grader and guardrails gpt-4.1-mini; judge gpt-4.1-mini. Declined = insufficient context, out of scope or blocked; these count as one outcome for status accuracy.

| id | type | expected | got | key facts | faithfulness |
|---|---|---|---|---|---|
| q01 | reg_fact | answered | answered | True | 1.0 |
| q02 | policy_fact | answered | answered | True | 1.0 |
| q03 | law_vs_policy | answered | answered | False | 1.0 |
| q04 | policy_fact | answered | answered | True | 1.0 |
| q05 | policy_fact | answered | answered | True | 1.0 |
| q06 | reg_fact | answered | answered | True | 1.0 |
| q07 | reg_fact | answered | answered | True | 1.0 |
| q08 | reg_fact | answered | answered | True | 1.0 |
| q09 | reg_fact | answered | answered | True | 1.0 |
| q10 | reg_fact | answered | answered | True | 1.0 |
| q11 | reg_fact | answered | answered | True | 1.0 |
| q12 | reg_fact | answered | answered | True | 1.0 |
| q13 | reg_fact | answered | answered | True | 1.0 |
| q14 | reg_fact | answered | answered | True | 1.0 |
| q15 | reg_fact | answered | answered | True | 1.0 |
| q16 | reg_fact | answered | answered | True | 1.0 |
| q17 | reg_fact | answered | answered | True | 1.0 |
| q18 | reg_fact | answered | answered | True | 1.0 |
| q19 | reg_fact | answered | answered | True | 1.0 |
| q20 | reg_fact | answered | answered | True | 1.0 |
| q21 | law_vs_policy | answered | answered | True | 1.0 |
| q22 | law_vs_policy | answered | answered | True | 1.0 |
| q23 | law_vs_policy | answered | answered | True | 1.0 |
| q24 | law_vs_policy | answered | insufficient_context | True | – |
| q25 | law_vs_policy | answered | answered | True | 1.0 |
| q26 | law_vs_policy | answered | answered | True | 1.0 |
| q27 | multi_doc | answered | unverified | True | 0.5 |
| q28 | multi_doc | answered | answered | True | 1.0 |
| q29 | multi_doc | answered | answered | True | 1.0 |
| q30 | multi_doc | answered | answered | True | 1.0 |
| q31 | multi_doc | answered | answered | True | 1.0 |
| q32 | multi_doc | answered | answered | True | 1.0 |
| q33 | multi_doc | answered | unverified | False | 1.0 |
| q34 | multi_doc | answered | answered | True | 1.0 |
| q35 | multi_doc | answered | answered | True | 1.0 |
| q36 | multi_doc | answered | answered | True | 1.0 |
| q37 | multi_doc | answered | answered | True | 1.0 |
| q38 | multi_doc | answered | answered | True | 1.0 |
| q39 | multi_doc | answered | answered | True | 1.0 |
| q40 | multi_doc | answered | answered | True | 1.0 |
| q41 | identifier | answered | answered | True | 1.0 |
| q42 | identifier | answered | unverified | True | 0.0 |
| q43 | identifier | answered | answered | True | 0.8 |
| q44 | identifier | answered | answered | True | 1.0 |
| q45 | identifier | answered | answered | True | 1.0 |
| q46 | identifier | answered | answered | True | 1.0 |
| q47 | identifier | answered | answered | True | 1.0 |
| q48 | identifier | answered | answered | True | 1.0 |
| q49 | version | answered | answered | True | 1.0 |
| q50 | version | answered | answered | True | 1.0 |
| q51 | version | answered | answered | True | 1.0 |
| q52 | version | answered | answered | True | 1.0 |
| q53 | version | answered | answered | True | 1.0 |
| q54 | conflict | answered | out_of_scope | True | – |
| q55 | conflict | answered | blocked | False | – |
| q56 | conflict | answered | answered | True | 1.0 |
| q57 | ambiguous | needs_clarification | needs_clarification | None | – |
| q58 | ambiguous | needs_clarification | needs_clarification | None | – |
| q59 | ambiguous | needs_clarification | needs_clarification | None | – |
| q60 | policy_fact | answered | answered | True | 1.0 |
| q61 | insufficient | insufficient_context | insufficient_context | None | – |
| q62 | insufficient | insufficient_context | insufficient_context | None | – |
| q63 | insufficient | insufficient_context | insufficient_context | None | – |
| q64 | insufficient | insufficient_context | insufficient_context | None | – |
| q65 | insufficient | insufficient_context | blocked | None | – |
| q66 | out_of_scope | out_of_scope | blocked | None | – |
| q67 | out_of_scope | out_of_scope | blocked | None | – |
| q68 | out_of_scope | out_of_scope | blocked | None | – |
| q69 | out_of_scope | out_of_scope | blocked | None | – |
| q70 | out_of_scope | out_of_scope | blocked | None | – |
| q71 | policy_fact | answered | answered | True | 1.0 |
| q72 | policy_fact | answered | answered | True | 1.0 |
| q73 | policy_fact | answered | answered | True | 1.0 |
| q74 | policy_fact | answered | answered | True | 1.0 |
