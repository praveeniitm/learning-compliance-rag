# Learning Compliance RAG

An **agentic RAG** assistant for **compliance-training questions**, built with LangGraph. It answers from OSHA and HIPAA regulations, California law, and an organization's policies, SOPs and course catalog. Every answer cites the exact regulation paragraph or policy clause, separates legal requirements from internal rules, and respects who is asking. The assistant remembers the conversation, can answer "as of" a past date, and is wrapped in guardrails.

**Stack:** LangChain · LangGraph · NeMo Guardrails · FAISS + BM25 · OpenAI (`gpt-4.1-nano` / `gpt-4.1-mini`, `text-embedding-3-small`; Ollama or vLLM also work through an OpenAI-compatible base URL) · FastAPI + Gradio.

## The problem

In an organization that runs a learning management system (LMS), compliance, L&D, HR and EHS teams ask questions like:

- *"Which training does a forklift operator need, how often, and which course satisfies it?"*
- *"Our policy says annual HIPAA refreshers. Is that the law or our own rule?"*
- *"A supervisor e-mailed that re-evaluations are every 24 months now. Is that right?"*
- *"What rule applied when this employee was hired in 2025?"* (audits need point-in-time answers)

The answer usually sits between regulations (the legal source of truth, cited by paragraph), internal policies (stricter, versioned, not public) and the course catalog and role matrix. Cross-referencing them by hand leads to misassigned training and failed audits. RAG fits because every answer must be traceable to a specific paragraph, and the rules change with every policy version. Fine-tuning would bake in rules that go stale, and a plain LLM cannot cite anything.

## Corpus

13 real regulation sections (eCFR point-in-time 2026-09-01, plus Cal. Gov. Code 12950.1) and 15 synthetic documents for a fictional 4,800-employee company. The synthetic set deliberately includes realistic problems:
- a **superseded** policy version
- an e-mail with a **wrong** claim
- an e-mail containing a **hidden prompt injection**
- **restricted** documents (audit memo, LMS configuration SOP)

Details: [data/README.md](data/README.md).

## Architecture

A custom LangGraph RAG agent: the model plans its own searches, while the code controls access, grounding and safety.

```
                     ┌──────────────── LangGraph (SQLite checkpointer = conversation memory) ────────────────┐
question, role, ───► │ contextualize ─► guard_input ─► plan ⇄ search ─► generate ─► grade ─► guard_output ─► │─► answer + [S#]
as_of                │ (rewrite          (NeMo: PII    (tool-calling   (structured (unsupported              │   + status
                     │  follow-up)        regex, jail-  agent, ≤3      output)     statements?               │
                     │                    break, topic) searches)           ▲       revise once) ──┘         │
                     └────────────────────────────────────────────────────────────────────────────────────────┘
search(query, source=any|regulation|internal)
       = BM25 (citation-aware) + FAISS dense ─► RRF ─► LLMListwiseRerank ─► top 5, filtered by the caller's ACL + as-of
ingest = Markdown + front matter ─► header-aware split + contextual header ─► LangChain index() (incremental) ─► FAISS
every request ─► logs/audit.jsonl (role, question, rewrite, agent searches, sources, status, tokens, $ cost, review flag)
```

**Why an agent rather than a fixed pipeline.** The hardest questions in this domain need more than one lookup, and a single top-5 retrieval serves them badly:
- *Law vs. policy* ("is the annual HIPAA refresher legally required?"): the agent searches `source="regulation"` and `source="internal"` separately. In one query, internal documents out-rank the regulation that paraphrases them.
- *Multi-hop* ("which courses does a Clinic Nurse take, and which recur yearly?"): role → curriculum, then course → interval.
- *Vocabulary gaps and noisy hits*: the agent re-queries with regulatory terms ("powered industrial truck") when results are poor or dominated by an e-mail or a superseded document.

**Control stays in code, not in the model:**
- The first turn must call the search tool (`tool_choice="required"`), so every answer is grounded.
- The role and as-of date are applied by the search node from the request. The agent can choose the query and the source, never the permissions.
- The loop is capped at 3 searches, and duplicate queries are refused.
- The generator, grader and guardrails are unchanged. They see only the accumulated evidence (at most 10 chunks).

**Trade-off:** 1–3 extra LLM calls per question (planning), about 2–4 s more latency than the fixed pipeline (typically 6–10 s end to end, around $0.003–0.008 per question). The measured results below are from the earlier fixed-pipeline version.

| File | Responsibility |
|---|---|
| `src/lcrag/corpus.py` | load documents and front matter; 3 chunking strategies |
| `src/lcrag/store.py` | incremental index sync, BM25 tokenizer, ACL and as-of filter, retriever modes |
| `src/lcrag/graph.py` | LangGraph agent (plan/search loop), prompts, memory, audit log, cost tracking |
| `src/lcrag/guardrails.py`, `rails/` | NeMo Guardrails config and prompts |
| `app/app.py` | FastAPI (`/api/ask`, `/api/feedback`, `/api/health`) + Gradio chat UI |
| `eval/evaluate.py` | retrieval, end-to-end and safety evaluation |
| `tests/` | offline unit tests (run in CI) |

## Features

| Capability | How it works |
|---|---|
| **Agentic retrieval** | A tool-calling planner decides the queries and the source (regulations vs. internal documents), splits multi-part questions and re-queries on poor results. It is bounded (at most 3 searches, no duplicates, first search mandatory), and the searches are shown in the UI and the audit log. |
| **Grounded, cited answers** | Structured output (answer with `[S#]` citations, status, clarifying question). The prompt enforces an authority order: regulation > current policy > SOP > FAQ > e-mail. |
| **Hallucination control** | A grader node lists statements the sources don't support. The generator revises once with that feedback. If the revision still fails, the answer is marked `unverified` and flagged `needs_review` in the audit log (a human review queue). |
| **Access control** | Each document carries an `access` field in its front matter. The filter runs *inside* retrieval (FAISS `filter=` and a per-role BM25 index), so restricted text never reaches the re-ranker, the prompt or a citation. Roles: `employee`, `manager`, `ehs`, `lms_admin`, `compliance`. |
| **Point-in-time answers** | Passing `as_of=2025-06-01` retrieves only documents in force on that date (effective, and not yet superseded), so it returns the 45-day rule from policy v3.1 instead of today's 30 days. |
| **Conversation memory** | A LangGraph `SqliteSaver` checkpointer, keyed by `thread_id`, survives restarts. A rewrite step turns "and what does OSHA require?" into a standalone question before retrieval. |
| **Guardrails (NeMo)** | Input: a regex rail blocks personal data (SSN, card numbers, DOB) without an LLM call, then a self-check blocks jailbreaks, off-topic requests, questions about a named person's records, and evasion ("backdate certifications"). Output: regex PII check plus a self-check for disclosure, evasion advice and injected instructions. Rails **fail closed**. |
| **Prompt-injection defence** | Three layers: sources are marked as data, not instructions, in the prompt; the grader checks the answer against the sources; the output rail catches injected instructions. Tested against a poisoned e-mail in the corpus. |
| **Index freshness** | LangChain Indexing API: content-hashed chunks, only new or changed ones embedded, removed files cleaned up. Adding one document re-embedded 2 chunks and skipped 964. |
| **Cost and observability** | Per-request token usage and $ cost (`get_usage_metadata_callback`), latency, sources and status in `logs/audit.jsonl`. User feedback goes to `logs/feedback.jsonl`. Set `LANGSMITH_TRACING=true` for full traces (no code change). |
| **Caching** | Exact-match LLM cache (SQLite). At temperature 0, repeated questions cost nothing. |
| **Model tiering** | Generation on `gpt-4.1-nano`. Decision and checking tasks (search planning, re-rank, grader, rails) on `gpt-4.1-mini`, where nano measurably failed; as the planner, nano skipped the internal-policy search on law-vs-policy questions. |
| **Resilience** | Re-rank falls back to RRF order if the LLM returns an invalid ranking; `max_retries=3`; rails fail closed. |
| **API + UI** | FastAPI REST endpoints with OpenAPI docs at `/docs`; Gradio chat with role selector, as-of date, sources, thumbs up/down. |
| **Quality gates** | Offline pytest in CI on every push. A manual workflow runs the sampled evaluation and fails if faithfulness < 0.9 or key-fact accuracy < 0.8. Dockerfile included. |

## Library and method choices

| Component | Choice | Alternatives considered | Why this one |
|---|---|---|---|
| Orchestration | **Custom LangGraph RAG agent** (plan ⇄ search loop inside a guarded graph) | fixed retrieve → generate pipeline; `create_react_agent`; fully autonomous agent | A fixed pipeline cannot run separate regulation and policy searches or multi-hop lookups. A prebuilt ReAct agent would also write the final answer, skipping structured output, the grounding grader and the rails. The custom graph gives the model only the search decisions, and keeps generation, verification and access control deterministic. |
| Framework | **LangChain + LangGraph** | LlamaIndex; plain SDK | LangChain provides every retrieval piece as a tested component: header-aware splitter, BM25, FAISS, RRF ensemble, listwise re-ranker and the Indexing API. LangGraph makes the loop, the guardrails and memory an explicit state graph with a built-in checkpointer. A hand-written first version was about 5x more code. |
| Chunking | **Header-aware split + 320-token cap + contextual header** | fixed windows; semantic chunking | A CFR paragraph "(l)" or policy section "7." is the natural unit of meaning. The header (citation, topic such as "forklift", section, version/SUPERSEDED) fixes the boundary problem: "(iii) ... every three years" doesn't say it is about forklift operators. Semantic chunking adds cost and non-determinism for no gain on structured text. |
| Embeddings | **text-embedding-3-small** | 3-large (measured); local bge/e5 (earlier iteration) | 3-large improves dense-only recall. In the full pipeline, the re-ranker closes most of the gap at 6.5x lower embedding cost. |
| Vector store | **FAISS** (flat, exact) | Chroma, pgvector, Weaviate | About 1k chunks: exact search takes under 1 ms, needs no server and is reproducible. ACLs are handled with a metadata filter during search. pgvector becomes worthwhile with concurrent writers or database-enforced row-level security. |
| Sparse | **BM25 with a citation-aware tokenizer** | default tokenizer; TF-IDF | Keeps "1910.178(l)(4)" and "EHS-FL-201" whole, including hierarchical prefixes, so exact-identifier questions match. |
| Fusion | **RRF** (`EnsembleRetriever`) | weighted score fusion | Rank-based, so no calibration between cosine and BM25 scores. |
| Re-rank | **`LLMListwiseRerank`** (gpt-4.1-mini) | cross-encoder; Cohere Rerank | No local models, no extra vendor, and it compares candidates against each other. At high volume, a hosted cross-encoder would be cheaper and faster. |
| Guardrails | **NeMo Guardrails** (`check_async`, rails only) | Guardrails AI; Llama Guard; prompt-only | Declarative YAML config with ready-made regex and self-check rails. `check()` runs rails without letting NeMo generate, so it slots in as graph nodes, and the same injected LLM keeps OpenAI/Ollama portability. Llama Guard needs a hosted safety model; Guardrails AI is output-validation focused. |
| Grounding check | **LLM grader node + one revision** | NeMo self-check-facts; NLI model | The grader sees the exact sources and returns the unsupported statements, which become revision feedback. NeMo's fact check would duplicate it. A local DeBERTa NLI checker (earlier iteration) misjudged paraphrases such as "forklift" vs. "powered industrial truck". |
| Memory | **LangGraph `SqliteSaver`** + query rewrite | `ConversationBufferMemory`; Redis | Native to the graph, persistent and keyed by thread. `PostgresSaver` is the drop-in for multi-instance deployment. |
| Evaluation | **Own structural gold + LLM judge + safety suite** | RAGAS | RAGAS 0.4.3 fails to import with the current `langchain-community`. Structural gold (document + paragraph/clause) scores all chunking strategies on the same ground truth. |
| Serving | **FastAPI + Gradio mounted together** | Streamlit; LangServe | One process gives a REST API (with OpenAPI docs) and a demo UI; LangServe is deprecated. |

## Results

Held-out set: 74 questions across 10 types (regulation facts, law vs. policy, multi-document, identifiers, versions, conflicts, ambiguous, insufficient, out of scope). The split is seeded and stratified: 27 dev / 47 test. To keep costs low, runs use a **stratified sample** (`--n`), so the confidence intervals are wide. Full tables: [retrieval](results/retrieval.md), [end to end](results/e2e.md), [safety](results/safety.md).

**Retrieval** (25-question sample, test part; k=5)

| Configuration | R@5 | MRR |
|---|---|---|
| dense only | 0.781 | 0.693 |
| BM25 only | 0.646 | 0.635 |
| hybrid RRF | 0.792 | 0.742 |
| **hybrid RRF + re-rank, gpt-4.1-mini (default)** | **0.906** | **0.969** |
| hybrid RRF + re-rank, gpt-4.1-nano | 0.844 | 0.869 |
| full pipeline, text-embedding-3-large | 0.958 | 0.885 |

On the full 61-question set (earlier run), hybrid + re-rank reached R@5 **0.944 vs. 0.791** for dense (+0.154, 95% CI [+0.056, +0.261]). Removing the contextual header cost 0.12 R@5. Fusion alone mainly improves MRR; the re-ranker gives the recall lift.

**End to end** (20-question sample; generator gpt-4.1-nano, checks gpt-4.1-mini, judge gpt-4.1-mini)

| Metric | dev (n=7) | test (n=13) |
|---|---|---|
| Status accuracy (answer / clarify / decline) | 1.00 | 0.77 |
| Key-fact accuracy (answerable) | 1.00 | 1.00 |
| Faithfulness (supported statements) | 0.91 | 1.00 |

**Safety suite** ([results/safety.md](results/safety.md)): guardrails 9/9 (6 attacks blocked, 3 legitimate questions allowed), ACL 5/5 (no restricted document retrieved, authorized roles get it), as-of 2/2, multi-turn memory 3/3, prompt injection resisted 2/2.

## Where it fails, and what I learned

- **Cheap models fail at checking before they fail at writing.** With nano as the grader, about 80% of faithful answers were flagged unsupported and status accuracy fell to 0.23. Moving only the checking tasks to mini restored 0.77–1.00. That is the reason for the model tiering.
- **Ambiguous questions** are still answered with a list of cases rather than a clarifying question: correct content, wrong status.
- **Rail false positives:** an early output rail blocked a legitimate audit answer ("11 operators ... badge integration failed"), and the input rail blocked "Which audit findings were raised?". Both were fixed with explicit allow-lists in the rail prompts, and the safety suite now tests both cases.
- **Insufficient-context questions** are sometimes caught by the input rail as off-topic (e.g. "training budget for Reno"). The user still gets a refusal, but the reason is wrong.
- **Conflict and version questions** have the lowest recall. The wrong e-mail and the old policy match the question's wording best; the authority rule lives in the prompt, not in retrieval.
- **Evaluation limits:** small samples, synthetic internal documents, and the same author for corpus and questions.

## Production notes

- **Scale (1,000 concurrent queries):** the bottleneck is 5–9 LLM calls per request (rewrite, 2 rails, 1–3 planning steps, a re-rank per search, generate, grade), 6–10 s total. FAISS and BM25 take milliseconds. Levers: run the input rail in parallel with retrieval, the LLM cache, async workers, per-provider rate-limit budgets, and a hosted cross-encoder in place of the LLM re-ranker.
- **State:** swap `SqliteSaver` for `PostgresSaver` and store FAISS on shared storage (or move to pgvector) for multiple instances.
- **Identity:** the role is chosen in the UI for the demo. In production it must come from SSO claims, never from the client.
- **Governance:** `needs_review` answers and 👎 feedback form a review queue. A LangGraph `interrupt` would add human approval before an answer changes an LMS rule.

## Reproduce

```bash
uv venv --python 3.11 .venv && uv pip install -r requirements.txt -e .
cp .env.example .env                        # add OPENAI_API_KEY (or point to Ollama/vLLM)
python -m lcrag.store                       # build the index (~1k chunks, a few cents)
pytest -q tests                             # offline unit tests
python eval/evaluate.py retrieval --n 25    # results/retrieval.md
python eval/evaluate.py e2e --n 20          # results/e2e.md
python eval/evaluate.py safety              # results/safety.md
python app/app.py                           # UI http://127.0.0.1:7860 · API docs /docs
docker build -t lcrag . && docker run -p 7860:7860 -e OPENAI_API_KEY lcrag
```

The regulations and synthetic documents are committed. `data/fetch_regulations.py` and `data/generate_synthetic.py` regenerate them deterministically. A sampled evaluation run costs well under $1.
