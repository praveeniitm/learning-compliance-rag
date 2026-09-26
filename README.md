# Learning Compliance RAG

A retrieval-augmented assistant for **compliance-training questions**. It answers from OSHA and HIPAA regulations, California law, and an organization's policies, SOPs and course catalog. Answers cite the exact regulation paragraph or policy clause, separate legal requirements from internal rules, and decline when the sources don't support an answer.

**Stack:** LangChain (loading, splitting, retrievers, indexing) · LangGraph (answer workflow) · FAISS · BM25 · OpenAI (`gpt-4.1-mini`, `text-embedding-3-small`; any OpenAI-compatible server such as Ollama or vLLM also works) · Gradio.

## The problem

In an organization that runs a learning management system (LMS), compliance, L&D, HR and EHS teams ask questions like:

- *"Which training does a forklift operator need, how often, and which course satisfies it?"*
- *"Our policy says annual HIPAA refreshers. Is that the law or our own rule?"*
- *"A supervisor e-mailed that re-evaluations are every 24 months now. Is that right?"*

The answer usually sits between three document types: regulations (the legal source of truth, cited by paragraph), internal policies (often stricter, and versioned), and the course catalog and role matrix. When they are cross-referenced by hand, errors creep in: training gets misassigned and audits fail. Retrieval with citations fits this problem because every answer must be checkable against a specific paragraph. Fine-tuning would bake in rules that change with every policy version, and a plain LLM cannot cite anything.

## Corpus

13 real regulation sections (eCFR point-in-time 2026-09-01, plus Cal. Gov. Code 12950.1) and 14 synthetic documents for a fictional 4,800-employee company. The synthetic set includes a **superseded** policy version and an e-mail containing a **wrong** claim, both on purpose. Details, generation method and gap to real data: [data/README.md](data/README.md).

## Architecture

```
Markdown corpus ──MarkdownHeaderTextSplitter + token splitter + contextual header──► chunks
chunks ──LangChain index() + SQLRecordManager (hash-based, incremental)──► FAISS
question ─► BM25 (citation-aware tokens) ┐
         └► dense (FAISS)                ├─ EnsembleRetriever (RRF) ─► LLMListwiseRerank ─► top 5
LangGraph:  retrieve ─► generate (structured output) ─► grade (unsupported statements?) ─┐
                            ▲───────────── revise once with feedback ◄───────────────────┘
```

| File | Lines of logic |
|---|---|
| `src/lcrag/corpus.py` | load + split (3 strategies) |
| `src/lcrag/store.py` | index sync, BM25 tokenizer, retriever modes |
| `src/lcrag/graph.py` | prompt, output schemas, LangGraph workflow |
| `eval/evaluate.py` | retrieval metrics and end-to-end evaluation |
| `app/app.py` | Gradio UI + index freshness demo |

## Library and method choices

Each choice names the alternatives considered and the reason for picking it.

| Component | Choice | Alternatives considered | Why this one |
|---|---|---|---|
| Framework | **LangChain + LangGraph** | LlamaIndex; plain SDK code | LangChain ships every part needed here as tested components: header-aware splitter, BM25, FAISS, RRF ensemble, LLM re-ranker and an incremental Indexing API. LangGraph makes the retry loop an explicit, inspectable graph instead of hidden control flow. LlamaIndex is comparable for pure retrieval, but its agent/workflow layer is less explicit than LangGraph's state graph. Hand-written code was tried first and was about 5x larger for the same behaviour. |
| Chunking | **Markdown-header split + token cap (320) + contextual header** | fixed windows; semantic chunking | Regulations and policies are already structured: a CFR paragraph "(l)" or a policy section "7." is the natural unit of meaning. The contextual header (citation, title, topic such as "forklift", section path, version/SUPERSEDED) fixes the main boundary problem. A child paragraph like "(iii) ... at least once every three years" doesn't say what it is about; the header does. Semantic chunking adds an embedding pass and is non-deterministic, for no gain on documents that are already structured. |
| Embeddings | **text-embedding-3-small** | text-embedding-3-large (measured); local bge/e5 (measured in an earlier iteration) | Measured below: 3-large improves dense-only recall, but in the full pipeline the gain is +0.018 R@5 at 6.5x the price. The re-ranker closes most of the gap. |
| Vector store | **FAISS** (flat, exact) | Chroma, pgvector, Weaviate | About 1k chunks: exact search takes under 1 ms, needs no server and is reproducible. The LangChain record manager adds incremental updates. For multi-tenant data with access controls and concurrent writers, **pgvector** is the production choice (vectors, metadata and ACLs in one transactional store), and it is a drop-in swap behind the same LangChain interface. |
| Sparse retrieval | **BM25** with a citation-aware tokenizer | TF-IDF; default BM25 tokenizer | Users paste identifiers such as "1910.178(l)(4)" and "EHS-FL-201". The default tokenizer splits them into noise; the custom tokenizer keeps them whole, including hierarchical prefixes. |
| Fusion | **Reciprocal rank fusion** (`EnsembleRetriever`) | weighted score fusion | RRF uses ranks only, so no calibration is needed between cosine and BM25 scores. Score fusion needed a tuned weight and was worse in an earlier experiment. |
| Re-ranking | **`LLMListwiseRerank`** (gpt-4.1-mini) | local cross-encoder; Cohere Rerank | No local models and no extra vendor. Listwise ranking compares candidates against each other, which helps with "which of these is the current rule". Cost: about 13k input tokens per query (≈$0.005) and 2–3 s latency. At high volume, a hosted cross-encoder (Cohere, Voyage) would be cheaper and faster. |
| Generation | **Structured output** (pydantic: answer, status, clarifying question) | free text + parsing | The status (answered / needs clarification / insufficient context / out of scope) drives the UI and the evaluation, and it never fails to parse. Placing `answer` before `status` in the schema made status choice noticeably more accurate. |
| Hallucination control | **Grounded prompt + LLM grader + one revision** (corrective/Self-RAG pattern in LangGraph) | NLI model; no check | The grader lists statements the sources don't support; the generator revises once. If the revision still fails, the answer is marked `unverified`. A local DeBERTa NLI checker was built in an earlier iteration and rejected: small NLI models misjudged paraphrases such as "forklift" vs. "powered industrial truck". |
| Evaluation | **Own structural gold + GPT-4.1 judge** | RAGAS | RAGAS 0.4.3 failed to import against the current `langchain-community`. Gold relevance is defined on document structure (doc + paragraph/clause), so different chunking strategies are scored on the same ground truth. The faithfulness judge is a stronger model than the generator. |
| UI | **Gradio** | Streamlit | Fewest lines for a Q&A demo with collapsible sources. |

## Results

Held-out set: 74 questions (61 with gold passages) across 10 types: regulation facts, law vs. policy, multi-document, exact identifiers, versions, conflicting sources, ambiguous, insufficient, and out of scope. The seeded, type-stratified split is 27 dev / 47 test. Full tables: [results/retrieval.md](results/retrieval.md), [results/e2e.md](results/e2e.md).

**Retrieval (test, k=5)**

| Configuration | P@5 | R@5 | MRR | ΔR@5 vs dense [95% CI] |
|---|---|---|---|---|
| dense only | 0.292 | 0.791 | 0.751 | – |
| BM25 only | 0.236 | 0.688 | 0.692 | −0.103 [−0.252, +0.051] |
| hybrid RRF | 0.287 | 0.799 | 0.796 | +0.009 [−0.073, +0.103] |
| **hybrid RRF + LLM re-rank (default)** | **0.344** | **0.944** | **0.957** | **+0.154 [+0.056, +0.261]** |
| full pipeline, no contextual header | 0.292 | 0.821 | 0.841 | +0.030 |
| full pipeline, fixed 320-token chunks | 0.303 | 0.885 | 0.835 | +0.094 |
| full pipeline, text-embedding-3-large | 0.359 | 0.962 | 0.949 | +0.171 |

- **Hybrid vs. dense:** fusion alone barely moves Recall@5 (+0.009) but raises Hit@5 (0.92 → 0.95) and MRR (0.75 → 0.80). BM25 wins on identifier questions, dense wins on paraphrases, and the gains cancel in the average. The **re-ranker is where the lift comes from**: it has a larger, more diverse candidate pool to choose from.
- **Contextual headers** are the most valuable chunking decision: removing them costs 0.12 R@5 in the full pipeline.
- P@5 looks low because most questions have 1–3 gold passages, so 5 retrieved chunks cap precision near 0.4–0.6.

**End to end** (generator gpt-4.1-mini, judge gpt-4.1)

| Metric | dev | test |
|---|---|---|
| Status accuracy (answer / clarify / decline) | 0.963 | 0.957 |
| Key-fact accuracy (answerable questions) | 0.955 | 1.000 |
| Faithfulness (share of statements supported by retrieved context) | 0.982 | 1.000 |

## Where it fails

- **Ambiguous questions** (3/3 missed): "How often is refresher training required?" gets a list of intervals by topic rather than a clarifying question. The answer is useful and faithful, but the status is wrong.
- **Out of scope vs. insufficient:** "reset my VPN password" is declined as *insufficient context* rather than *out of scope*. The user still gets a correct refusal; the label is wrong (the strict label is counted in `results/e2e.md`).
- **Conflict and version questions** have the lowest recall (0.78–0.80). The informal e-mail and the superseded policy rank high because they match the question's wording best. The generator still answered correctly, following the authority rule in the prompt, but that rule lives in the prompt, not in retrieval.
- **Evaluation limits:** 74 questions give wide confidence intervals, the synthetic documents are cleaner than real ones, and the same person wrote the corpus and the questions.

## Production notes

- **Freshness:** `sync_index()` uses LangChain's Indexing API. Chunks are hashed, only new or changed ones are embedded, and chunks of removed files are deleted. Adding one document re-embedded 2 chunks and skipped 964 (demo in the app). Run it from a document-management webhook or a nightly eCFR check.
- **1,000 concurrent queries:** the bottleneck is LLM calls (re-rank, generate, grade: 3 per query, 5–8 s). FAISS and BM25 take milliseconds. To scale: async workers with `abatch`, provider rate-limit budgeting, caching of frequent questions, and a hosted cross-encoder in place of the LLM re-ranker. Move to pgvector for multi-writer, ACL-filtered retrieval.
- **Before real deployment:** add document-level access control (HR policy vs. EHS SOPs), log questions and answers for review, and add human sign-off before an answer changes an LMS assignment rule.
- **Security:** the Gradio demo binds to localhost with no authentication. Put it behind SSO before exposing it.

## Reproduce

```bash
uv venv --python 3.11 .venv && uv pip install -r requirements.txt -e .
cp .env.example .env                   # add OPENAI_API_KEY (or point to Ollama/vLLM)
python data/fetch_regulations.py       # optional: regulations are committed; this re-fetches them
python data/generate_synthetic.py      # optional: synthetic docs are committed; deterministic
python -m lcrag.store                  # build the index (about 1k chunks, a few cents)
python eval/evaluate.py retrieval      # results/retrieval.md
python eval/evaluate.py e2e            # results/e2e.md
python app/app.py                      # http://127.0.0.1:7860
```

A full evaluation run costs roughly $1–2 in OpenAI usage.
