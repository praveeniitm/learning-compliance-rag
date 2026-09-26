"""LangGraph workflow (corrective / self-RAG pattern):

    START -> retrieve -> generate -> grade --(all statements supported, or declined)--> END
                            ^------------(unsupported statements, first attempt)--/

generate: structured output with an answer and [S#] citations, a status and an optional clarifying
question. grade: an LLM grader lists the answer statements the sources do not support; if there
are any, the generator revises once with that feedback. If it still fails, the status becomes
"unverified" so the UI warns instead of presenting a confident answer.
"""

from typing import Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .config import chat_model

SYSTEM = """You are a compliance-training assistant for Northwind Manufacturing & Health. Answer questions from compliance, HR, EHS and LMS administrators using ONLY the numbered sources.
1. Use only facts in the sources, never outside knowledge. Cite every factual sentence, e.g. [S2].
2. Separate what the LAW requires (regulation/statute) from what Northwind POLICY requires (often stricter). State both when available.
3. Authority when sources disagree: regulation/statute > current policy > SOP/catalog/matrix > audit/FAQ/release notes > e-mail. Never rely on a SUPERSEDED document or an e-mail when a current authoritative source covers the point, and point out the contradiction.
4. If the question does not name the training topic, role, site or state it is about and the answer differs by case, set status needs_clarification: ask one short question and list at most 3 example cases. Example: "How often is refresher training required?" names no topic, and intervals differ by topic.
5. If the sources do not contain the answer, say what is missing. Do not guess.
6. At most 6 sentences. Quote exact intervals, day counts and course codes. Cite as [S1][S2], not [S1, S2]."""


class RAGAnswer(BaseModel):
    answer: str = Field(description="Answer with inline [S#] citations")  # answer before status: better status choice
    status: Literal["answered", "needs_clarification", "insufficient_context", "out_of_scope"] = Field(
        description="answered (default): the requested facts are given. needs_clarification: rule 4. "
                    "insufficient_context: sources lack the answer. out_of_scope: not about compliance training.")
    clarifying_question: str | None = Field(None, description="Only when status is needs_clarification")


class Grade(BaseModel):
    unsupported_statements: list[str] = Field(description="Answer statements NOT supported by the sources")


class State(TypedDict, total=False):
    question: str
    docs: list[Document]
    result: RAGAnswer
    unsupported: list[str]
    attempts: int
    status: str


def format_sources(docs: list[Document]) -> str:
    return "\n\n".join(f"[S{i}] ({d.metadata['doc_type']}, {d.metadata.get('status', 'current')})\n{d.page_content}"
                       for i, d in enumerate(docs, start=1))


def build_graph(retriever):
    generator = chat_model().with_structured_output(RAGAnswer)
    grader = chat_model().with_structured_output(Grade)

    def retrieve(s: State) -> State:
        return {"docs": retriever.invoke(s["question"]), "attempts": 0}

    def generate(s: State) -> State:
        msgs = [("system", SYSTEM), ("human", f"Sources:\n\n{format_sources(s['docs'])}\n\nQuestion: {s['question']}")]
        if s.get("unsupported"):
            msgs += [("ai", s["result"].model_dump_json()), ("human", "Not supported by the sources: "
                     + "; ".join(s["unsupported"]) + ". Revise: remove or correct these, using only the sources.")]
        res = generator.invoke(msgs)
        return {"result": res, "attempts": s["attempts"] + 1, "status": res.status}

    def grade(s: State) -> State:
        if s["result"].status in ("insufficient_context", "out_of_scope"):
            return {"unsupported": []}
        bad = grader.invoke([("system", "List every statement in the ANSWER that the SOURCES do not support "
                                        "(not stated, or contradicted). Supported: restating the question, "
                                        "reporting what a source says (even an incorrect e-mail), and conclusions "
                                        "that follow directly from the sources."),
                             ("human", f"SOURCES:\n{format_sources(s['docs'])}\n\nANSWER:\n{s['result'].answer}")
                             ]).unsupported_statements
        return {"unsupported": bad, "status": "unverified" if bad and s["attempts"] >= 2 else s["result"].status}

    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("grade", grade)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "grade")
    g.add_conditional_edges("grade", lambda s: "generate" if s["unsupported"] and s["attempts"] < 2 else END,
                            ["generate", END])
    return g.compile()


def build_app(strategy: str = "structure", mode: str = "hybrid_rerank"):
    from .store import make_retriever, sync_index

    return build_graph(make_retriever(sync_index(strategy)[0], mode))
