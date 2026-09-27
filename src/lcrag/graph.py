"""Agentic RAG with LangGraph.

  START -> contextualize -> guard_input --blocked--> finalize -> END
                                 \\-> plan <-> search            (agent loop, at most MAX_SEARCHES)
                                        \\-> clarify -> finalize (underspecified question)
                                        \\-> small_talk -> finalize (greeting, thanks: fixed reply, no search)
                                        \\-> generate -> grade --unsupported, 1st try--> generate
                                                           \\-> guard_output -> finalize -> END

contextualize  rewrite a follow-up into a standalone question (memory = SQLite checkpointer per thread)
guard_input    NeMo input rails: PII regex, jailbreak / off-topic / evasion
plan           tool-calling agent chooses what to search and where (regulations vs internal documents),
               or asks the user to clarify an underspecified question; its first turn must call a tool
search         hybrid retrieval with the caller's role and as-of date (the agent cannot change them)
generate       structured answer with [S#] citations and a status
grade          lists unsupported statements -> one revision, else status "unverified" (needs review)
guard_output   NeMo output rails: PII regex, disclosure / evasion / injected instructions
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from . import prompts
from .config import CHECK_MODEL, chat_model
from .guardrails import check_input, check_output
from .retrieval import make_retriever

MAX_SEARCHES, MAX_DOCS = 3, 10  # bound the cost of the agent loop and the prompt size


class Search(BaseModel):
    """Search the compliance knowledge base. Returns the most relevant passages."""

    query: str = Field(description="Focused search query")
    source: Literal["any", "regulation", "internal"] = Field(
        "any", description="regulation = laws and regulations only; internal = Northwind documents only")


class Clarify(BaseModel):
    """Ask the user one short question when the answer depends on information they did not give."""

    question: str = Field(description="The clarifying question")
    examples: list[str] = Field(description="At most 3 example cases the user might mean")


class SmallTalk(BaseModel):
    """Reply to a greeting, thanks, goodbye or a question about what the assistant can do. No search."""

    kind: Literal["greeting", "thanks", "goodbye", "capabilities"]


class RAGAnswer(BaseModel):
    answer: str = Field(description="Answer with inline [S#] citations")  # answer before status: better status choice
    status: Literal["answered", "needs_clarification", "insufficient_context", "out_of_scope"] = Field(
        description="answered (default): the requested facts are given. needs_clarification: rule 4. "
                    "insufficient_context: sources lack the answer. out_of_scope: not about compliance training.")
    clarifying_question: str | None = Field(None, description="Only when status is needs_clarification")


class Grade(BaseModel):
    unsupported_statements: list[str] = Field(description="Answer statements NOT supported by the sources")


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # conversation memory (persisted)
    question: str
    role: str
    as_of: str | None
    # per-turn working state, reset in contextualize
    standalone: str
    scratch: list[AnyMessage]  # the agent's tool-calling transcript
    searches: list[dict]  # [{"query", "source"}]
    docs: list[Document]  # evidence accumulated across searches
    answer: str
    clarifying_question: str | None
    status: str  # RAGAnswer.status, or "unverified" / "blocked" / "small_talk"
    unsupported: list[str]
    attempts: int
    blocked_by: str | None


def format_sources(docs: list[Document]) -> str:
    return "\n\n".join(f"[S{i}] ({d.metadata['doc_type']}, {d.metadata.get('status', 'current')})\n{d.page_content}"
                       for i, d in enumerate(docs[:MAX_DOCS], start=1))


def build_graph(vs, checkpointer=None):
    generator = chat_model().with_structured_output(RAGAnswer)
    grader = chat_model(CHECK_MODEL).with_structured_output(Grade)
    planner = chat_model(CHECK_MODEL)  # planning is a decision task; nano skipped needed searches
    rewriter = chat_model()
    retrievers = {}  # (role, as_of, source) -> retriever

    def contextualize(s: State) -> State:
        reset = {"scratch": [], "searches": [], "docs": [], "unsupported": [], "attempts": 0, "blocked_by": None,
                 "clarifying_question": None}
        history = s.get("messages", [])[-6:]  # last 3 turns
        if not history:
            return reset | {"standalone": s["question"]}
        chat = "\n".join(f"{m.type}: {m.content}" for m in history)
        q = rewriter.invoke([("system", prompts.REWRITE),
                             ("human", f"Chat history:\n{chat}\n\nLatest question: {s['question']}")]).content
        return reset | {"standalone": q.strip()}

    def blocked(rail: str | None) -> State:
        return {"blocked_by": rail, "status": "blocked", "answer": prompts.REFUSAL} if rail else {}

    def guard_input(s: State) -> State:
        return blocked(check_input(s["standalone"]))

    def plan(s: State) -> State:
        when = f" (as of {s['as_of']})" if s.get("as_of") else ""
        llm = planner.bind_tools([Search, Clarify, SmallTalk], tool_choice="required" if not s["scratch"] else "auto")
        ai = llm.invoke([("system", prompts.AGENT), ("human", f"Question: {s['standalone']}{when}"), *s["scratch"]])
        return {"scratch": s["scratch"] + [ai]}

    def search(s: State) -> State:
        docs, searches, replies = list(s["docs"]), list(s["searches"]), []
        for call in s["scratch"][-1].tool_calls:
            if call["name"] != "Search":
                continue
            args = Search(**call["args"])
            if args.model_dump() in searches:
                replies.append(ToolMessage("Already searched. Use different wording or reply done.",
                                           tool_call_id=call["id"]))
                continue
            key = (s.get("role", "compliance"), s.get("as_of"), args.source)
            if key not in retrievers:  # role and as-of come from the request, never from the model
                retrievers[key] = make_retriever(vs, role=key[0], as_of=key[1], source=key[2])
            hits = retrievers[key].invoke(args.query)
            docs += [d for d in hits if d not in docs]
            searches.append(args.model_dump())
            listing = "\n".join(f"- {d.page_content[:350]}" for d in hits) or "No results."
            replies.append(ToolMessage(listing, tool_call_id=call["id"]))
        return {"docs": docs, "searches": searches, "scratch": s["scratch"] + replies}

    def after_plan(s: State) -> str:
        calls = s["scratch"][-1].tool_calls
        if any(c["name"] == "Clarify" for c in calls):
            return "clarify"
        if any(c["name"] == "SmallTalk" for c in calls) and not s["searches"]:
            return "small_talk"
        return "search" if calls and len(s["searches"]) < MAX_SEARCHES else "generate"

    def clarify(s: State) -> State:
        c = Clarify(**next(c["args"] for c in s["scratch"][-1].tool_calls if c["name"] == "Clarify"))
        cases = "; ".join(c.examples[:3])
        return {"status": "needs_clarification", "clarifying_question": c.question,
                "answer": f"The answer depends on the case{f' (for example: {cases})' if cases else ''}."}

    def small_talk(s: State) -> State:
        kind = next(c["args"]["kind"] for c in s["scratch"][-1].tool_calls if c["name"] == "SmallTalk")
        return {"status": "small_talk", "answer": prompts.SMALL_TALK.get(kind, prompts.SMALL_TALK["greeting"])}

    def generate(s: State) -> State:
        when = f"\nAs-of date: {s['as_of']}" if s.get("as_of") else ""
        msgs = [("system", prompts.ANSWER),
                ("human", f"Sources:\n\n{format_sources(s['docs'])}\n\nQuestion: {s['standalone']}{when}")]
        if s["unsupported"]:  # revision round: show the previous answer and what the grader rejected
            msgs += [("ai", s["answer"]), ("human", "Not supported by the sources: " + "; ".join(s["unsupported"])
                                          + ". Revise: remove or correct these, using only the sources.")]
        r = generator.invoke(msgs)
        return {"answer": r.answer, "status": r.status, "clarifying_question": r.clarifying_question,
                "attempts": s["attempts"] + 1}

    def grade(s: State) -> State:
        if s["status"] in ("insufficient_context", "out_of_scope"):
            return {"unsupported": []}
        bad = grader.invoke([("system", prompts.GRADE),
                             ("human", f"SOURCES:\n{format_sources(s['docs'])}\n\nANSWER:\n{s['answer']}")
                             ]).unsupported_statements
        return {"unsupported": bad, "status": "unverified" if bad and s["attempts"] >= 2 else s["status"]}

    def guard_output(s: State) -> State:
        return blocked(check_output(s["standalone"], s["answer"]))

    def finalize(s: State) -> State:
        return {"messages": [HumanMessage(s["question"]), AIMessage(s["answer"])]}

    g = StateGraph(State)
    for fn in (contextualize, guard_input, plan, search, clarify, small_talk, generate, grade, guard_output, finalize):
        g.add_node(fn.__name__, fn)
    g.add_edge(START, "contextualize")
    g.add_edge("contextualize", "guard_input")
    g.add_conditional_edges("guard_input", lambda s: "finalize" if s["blocked_by"] else "plan", ["finalize", "plan"])
    g.add_conditional_edges("plan", after_plan, ["search", "clarify", "small_talk", "generate"])
    g.add_edge("search", "plan")
    g.add_edge("clarify", "finalize")
    g.add_edge("small_talk", "finalize")
    g.add_edge("generate", "grade")
    g.add_conditional_edges("grade", lambda s: "generate" if s["unsupported"] and s["attempts"] < 2 else "guard_output",
                            ["generate", "guard_output"])
    g.add_edge("guard_output", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
