"""End-to-end question answering: query gate -> retrieval -> grounded generation -> guardrails.

The generator returns structured JSON: a status, an answer with inline [S#] tags, and an optional
clarifying question. For verification, the answer the user will see is split into sentences, and
each sentence is checked against the sources it cites. Model-declared "claims" lists were tried
first and rejected: the model can drop a wrong statement from the claims list while keeping it in
the answer text, so what gets verified is not what gets shown.

Statuses
  answered              grounded answer produced
  needs_clarification   the answer depends on facts the user did not give (role, site, ...); the
                        model asks one question and may give a conditional answer
  insufficient_context  in-domain, but the retrieved sources do not contain the answer
  unverified            an answer was generated but the grounding checks failed even after one
                        revision; it is shown with a warning, never as a confident answer
  out_of_scope          rejected before generation: the best re-ranker score is below a threshold
                        calibrated on the dev split (no LLM call is spent)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from .config import get_llm_config
from .guardrails import check_grounding
from .llm import chat, parse_json
from .query import is_underspecified
from .retrieval import Hit, Retriever

OOS_THRESHOLD = 0.002  # calibrated on the dev split (eval/calibrate_oos.py); re-ranker relevance in [0, 1]

AUTHORITY_LABEL = {
    "regulation": "REGULATION (legally binding)",
    "statute": "STATUTE (legally binding)",
    "policy": "INTERNAL POLICY",
    "sop": "INTERNAL PROCEDURE (SOP)",
    "catalog": "COURSE CATALOG",
    "matrix": "ROLE CURRICULUM MATRIX",
    "audit": "INTERNAL AUDIT REPORT",
    "faq": "FAQ / GUIDANCE",
    "release_notes": "LMS RELEASE NOTES",
    "email": "INFORMAL E-MAIL (not authoritative)",
}

SYSTEM_PROMPT = """You are a compliance-training assistant for Northwind Manufacturing & Health. You answer questions from compliance, HR, EHS and LMS administrators using ONLY the numbered sources provided.

Rules:
1. Use only facts stated in the sources. Do not use outside knowledge, even if you believe it is correct.
2. Every claim must cite its source tags, e.g. [S2]. Cite only tags that appear in the sources.
3. Distinguish what the LAW requires (regulation/statute sources) from what Northwind POLICY requires. Northwind policy is often stricter; when both are available, state both.
4. Authority order when sources disagree: regulation/statute > current internal policy > SOP/catalog/matrix > audit/FAQ/release notes > e-mail. Never base an answer on a SUPERSEDED document or an informal e-mail when a current authoritative source covers the point. If a lower-authority source contradicts a higher one, say so briefly.
5. If the question is underspecified, i.e. the answer depends on something the user did not say (which training topic, job role, site or state) and different cases have different answers, set status to "needs_clarification", ask one short clarifying question, and briefly summarise the main cases. Example: "How often is refresher training required?" does not say which training, and intervals differ by topic, so ask which training is meant.
6. If the sources do not contain the answer, set status to "insufficient_context" and say what is missing. Do not guess. If the question is not about compliance training, training policy, courses or LMS procedures at all (e.g. benefits, IT support, general knowledge), set status to "out_of_scope".
7. Be concise: at most 6 sentences. Quote exact intervals, day counts and course codes.
8. Put the [S#] tag(s) at the end of every sentence that states a fact; each sentence is verified against the sources it cites. Write self-contained sentences that name their subject (e.g. "Forklift operators must be re-evaluated every 36 months [S2].").

Training topics in this knowledge base (a question about "training" or "refresher" that names none of them is underspecified): bloodborne pathogens, HIPAA privacy, HIPAA security, harassment prevention, forklift / powered industrial trucks, respiratory protection, lockout/tagout, hazard communication, fire extinguishers, hearing conservation, confined spaces, emergency action plan, code of conduct.

Return a JSON object only:
{
  "status": "answered" | "needs_clarification" | "insufficient_context" | "out_of_scope",
  "answer": "<answer text with inline [S#] tags>",
  "clarifying_question": "<question or null>"
}"""


_SENT_RE = re.compile(r"(?<=[.!?])(?:\s*\[S\d+\](?:\s*(?:,|and)?\s*\[S\d+\])*)?\s+(?=[A-Z])")


def answer_claims(answer: str, tags: list[str]) -> list[dict]:
    """Split an answer into sentence-level claims with the source tags each sentence cites.

    Tags written after the period ("... 36 months. [S2]") are attached to the sentence before them.
    A sentence without tags inherits the previous sentence's tags; an answer without any tags is
    checked against all sources, and the missing citations are reported separately.
    """
    parts, last = [], 0
    for m in _SENT_RE.finditer(answer):
        parts.append(answer[last : m.end()].strip())
        last = m.end()
    parts.append(answer[last:].strip())
    claims, prev = [], []
    for sent in (p for p in parts if p):
        found = re.findall(r"\[(S\d+)\]", sent)
        text = re.sub(r"\s*\[S\d+\]", "", sent)
        text = re.sub(r"\s*(,|\band\b)\s*(?=[.!?]$)", "", text).strip()
        if len(re.findall(r"[A-Za-z]+", text)) < 4:  # fragments such as "Yes." carry no checkable fact
            continue
        srcs = found or prev or tags
        claims.append({"text": text, "sources": srcs, "cited": bool(found or prev)})
        prev = found or prev
    return claims


def format_sources(hits: list[Hit]) -> tuple[str, dict[str, str], dict[str, Hit]]:
    blocks, texts, by_tag = [], {}, {}
    for i, h in enumerate(hits, start=1):
        tag = f"S{i}"
        c = h.chunk
        label = AUTHORITY_LABEL.get(c.meta.get("doc_type", ""), c.meta.get("doc_type", "").upper())
        if c.meta.get("status") == "superseded":
            label = f"SUPERSEDED {label} (historical only, replaced on {c.meta.get('superseded_on')})"
        date = c.meta.get("effective_date") or c.meta.get("as_of") or ""
        blocks.append(f"[{tag}] {c.cite} | {label}{' | effective ' + date if date else ''}\n{c.body}")
        texts[tag] = f"{c.header}\n\n{c.body}" if c.header else c.body
        by_tag[tag] = h
    return "\n\n".join(blocks), texts, by_tag


@dataclass
class Answer:
    question: str
    status: str
    answer: str
    clarifying_question: str | None = None
    citations: list[dict] = field(default_factory=list)  # [{"tag", "cite", "chunk_id"}]
    grounding: dict = field(default_factory=dict)
    retrieval: list[dict] = field(default_factory=list)
    contexts: list[dict] = field(default_factory=list)  # [{"tag", "cite", "text"}]
    timings_ms: dict = field(default_factory=dict)
    model: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


class Assistant:
    def __init__(self, retriever: Retriever, oos_threshold: float = OOS_THRESHOLD, k: int | None = None,
                 verify: bool = True, retry: bool = True) -> None:
        self.retriever, self.oos_threshold, self.k, self.verify = retriever, oos_threshold, k, verify
        self.retry = retry

    def _generate(self, messages: list[dict], texts: dict[str, str]) -> dict:
        raw = chat(messages, json_mode=True)
        try:
            out = parse_json(raw)
        except (json.JSONDecodeError, ValueError):
            out = {"status": "insufficient_context", "answer": "The model returned an unparseable response.",
                   "claims": []}
        status = out.get("status", "answered")
        if status not in ("answered", "needs_clarification", "insufficient_context", "out_of_scope"):
            status = "answered"
        answer_text = str(out.get("answer", "")).strip()
        claims = answer_claims(answer_text, list(texts))
        cited = sorted(set(re.findall(r"\[(S\d+)\]", answer_text)))
        grounding = {}
        if self.verify and status not in ("insufficient_context", "out_of_scope"):
            grounding = check_grounding(claims, texts, cited).to_dict()
        return {"raw": raw, "out": out, "status": status, "answer": answer_text, "cited": cited,
                "grounding": grounding}

    def ask(self, question: str) -> Answer:
        t_all = time.perf_counter()
        res = self.retriever.retrieve(question, k=self.k)
        timings = dict(res.timings_ms)
        retrieval = [h.to_dict() for h in res.hits]
        top = res.top_rerank_score
        model = get_llm_config().model

        if top is not None and top < self.oos_threshold:
            return Answer(question, "out_of_scope",
                          "I can only answer questions about compliance-training requirements, policies, courses and "
                          "LMS procedures covered by the indexed documents. I could not find anything relevant to this "
                          "question.", retrieval=retrieval, timings_ms=timings | {"total": _ms(t_all)}, model=model)

        context, texts, by_tag = format_sources(res.hits)
        hint = ("\n\nQuery analysis: the question does not name a training topic, course or rule. Retrieval results "
                "may cover only one topic by chance; treat the question as underspecified (rule 5)."
                if is_underspecified(question) else "")
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Sources:\n\n{context}\n\nQuestion: {question}{hint}"}]
        t0 = time.perf_counter()
        gen = self._generate(messages, texts)
        timings["llm"] = _ms(t0)

        # One self-correction round, only when the checks fail: the unsupported claims are fed back
        # and the model is asked to revise using only the sources. The revision is kept only if its
        # support rate is higher. Cost is paid on failures only.
        retried = False
        if self.verify and self.retry and gen["grounding"].get("verdict") in ("unsupported", "contradicted",
                                                                              "invalid_citation"):
            bad = [c["text"] for c in gen["grounding"]["claims"] if c["label"] != "supported"]
            feedback = ("Automatic verification could not find support in the cited sources for these claims:\n- "
                        + "\n- ".join(bad or gen["grounding"].get("invalid_citations", []))
                        + "\nRevise the answer. Remove or correct any claim that the sources do not state explicitly, "
                          "and keep the same JSON format.")
            t0 = time.perf_counter()
            gen2 = self._generate(messages + [{"role": "assistant", "content": gen["raw"]},
                                              {"role": "user", "content": feedback}], texts)
            timings["llm_retry"] = _ms(t0)
            retried = True
            if _quality(gen2["grounding"]) > _quality(gen["grounding"]):
                gen = gen2
        status, answer_text, grounding, out = gen["status"], gen["answer"], gen["grounding"], gen["out"]
        if grounding:
            grounding["retried"] = retried
        if grounding.get("verdict") in ("unsupported", "contradicted", "invalid_citation") and status == "answered":
            status = "unverified"  # shown to the user with a warning, never as a confident answer
        cited = gen["cited"]

        citations = [{"tag": t, "cite": by_tag[t].chunk.cite, "chunk_id": by_tag[t].chunk.chunk_id}
                     for t in cited if t in by_tag]
        contexts = [{"tag": t, "cite": h.chunk.cite, "doc_type": h.chunk.meta.get("doc_type"),
                     "status": h.chunk.meta.get("status"), "text": h.chunk.body} for t, h in by_tag.items()]
        timings["total"] = _ms(t_all)
        return Answer(question, status, answer_text, out.get("clarifying_question") or None, citations, grounding,
                      retrieval, contexts, timings, model)


def _quality(g: dict) -> tuple:
    """Order revisions: no contradicted claims first, then higher support rate."""
    contradicted = sum(c["label"] == "contradicted" for c in g.get("claims", []))
    return (g.get("verdict") not in ("contradicted", "invalid_citation"), -contradicted, g.get("support_rate", 0))


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def build_default_assistant(chunks_name: str = "structure350", model: str | None = None) -> Assistant:
    from .config import get_retrieval_config
    from .index import CorpusIndex

    model = model or get_retrieval_config().embed_model
    return Assistant(Retriever(CorpusIndex.load(chunks_name, model), mode="hybrid_rerank"))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Ask a question from the command line")
    ap.add_argument("question")
    args = ap.parse_args()
    ans = build_default_assistant().ask(args.question)
    print(json.dumps({k: v for k, v in ans.to_dict().items() if k != "contexts"}, indent=2))


if __name__ == "__main__":
    main()
