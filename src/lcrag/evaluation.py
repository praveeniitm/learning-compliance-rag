"""Evaluation utilities: QA loading, structural relevance, retrieval metrics, answer checks.

Retrieval metrics (per question with gold passages, then macro-averaged):
  Precision@k  fraction of the top-k chunks that are relevant to at least one gold passage
  Recall@k     fraction of gold passages covered by at least one of the top-k chunks
  Hit@k        1 if any gold passage is covered in the top-k
  MRR          reciprocal rank of the first relevant chunk (within top-20)
  nDCG@k       graded by "number of gold passages newly covered" at each rank

Answer metrics:
  key-fact accuracy  every key-fact regex matches the answer (answered questions only)
  status accuracy    predicted status == expected status (all questions)
  faithfulness       share of answer sentences supported by the retrieved context, judged by an
                     LLM (offline), with the runtime NLI verdict reported alongside
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import EVAL_DIR
from .schema import Chunk


@dataclass
class QA:
    id: str
    type: str
    question: str
    gold: list[tuple[str, str]]
    key_facts: list[str]
    expect: str
    split: str = "test"
    meta: dict = field(default_factory=dict)


def load_qa(path: Path | None = None, split: str | None = None) -> list[QA]:
    path = path or EVAL_DIR / "qa.yaml"
    rows = yaml.safe_load(path.read_text(encoding="utf-8"))
    splits = load_splits()
    out = []
    for r in rows:
        qa = QA(r["id"], r["type"], r["question"], [tuple(g) for g in r.get("gold") or []],
                r.get("key_facts") or [], r["expect"], splits.get(r["id"], "test"))
        if split is None or qa.split == split:
            out.append(qa)
    return out


def load_splits() -> dict[str, str]:
    p = EVAL_DIR / "splits.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def assign_splits(qas: list[QA], dev_fraction: float = 0.35, seed: str = "lcrag-v1") -> dict[str, str]:
    """Stratified by question type; deterministic ordering by a seeded hash of the id."""
    by_type: dict[str, list[QA]] = {}
    for q in qas:
        by_type.setdefault(q.type, []).append(q)
    out = {}
    for items in by_type.values():
        items = sorted(items, key=lambda q: hashlib.sha1(f"{seed}:{q.id}".encode()).hexdigest())
        n_dev = max(1, round(len(items) * dev_fraction))
        for i, q in enumerate(items):
            out[q.id] = "dev" if i < n_dev else "test"
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------------------------
# relevance
# --------------------------------------------------------------------------------------------
def anchor_covers(gold: str, anchor: str) -> bool:
    """Is `anchor` equal to or nested under `gold`?  '(l)(4)' covers '(l)(4)(iii)'; '7' covers '7.2'."""
    if gold == "":
        return True
    if anchor == gold:
        return True
    if gold.startswith("("):
        return anchor.startswith(gold)
    return anchor.startswith(gold + ".")


def covered_gold(chunk: Chunk, gold: list[tuple[str, str]]) -> set[int]:
    return {i for i, (doc, anc) in enumerate(gold)
            if chunk.doc_id == doc and any(anchor_covers(anc, a) for a in (chunk.anchors or [""]))}


def retrieval_metrics(ranked: list[Chunk], gold: list[tuple[str, str]], ks=(1, 3, 5, 10)) -> dict:
    cover = [covered_gold(c, gold) for c in ranked]
    m: dict[str, float] = {}
    for k in ks:
        top = cover[:k]
        m[f"P@{k}"] = sum(bool(c) for c in top) / k
        got = set().union(*top) if top else set()
        m[f"R@{k}"] = len(got) / len(gold)
        m[f"Hit@{k}"] = float(bool(got))
    first = next((i for i, c in enumerate(cover[:20]) if c), None)
    m["MRR"] = 0.0 if first is None else 1.0 / (first + 1)
    # nDCG@5 with gain = number of gold passages newly covered at that rank
    seen: set[int] = set()
    dcg = 0.0
    for i, c in enumerate(cover[:5]):
        new = c - seen
        seen |= c
        dcg += len(new) / math.log2(i + 2)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold), 5)))
    m["nDCG@5"] = dcg / ideal if ideal else 0.0
    return m


def mean_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in rows[0]}


# --------------------------------------------------------------------------------------------
# answers
# --------------------------------------------------------------------------------------------
def key_fact_hits(answer: str, key_facts: list[str]) -> list[bool]:
    text = re.sub(r"\[S\d+\]", "", answer)
    return [bool(re.search(p, text, flags=re.I)) for p in key_facts]


JUDGE_PROMPT = """You are auditing a retrieval-augmented answer for faithfulness. For each numbered answer sentence, decide whether it is fully supported by the SOURCES (explicitly stated or directly implied). Restating the user's question context (e.g. a site name the user mentioned) is fine. Outside knowledge does not count as support, even if true.

SOURCES:
{sources}

ANSWER SENTENCES:
{sentences}

Return JSON only: {{"judgements": [{{"n": 1, "supported": true, "reason": "<short>"}}, ...]}}"""
