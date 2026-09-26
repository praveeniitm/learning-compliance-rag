"""BM25 keyword retrieval with a tokenizer that understands regulatory citations and course codes.

Why a custom tokenizer: generic word tokenizers split "1910.178(l)(4)" into meaningless pieces
("1910", "178", "l", "4") and "EHS-FL-201" into "ehs", "fl", "201". Exact identifiers are exactly
the queries where keyword search beats dense retrieval, so they are kept intact:

  * CFR section numbers:   "1910.178", plus hierarchical paragraph cites derived from each chunk's
                           anchors: "1910.178(l)", "1910.178(l)(4)", "1910.178(l)(4)(iii)"
  * Statute numbers:       "12950.1"
  * Internal identifiers:  "ehs-fl-201", "pol-ct-001", "cln-tel", "f-2"
  * Numbers with units stay as tokens ("36", "months")

Ordinary words are lowercased, stop-word filtered and lightly stemmed (plural and common suffixes).
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from .schema import Chunk

_STOP = set(
    """a an and are as at be by for from has have how i if in into is it its of on or that the their them
    then there these this to was were what when where which who why will with do does did must shall
    should can may our we you your my me any all each such than so not no""".split()
)
_ID_RE = re.compile(
    r"\b\d{2,5}\.\d+(?:\([a-zA-Z0-9]{1,5}\))*"  # 1910.178(l)(4), 164.530, 12950.1
    r"|\b[A-Za-z]{1,4}(?:-[A-Za-z0-9]{1,6})+\b"  # EHS-FL-201, POL-CT-001, CLN-TEL, F-2
)
_WORD_RE = re.compile(r"[a-z0-9]+")


def _stem(w: str) -> str:
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + rep
    return w


def _expand_id(tok: str) -> list[str]:
    """'1910.178(l)(4)' -> ['1910.178', '1910.178(l)', '1910.178(l)(4)']."""
    parts = re.findall(r"\([^)]+\)", tok)
    if not parts:
        return [tok]
    base = tok[: tok.index("(")]
    out, cur = [base], base
    for p in parts:
        cur += p
        out.append(cur)
    return out


def _is_identifier(tok: str) -> bool:
    # Keep "EHS-FL-201", "CLN-TEL", "1910.178"; let ordinary hyphenated words ("near-miss",
    # "e-learning") fall through to the word tokenizer so they still match "near miss".
    return any(ch.isdigit() for ch in tok) or tok.replace("-", "").isupper()


def tokenize(text: str) -> list[str]:
    toks: list[str] = []
    spans: list[tuple[int, int]] = []
    for m in _ID_RE.finditer(text):
        if _is_identifier(m.group(0)):
            toks.extend(t.lower() for t in _expand_id(m.group(0)))
            spans.append(m.span())
    rest = text
    for s, e in reversed(spans):
        rest = rest[:s] + " " + rest[e:]
    rest = rest.lower()
    toks.extend(_stem(w) for w in _WORD_RE.findall(rest) if w not in _STOP)
    return toks


def sparse_text(chunk: Chunk) -> str:
    """Text indexed by BM25: chunk text plus the full citations of every anchor it covers."""
    cite_prefix = chunk.meta.get("citation", chunk.doc_id)
    section = re.search(r"\d{2,5}\.\d+", cite_prefix)
    extra = []
    for a in chunk.anchors:
        if section and a.startswith("("):
            extra.append(section.group(0) + a)
    return chunk.text + "\n" + " ".join(extra)


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75) -> None:
        self.ids = [c.chunk_id for c in chunks]
        self.bm25 = BM25Okapi([tokenize(sparse_text(c)) for c in chunks], k1=k1, b=b)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = tokenize(query)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]
