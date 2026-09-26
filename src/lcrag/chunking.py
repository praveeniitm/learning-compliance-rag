"""Chunking strategies.

`structure` (default)
    Structure-aware packing. Consecutive blocks are packed into a chunk until a token budget is
    reached, but a chunk never crosses a top-level unit (a CFR paragraph such as "(l)", or a policy
    section such as "7."), and once a chunk has reached `min_tokens` it closes at the next
    second-level boundary (e.g. between "(l)(3)" and "(l)(4)"). Each chunk's embedded text gets a
    contextual header (citation, document title, heading trail, version/status) so a child
    paragraph like "(iii) ... at least once every three years" still says it is about *forklift
    operator evaluation*. Without that header the paragraph cannot be retrieved for "how often are
    forklift operators evaluated".

`fixed`
    Baseline: sliding windows of `size` tokens with `overlap` over the concatenated document
    text, no header. This is the common default in RAG tutorials and the comparison point.

Both strategies record which block anchors each chunk covers. Evaluation relevance is defined on
(doc_id, anchor), so the two strategies are compared on the same ground truth.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from functools import lru_cache

from .schema import Block, Chunk, Document

TOKENIZER_NAME = "BAAI/bge-small-en-v1.5"  # BERT WordPiece; counts are close for other BERT-family encoders


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def count_tokens(text: str) -> int:
    return len(_tokenizer()(text, add_special_tokens=False)["input_ids"])


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "structure"  # "structure" | "fixed"
    max_tokens: int = 350  # leaves headroom under the 512-token limit of BERT-style encoders for the header
    contextual_header: bool = True
    size: int = 256  # fixed strategy
    overlap: int = 48  # fixed strategy

    @property
    def name(self) -> str:
        if self.strategy == "fixed":
            return f"fixed{self.size}o{self.overlap}"
        return f"structure{self.max_tokens}" + ("" if self.contextual_header else "-nohdr")


def _header(doc: Document, trail: list[str]) -> str:
    meta = doc.meta
    status = meta.get("status", "current")
    parts = [f"[{doc.citation}] {doc.title}" + (f" ({meta['topic']})" if meta.get("topic") else "")]
    if doc.doc_type not in ("regulation", "statute"):
        v = f"{doc.doc_type}, version {meta.get('version', '?')}, effective {meta.get('effective_date', '?')}"
        parts.append(v + (f", SUPERSEDED on {meta.get('superseded_on')}" if status == "superseded" else ""))
    ctx = [t for t in trail[1:] if t]  # trail[0] is the document title, already present
    if ctx:
        parts.append(" > ".join(ctx))
    return "\n".join(parts)


def _sub_unit(anchor: str, doc: Document) -> str:
    if doc.doc_type in ("regulation", "statute"):
        groups = re.findall(r"\([^)]+\)", anchor)
        return "".join(groups[:2]) or anchor
    return anchor


def _doc_meta(doc: Document) -> dict:
    keep = ("status", "version", "effective_date", "superseded_on", "authority", "source_url", "as_of")
    return {"citation": doc.citation, "doc_type": doc.doc_type, "title": doc.title} | {
        k: doc.meta[k] for k in keep if k in doc.meta
    }


def _make_chunk(doc: Document, idx: int, blocks: list[Block], body: str, cfg: ChunkingConfig) -> Chunk:
    anchors: list[str] = []
    for b in blocks:
        if b.anchor and b.anchor not in anchors:
            anchors.append(b.anchor)
    # Header uses the heading trail shared by every block in the chunk (common prefix), so a chunk
    # spanning (l)(5)-(l)(7) is labelled "(l) Operator training", not with (l)(5)'s title.
    trail = list(blocks[0].trail)
    for b in blocks[1:]:
        n = 0
        while n < min(len(trail), len(b.trail)) and trail[n] == b.trail[n]:
            n += 1
        trail = trail[:n]
    header = _header(doc, trail) if cfg.contextual_header else ""
    text = f"{header}\n\n{body}" if header else body
    meta = _doc_meta(doc) | {"hash": hashlib.sha1(text.encode()).hexdigest(), "appendix": blocks[0].appendix}
    return Chunk(chunk_id=f"{doc.doc_id}::{idx:04d}", doc_id=doc.doc_id, text=text, body=body,
                 anchors=anchors, header=header, meta=meta)


def _split_long(text: str, max_tokens: int, is_table: bool = False) -> list[str]:
    """Split an oversized block on sentence boundaries, or on rows for tables.

    Table pieces repeat the header row so that each piece remains interpretable on its own
    (a row "EHS-FL-201 | ... | every 36 months" is meaningless without its column names).
    """
    if is_table:
        header, *units = text.split("\n")
        sep, head_tok = "\n", count_tokens(header)
    else:
        header, units = "", re.split(r"(?<=[.;:])\s+(?=[A-Z(])", text)
        sep, head_tok = " ", 0
    out, cur, cur_tok = [], [], head_tok
    for u in units:
        t = count_tokens(u)
        if cur and cur_tok + t > max_tokens:
            out.append(sep.join(([header] if header else []) + cur))
            cur, cur_tok = [], head_tok
        cur.append(u)
        cur_tok += t
    if cur:
        out.append(sep.join(([header] if header else []) + cur))
    return out


def _groups(blocks: list[Block], key) -> list[list[Block]]:
    out: list[list[Block]] = []
    for b in blocks:
        if out and key(out[-1][-1]) == key(b):
            out[-1].append(b)
        else:
            out.append([b])
    return out


def chunk_structure(doc: Document, cfg: ChunkingConfig) -> list[Chunk]:
    """Hierarchical packing: top-level unit -> second-level units -> blocks -> sentences.

    Whole second-level units (e.g. all of 1910.178(l)(4)) are kept together whenever they fit;
    small neighbouring units are merged; only units larger than the budget are split, first on
    block (paragraph) boundaries and, for a single oversized paragraph or table, on sentences/rows.
    """
    chunks: list[Chunk] = []
    header_budget = count_tokens(_header(doc, ["", "x" * 60])) if cfg.contextual_header else 0
    budget = cfg.max_tokens - header_budget
    ntok = {id(b): count_tokens(b.text) for b in doc.blocks}

    cur: list[Block] = []
    cur_tok = 0

    def flush() -> None:
        nonlocal cur, cur_tok
        if cur:
            chunks.append(_make_chunk(doc, len(chunks), cur, "\n".join(b.text for b in cur), cfg))
        cur, cur_tok = [], 0

    def add(blocks: list[Block], tokens: int) -> None:
        nonlocal cur_tok
        if cur and cur_tok + tokens > budget:
            flush()
        cur.extend(blocks)
        cur_tok += tokens

    content = [b for b in doc.blocks if b.kind != "heading"]
    for top_group in _groups(content, key=lambda b: b.top):
        flush()  # never cross a top-level unit
        for unit in _groups(top_group, key=lambda b: _sub_unit(b.anchor, doc)):
            unit_tok = sum(ntok[id(b)] for b in unit)
            if unit_tok <= budget:
                add(unit, unit_tok)
                continue
            flush()
            for b in unit:
                t = ntok[id(b)]
                if t <= budget:
                    add([b], t)
                    continue
                flush()
                for piece in _split_long(b.text, budget, is_table=b.kind == "table"):
                    chunks.append(_make_chunk(doc, len(chunks), [b], piece, cfg))
            flush()
    flush()
    return chunks


def chunk_fixed(doc: Document, cfg: ChunkingConfig) -> list[Chunk]:
    cfg = replace(cfg, contextual_header=False)
    tok = _tokenizer()
    content = [b for b in doc.blocks if b.kind != "heading"]
    text, spans = "", []
    for b in content:
        start = len(text)
        text += b.text + "\n"
        spans.append((start, len(text), b))
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    chunks: list[Chunk] = []
    step = cfg.size - cfg.overlap
    for i in range(0, max(len(offsets), 1), step):
        window = offsets[i : i + cfg.size]
        if not window:
            break
        s, e = window[0][0], window[-1][1]
        blocks = [b for (bs, be, b) in spans if bs < e and be > s]
        chunks.append(_make_chunk(doc, len(chunks), blocks, text[s:e], cfg))
        if i + cfg.size >= len(offsets):
            break
    return chunks


def chunk_documents(docs: list[Document], cfg: ChunkingConfig) -> list[Chunk]:
    fn = chunk_fixed if cfg.strategy == "fixed" else chunk_structure
    out: list[Chunk] = []
    for d in docs:
        out.extend(fn(d, cfg))
    return out
