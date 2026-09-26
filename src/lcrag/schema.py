"""Core data types shared by ingestion, indexing, retrieval and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Block:
    """Smallest structural unit of a document: a regulation paragraph, a policy clause, a table.

    `anchor` is the citable address inside the document, e.g. "(l)(4)(iii)" for a CFR
    paragraph or "7.1" for a policy clause. `trail` is the chain of headings above the block
    (outermost first), used to give chunks their context header.
    """

    kind: str  # "heading" | "para" | "table"
    text: str
    anchor: str = ""
    trail: list[str] = field(default_factory=list)
    top: str = ""  # top-level structural unit, e.g. "(l)" or "7"; chunks never cross it
    appendix: bool = False


@dataclass
class Document:
    doc_id: str
    title: str
    doc_type: str  # regulation | statute | policy | sop | catalog | matrix | audit | faq | email | release_notes
    citation: str  # human-readable prefix used in citations, e.g. "29 CFR 1910.178"
    blocks: list[Block]
    meta: dict = field(default_factory=dict)  # version, status, effective_date, source_url, sha256, ...

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Document":
        return Document(**{**d, "blocks": [Block(**b) for b in d["blocks"]]})


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str  # what is embedded / indexed (may include a contextual header)
    body: str  # the document text only, shown to the LLM and the user
    anchors: list[str]  # anchors of all blocks fully or partly inside the chunk
    header: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def cite(self) -> str:
        """Citation label for the chunk, e.g. '29 CFR 1910.178(l)(4)'."""
        prefix = self.meta.get("citation", self.doc_id)
        if not self.anchors:
            return prefix
        return format_cite(prefix, self.anchors[0])

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        return Chunk(**d)


def format_cite(prefix: str, anchor: str) -> str:
    """'29 CFR 1910.178' + '(l)(4)' -> '29 CFR 1910.178(l)(4)'; 'POL-CT-001-v3.2' + '7.1' -> '... §7.1'."""
    if not anchor:
        return prefix
    if anchor.startswith("("):
        return f"{prefix}{anchor}"
    if anchor[0].isdigit():
        return f"{prefix} §{anchor}"
    return f"{prefix} {anchor}"
