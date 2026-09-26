"""Load the Markdown corpus and split it with LangChain text splitters.

Strategies (compared in eval/evaluate.py):
  structure  MarkdownHeaderTextSplitter: one section per regulation paragraph "(l)" or policy
             section "7.". Token-bounded recursive split inside a section. A contextual header on
             every chunk (citation, title, practitioner topic, section path, version/status).
  nohdr      same splits without the header (ablation)
  fixed      token-bounded recursive split of the whole document (common baseline)
"""

from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .config import CORPUS_DIRS


def load_documents(dirs: list[Path] = CORPUS_DIRS) -> list[Document]:
    docs = []
    for path in (p for d in dirs if d.exists() for p in sorted(d.glob("*.md"))):
        _, fm, body = path.read_text(encoding="utf-8").split("---\n", 2)
        meta = {k: str(v) for k, v in yaml.safe_load(fm).items()} | {"source": path.name}
        docs.append(Document(page_content=body, metadata=meta))
    return docs


def _header(m: dict) -> str:
    head = f"[{m.get('citation', m['doc_id'])}] {m.get('title', '')}" + (f" ({m['topic']})" if m.get("topic") else "")
    if m["doc_type"] not in ("regulation", "statute"):
        head += f" | {m['doc_type']} v{m.get('version')}, effective {m.get('effective_date')}"
    if m.get("status") == "superseded":
        head += f" | SUPERSEDED on {m.get('superseded_on')}"
    path = " > ".join(m[h] for h in ("h2", "h3") if m.get(h))
    return head + (f"\n{path}" if path else "")


def split_documents(docs: list[Document], strategy: str = "structure", tokens: int = 320) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=tokens, chunk_overlap=32)
    if strategy == "fixed":
        return splitter.split_documents(docs)
    md = MarkdownHeaderTextSplitter([("#", "h1"), ("##", "h2"), ("###", "h3")])
    return [Document(page_content=piece if strategy == "nohdr" else f"{_header(meta)}\n{piece}", metadata=meta)
            for doc in docs for sec in md.split_text(doc.page_content)
            for meta in [doc.metadata | sec.metadata] for piece in splitter.split_text(sec.page_content)]
