"""Parse the corpus and write chunk files for one or more chunking strategies.

Usage:
    python -m lcrag.ingest                      # default structure-aware chunks
    python -m lcrag.ingest --all                # every strategy used in experiments
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .chunking import ChunkingConfig, chunk_documents, count_tokens
from .config import PROCESSED_DIR, RAW_DIR, SYNTH_DIR
from .parsers import load_corpus
from .schema import Chunk, Document

EXPERIMENT_CHUNKERS = [
    ChunkingConfig(strategy="structure", max_tokens=350),
    ChunkingConfig(strategy="structure", max_tokens=350, contextual_header=False),
    ChunkingConfig(strategy="structure", max_tokens=200),
    ChunkingConfig(strategy="fixed", size=256, overlap=48),
    ChunkingConfig(strategy="fixed", size=512, overlap=64),
]


def docs_path() -> Path:
    return PROCESSED_DIR / "docs.jsonl"


def chunks_path(name: str) -> Path:
    return PROCESSED_DIR / f"chunks_{name}.jsonl"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_docs() -> list[Document]:
    return [Document.from_dict(d) for d in read_jsonl(docs_path())]


def load_chunks(name: str) -> list[Chunk]:
    return [Chunk.from_dict(d) for d in read_jsonl(chunks_path(name))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="write chunks for all experiment strategies")
    args = ap.parse_args()

    docs = load_corpus(RAW_DIR, SYNTH_DIR / "docs")
    write_jsonl(docs_path(), [d.to_dict() for d in docs])
    print(f"{len(docs)} documents, {sum(len(d.blocks) for d in docs)} blocks -> {docs_path()}")

    configs = EXPERIMENT_CHUNKERS if args.all else EXPERIMENT_CHUNKERS[:1]
    for cfg in configs:
        chunks = chunk_documents(docs, cfg)
        write_jsonl(chunks_path(cfg.name), [c.to_dict() for c in chunks])
        toks = [count_tokens(c.text) for c in chunks]
        print(f"{cfg.name:20s} chunks={len(chunks):5d} tokens median={statistics.median(toks):.0f} "
              f"p95={sorted(toks)[int(0.95 * len(toks))]} max={max(toks)}")


if __name__ == "__main__":
    main()
