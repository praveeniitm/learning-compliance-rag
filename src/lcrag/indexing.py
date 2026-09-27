"""Indexing pipeline: load -> chunk -> embed -> store, with delta updates.

    python -m lcrag.indexing        # first run builds the index; later runs apply only the changes

How changes are handled (LangChain Indexing API with a SQLite "record manager"):
  * Every chunk is hashed (text + metadata), and the record manager remembers which hashes are in
    FAISS and which source file each came from.
  * New file             -> its chunks are embedded and added.
  * Unchanged file       -> every chunk hash is already known: skipped, nothing is embedded.
  * Edited file          -> only chunks whose text changed are embedded; the file's old chunks are deleted.
  * Deleted file         -> all of its chunks are deleted from FAISS.
  * Superseded version   -> stays indexed (needed for "what applied before?"), but is labelled
                            SUPERSEDED in every chunk and filtered out by as-of queries.
cleanup="full" means "the corpus passed in is the complete corpus", which is what makes the
deletion of removed files possible. The whole corpus is re-read and re-hashed each run (cheap:
about 1 s); only changed text costs embedding calls.
"""

import faiss
from langchain_classic.indexes import SQLRecordManager, index
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from .config import EMBED_MODEL, INDEX_DIR, embedding_model
from .corpus import load_documents, split_documents


def sync_index(strategy: str = "structure", model: str = EMBED_MODEL) -> tuple[FAISS, dict]:
    """Bring the FAISS index in line with the corpus. Returns the store and the change counts."""
    path = INDEX_DIR / f"{strategy}__{model}"
    emb = embedding_model(model)
    if (path / "index.faiss").exists():
        vs = FAISS.load_local(str(path), emb, allow_dangerous_deserialization=True)  # a file we wrote ourselves
    else:
        vs = FAISS(emb, faiss.IndexFlatL2(len(emb.embed_query("x"))), InMemoryDocstore(), {})
    path.mkdir(parents=True, exist_ok=True)
    records = SQLRecordManager(f"lcrag/{path.name}", db_url=f"sqlite:///{path / 'records.sqlite'}")
    records.create_schema()
    chunks = split_documents(load_documents(), strategy)
    stats = index(chunks, records, vs, cleanup="full", source_id_key="source")
    vs.save_local(str(path))
    return vs, stats


if __name__ == "__main__":
    vs, s = sync_index()
    print(f"added {s['num_added']}, updated {s['num_updated']}, skipped (unchanged) {s['num_skipped']}, "
          f"deleted {s['num_deleted']} -> {vs.index.ntotal} chunks in the index")
