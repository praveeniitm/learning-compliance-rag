"""The assistant used by the app and API: graph + persistent memory + audit log."""

import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone

from langchain_core.callbacks import get_usage_metadata_callback
from langgraph.checkpoint.sqlite import SqliteSaver

from .config import LOG_DIR, ROOT
from .graph import MAX_DOCS, build_graph
from .indexing import sync_index

def _log(name: str, record: dict) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_DIR / name, "a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds")} | record) + "\n")


class Assistant:
    def __init__(self):
        conn = sqlite3.connect(ROOT / ".cache" / "memory.sqlite", check_same_thread=False)
        self.memory = SqliteSaver(conn)
        self.reload()

    def reload(self) -> dict:
        """Sync the index with the corpus (only changed chunks are embedded) and rebuild the graph."""
        self.vs, stats = sync_index()
        self.graph = build_graph(self.vs, checkpointer=self.memory)
        return stats

    def ask(self, question: str, thread_id: str | None = None, role: str = "compliance", as_of: str | None = None):
        thread_id = thread_id or uuid.uuid4().hex
        t0 = time.perf_counter()
        with get_usage_metadata_callback() as cb:
            out = self.graph.invoke({"question": question, "role": role, "as_of": as_of or None},
                                    {"configurable": {"thread_id": thread_id}})
        out["thread_id"] = thread_id
        out["latency_s"] = round(time.perf_counter() - t0, 2)
        out["sources"] = [f"{d.metadata['doc_id']} | {d.metadata.get('h2', '')}" for d in out["docs"][:MAX_DOCS]]
        # audit log: who asked what, what was searched and cited, token usage; "unverified" answers form the review queue
        _log("audit.jsonl", {k: out.get(k) for k in (
            "thread_id", "role", "as_of", "question", "standalone", "status", "blocked_by", "unsupported",
            "answer", "searches", "sources", "latency_s")} | {"tokens": cb.usage_metadata})
        return out

    def feedback(self, thread_id: str, rating: str, comment: str = "") -> None:
        _log("feedback.jsonl", {"thread_id": thread_id, "rating": rating, "comment": comment})
