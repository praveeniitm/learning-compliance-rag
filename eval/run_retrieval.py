"""Retrieval experiments (no LLM needed): retrievers, fusion, re-rankers, chunking, embeddings, ablations.

Every configuration is scored on the same structural gold passages. Results are written to
results/retrieval.json (all numbers) and results/retrieval.md (tables), split by dev/test and
broken down by question type.

Usage:
    python eval/run_retrieval.py                  # all experiment groups
    python eval/run_retrieval.py --group retriever
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import replace

from lcrag.config import RESULTS_DIR, get_retrieval_config
from lcrag.evaluation import load_qa, mean_metrics, retrieval_metrics
from lcrag.index import CorpusIndex
from lcrag.ingest import load_chunks
from lcrag.retrieval import Retriever

BASE = dict(chunks="structure350", model="BAAI/bge-small-en-v1.5", mode="hybrid_rerank",
            reranker="cross-encoder/ms-marco-MiniLM-L-6-v2", expand_query=True, legal_slot=False,
            source_policy=True, alpha=0.5)

GROUPS: dict[str, list[dict]] = {
    "retriever": [
        {"name": "dense (bge-small)", "mode": "dense"},
        {"name": "sparse (BM25)", "mode": "sparse"},
        {"name": "hybrid RRF", "mode": "hybrid"},
        {"name": "hybrid convex a=0.5", "mode": "hybrid_convex", "alpha": 0.5},
        {"name": "hybrid convex a=0.7", "mode": "hybrid_convex", "alpha": 0.7},
        {"name": "dense + rerank", "mode": "dense_rerank"},
        {"name": "hybrid RRF + rerank (default)", "mode": "hybrid_rerank"},
    ],
    "reranker": [
        {"name": "ms-marco-MiniLM-L-6 (22M)", "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
        {"name": "bge-reranker-base (278M)", "reranker": "BAAI/bge-reranker-base"},
    ],
    "ablation": [
        {"name": "default (all components)"},
        {"name": "- query expansion", "expand_query": False},
        {"name": "+ legal-source slot", "legal_slot": True},
        {"name": "+ legal-source slot, bge-reranker-base", "legal_slot": True, "reranker": "BAAI/bge-reranker-base"},
        {"name": "- superseded demotion", "source_policy": False},
        {"name": "- contextual headers", "chunks": "structure350-nohdr"},
        {"name": "- expansion, demotion, headers", "expand_query": False, "source_policy": False,
         "chunks": "structure350-nohdr"},
    ],
    "chunking": [
        {"name": f"{c} / {m}", "chunks": c, "mode": m}
        for c in ("structure350", "structure200", "structure350-nohdr", "fixed256o48", "fixed512o64")
        for m in ("dense", "hybrid_rerank")
    ],
    "embedding": [
        {"name": f"{e.split('/')[-1]} / {m}", "model": e, "mode": m}
        for e in ("sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5",
                  "intfloat/e5-base-v2")
        for m in ("dense", "hybrid_rerank")
    ],
}

_INDEX_CACHE: dict[tuple[str, str], CorpusIndex] = {}


def get_index(chunks: str, model: str) -> CorpusIndex:
    key = (chunks, model)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = CorpusIndex.build(load_chunks(chunks), chunks, model)
    return _INDEX_CACHE[key]


def run_config(cfg: dict, qas) -> dict:
    c = {**BASE, **cfg}
    idx = get_index(c["chunks"], c["model"])
    rcfg = replace(get_retrieval_config(), rerank_model=c["reranker"])
    r = Retriever(idx, mode=c["mode"], cfg=rcfg, source_policy=c["source_policy"], alpha=c["alpha"],
                  expand_query=c["expand_query"], legal_slot=c["legal_slot"])
    r.retrieve("warm-up query about forklift training")  # load models outside the timed loop
    per_q, lat = [], []
    for q in qas:
        t0 = time.perf_counter()
        res = r.retrieve(q.question, k=10)
        lat.append((time.perf_counter() - t0) * 1000)
        m = retrieval_metrics([h.chunk for h in res.hits], q.gold)
        per_q.append({"id": q.id, "type": q.type, "split": q.split, **m,
                      "top5": [h.chunk.cite for h in res.hits[:5]]})
    out = {"name": cfg["name"], "config": c, "latency_ms_p50": round(sorted(lat)[len(lat) // 2], 1),
           "per_question": per_q}
    metric_keys = [k for k in per_q[0] if k not in ("id", "type", "split", "top5")]
    for split in ("dev", "test", "all"):
        rows = [p for p in per_q if split == "all" or p["split"] == split]
        out[split] = mean_metrics([{k: p[k] for k in metric_keys} for p in rows])
    out["by_type_test"] = {
        t: mean_metrics([{k: p[k] for k in ("R@5", "MRR")} for p in per_q if p["type"] == t and p["split"] == "test"])
        for t in sorted({p["type"] for p in per_q})
    }
    return out


def paired_bootstrap(ref: dict, other: dict, metric: str = "R@5", n: int = 2000, seed: int = 0):
    """Mean difference (other - ref) on test questions with a 95% percentile interval."""
    rng = random.Random(seed)
    a = {p["id"]: p[metric] for p in ref["per_question"] if p["split"] == "test"}
    b = {p["id"]: p[metric] for p in other["per_question"] if p["split"] == "test"}
    ids = sorted(a)
    diffs = [b[i] - a[i] for i in ids]
    means = sorted(sum(rng.choice(diffs) for _ in ids) / len(ids) for _ in range(n))
    return sum(diffs) / len(diffs), means[int(0.025 * n)], means[int(0.975 * n)]


def to_markdown(results: dict[str, list[dict]]) -> str:
    lines = ["# Retrieval results", "",
             "Test split unless noted; questions with gold passages only (ambiguous / insufficient / "
             "out-of-scope questions have none). Latency is the median per query on an Apple M-series "
             "laptop (MPS), excluding model load.", ""]
    cols = ["P@5", "R@5", "R@10", "Hit@5", "MRR", "nDCG@5"]
    for group, rows in results.items():
        lines += [f"## {group}", "", "| Configuration | " + " | ".join(cols) + " | dev R@5 | p50 ms |",
                  "|---|" + "---|" * (len(cols) + 2)]
        for r in rows:
            vals = " | ".join(f"{r['test'][c]:.3f}" for c in cols)
            lines.append(f"| {r['name']} | {vals} | {r['dev']['R@5']:.3f} | {r['latency_ms_p50']:.0f} |")
        lines.append("")
        # reference = the default configuration (same settings as BASE)
        ref = next((r for r in rows if {k: v for k, v in r["config"].items() if k != "name"} == BASE), rows[0])
        diffs = [(r["name"], *paired_bootstrap(ref, r)) for r in rows if r is not ref]
        if diffs:
            lines += [f"Paired bootstrap (test R@5, 2,000 resamples) vs. **{ref['name']}**: "
                      + "; ".join(f"{n}: Δ={d:+.3f} [{lo:+.3f}, {hi:+.3f}]" for n, d, lo, hi in diffs), ""]
    base = next((r for r in results.get("retriever", []) if "default" in r["name"]), None)
    if base:
        lines += ["## Recall@5 by question type (default configuration, test)", "", "| Type | R@5 | MRR |", "|---|---|---|"]
        for t, m in base["by_type_test"].items():
            if m:
                lines.append(f"| {t} | {m['R@5']:.3f} | {m['MRR']:.3f} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(GROUPS), action="append")
    args = ap.parse_args()
    groups = args.group or list(GROUPS)
    qas = [q for q in load_qa() if q.gold]
    path = RESULTS_DIR / "retrieval.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for g in groups:
        results[g] = []
        for cfg in GROUPS[g]:
            res = run_config(cfg, qas)
            results[g].append(res)
            print(f"[{g}] {res['name']:40s} test R@5={res['test']['R@5']:.3f} P@5={res['test']['P@5']:.3f} "
                  f"MRR={res['test']['MRR']:.3f} | dev R@5={res['dev']['R@5']:.3f} | {res['latency_ms_p50']:.0f} ms",
                  flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ordered = {g: results[g] for g in GROUPS if g in results}
    path.write_text(json.dumps(ordered, indent=1))
    (RESULTS_DIR / "retrieval.md").write_text(to_markdown(ordered))


if __name__ == "__main__":
    main()
