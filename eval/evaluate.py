"""Evaluation on the held-out QA set (eval/qa.yaml; dev/test split in eval/splits.yaml).

  python eval/evaluate.py retrieval   # Precision@5, Recall@5, MRR: retrievers, chunking, embeddings
  python eval/evaluate.py e2e         # answers: status, key facts, faithfulness (LLM judge)

Relevance is structural: a chunk is relevant if it comes from the gold document and contains the
gold paragraph cite ("29 CFR 1910.178(l)(4)", which also covers (l)(4)(iii)) or the gold clause or
section ("7.2", "EHS-FL-201", "Q3"). Every chunking strategy is therefore scored against the same
ground truth. Headline numbers are on the test split.
"""

import json
import random
import re
import sys

import yaml
from pydantic import BaseModel

from lcrag.config import EVAL_DIR, JUDGE_MODEL, RESULTS_DIR, chat_model
from lcrag.graph import build_graph, format_sources
from lcrag.store import make_retriever, sync_index

SPLITS = yaml.safe_load((EVAL_DIR / "splits.yaml").read_text())
QA = [q | {"split": SPLITS[q["id"]]} for q in yaml.safe_load((EVAL_DIR / "qa.yaml").read_text())]
RUN = {"max_concurrency": 8}


def covered(doc, gold) -> set[int]:
    m, text, heads = doc.metadata, doc.page_content, [doc.metadata.get("h2", ""), doc.metadata.get("h3", "")]
    return {i for i, (did, a) in enumerate(gold) if m["doc_id"] == did and (
        not a or f"{m.get('citation')}{a}" in text or any(h == a or h.startswith((a + ".", a + " ")) for h in heads)
        or re.search(rf"(^|\n)(#+ |\*\*)?{re.escape(a)}([ .\n]|$)", text))}


def retrieval_metrics(docs, gold, k=5) -> dict:
    cov = [covered(d, gold) for d in docs[:k]]
    first = next((i for i, c in enumerate(cov) if c), None)
    return {"P@5": sum(map(bool, cov)) / k, "R@5": len(set().union(*cov)) / len(gold),
            "Hit@5": float(first is not None), "MRR": 0.0 if first is None else 1 / (first + 1)}


def mean(rows, key, split="test"):
    xs = [r[key] for r in rows if r["split"] == split and r.get(key) is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def bootstrap(a, b, n=2000):  # paired 95% CI of mean(b - a)
    d = [y - x for x, y in zip(a, b)]
    ms = sorted(sum(random.Random(i).choices(d, k=len(d))) / len(d) for i in range(n))
    return round(sum(d) / len(d), 3), round(ms[int(n * .025)], 3), round(ms[int(n * .975)], 3)


CONFIGS = [  # (name, chunking, embedding model, retriever mode)
    ("dense", "structure", "text-embedding-3-small", "dense"),
    ("BM25", "structure", "text-embedding-3-small", "bm25"),
    ("hybrid RRF", "structure", "text-embedding-3-small", "hybrid"),
    ("hybrid RRF + LLM re-rank (default)", "structure", "text-embedding-3-small", "hybrid_rerank"),
    ("chunking: no contextual header, dense", "nohdr", "text-embedding-3-small", "dense"),
    ("chunking: no contextual header, full", "nohdr", "text-embedding-3-small", "hybrid_rerank"),
    ("chunking: fixed 320 tokens, dense", "fixed", "text-embedding-3-small", "dense"),
    ("chunking: fixed 320 tokens, full", "fixed", "text-embedding-3-small", "hybrid_rerank"),
    ("embedding: 3-large, dense", "structure", "text-embedding-3-large", "dense"),
    ("embedding: 3-large, full", "structure", "text-embedding-3-large", "hybrid_rerank"),
]


def run_retrieval():
    qs = [q for q in QA if q["gold"]]
    results, per_q = [], {}
    for name, strategy, model, mode in CONFIGS:
        docs = make_retriever(sync_index(strategy, model)[0], mode).batch([q["question"] for q in qs], RUN)
        rows = [{"id": q["id"], "type": q["type"], "split": q["split"], **retrieval_metrics(d, q["gold"])}
                for q, d in zip(qs, docs)]
        per_q[name] = rows
        res = {"config": name, **{f"test {k}": mean(rows, k) for k in ("P@5", "R@5", "Hit@5", "MRR")},
               "dev R@5": mean(rows, "R@5", "dev")}
        results.append(res)
        print(res, flush=True)
    base = [r["R@5"] for r in per_q["dense"] if r["split"] == "test"]
    ci = {n: bootstrap(base, [r["R@5"] for r in rows if r["split"] == "test"]) for n, rows in per_q.items() if n != "dense"}
    by_type = {}
    for r in per_q["hybrid RRF + LLM re-rank (default)"]:
        by_type.setdefault(r["type"], []).append(r["R@5"])
    md = ["| Configuration | P@5 | R@5 | Hit@5 | MRR | dev R@5 | ΔR@5 vs dense [95% CI] |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['config']} | {r['test P@5']} | {r['test R@5']} | {r['test Hit@5']} | {r['test MRR']} | {r['dev R@5']} | "
           f"{ci.get(r['config'], '-')} |" for r in results]
    md += ["", "Recall@5 by question type (default, dev+test): "
           + ", ".join(f"{t} {sum(v) / len(v):.2f} (n={len(v)})" for t, v in sorted(by_type.items()))]
    (RESULTS_DIR / "retrieval.md").write_text("# Retrieval (test split, k=5)\n\n" + "\n".join(md) + "\n")
    (RESULTS_DIR / "retrieval.json").write_text(json.dumps({"summary": results, "per_question": per_q}, indent=1))


class Judged(BaseModel):
    statements: list[str]
    supported: list[bool]


def run_e2e():
    app = build_graph(make_retriever(sync_index()[0]))
    outs = app.batch([{"question": q["question"]} for q in QA], RUN)
    judge = chat_model(JUDGE_MODEL).with_structured_output(Judged)
    decline = {"insufficient_context", "out_of_scope"}
    rows, to_judge = [], []
    for q, o in zip(QA, outs):
        ans = o["result"].answer
        r = {"id": q["id"], "type": q["type"], "split": q["split"], "question": q["question"], "expect": q["expect"],
             "status": o["status"], "answer": ans, "clarifying_question": o["result"].clarifying_question,
             "revised": o["attempts"] > 1, "sources": [d.page_content for d in o["docs"]],
             "status_ok": o["status"] == q["expect"] or (q["expect"] in decline and o["status"] in decline),
             "key_facts_ok": all(re.search(p, ans, re.I) for p in q["key_facts"]) if q["expect"] == "answered" else None}
        rows.append(r)
        if o["status"] not in decline:
            to_judge.append((r, [("system", "Split the ANSWER into atomic factual statements. For each, decide if the "
                                  "SOURCES support it (stated or directly implied). Outside knowledge does not count."),
                                 ("human", f"SOURCES:\n{format_sources(o['docs'])}\n\nANSWER:\n{ans}")]))
    for (r, _), j in zip(to_judge, judge.batch([m for _, m in to_judge], RUN)):
        r["faithfulness"] = sum(j.supported) / max(len(j.supported), 1)
        r["unsupported"] = [s for s, ok in zip(j.statements, j.supported) if not ok]
    for r in rows:
        r["unverified_flag"] = r["status"] == "unverified"
    summary = {split: {"n": sum(r["split"] == split for r in rows),
                       "status accuracy": mean(rows, "status_ok", split),
                       "key-fact accuracy (answerable)": mean(rows, "key_facts_ok", split),
                       "faithfulness (judge, share of supported statements)": mean(rows, "faithfulness", split),
                       "answers revised by grader loop": mean(rows, "revised", split)} for split in ("dev", "test")}
    print(json.dumps(summary, indent=1))
    (RESULTS_DIR / "e2e.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    md = ["| Metric | dev | test |", "|---|---|---|"] + [f"| {k} | {summary['dev'][k]} | {summary['test'][k]} |"
                                                          for k in summary["test"]]
    md += ["", "| id | type | expected | got | key facts | faithfulness |", "|---|---|---|---|---|---|"]
    md += [f"| {r['id']} ({r['split']}) | {r['type']} | {r['expect']} | {r['status']} | {r['key_facts_ok']} | "
           f"{'-' if r.get('faithfulness') is None else round(r['faithfulness'], 2)} |" for r in rows]
    (RESULTS_DIR / "e2e.md").write_text(f"# End-to-end (generator {chat_model().model_name}, judge {JUDGE_MODEL})\n\n"
                                        + "\n".join(md) + "\n")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    {"retrieval": run_retrieval, "e2e": run_e2e}[sys.argv[1]]()
