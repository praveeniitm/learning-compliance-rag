"""Calibrate the out-of-scope gate on the dev split.

The gate rejects a query without calling the LLM when the best re-ranker relevance among the
retrieved candidates is below a threshold. The threshold is picked on dev to maximise balanced
accuracy between out-of-scope and in-domain questions (answerable, ambiguous and insufficient
questions all count as in-domain). Ties are broken by the largest log-space margin to the nearest dev score. Queries that
look in-domain lexically ("parental leave policy" retrieves the leave-of-absence clause) are left
to the generator, which should answer insufficient_context. Falsely rejecting a real compliance
question costs more than letting the LLM decline an odd one.
Test-split numbers are reported for the chosen threshold.
"""

from __future__ import annotations

import json
import math

from lcrag.config import RESULTS_DIR, get_retrieval_config
from lcrag.evaluation import load_qa
from lcrag.index import CorpusIndex
from lcrag.retrieval import Retriever


def balanced_acc(rows, th):
    oos = [r for r in rows if r["oos"]]
    ins = [r for r in rows if not r["oos"]]
    tpr = sum(r["score"] < th for r in oos) / len(oos)
    tnr = sum(r["score"] >= th for r in ins) / len(ins)
    return (tpr + tnr) / 2, tpr, tnr


def main() -> None:
    cfg = get_retrieval_config()
    r = Retriever(CorpusIndex.load("structure350", cfg.embed_model), "hybrid_rerank")
    rows = []
    for q in load_qa():
        res = r.retrieve(q.question)
        rows.append({"id": q.id, "split": q.split, "type": q.type, "oos": q.expect == "out_of_scope",
                     "score": res.top_rerank_score})
    dev = [x for x in rows if x["split"] == "dev"]
    test = [x for x in rows if x["split"] == "test"]
    candidates = sorted({0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2})
    table = [(th, *balanced_acc(dev, th)) for th in candidates]
    # Among thresholds with the best dev balanced accuracy, take the one with the largest margin in
    # log space to the nearest dev score, so small score drift does not flip decisions.
    top_acc = max(t[1] for t in table)
    tied = [t[0] for t in table if t[1] == top_acc]
    logs = [math.log10(max(x["score"], 1e-9)) for x in dev]
    best = max(tied, key=lambda th: min(abs(math.log10(th) - l) for l in logs))
    report = {"threshold": best, "dev": dict(zip(("bal_acc", "oos_recall", "in_domain_pass"), balanced_acc(dev, best))),
              "test": dict(zip(("bal_acc", "oos_recall", "in_domain_pass"), balanced_acc(test, best))),
              "sweep_dev": [{"th": t, "bal_acc": round(a, 3), "oos_recall": round(p, 3), "in_domain_pass": round(n, 3)}
                            for t, a, p, n in table],
              "scores": sorted(rows, key=lambda x: x["score"])}
    for x in report["scores"][:14]:
        print(f"{x['score']:.5f} {x['id']} {x['split']} {x['type']}")
    print(json.dumps({k: report[k] for k in ("threshold", "dev", "test")}, indent=1))
    (RESULTS_DIR / "oos_calibration.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
