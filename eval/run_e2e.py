"""End-to-end evaluation: answers, status handling, key facts, faithfulness (LLM judge + runtime NLI).

Usage:
    python eval/run_e2e.py                 # all questions, model from .env / environment
    python eval/run_e2e.py --split test
    JUDGE_MODEL=gpt-4o python eval/run_e2e.py   # use a different (ideally stronger) judge

Outputs results/e2e_<model>.json (every answer, context, check) and results/e2e_<model>.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time

from lcrag.config import RESULTS_DIR, get_llm_config
from lcrag.evaluation import JUDGE_PROMPT, key_fact_hits, load_qa
from lcrag.llm import chat, parse_json
from lcrag.pipeline import answer_claims, build_default_assistant

DECLINE = {"insufficient_context", "out_of_scope"}


def judge(answer: str, contexts: list[dict]) -> list[dict]:
    claims = answer_claims(answer, [c["tag"] for c in contexts])
    if not claims:
        return []
    sources = "\n\n".join(f"[{c['tag']}] {c['cite']}\n{c['text']}" for c in contexts)
    sentences = "\n".join(f"{i}. {c['text']}" for i, c in enumerate(claims, start=1))
    model = os.getenv("JUDGE_MODEL") or get_llm_config().model
    raw = chat([{"role": "user", "content": JUDGE_PROMPT.format(sources=sources, sentences=sentences)}],
               json_mode=True, model=model, max_tokens=900)
    try:
        js = parse_json(raw).get("judgements", [])
    except Exception:  # noqa: BLE001 - a judge failure should not abort the run
        js = []
    by_n = {int(j.get("n", 0)): j for j in js if isinstance(j, dict)}
    return [{"text": c["text"], "supported": bool(by_n.get(i, {}).get("supported", False)),
             "reason": by_n.get(i, {}).get("reason", "no judgement")} for i, c in enumerate(claims, start=1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"])
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    assistant = build_default_assistant()
    assistant.ask("warm-up: how often is forklift evaluation required?")
    qas = load_qa(split=args.split)
    rows = []
    for i, q in enumerate(qas, start=1):
        t0 = time.perf_counter()
        ans = assistant.ask(q.question)
        row = {"id": q.id, "type": q.type, "split": q.split, "question": q.question, "expect": q.expect,
               **{k: v for k, v in ans.to_dict().items() if k != "question"}}
        row["key_facts"] = q.key_facts
        row["key_fact_hits"] = key_fact_hits(ans.answer, q.key_facts) if q.key_facts else []
        if not args.no_judge and ans.status not in DECLINE and ans.answer:
            row["judge"] = judge(ans.answer, ans.contexts)
        row["wall_s"] = round(time.perf_counter() - t0, 1)
        rows.append(row)
        kf = f"{sum(row['key_fact_hits'])}/{len(row['key_fact_hits'])}" if row["key_fact_hits"] else "-"
        print(f"[{i:2d}/{len(qas)}] {q.id} {q.type:13s} expect={q.expect:20s} got={ans.status:20s} kf={kf:5s} "
              f"verdict={ans.grounding.get('verdict', '-'):20s} {row['wall_s']}s", flush=True)

    summary = summarise(rows)
    model = get_llm_config().model
    slug = re.sub(r"[^A-Za-z0-9.-]+", "_", model)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"e2e_{slug}.json").write_text(json.dumps({"model": model, "summary": summary, "rows": rows},
                                                             indent=1))
    (RESULTS_DIR / f"e2e_{slug}.md").write_text(to_markdown(model, summary, rows))
    print(json.dumps(summary, indent=1))


def _rate(xs: list[bool]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def summarise(rows: list[dict]) -> dict:
    out = {}
    for split in ("dev", "test", "all"):
        rs = [r for r in rows if split == "all" or r["split"] == split]
        if not rs:
            continue
        ans_rows = [r for r in rs if r["expect"] == "answered"]
        judged = [r for r in rs if r.get("judge")]
        faith = [sum(j["supported"] for j in r["judge"]) / len(r["judge"]) for r in judged]
        nli_vs_judge = [(c["label"] == "supported", j["supported"])
                        for r in judged for c, j in zip(r["grounding"].get("claims", []), r["judge"])
                        if c["text"] == j["text"]]
        lat = [r["timings_ms"].get("total", 0) / 1000 for r in rs if r["status"] != "out_of_scope"]
        out[split] = {
            "n": len(rs),
            "status_accuracy": _rate([r["status"] == r["expect"] or (r["expect"] in DECLINE and r["status"] in DECLINE)
                                      for r in rs]),
            "status_accuracy_strict": _rate([r["status"] == r["expect"] for r in rs]),
            "answered_key_fact_all": _rate([all(r["key_fact_hits"]) for r in ans_rows if r["key_fact_hits"]]),
            "answered_key_fact_mean": round(statistics.mean(
                sum(r["key_fact_hits"]) / len(r["key_fact_hits"]) for r in ans_rows if r["key_fact_hits"]), 3)
            if ans_rows else None,
            "false_decline_rate": _rate([r["status"] in DECLINE for r in ans_rows]),
            "decline_recall": _rate([r["status"] in DECLINE for r in rs if r["expect"] in DECLINE]),
            "clarification_recall": _rate([r["status"] == "needs_clarification" for r in rs
                                           if r["expect"] == "needs_clarification"]),
            "faithfulness_judge": round(statistics.mean(faith), 3) if faith else None,
            "fully_faithful_answers": _rate([f == 1.0 for f in faith]),
            "runtime_verdicts": _count(r["grounding"].get("verdict", "none") for r in rs),
            "retry_rate": _rate([bool(r["grounding"].get("retried")) for r in rs if r["grounding"]]),
            "nli_judge_agreement": _rate([a == b for a, b in nli_vs_judge]),
            "nli_precision_vs_judge": _rate([b for a, b in nli_vs_judge if a]),
            "nli_recall_of_unsupported": _rate([not a for a, b in nli_vs_judge if not b]),
            "latency_s_p50": round(statistics.median(lat), 1) if lat else None,
            "latency_s_p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1) if lat else None,
        }
    return out


def _count(items) -> dict:
    d: dict[str, int] = {}
    for x in items:
        d[x] = d.get(x, 0) + 1
    return dict(sorted(d.items()))


def to_markdown(model: str, summary: dict, rows: list[dict]) -> str:
    lines = [f"# End-to-end results: generator `{model}`", "",
             "Status accuracy counts `insufficient_context` and `out_of_scope` as the same *decline* outcome "
             "(strict accuracy separates them). Key-fact accuracy is on questions expected to be answered. "
             "Faithfulness is the share of answer sentences an LLM judge finds supported by the retrieved "
             "context; the runtime NLI check is compared against the judge.", "",
             "| Metric | dev | test |", "|---|---|---|"]
    keys = [k for k in summary.get("test", summary.get("all", {})) if k not in ("runtime_verdicts",)]
    for k in keys:
        lines.append(f"| {k} | {summary.get('dev', {}).get(k, '')} | {summary.get('test', {}).get(k, '')} |")
    lines += ["", f"Runtime verdicts (test): {summary.get('test', {}).get('runtime_verdicts')}", "",
              "## Per question", "", "| id | type | expect | got | key facts | runtime verdict | judge |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        kf = f"{sum(r['key_fact_hits'])}/{len(r['key_fact_hits'])}" if r["key_fact_hits"] else "-"
        jd = f"{sum(j['supported'] for j in r['judge'])}/{len(r['judge'])}" if r.get("judge") else "-"
        lines.append(f"| {r['id']} ({r['split']}) | {r['type']} | {r['expect']} | {r['status']} | {kf} | "
                     f"{r['grounding'].get('verdict', '-')} | {jd} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
