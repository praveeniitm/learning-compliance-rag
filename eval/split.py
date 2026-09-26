"""Assign the dev/test split once (seeded, stratified by question type) and write eval/splits.yaml.

The dev split is used for every tuning decision (OOS threshold, fusion weights, re-ranker
choice); headline numbers are reported on test only.
"""

from collections import Counter

import yaml

from lcrag.config import EVAL_DIR
from lcrag.evaluation import assign_splits, load_qa

if __name__ == "__main__":
    qas = load_qa()
    splits = assign_splits(qas)
    (EVAL_DIR / "splits.yaml").write_text(yaml.safe_dump(splits, sort_keys=True))
    by = Counter((q.type, splits[q.id]) for q in qas)
    for t in sorted({q.type for q in qas}):
        print(f"{t:14s} dev={by[(t, 'dev')]:2d} test={by[(t, 'test')]:2d}")
    print("total", Counter(splits.values()))
