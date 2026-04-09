#!/usr/bin/env python3
"""
Summarize a regime sweep job directory of the form:
  Imbalance_data_outputs/<JOBID>/cfg_<TASKID>/summary_last_time_mean_ci.csv

Prints:
- best algorithm per cfg by test_bal_acc_mean
- whether HG-MS is #1 among {G-PBGD, IAPTT-GM, V-PBGD, HG-MS, HG-baseline}
"""

from __future__ import annotations

import csv
import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple


WANT = ["HG-MS", "HG-baseline", "G-PBGD", "IAPTT-GM", "V-PBGD"]


def read_summary(path: str) -> Dict[str, float]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out: Dict[str, float] = {}
    for r in rows:
        alg = r["algorithm"]
        try:
            out[alg] = float(r["test_bal_acc_mean"])
        except Exception:
            continue
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    parser.add_argument("--root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"))
    args = parser.parse_args()
    job = args.job_id.strip()
    base = os.path.join(args.root, job)
    if not os.path.isdir(base):
        print(f"Not found: {base}", file=sys.stderr)
        return 2

    cfg_dirs = sorted(glob.glob(os.path.join(base, "cfg_*")))
    if not cfg_dirs:
        print(f"No cfg_* dirs under {base}", file=sys.stderr)
        return 2

    rows: List[Tuple[int, str, float, Dict[str, float]]] = []
    for d in cfg_dirs:
        m = os.path.basename(d).split("_", 1)[-1]
        try:
            cfg_id = int(m)
        except Exception:
            continue
        summ = os.path.join(d, "summary_last_time_mean_ci.csv")
        if not os.path.exists(summ):
            continue
        scores = read_summary(summ)
        if not scores:
            continue
        best_alg = max(scores, key=scores.get)
        rows.append((cfg_id, best_alg, float(scores[best_alg]), scores))

    rows.sort(key=lambda x: x[2], reverse=True)
    print(f"Job {job}: {len(rows)} cfgs with summaries")
    for cfg_id, best_alg, best_val, scores in rows:
        have_all = all(a in scores for a in WANT)
        tag = ""
        if have_all:
            best_want = max(WANT, key=lambda a: scores[a])
            if best_want == "HG-MS":
                tag = "  <-- HG-MS #1 (wanted set)"
        parts = [f"{a}={scores.get(a, float('nan')):.4f}" for a in WANT if a in scores]
        print(f"cfg_{cfg_id:02d}: best={best_alg} {best_val:.4f} | " + " ".join(parts) + tag)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
