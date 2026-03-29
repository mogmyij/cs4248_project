#!/usr/bin/env python3
"""Filter synthetic sarcasm pairs by sarcasm and context scores.

Reads a combined scores CSV and splits pairs into passing/failing based on
OR logic: a pair fails if either score is below its threshold.

Outputs:
  - <filtered-output>: passing pairs as {"input": ..., "output": ...} JSONL
  - <failed-output>:   failing originals as {"input": ...} JSONL

The filtered output can be used directly as the --output target when
re-running generate_sarcastic.py; load_already_done() will skip everything
already present and regenerate only the missing (failed) pairs.
"""

import argparse
import json

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter sarcasm pairs by score thresholds"
    )
    parser.add_argument("--scores-csv", required=True,
                        help="Combined scores CSV with columns: input, output, sarcasm_score, context_score")
    parser.add_argument(
        "--sarcasm-threshold", type=float, required=True,
        help="Minimum sarcasm score (pairs below this fail)",
    )
    parser.add_argument(
        "--context-threshold", type=float, required=True,
        help="Minimum context score (pairs below this fail)",
    )
    parser.add_argument("--filtered-output", default="filtered.jsonl",
                        help="Output JSONL for passing pairs (default: filtered.jsonl)")
    parser.add_argument("--failed-output", default="failed.jsonl",
                        help="Output JSONL of failing inputs (default: failed.jsonl)")
    args = parser.parse_args()

    df = pd.read_csv(args.scores_csv)
    df.columns = [c.lower() for c in df.columns]

    print(f"Loaded {len(df)} pairs from {args.scores_csv}")
    print(f"Sarcasm threshold: {args.sarcasm_threshold}")
    print(f"Context threshold:  {args.context_threshold}")

    fail_mask = (
        df["sarcasm_score"].isna() | (df["sarcasm_score"] < args.sarcasm_threshold) |
        df["context_score"].isna() | (df["context_score"] < args.context_threshold)
    )

    passing = df[~fail_mask]
    failing = df[fail_mask]

    n_fail_sarcasm = ((df["sarcasm_score"] < args.sarcasm_threshold) & ~(df["context_score"] < args.context_threshold)).sum()
    n_fail_context = (~(df["sarcasm_score"] < args.sarcasm_threshold) & (df["context_score"] < args.context_threshold)).sum()
    n_fail_both = ((df["sarcasm_score"] < args.sarcasm_threshold) & (df["context_score"] < args.context_threshold)).sum()

    print(f"\nResults:")
    print(f"  Passed:            {len(passing)}")
    print(f"  Failed total:      {len(failing)}")
    print(f"    Sarcasm only:    {n_fail_sarcasm}")
    print(f"    Context only:    {n_fail_context}")
    print(f"    Both:            {n_fail_both}")

    with open(args.filtered_output, "w") as f:
        for _, row in passing.iterrows():
            f.write(json.dumps({"input": row["input"], "output": row["output"]}, ensure_ascii=False) + "\n")
    print(f"\nPassing pairs → {args.filtered_output}")

    with open(args.failed_output, "w") as f:
        for _, row in failing.iterrows():
            f.write(json.dumps({"input": row["input"]}, ensure_ascii=False) + "\n")
    print(f"Failing inputs  → {args.failed_output}")


if __name__ == "__main__":
    main()
