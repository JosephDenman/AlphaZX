#!/usr/bin/env python3
"""
Export TensorBoard scalar metrics to CSV files.

Reads all event files from a TensorBoard log directory and writes
one CSV per metric tag, plus a combined wide-format CSV with all
metrics aligned by training step.

Usage:
    python export_tb_to_csv.py                          # defaults: runs/alphazx → tb_export/
    python export_tb_to_csv.py --logdir runs/alphazx --outdir tb_export
    python export_tb_to_csv.py --combined-only           # skip per-metric CSVs
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from tbparse import SummaryReader
except ImportError:
    print("tbparse is required: pip install tbparse", file=sys.stderr)
    sys.exit(1)

import pandas as pd


def export(logdir: str, outdir: str, combined_only: bool = False) -> Path:
    logdir = Path(logdir)
    outdir = Path(outdir)

    if not logdir.exists():
        print(f"Error: TensorBoard log directory not found: {logdir}", file=sys.stderr)
        sys.exit(1)

    reader = SummaryReader(str(logdir), pivot=False)
    df = reader.scalars

    if df.empty:
        print("No scalar metrics found in the event files.", file=sys.stderr)
        sys.exit(1)

    # Detect the wall-time column name (varies across tbparse versions)
    wall_col = None
    for candidate in ("wall_time", "wall time", "time"):
        if candidate in df.columns:
            wall_col = candidate
            break

    outdir.mkdir(parents=True, exist_ok=True)

    tags = sorted(df["tag"].unique())
    print(f"Found {len(tags)} metrics across {len(df)} data points:\n")
    for tag in tags:
        count = len(df[df["tag"] == tag])
        print(f"  {tag:50s}  ({count:,} points)")

    # Per-metric CSVs
    if not combined_only:
        per_metric_dir = outdir / "per_metric"
        per_metric_dir.mkdir(exist_ok=True)
        keep_cols = ["step", "value"]
        if wall_col:
            keep_cols.append(wall_col)
        for tag in tags:
            subset = df[df["tag"] == tag][keep_cols].copy()
            subset = subset.sort_values("step").reset_index(drop=True)
            safe_name = tag.replace("/", "__").replace(" ", "_")
            path = per_metric_dir / f"{safe_name}.csv"
            subset.to_csv(path, index=False)
        print(f"\nPer-metric CSVs written to: {per_metric_dir}/")

    # Combined wide-format CSV (metrics as columns, steps as rows)
    wide = df.pivot_table(index="step", columns="tag", values="value", aggfunc="last")
    wide = wide.sort_index()
    wide.index.name = "step"
    combined_path = outdir / "all_metrics.csv"
    wide.to_csv(combined_path)

    print(f"Combined CSV written to:    {combined_path}")
    print(f"\nShape: {wide.shape[0]} steps × {wide.shape[1]} metrics")

    # Print a quick summary of latest values
    print("\n--- Latest values ---")
    last_row = wide.iloc[-1]
    for col in wide.columns:
        val = last_row[col]
        if pd.notna(val):
            print(f"  {col:50s}  {val:.6g}")

    return combined_path


def main():
    parser = argparse.ArgumentParser(description="Export TensorBoard metrics to CSV")
    parser.add_argument("--logdir", default="runs/alphazx",
                        help="TensorBoard log directory (default: runs/alphazx)")
    parser.add_argument("--outdir", default="tb_export",
                        help="Output directory for CSVs (default: tb_export)")
    parser.add_argument("--combined-only", action="store_true",
                        help="Only write the combined wide-format CSV")
    args = parser.parse_args()
    export(args.logdir, args.outdir, args.combined_only)


if __name__ == "__main__":
    main()
