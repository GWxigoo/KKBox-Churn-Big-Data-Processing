#!/usr/bin/env python3
"""Validate that Spark and pandas scaling summaries represent the same outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spark-summary", required=True)
    parser.add_argument("--pandas-summary", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = arguments()
    spark = pd.read_csv(args.spark_summary)
    pandas = pd.read_csv(args.pandas_summary)
    merged = spark.merge(
        pandas,
        on="percentage",
        suffixes=("_spark", "_pandas"),
        validate="one_to_one",
    )
    exact_columns = [
        "output_rows",
        "distinct_users",
        "transaction_count_checksum",
        "active_days_checksum",
    ]
    for column in exact_columns:
        merged[f"{column}_matches"] = (
            merged[f"{column}_spark"] == merged[f"{column}_pandas"]
        )
    difference = (
        merged["listening_seconds_checksum_spark"]
        - merged["listening_seconds_checksum_pandas"]
    ).abs()
    tolerance = (
        merged["listening_seconds_checksum_spark"].abs() * 1e-9
    ).clip(lower=0.01)
    merged["listening_seconds_checksum_matches"] = difference <= tolerance
    match_columns = [c for c in merged.columns if c.endswith("_matches")]
    merged["all_outputs_match"] = merged[match_columns].all(axis=1)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(merged[["percentage", *match_columns, "all_outputs_match"]])
    if not merged["all_outputs_match"].all():
        raise AssertionError("At least one Spark/pandas output checksum differs")


if __name__ == "__main__":
    main()
