#!/usr/bin/env python3
"""Create a compact index proving which pipeline outputs were produced."""

from __future__ import annotations

import argparse
import time

from kkbox_common import build_spark, path_stats, show_and_write


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-base", required=True)
    return parser.parse_args()


def main():
    args = arguments()
    started = time.perf_counter()
    spark = build_spark("KKBox-04-Result-Index", 40)
    run = args.run_base.rstrip("/")
    results = f"{run}/results"
    expected = [
        ("audit", f"{results}/audit"),
        ("churn_analysis", f"{results}/analysis"),
        ("big_data_experiments", f"{results}/experiments"),
        ("scaling_benchmark", f"{results}/scaling"),
        ("pandas_scaling_inputs", f"{results}/scaling_inputs"),
        ("logistic_regression_results", f"{results}/models"),
        ("local_logistic_inputs", f"{results}/local_ml_input"),
        ("job_runtimes", f"{results}/runtime"),
        ("saved_spark_models", f"{run}/models"),
        ("curated_parquet", f"{run}/curated"),
    ]
    rows = []
    for category, uri in expected:
        files, size = path_stats(spark, uri)
        rows.append(
            (
                category,
                uri,
                files,
                size,
                size / (1024**3),
                files > 0,
            )
        )
    index = spark.createDataFrame(
        rows,
        ["category", "s3_path", "files", "bytes", "gibibytes", "exists"],
    )
    if any(not row[-1] for row in rows):
        missing = [row[0] for row in rows if not row[-1]]
        raise AssertionError(f"Missing expected outputs: {missing}")
    show_and_write(
        index,
        "04A - Complete result index",
        f"{results}/result_index",
    )
    runtime = spark.createDataFrame(
        [
            (
                "04_build_result_index",
                time.perf_counter() - started,
                "complete",
            )
        ],
        ["job", "runtime_seconds", "status"],
    )
    show_and_write(
        runtime,
        "04B - Result indexing completed",
        f"{results}/runtime/job_04",
    )
    spark.stop()


if __name__ == "__main__":
    main()
