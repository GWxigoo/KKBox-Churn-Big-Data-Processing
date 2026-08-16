#!/usr/bin/env python3
"""Matched 1%, 5%, 10%, and 20% pandas feature-engineering benchmark."""

from __future__ import annotations

import argparse
import glob
import json
import os
import threading
import time
from pathlib import Path

import pandas as pd
import psutil


CUTOFF = pd.Timestamp("2017-01-31")
FEATURE_COLUMNS = [
    "membership_tenure_days",
    "transaction_count_90",
    "auto_renew_rate_90",
    "cancellation_count_90",
    "days_since_last_transaction",
    "days_until_expiration",
    "no_recent_transaction",
    "active_days_30",
    "total_secs_30",
    "days_since_last_listen",
    "no_recent_listening",
]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-base", required=True)
    parser.add_argument("--percentages", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser.parse_args()


def csv_parts(folder):
    parts = sorted(glob.glob(os.path.join(folder, "*.csv*")))
    if not parts:
        raise FileNotFoundError(f"No CSV parts found below {folder}")
    return parts


class PeakMemory:
    def __init__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop_event.wait(0.1):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def read_small(folder, **kwargs):
    return pd.concat(
        [pd.read_csv(path, **kwargs) for path in csv_parts(folder)],
        ignore_index=True,
    )


def aggregate_transactions(folder, chunksize):
    partials = []
    input_rows = 0
    chunks = 0
    for path in csv_parts(folder):
        for frame in pd.read_csv(
            path,
            chunksize=chunksize,
            dtype={
                "msno": "string",
                "is_auto_renew": "int8",
                "is_cancel": "int8",
            },
        ):
            chunks += 1
            input_rows += len(frame)
            frame["transaction_date"] = pd.to_datetime(
                frame["transaction_date_raw"], errors="coerce"
            )
            frame["membership_expire_date"] = pd.to_datetime(
                frame["membership_expire_date_raw"], errors="coerce"
            )
            frame = frame.dropna(
                subset=["msno", "transaction_date", "membership_expire_date"]
            )
            grouped = frame.groupby("msno", observed=True).agg(
                transaction_count_90=("msno", "size"),
                auto_renew_sum=("is_auto_renew", "sum"),
                cancellation_count_90=("is_cancel", "sum"),
                last_transaction_date=("transaction_date", "max"),
                latest_expiration=("membership_expire_date", "max"),
            )
            partials.append(grouped.reset_index())
    combined = pd.concat(partials, ignore_index=True)
    final = combined.groupby("msno", as_index=False, observed=True).agg(
        transaction_count_90=("transaction_count_90", "sum"),
        auto_renew_sum=("auto_renew_sum", "sum"),
        cancellation_count_90=("cancellation_count_90", "sum"),
        last_transaction_date=("last_transaction_date", "max"),
        latest_expiration=("latest_expiration", "max"),
    )
    final["auto_renew_rate_90"] = (
        final["auto_renew_sum"] / final["transaction_count_90"]
    )
    final["days_since_last_transaction"] = (
        CUTOFF - final["last_transaction_date"]
    ).dt.days
    final["days_until_expiration"] = (
        final["latest_expiration"] - CUTOFF
    ).dt.days
    return (
        final[
            [
                "msno",
                "transaction_count_90",
                "auto_renew_rate_90",
                "cancellation_count_90",
                "days_since_last_transaction",
                "days_until_expiration",
            ]
        ],
        input_rows,
        chunks,
    )


def aggregate_logs(folder, chunksize):
    partials = []
    input_rows = 0
    chunks = 0
    for path in csv_parts(folder):
        for frame in pd.read_csv(
            path,
            chunksize=chunksize,
            dtype={"msno": "string", "total_secs": "float64"},
        ):
            chunks += 1
            input_rows += len(frame)
            frame["event_date"] = pd.to_datetime(
                frame["event_date_raw"], errors="coerce"
            )
            frame = frame.dropna(subset=["msno", "event_date", "total_secs"])
            frame = frame[frame["total_secs"] >= 0]
            # The Spark sample preparation deliberately writes one row per
            # subscriber-day, so summing per-chunk day counts is equivalent to
            # COUNT(DISTINCT event_date).
            grouped = frame.groupby("msno", observed=True).agg(
                active_days_30=("event_date", "count"),
                total_secs_30=("total_secs", "sum"),
                last_listen_date=("event_date", "max"),
            )
            partials.append(grouped.reset_index())
    combined = pd.concat(partials, ignore_index=True)
    final = combined.groupby("msno", as_index=False, observed=True).agg(
        active_days_30=("active_days_30", "sum"),
        total_secs_30=("total_secs_30", "sum"),
        last_listen_date=("last_listen_date", "max"),
    )
    final["days_since_last_listen"] = (
        CUTOFF - final["last_listen_date"]
    ).dt.days
    return (
        final[
            [
                "msno",
                "active_days_30",
                "total_secs_30",
                "days_since_last_listen",
            ]
        ],
        input_rows,
        chunks,
    )


def run_one(input_base, percentage, output_dir, chunksize):
    base = os.path.join(input_base, f"pct_{percentage:02d}")
    stage = {}
    total_started = time.perf_counter()
    with PeakMemory() as memory:
        started = time.perf_counter()
        labels = read_small(
            os.path.join(base, "labels"),
            dtype={"msno": "string", "is_churn": "int8"},
        ).drop_duplicates("msno")
        members = read_small(
            os.path.join(base, "members"), dtype={"msno": "string"}
        )
        members["registration_date"] = pd.to_datetime(
            members["registration_date_raw"], errors="coerce"
        )
        member_features = (
            members.groupby("msno", as_index=False, observed=True)
            .agg(registration_date=("registration_date", "max"))
        )
        member_features["membership_tenure_days"] = (
            CUTOFF - member_features["registration_date"]
        ).dt.days
        member_features = member_features[
            ["msno", "membership_tenure_days"]
        ]
        stage["labels_and_members_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        transaction_features, tx_rows, tx_chunks = aggregate_transactions(
            os.path.join(base, "transactions"), chunksize
        )
        stage["transaction_aggregation_seconds"] = (
            time.perf_counter() - started
        )

        started = time.perf_counter()
        listening_features, log_rows, log_chunks = aggregate_logs(
            os.path.join(base, "logs"), chunksize
        )
        stage["listening_aggregation_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        features = (
            labels.merge(
                member_features, on="msno", how="left", validate="one_to_one"
            )
            .merge(
                transaction_features,
                on="msno",
                how="left",
                validate="one_to_one",
                indicator="_transaction_merge",
            )
            .merge(
                listening_features,
                on="msno",
                how="left",
                validate="one_to_one",
                indicator="_listening_merge",
            )
        )
        features["no_recent_transaction"] = (
            features["_transaction_merge"] == "left_only"
        ).astype("int8")
        features["no_recent_listening"] = (
            features["_listening_merge"] == "left_only"
        ).astype("int8")
        features = features.drop(
            columns=["_transaction_merge", "_listening_merge"]
        )
        zero_columns = [
            "transaction_count_90",
            "auto_renew_rate_90",
            "cancellation_count_90",
            "active_days_30",
            "total_secs_30",
        ]
        features[zero_columns] = features[zero_columns].fillna(0)
        if len(features) != len(labels):
            raise AssertionError("Output rows do not equal label rows")
        if features["msno"].nunique() != len(features):
            raise AssertionError("Output contains duplicate msno values")
        output_path = output_dir / f"pandas_features_{percentage:02d}.parquet"
        features.to_parquet(output_path, index=False)
        stage["join_and_write_seconds"] = time.perf_counter() - started

    result = {
        "percentage": percentage,
        "platform": "Python/pandas local",
        "label_rows": int(len(labels)),
        "member_rows": int(len(members)),
        "transaction_rows": int(tx_rows),
        "listening_rows": int(log_rows),
        "transaction_chunks": int(tx_chunks),
        "listening_chunks": int(log_chunks),
        "output_rows": int(len(features)),
        "distinct_users": int(features["msno"].nunique()),
        "transaction_count_checksum": int(
            features["transaction_count_90"].sum()
        ),
        "active_days_checksum": int(features["active_days_30"].sum()),
        "listening_seconds_checksum": float(
            features["total_secs_30"].sum()
        ),
        "peak_rss_bytes": int(memory.peak),
        "end_to_end_runtime_seconds": time.perf_counter() - total_started,
        **stage,
    }
    return result


def main():
    args = arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for percentage in args.percentages:
        print(f"Running pandas scaling benchmark at {percentage}%")
        result = run_one(
            args.input_base,
            percentage,
            output_dir,
            args.chunksize,
        )
        results.append(result)
        (output_dir / f"pandas_runtime_{percentage:02d}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
    pd.DataFrame(results).to_csv(
        output_dir / "pandas_scaling_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
