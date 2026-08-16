#!/usr/bin/env python3
"""Run join, file-format, and controlled Spark-versus-pandas experiments."""

from __future__ import annotations

import argparse
import time

from pyspark.sql import functions as F, types as T

from kkbox_common import (
    LOG_SCHEMA,
    TRAIN_SCHEMA,
    build_spark,
    deterministic_bucket,
    path_stats,
    show_and_write,
)


SAMPLE_MEMBER_SCHEMA = T.StructType(
    [
        T.StructField("msno", T.StringType(), True),
        T.StructField("registration_date_raw", T.StringType(), True),
    ]
)
SAMPLE_TX_SCHEMA = T.StructType(
    [
        T.StructField("msno", T.StringType(), True),
        T.StructField("transaction_date_raw", T.StringType(), True),
        T.StructField("membership_expire_date_raw", T.StringType(), True),
        T.StructField("is_auto_renew", T.IntegerType(), True),
        T.StructField("is_cancel", T.IntegerType(), True),
    ]
)
SAMPLE_LOG_SCHEMA = T.StructType(
    [
        T.StructField("msno", T.StringType(), True),
        T.StructField("event_date_raw", T.StringType(), True),
        T.StructField("total_secs", T.DoubleType(), True),
    ]
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-base", required=True)
    parser.add_argument("--run-base", required=True)
    parser.add_argument("--shuffle-partitions", type=int, default=400)
    return parser.parse_args()


def materialize_with_time(df):
    started = time.perf_counter()
    cached = df.cache()
    rows = cached.count()
    elapsed = time.perf_counter() - started
    return cached, rows, elapsed


def gzip_csv(df, path, partitions):
    (
        df.repartition(partitions, "msno")
        .write.mode("overwrite")
        .option("header", True)
        .option("compression", "gzip")
        .csv(path)
    )


def build_scaling_inputs(
    spark, labels, members, transactions, logs, results, percentages
):
    preparation_rows = []
    for percentage in percentages:
        started = time.perf_counter()
        selected = labels.filter(
            deterministic_bucket("msno", 100) < F.lit(percentage)
        ).cache()
        selected.createOrReplaceTempView("selected_labels")
        members.createOrReplaceTempView("all_members")
        transactions.createOrReplaceTempView("all_transactions")
        logs.createOrReplaceTempView("all_logs")

        sample_members = spark.sql(
            """
            SELECT m.msno,
                   DATE_FORMAT(m.registration_date, 'yyyy-MM-dd')
                     AS registration_date_raw
            FROM all_members m
            LEFT SEMI JOIN selected_labels l ON m.msno = l.msno
            """
        )
        sample_transactions = spark.sql(
            """
            SELECT t.msno,
                   DATE_FORMAT(t.transaction_date, 'yyyy-MM-dd')
                     AS transaction_date_raw,
                   DATE_FORMAT(t.membership_expire_date, 'yyyy-MM-dd')
                     AS membership_expire_date_raw,
                   t.is_auto_renew,
                   t.is_cancel
            FROM all_transactions t
            LEFT SEMI JOIN selected_labels l ON t.msno = l.msno
            WHERE t.transaction_date BETWEEN DATE '2016-11-03'
                                         AND DATE '2017-01-31'
            """
        )
        sample_logs = spark.sql(
            """
            SELECT g.msno,
                   DATE_FORMAT(g.event_date, 'yyyy-MM-dd') AS event_date_raw,
                   SUM(g.total_secs) AS total_secs
            FROM all_logs g
            LEFT SEMI JOIN selected_labels l ON g.msno = l.msno
            WHERE g.event_date BETWEEN DATE '2017-01-02'
                                   AND DATE '2017-01-31'
            GROUP BY g.msno, g.event_date
            """
        )
        base = f"{results}/scaling_inputs/pct_{percentage:02d}"
        partitions = max(2, percentage)
        gzip_csv(selected, f"{base}/labels", 2)
        gzip_csv(sample_members, f"{base}/members", 2)
        gzip_csv(sample_transactions, f"{base}/transactions", partitions)
        gzip_csv(sample_logs, f"{base}/logs", partitions)

        label_rows = selected.count()
        member_rows = sample_members.count()
        tx_rows = sample_transactions.count()
        log_rows = sample_logs.count()
        files, size = path_stats(spark, base)
        elapsed = time.perf_counter() - started
        preparation_rows.append(
            (
                percentage,
                label_rows,
                member_rows,
                tx_rows,
                log_rows,
                files,
                size,
                elapsed,
            )
        )
        print(
            f"[MILESTONE] 02 scaling input {percentage}%: "
            f"{label_rows} labels, {tx_rows} transactions, {log_rows} logs"
        )
        selected.unpersist()
    return spark.createDataFrame(
        preparation_rows,
        [
            "percentage",
            "label_rows",
            "member_rows",
            "transaction_rows",
            "listening_rows",
            "input_files",
            "compressed_input_bytes",
            "preparation_seconds_excluded_from_benchmark",
        ],
    )


def run_spark_scaling(spark, results, percentages):
    records = []
    for percentage in percentages:
        base = f"{results}/scaling_inputs/pct_{percentage:02d}"
        started = time.perf_counter()
        labels = (
            spark.read.schema(TRAIN_SCHEMA)
            .option("header", True)
            .csv(f"{base}/labels")
            .dropDuplicates(["msno"])
        )
        members = (
            spark.read.schema(SAMPLE_MEMBER_SCHEMA)
            .option("header", True)
            .csv(f"{base}/members")
            .withColumn(
                "registration_date", F.to_date("registration_date_raw")
            )
        )
        transactions = (
            spark.read.schema(SAMPLE_TX_SCHEMA)
            .option("header", True)
            .csv(f"{base}/transactions")
            .withColumn("transaction_date", F.to_date("transaction_date_raw"))
            .withColumn(
                "membership_expire_date",
                F.to_date("membership_expire_date_raw"),
            )
        )
        logs = (
            spark.read.schema(SAMPLE_LOG_SCHEMA)
            .option("header", True)
            .csv(f"{base}/logs")
            .withColumn("event_date", F.to_date("event_date_raw"))
            .filter(F.col("total_secs") >= 0)
        )
        labels.createOrReplaceTempView("scale_labels")
        members.createOrReplaceTempView("scale_members")
        transactions.createOrReplaceTempView("scale_transactions")
        logs.createOrReplaceTempView("scale_logs")
        features = spark.sql(
            """
            WITH member_features AS (
              SELECT msno,
                     DATEDIFF(DATE '2017-01-31', MAX(registration_date))
                       AS membership_tenure_days
              FROM scale_members
              GROUP BY msno
            ),
            transaction_features AS (
              SELECT msno,
                     COUNT(*) AS transaction_count_90,
                     AVG(CAST(is_auto_renew AS DOUBLE)) AS auto_renew_rate_90,
                     SUM(CASE WHEN is_cancel = 1 THEN 1 ELSE 0 END)
                       AS cancellation_count_90,
                     DATEDIFF(DATE '2017-01-31', MAX(transaction_date))
                       AS days_since_last_transaction,
                     DATEDIFF(MAX(membership_expire_date), DATE '2017-01-31')
                       AS days_until_expiration
              FROM scale_transactions
              GROUP BY msno
            ),
            listening_features AS (
              SELECT msno,
                     COUNT(DISTINCT event_date) AS active_days_30,
                     SUM(total_secs) AS total_secs_30,
                     DATEDIFF(DATE '2017-01-31', MAX(event_date))
                       AS days_since_last_listen
              FROM scale_logs
              GROUP BY msno
            )
            SELECT l.msno, l.is_churn,
                   m.membership_tenure_days,
                   COALESCE(t.transaction_count_90, 0)
                     AS transaction_count_90,
                   COALESCE(t.auto_renew_rate_90, 0.0) AS auto_renew_rate_90,
                   COALESCE(t.cancellation_count_90, 0)
                     AS cancellation_count_90,
                   t.days_since_last_transaction,
                   t.days_until_expiration,
                   CASE WHEN t.msno IS NULL THEN 1 ELSE 0 END
                     AS no_recent_transaction,
                   COALESCE(g.active_days_30, 0) AS active_days_30,
                   COALESCE(g.total_secs_30, 0.0) AS total_secs_30,
                   g.days_since_last_listen,
                   CASE WHEN g.msno IS NULL THEN 1 ELSE 0 END
                     AS no_recent_listening
            FROM scale_labels l
            LEFT JOIN member_features m ON l.msno = m.msno
            LEFT JOIN transaction_features t ON l.msno = t.msno
            LEFT JOIN listening_features g ON l.msno = g.msno
            """
        ).cache()
        summary = features.agg(
            F.count("*").alias("output_rows"),
            F.countDistinct("msno").alias("distinct_users"),
            F.sum("transaction_count_90").alias("transaction_count_checksum"),
            F.sum("active_days_30").alias("active_days_checksum"),
            F.sum("total_secs_30").alias("listening_seconds_checksum"),
        ).first()
        features.write.mode("overwrite").parquet(
            f"{results}/scaling/spark_features/pct_{percentage:02d}"
        )
        elapsed = time.perf_counter() - started
        records.append(
            (
                percentage,
                "Spark SQL on EMR",
                summary["output_rows"],
                summary["distinct_users"],
                int(summary["transaction_count_checksum"] or 0),
                int(summary["active_days_checksum"] or 0),
                float(summary["listening_seconds_checksum"] or 0.0),
                elapsed,
            )
        )
        print(
            f"[MILESTONE] 02 Spark scaling {percentage}% complete in "
            f"{elapsed:.3f} seconds"
        )
        features.unpersist()
    return spark.createDataFrame(
        records,
        [
            "percentage",
            "platform",
            "output_rows",
            "distinct_users",
            "transaction_count_checksum",
            "active_days_checksum",
            "listening_seconds_checksum",
            "end_to_end_runtime_seconds",
        ],
    )


def main():
    args = arguments()
    overall_started = time.perf_counter()
    spark = build_spark("KKBox-02-Big-Data-Experiments", args.shuffle_partitions)
    raw = args.raw_base.rstrip("/")
    run = args.run_base.rstrip("/")
    curated = f"{run}/curated"
    results = f"{run}/results"

    labels = spark.read.parquet(f"{curated}/labels/february")
    members = spark.read.parquet(f"{curated}/typed/members")
    transactions = spark.read.parquet(f"{curated}/typed/transactions")
    logs = spark.read.parquet(f"{curated}/typed/user_logs")
    labels.createOrReplaceTempView("labels")
    transactions.createOrReplaceTempView("transactions")
    logs.createOrReplaceTempView("user_logs")

    # RQ1 uses a deterministic 0.1% subscriber sample because a full raw
    # many-to-many join is intentionally unsafe.
    join_labels = labels.filter(
        deterministic_bucket("msno", 1000) < F.lit(1)
    )
    join_labels.createOrReplaceTempView("join_labels")
    join_logs = spark.sql(
        """
        SELECT g.msno, g.event_date, g.total_secs
        FROM user_logs g
        LEFT SEMI JOIN join_labels l ON g.msno = l.msno
        WHERE g.event_date BETWEEN DATE '2017-01-02' AND DATE '2017-01-31'
        """
    )
    join_tx = spark.sql(
        """
        SELECT t.msno, t.transaction_date, t.is_auto_renew, t.is_cancel
        FROM transactions t
        LEFT SEMI JOIN join_labels l ON t.msno = l.msno
        WHERE t.transaction_date BETWEEN DATE '2016-11-03'
                                     AND DATE '2017-01-31'
        """
    )
    join_logs.createOrReplaceTempView("join_logs")
    join_tx.createOrReplaceTempView("join_transactions")

    raw_join = spark.sql(
        """
        SELECT l.msno, g.event_date, g.total_secs,
               t.transaction_date, t.is_auto_renew, t.is_cancel
        FROM join_labels l
        INNER JOIN join_logs g ON l.msno = g.msno
        INNER JOIN join_transactions t ON l.msno = t.msno
        """
    )
    spark.catalog.clearCache()
    _, raw_rows, raw_seconds = materialize_with_time(raw_join)

    aggregate_first = spark.sql(
        """
        WITH listening AS (
          SELECT msno, COUNT(DISTINCT event_date) AS active_days_30,
                 SUM(total_secs) AS total_secs_30
          FROM join_logs GROUP BY msno
        ),
        transaction AS (
          SELECT msno, COUNT(*) AS transaction_count_90,
                 AVG(CAST(is_auto_renew AS DOUBLE)) AS auto_renew_rate_90,
                 SUM(CASE WHEN is_cancel = 1 THEN 1 ELSE 0 END)
                   AS cancellation_count_90
          FROM join_transactions GROUP BY msno
        )
        SELECT l.msno, g.active_days_30, g.total_secs_30,
               t.transaction_count_90, t.auto_renew_rate_90,
               t.cancellation_count_90
        FROM join_labels l
        LEFT JOIN listening g ON l.msno = g.msno
        LEFT JOIN transaction t ON l.msno = t.msno
        """
    )
    spark.catalog.clearCache()
    _, aggregate_rows, aggregate_seconds = materialize_with_time(
        aggregate_first
    )
    input_users = join_labels.count()
    input_logs = join_logs.count()
    input_tx = join_tx.count()
    join_metrics = spark.createDataFrame(
        [
            (
                "raw_many_to_many_join",
                input_users,
                input_logs,
                input_tx,
                raw_rows,
                raw_seconds,
                raw_rows / max(input_users, 1),
            ),
            (
                "aggregate_before_join",
                input_users,
                input_logs,
                input_tx,
                aggregate_rows,
                aggregate_seconds,
                aggregate_rows / max(input_users, 1),
            ),
        ],
        [
            "strategy",
            "sample_subscribers",
            "input_listening_rows",
            "input_transaction_rows",
            "output_rows",
            "runtime_seconds",
            "output_rows_per_subscriber",
        ],
    )
    show_and_write(
        join_metrics,
        "02A - Raw join versus aggregate-before-join",
        f"{results}/experiments/join_strategy",
    )

    # RQ2: identical 30-day aggregation from raw CSV and curated Parquet.
    csv_logs = (
        spark.read.schema(LOG_SCHEMA)
        .option("header", True)
        .csv(f"{raw}/user_logs.csv")
        .withColumn("event_date", F.to_date("date", "yyyyMMdd"))
    )
    csv_logs.createOrReplaceTempView("csv_logs")
    logs.createOrReplaceTempView("parquet_logs")
    csv_query = spark.sql(
        """
        SELECT msno, COUNT(DISTINCT event_date) AS active_days_30,
               SUM(total_secs) AS total_secs_30
        FROM csv_logs
        WHERE event_date BETWEEN DATE '2017-01-02' AND DATE '2017-01-31'
          AND total_secs >= 0
        GROUP BY msno
        """
    )
    parquet_query = spark.sql(
        """
        SELECT msno, COUNT(DISTINCT event_date) AS active_days_30,
               SUM(total_secs) AS total_secs_30
        FROM parquet_logs
        WHERE event_date BETWEEN DATE '2017-01-02' AND DATE '2017-01-31'
        GROUP BY msno
        """
    )
    spark.catalog.clearCache()
    csv_cached, csv_rows, csv_seconds = materialize_with_time(csv_query)
    csv_checksum = csv_cached.agg(F.sum("total_secs_30")).first()[0]
    csv_plan = csv_query._jdf.queryExecution().executedPlan().toString()
    csv_cached.unpersist()
    spark.catalog.clearCache()
    parquet_cached, parquet_rows, parquet_seconds = materialize_with_time(
        parquet_query
    )
    parquet_checksum = parquet_cached.agg(F.sum("total_secs_30")).first()[0]
    parquet_plan = parquet_query._jdf.queryExecution().executedPlan().toString()
    parquet_cached.unpersist()
    format_metrics = spark.createDataFrame(
        [
            ("CSV", csv_rows, float(csv_checksum), csv_seconds),
            ("Parquet", parquet_rows, float(parquet_checksum), parquet_seconds),
        ],
        ["format", "output_users", "seconds_checksum", "runtime_seconds"],
    ).withColumn(
        "results_match",
        (F.lit(csv_rows) == F.lit(parquet_rows))
        & (
            F.abs(
                F.lit(float(csv_checksum)) - F.lit(float(parquet_checksum))
            )
            < F.lit(0.01)
        ),
    )
    show_and_write(
        format_metrics,
        "02B - CSV versus Parquet",
        f"{results}/experiments/csv_vs_parquet",
    )
    spark.createDataFrame(
        [("CSV", csv_plan), ("Parquet", parquet_plan)],
        ["format", "physical_plan"],
    ).write.mode("overwrite").json(f"{results}/experiments/physical_plans")

    percentages = [1, 5, 10, 20]
    preparation = build_scaling_inputs(
        spark,
        labels,
        members,
        transactions,
        logs,
        results,
        percentages,
    )
    show_and_write(
        preparation,
        "02C - Controlled scaling input inventory",
        f"{results}/scaling/input_inventory",
    )
    scaling = run_spark_scaling(spark, results, percentages)
    show_and_write(
        scaling,
        "02D - Spark SQL controlled scaling results",
        f"{results}/scaling/spark_runtime",
    )

    runtime = spark.createDataFrame(
        [
            (
                "02_run_big_data_experiments",
                time.perf_counter() - overall_started,
                "complete",
            )
        ],
        ["job", "runtime_seconds", "status"],
    )
    show_and_write(
        runtime,
        "02E - Big-data experiments completed",
        f"{results}/runtime/job_02",
    )
    spark.stop()


if __name__ == "__main__":
    main()
