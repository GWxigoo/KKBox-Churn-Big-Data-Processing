#!/usr/bin/env python3
"""Audit, clean, convert, aggregate, and join the complete KKBox dataset."""

from __future__ import annotations

import argparse
import time

from pyspark.sql import functions as F

from kkbox_common import (
    FEATURE_COLUMNS,
    LOG_SCHEMA,
    MEMBER_SCHEMA,
    TRAIN_SCHEMA,
    TRANSACTION_SCHEMA,
    build_spark,
    path_stats,
    read_sql,
    show_and_write,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-base", required=True)
    parser.add_argument("--run-base", required=True)
    parser.add_argument("--sql-base", required=True)
    parser.add_argument("--shuffle-partitions", type=int, default=400)
    return parser.parse_args()


def typed_sources(spark, raw):
    train = spark.read.schema(TRAIN_SCHEMA).option("header", True).csv(
        f"{raw}/train.csv"
    )
    train_v2 = spark.read.schema(TRAIN_SCHEMA).option("header", True).csv(
        f"{raw}/train_v2.csv"
    )
    raw_members = spark.read.schema(MEMBER_SCHEMA).option("header", True).csv(
        f"{raw}/members_v3.csv"
    )
    raw_transactions = (
        spark.read.schema(TRANSACTION_SCHEMA)
        .option("header", True)
        .csv(f"{raw}/transactions.csv")
    )
    raw_logs = spark.read.schema(LOG_SCHEMA).option("header", True).csv(
        f"{raw}/user_logs.csv"
    )

    typed_members = raw_members.withColumn(
        "registration_date",
        F.to_date(F.col("registration_init_time"), "yyyyMMdd"),
    )
    typed_transactions = (
        raw_transactions
        .withColumnRenamed("transaction_date", "transaction_date_raw")
        .withColumnRenamed(
            "membership_expire_date", "membership_expire_date_raw"
        )
        .withColumn(
            "transaction_date",
            F.to_date("transaction_date_raw", "yyyyMMdd"),
        )
        .withColumn(
            "membership_expire_date",
            F.to_date("membership_expire_date_raw", "yyyyMMdd"),
        )
    )
    typed_logs = (
        raw_logs.withColumnRenamed("date", "event_date_raw")
        .withColumn("event_date", F.to_date("event_date_raw", "yyyyMMdd"))
    )

    views = {
        "raw_train": train,
        "raw_train_v2": train_v2,
        "raw_members": raw_members,
        "typed_members": typed_members,
        "typed_transactions": typed_transactions,
        "typed_user_logs": typed_logs,
    }
    for name, frame in views.items():
        frame.createOrReplaceTempView(name)
    return views


def clean_labels(frame):
    return (
        frame.filter(
            F.col("msno").isNotNull()
            & (F.trim("msno") != "")
            & F.col("is_churn").isin(0, 1)
        )
        .groupBy("msno")
        .agg(F.max("is_churn").cast("int").alias("is_churn"))
    )


def treatment_rows(quality_rows, duplicate_rows, after_counts):
    q = {row["table_name"]: row.asDict() for row in quality_rows}
    d = {row["table_name"]: row.asDict() for row in duplicate_rows}
    rows = []

    def add(table, check, affected, treatment):
        rows.append(
            (
                table,
                check,
                int(affected or 0),
                treatment,
                int(after_counts[table]),
            )
        )

    for table in ("train", "train_v2", "members_v3", "transactions", "user_logs"):
        add(
            table,
            "Null or blank msno",
            q[table]["invalid_msno"],
            "Exclude row before aggregation",
        )
        add(
            table,
            "Exact duplicate row",
            d[table]["exact_duplicate_rows"],
            "Retain one exact copy",
        )
    for table in ("train", "train_v2"):
        add(
            table,
            "Target outside {0,1}",
            q[table]["invalid_target"],
            "Exclude invalid labelled row",
        )
    add(
        "members_v3",
        "Invalid registration date",
        q["members_v3"]["invalid_primary_date"],
        "Retain subscriber but leave tenure missing",
    )
    add(
        "members_v3",
        "Duplicate subscriber membership row",
        q["members_v3"]["input_rows"] - after_counts["members_v3"],
        "Keep latest valid registration deterministically",
    )
    add(
        "transactions",
        "Invalid transaction date",
        q["transactions"]["invalid_primary_date"],
        "Exclude from dated aggregation",
    )
    add(
        "transactions",
        "Invalid expiration date",
        q["transactions"]["invalid_secondary_date"],
        "Exclude from dated aggregation",
    )
    add(
        "user_logs",
        "Missing or negative total_secs",
        q["user_logs"]["negative_total_secs"],
        "Exclude from listening aggregation",
    )
    add(
        "user_logs",
        "Missing or negative play count",
        q["user_logs"]["negative_play_counts"],
        "Exclude invalid daily listening row",
    )
    return rows


def build_features(
    spark,
    sql_base,
    labels,
    label_view,
    cohort,
    cutoff,
    curated,
):
    labels.createOrReplaceTempView(label_view)
    member = spark.sql(
        read_sql(
            spark,
            sql_base,
            "06_membership_features.sql",
            LABEL_VIEW=label_view,
            CUTOFF=cutoff,
        )
    )
    transaction = spark.sql(
        read_sql(
            spark,
            sql_base,
            "07_transaction_features.sql",
            LABEL_VIEW=label_view,
            CUTOFF=cutoff,
        )
    )
    listening = spark.sql(
        read_sql(
            spark,
            sql_base,
            "08_listening_features.sql",
            LABEL_VIEW=label_view,
            CUTOFF=cutoff,
        )
    )
    member_view = f"{cohort}_membership_features"
    transaction_view = f"{cohort}_transaction_features"
    listening_view = f"{cohort}_listening_features"
    member.createOrReplaceTempView(member_view)
    transaction.createOrReplaceTempView(transaction_view)
    listening.createOrReplaceTempView(listening_view)
    features = spark.sql(
        read_sql(
            spark,
            sql_base,
            "09_feature_join.sql",
            LABEL_VIEW=label_view,
            MEMBERSHIP_VIEW=member_view,
            TRANSACTION_VIEW=transaction_view,
            LISTENING_VIEW=listening_view,
        )
    )
    features.write.mode("overwrite").parquet(f"{curated}/features/{cohort}")
    return spark.read.parquet(f"{curated}/features/{cohort}")


def main():
    args = arguments()
    job_started = time.perf_counter()
    spark = build_spark("KKBox-01-Audit-Clean-Features", args.shuffle_partitions)
    raw = args.raw_base.rstrip("/")
    run = args.run_base.rstrip("/")
    curated = f"{run}/curated"
    results = f"{run}/results"

    print("[MILESTONE] 01A - Reading all five raw CSV files")
    sources = typed_sources(spark, raw)

    quality = spark.sql(
        read_sql(spark, args.sql_base, "01_quality_profile.sql")
    ).cache()
    show_and_write(
        quality,
        "01B - Data-quality profile before treatment",
        f"{results}/audit/quality_check_counts",
    )

    duplicates = spark.sql(
        read_sql(spark, args.sql_base, "02_duplicate_profile.sql")
    ).cache()
    show_and_write(
        duplicates,
        "01C - Exact-duplicate profile before treatment",
        f"{results}/audit/duplicate_summary",
    )

    print("[MILESTONE] 01D - Applying documented Spark SQL treatments")
    members = spark.sql(
        read_sql(spark, args.sql_base, "03_clean_members.sql")
    )
    transactions = spark.sql(
        read_sql(spark, args.sql_base, "04_clean_transactions.sql")
    )
    logs = spark.sql(
        read_sql(spark, args.sql_base, "05_clean_user_logs.sql")
    )

    members.write.mode("overwrite").parquet(f"{curated}/typed/members")
    (
        transactions.withColumn("tx_year", F.year("transaction_date"))
        .withColumn("tx_month", F.month("transaction_date"))
        .write.mode("overwrite")
        .partitionBy("tx_year", "tx_month")
        .parquet(f"{curated}/typed/transactions")
    )
    (
        logs.withColumn("event_year", F.year("event_date"))
        .withColumn("event_month", F.month("event_date"))
        .write.mode("overwrite")
        .partitionBy("event_year", "event_month")
        .parquet(f"{curated}/typed/user_logs")
    )

    clean_members = spark.read.parquet(f"{curated}/typed/members")
    clean_transactions = spark.read.parquet(f"{curated}/typed/transactions")
    clean_logs = spark.read.parquet(f"{curated}/typed/user_logs")
    clean_train = clean_labels(sources["raw_train"])
    clean_train_v2 = clean_labels(sources["raw_train_v2"])
    clean_train.write.mode("overwrite").parquet(f"{curated}/labels/february")
    clean_train_v2.write.mode("overwrite").parquet(f"{curated}/labels/march")

    after_counts = {
        "train": clean_train.count(),
        "train_v2": clean_train_v2.count(),
        "members_v3": clean_members.count(),
        "transactions": clean_transactions.count(),
        "user_logs": clean_logs.count(),
    }
    treatment = spark.createDataFrame(
        treatment_rows(
            quality.collect(),
            duplicates.collect(),
            after_counts,
        ),
        [
            "table_name",
            "quality_check",
            "affected_rows",
            "treatment",
            "clean_rows_remaining",
        ],
    )
    show_and_write(
        treatment,
        "01E - Treatment summary and rows retained",
        f"{results}/audit/quality_treatment_summary",
    )

    clean_members.createOrReplaceTempView("members")
    clean_transactions.createOrReplaceTempView("transactions")
    clean_logs.createOrReplaceTempView("user_logs")

    date_ranges = spark.createDataFrame(
        [
            (
                "transactions",
                str(clean_transactions.agg(F.min("transaction_date")).first()[0]),
                str(clean_transactions.agg(F.max("transaction_date")).first()[0]),
            ),
            (
                "user_logs",
                str(clean_logs.agg(F.min("event_date")).first()[0]),
                str(clean_logs.agg(F.max("event_date")).first()[0]),
            ),
        ],
        ["table_name", "minimum_valid_date", "maximum_valid_date"],
    )
    show_and_write(
        date_ranges,
        "01F - Valid date ranges",
        f"{results}/audit/date_range_summary",
    )

    print("[MILESTONE] 01G - Building February features with 2017-01-31 cutoff")
    february = build_features(
        spark,
        args.sql_base,
        clean_train,
        "february_labels",
        "february",
        "2017-01-31",
        curated,
    )
    print("[MILESTONE] 01H - Building March features with 2017-02-28 cutoff")
    march = build_features(
        spark,
        args.sql_base,
        clean_train_v2,
        "march_labels",
        "march",
        "2017-02-28",
        curated,
    )

    validations = []
    null_rows = []
    cutoffs = {"february": "2017-01-31", "march": "2017-02-28"}
    for cohort, frame, label_count in [
        ("february", february, clean_train.count()),
        ("march", march, clean_train_v2.count()),
    ]:
        summary = frame.agg(
            F.count("*").alias("rows"),
            F.countDistinct("msno").alias("distinct_users"),
            F.sum("is_churn").alias("churners"),
        ).first()
        passed = (
            summary["rows"] == summary["distinct_users"] == label_count
        )
        validations.append(
            (
                cohort,
                cutoffs[cohort],
                label_count,
                summary["rows"],
                summary["distinct_users"],
                int(summary["churners"]),
                bool(passed),
            )
        )
        for feature in FEATURE_COLUMNS:
            count = frame.filter(F.col(feature).isNull()).count()
            null_rows.append((cohort, feature, count))
        if not passed:
            raise AssertionError(f"{cohort} feature-table grain check failed")

    validation_df = spark.createDataFrame(
        validations,
        [
            "cohort",
            "feature_cutoff",
            "label_rows",
            "feature_rows",
            "distinct_feature_users",
            "churners",
            "grain_check_passed",
        ],
    )
    show_and_write(
        validation_df,
        "01I - Cohort and one-row-per-subscriber validation",
        f"{results}/audit/cohort_validation",
    )
    null_df = spark.createDataFrame(
        null_rows, ["cohort", "feature", "null_rows"]
    )
    show_and_write(
        null_df,
        "01J - Feature null summary after joins",
        f"{results}/audit/feature_null_summary",
    )

    march.createOrReplaceTempView("march_features")
    segments = spark.sql(
        read_sql(spark, args.sql_base, "10_churn_segments.sql")
    )
    show_and_write(
        segments,
        "01K - March churn segments",
        f"{results}/analysis/churn_segments",
    )

    storage_rows = []
    for label, uri in [
        ("raw_csv", raw),
        ("curated_parquet", curated),
    ]:
        files, size = path_stats(spark, uri)
        storage_rows.append((label, uri, files, size, size / (1024**3)))
    storage = spark.createDataFrame(
        storage_rows,
        ["storage_layer", "path", "files", "bytes", "gibibytes"],
    )
    show_and_write(
        storage,
        "01L - Raw and curated storage inventory",
        f"{results}/audit/storage_summary",
    )

    elapsed = time.perf_counter() - job_started
    runtime = spark.createDataFrame(
        [("01_build_curated_and_features", elapsed, "complete")],
        ["job", "runtime_seconds", "status"],
    )
    show_and_write(
        runtime,
        "01M - Full audit/feature job completed",
        f"{results}/runtime/job_01",
    )
    quality.unpersist()
    duplicates.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
