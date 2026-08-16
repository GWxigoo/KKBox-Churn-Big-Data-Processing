#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from typing import Iterable

from pyspark.sql import DataFrame, SparkSession, functions as F, types as T


TRAIN_SCHEMA = T.StructType(
    [
        T.StructField("msno", T.StringType(), True),
        T.StructField("is_churn", T.IntegerType(), True),
    ]
)

MEMBER_SCHEMA = T.StructType(
    [
        T.StructField("msno", T.StringType(), True),
        T.StructField("city", T.IntegerType(), True),
        T.StructField("bd", T.IntegerType(), True),
        T.StructField("gender", T.StringType(), True),
        T.StructField("registered_via", T.IntegerType(), True),
        T.StructField("registration_init_time", T.StringType(), True),
    ]
)

TRANSACTION_SCHEMA = T.StructType(
    [
        T.StructField("msno", T.StringType(), True),
        T.StructField("payment_method_id", T.IntegerType(), True),
        T.StructField("payment_plan_days", T.IntegerType(), True),
        T.StructField("plan_list_price", T.IntegerType(), True),
        T.StructField("actual_amount_paid", T.IntegerType(), True),
        T.StructField("is_auto_renew", T.IntegerType(), True),
        T.StructField("transaction_date", T.StringType(), True),
        T.StructField("membership_expire_date", T.StringType(), True),
        T.StructField("is_cancel", T.IntegerType(), True),
    ]
)

LOG_SCHEMA = T.StructType(
    [
        T.StructField("msno", T.StringType(), True),
        T.StructField("date", T.StringType(), True),
        T.StructField("num_25", T.LongType(), True),
        T.StructField("num_50", T.LongType(), True),
        T.StructField("num_75", T.LongType(), True),
        T.StructField("num_985", T.LongType(), True),
        T.StructField("num_100", T.LongType(), True),
        T.StructField("num_unq", T.LongType(), True),
        T.StructField("total_secs", T.DoubleType(), True),
    ]
)

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

NULLABLE_FEATURES = [
    "membership_tenure_days",
    "days_since_last_transaction",
    "days_until_expiration",
    "days_since_last_listen",
]


def build_spark(app_name: str, shuffle_partitions: int = 400) -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def read_sql(
    spark: SparkSession, sql_base: str, filename: str, **values: object
) -> str:
    uri = f"{sql_base.rstrip('/')}/{filename}"
    pairs = spark.sparkContext.wholeTextFiles(uri).collect()
    if len(pairs) != 1:
        raise FileNotFoundError(f"Expected exactly one SQL file at {uri}")
    text = pairs[0][1]
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    if "{{" in text:
        raise ValueError(f"Unresolved SQL placeholder in {filename}")
    return text


def write_single_csv(df: DataFrame, path: str) -> None:
    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(path)
    )


def show_and_write(df: DataFrame, label: str, path: str) -> None:
    print(f"\n[MILESTONE] {label}")
    df.show(200, truncate=False)
    write_single_csv(df, path)


def timed_materialize(df: DataFrame) -> tuple[int, float]:
    started = time.perf_counter()
    rows = df.count()
    return rows, time.perf_counter() - started


def deterministic_bucket(column: str = "msno", modulus: int = 100):
    # The first 15 hex digits fit in a signed 64-bit integer. The same formula
    # is implemented with hashlib.sha256 in the pandas job.
    digest = F.sha2(F.col(column), 256)
    decimal = F.conv(F.substring(digest, 1, 15), 16, 10).cast("long")
    return F.pmod(decimal, F.lit(modulus))


def path_stats(spark: SparkSession, uri: str) -> tuple[int, int]:
    jvm = spark.sparkContext._jvm
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    path = jvm.org.apache.hadoop.fs.Path(uri)
    fs = path.getFileSystem(hadoop_conf)
    if not fs.exists(path):
        return 0, 0
    iterator = fs.listFiles(path, True)
    files = 0
    size = 0
    while iterator.hasNext():
        status = iterator.next()
        files += 1
        size += status.getLen()
    return files, size


def write_json_via_spark(
    spark: SparkSession, payload: dict, path: str, field: str = "json"
) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    spark.createDataFrame([(text,)], [field]).coalesce(1).write.mode(
        "overwrite"
    ).text(path)


def union_all(frames: Iterable[DataFrame]) -> DataFrame:
    iterator = iter(frames)
    result = next(iterator)
    for frame in iterator:
        result = result.unionByName(frame, allowMissingColumns=True)
    return result
