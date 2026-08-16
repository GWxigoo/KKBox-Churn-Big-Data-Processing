#!/usr/bin/env python3
"""Train one class-weighted Logistic Regression model with all combined features."""

from __future__ import annotations

import argparse
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from kkbox_common import (
    FEATURE_COLUMNS,
    NULLABLE_FEATURES,
    build_spark,
    deterministic_bucket,
    show_and_write,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-base", required=True)
    parser.add_argument("--shuffle-partitions", type=int, default=400)
    return parser.parse_args()


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def add_class_weights(frame):
    counts = {
        int(row["is_churn"]): int(row["count"])
        for row in frame.groupBy("is_churn").count().collect()
    }
    total = counts[0] + counts[1]
    negative_weight = total / (2.0 * counts[0])
    positive_weight = total / (2.0 * counts[1])
    weighted = frame.withColumn(
        "class_weight",
        F.when(F.col("is_churn") == 1, F.lit(positive_weight)).otherwise(
            F.lit(negative_weight)
        ),
    )
    return weighted, counts, negative_weight, positive_weight


def preprocessing_stages():
    nullable = [c for c in FEATURE_COLUMNS if c in NULLABLE_FEATURES]
    imputed = [f"{c}_imputed" for c in nullable]
    assembler_inputs = [
        f"{c}_imputed" if c in nullable else c for c in FEATURE_COLUMNS
    ]
    return [
        Imputer(inputCols=nullable, outputCols=imputed, strategy="median"),
        VectorAssembler(
            inputCols=assembler_inputs,
            outputCol="assembled_features",
            handleInvalid="error",
        ),
        StandardScaler(
            inputCol="assembled_features",
            outputCol="features",
            withMean=True,
            withStd=True,
        ),
    ]


def with_probability(frame):
    return frame.withColumn(
        "p1", vector_to_array("probability")[1].cast("double")
    )


def choose_threshold(spark, validation_predictions):
    scored = with_probability(validation_predictions).select(
        "is_churn", "p1"
    ).cache()
    candidates = spark.createDataFrame(
        [(float(value) / 100.0,) for value in range(10, 91, 5)],
        ["threshold"],
    )
    summary = (
        scored.crossJoin(F.broadcast(candidates))
        .withColumn(
            "predicted", (F.col("p1") >= F.col("threshold")).cast("int")
        )
        .groupBy("threshold")
        .agg(
            F.sum(
                F.when(
                    (F.col("is_churn") == 1) & (F.col("predicted") == 1), 1
                ).otherwise(0)
            ).alias("tp"),
            F.sum(
                F.when(
                    (F.col("is_churn") == 0) & (F.col("predicted") == 1), 1
                ).otherwise(0)
            ).alias("fp"),
            F.sum(
                F.when(
                    (F.col("is_churn") == 1) & (F.col("predicted") == 0), 1
                ).otherwise(0)
            ).alias("fn"),
        )
        .withColumn(
            "precision",
            F.when(
                (F.col("tp") + F.col("fp")) > 0,
                F.col("tp") / (F.col("tp") + F.col("fp")),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "recall",
            F.when(
                (F.col("tp") + F.col("fn")) > 0,
                F.col("tp") / (F.col("tp") + F.col("fn")),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "f1",
            F.when(
                (F.col("precision") + F.col("recall")) > 0,
                2 * F.col("precision") * F.col("recall")
                / (F.col("precision") + F.col("recall")),
            ).otherwise(F.lit(0.0)),
        )
    ).cache()
    best = summary.orderBy(
        F.desc("f1"), F.desc("recall"), F.asc("threshold")
    ).first()
    scored.unpersist()
    return float(best["threshold"]), summary


def evaluate(predictions, threshold):
    scored = (
        with_probability(predictions)
        .withColumn(
            "predicted_label",
            (F.col("p1") >= F.lit(threshold)).cast("int"),
        )
        .cache()
    )
    pr_auc = BinaryClassificationEvaluator(
        labelCol="is_churn",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR",
    ).evaluate(scored)
    roc_auc = BinaryClassificationEvaluator(
        labelCol="is_churn",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    ).evaluate(scored)
    clipped = F.greatest(
        F.lit(1e-15), F.least(F.col("p1"), F.lit(1.0 - 1e-15))
    )
    row = scored.agg(
        F.count("*").alias("rows"),
        F.sum(F.when((F.col("is_churn") == 1) & (F.col("predicted_label") == 1), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col("is_churn") == 0) & (F.col("predicted_label") == 1), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("is_churn") == 0) & (F.col("predicted_label") == 0), 1).otherwise(0)).alias("tn"),
        F.sum(F.when((F.col("is_churn") == 1) & (F.col("predicted_label") == 0), 1).otherwise(0)).alias("fn"),
        F.avg(
            -(F.col("is_churn") * F.log(clipped)
              + (1 - F.col("is_churn")) * F.log(1 - clipped))
        ).alias("log_loss"),
    ).first()
    precision = safe_divide(row["tp"], row["tp"] + row["fp"])
    recall = safe_divide(row["tp"], row["tp"] + row["fn"])
    f1 = safe_divide(2 * precision * recall, precision + recall)
    metrics = {
        "model": "Logistic Regression - all combined features",
        "test_rows": int(row["rows"]),
        "feature_count": len(FEATURE_COLUMNS),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "log_loss": float(row["log_loss"]),
        "tp": int(row["tp"]),
        "fp": int(row["fp"]),
        "tn": int(row["tn"]),
        "fn": int(row["fn"]),
    }
    return scored, metrics


def main():
    args = arguments()
    overall_started = time.perf_counter()
    spark = build_spark("KKBox-03-Logistic-Regression", args.shuffle_partitions)
    run = args.run_base.rstrip("/")
    curated = f"{run}/curated"
    results = f"{run}/results"

    february = spark.read.parquet(f"{curated}/features/february").withColumn(
        "split_bucket", deterministic_bucket("msno", 100)
    )
    march = spark.read.parquet(f"{curated}/features/march")
    development_train = february.filter(F.col("split_bucket") < 80).drop("split_bucket")
    validation = february.filter(F.col("split_bucket") >= 80).drop("split_bucket")
    weighted_train, counts, negative_weight, positive_weight = add_class_weights(
        development_train
    )

    split_summary = spark.createDataFrame(
        [
            ("February model training", development_train.count()),
            ("February threshold validation", validation.count()),
            ("March untouched temporal test", march.count()),
        ],
        ["dataset", "rows"],
    )
    show_and_write(
        split_summary,
        "03A - Deterministic temporal model split",
        f"{results}/models/data_split",
    )
    weights = spark.createDataFrame(
        [
            ("non_churn", counts[0], negative_weight),
            ("churn", counts[1], positive_weight),
        ],
        ["class", "training_rows", "class_weight"],
    )
    show_and_write(
        weights,
        "03B - Logistic Regression class weights",
        f"{results}/models/class_weights",
    )

    classifier = LogisticRegression(
        labelCol="is_churn",
        featuresCol="features",
        weightCol="class_weight",
        maxIter=100,
        regParam=0.0,
        elasticNetParam=0.0,
        standardization=False,
    )
    pipeline = Pipeline(stages=preprocessing_stages() + [classifier])

    print("[MILESTONE] 03C - Training one combined-feature Logistic Regression model")
    training_started = time.perf_counter()
    fitted = pipeline.fit(weighted_train)
    training_seconds = time.perf_counter() - training_started

    validation_started = time.perf_counter()
    validation_predictions = fitted.transform(validation).cache()
    validation_predictions.count()
    validation_seconds = time.perf_counter() - validation_started
    threshold, threshold_table = choose_threshold(spark, validation_predictions)
    validation_predictions.unpersist()

    show_and_write(
        threshold_table.orderBy("threshold"),
        "03D - Logistic Regression validation threshold search",
        f"{results}/models/threshold_search",
    )

    inference_started = time.perf_counter()
    test_predictions = fitted.transform(march).cache()
    test_predictions.count()
    inference_seconds = time.perf_counter() - inference_started
    scored, metrics = evaluate(test_predictions, threshold)
    test_predictions.unpersist()
    metrics["training_seconds"] = training_seconds
    metrics["validation_prediction_seconds"] = validation_seconds
    metrics["test_inference_seconds"] = inference_seconds

    metric_columns = list(metrics.keys())
    metric_frame = spark.createDataFrame(
        [tuple(metrics[column] for column in metric_columns)], metric_columns
    )
    show_and_write(
        metric_frame,
        "03E - Logistic Regression March temporal-test metrics",
        f"{results}/models/metrics",
    )
    (
        scored.select("msno", "is_churn", "p1", "predicted_label")
        .write.mode("overwrite")
        .parquet(f"{results}/models/predictions/logistic_regression_all_features")
    )
    scored.unpersist()
    fitted.write().overwrite().save(f"{run}/models/logistic_regression_all_features")

    print("[MILESTONE] 03F - Exporting matched local Logistic Regression inputs")
    (
        february.write.mode("overwrite")
        .option("header", True)
        .option("compression", "gzip")
        .csv(f"{results}/local_ml_input/february")
    )
    (
        march.write.mode("overwrite")
        .option("header", True)
        .option("compression", "gzip")
        .csv(f"{results}/local_ml_input/march")
    )

    runtime = spark.createDataFrame(
        [("03_train_logistic_regression", time.perf_counter() - overall_started, "complete")],
        ["job", "runtime_seconds", "status"],
    )
    show_and_write(
        runtime,
        "03G - Logistic Regression job completed",
        f"{results}/runtime/job_03",
    )
    spark.stop()


if __name__ == "__main__":
    main()
