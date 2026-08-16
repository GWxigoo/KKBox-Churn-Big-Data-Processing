# Focused-scope AWS rerun

This revision keeps Spark SQL cleaning, Parquet preparation, aggregate-before-
join, CSV-versus-Parquet, controlled Spark-versus-pandas scaling, and a matched
Spark-ML-versus-local-scikit-learn Logistic Regression comparison. 

## 1. Use a new code prefix and run ID

```bash
export AWS_DEFAULT_REGION=us-east-1
export BUCKET=your-bucket-name
export CODE_BASE=s3://$BUCKET/code/v3
export RAW_BASE=s3://$BUCKET/raw
export RUN_ID=YYYYMMDD-rerun-01
export RUN_BASE=s3://$BUCKET/runs/$RUN_ID
export CLUSTER_ID=PASTE_YOUR_ACTIVE_CLUSTER_ID
```

Confirm the run prefix is empty with `aws s3 ls "$RUN_BASE/" --recursive`. If
objects appear, choose another run ID rather than deleting an old run.

## 2. Upload the revised source from Windows PowerShell

```powershell
$BUCKET = "your-bucket-name"
$CODE = "s3://$BUCKET/code/v3"
aws s3 cp .\common\kkbox_common.py "$CODE/kkbox_common.py"
aws s3 cp .\jobs\01_build_curated_and_features.py "$CODE/01_build_curated_and_features.py"
aws s3 cp .\jobs\02_run_big_data_experiments.py "$CODE/02_run_big_data_experiments.py"
aws s3 cp .\jobs\03_train_spark_models.py "$CODE/03_train_spark_models.py"
aws s3 cp .\jobs\04_build_result_index.py "$CODE/04_build_result_index.py"
aws s3 cp .\sql "$CODE/sql" --recursive
```

Verify with `aws s3 ls "$CODE_BASE/" --recursive` in CloudShell.

## 3. Submit Step 01

```bash
export STEP1=$(aws emr add-steps --cluster-id "$CLUSTER_ID" \
  --steps Type=CUSTOM_JAR,Name="FOCUSED-01-Clean-Parquet-Features",ActionOnFailure=CONTINUE,Jar=command-runner.jar,Args=[spark-submit,--master,yarn,--deploy-mode,client,--py-files,"$CODE_BASE/kkbox_common.py","$CODE_BASE/01_build_curated_and_features.py",--raw-base,"$RAW_BASE",--run-base,"$RUN_BASE",--sql-base,"$CODE_BASE",--shuffle-partitions,400] \
  --query 'StepIds[0]' --output text)
echo "STEP1=$STEP1"
```

## 4. Submit Step 02 after Step 01 completes

```bash
export STEP2=$(aws emr add-steps --cluster-id "$CLUSTER_ID" \
  --steps Type=CUSTOM_JAR,Name="FOCUSED-02-Big-Data-Experiments",ActionOnFailure=CONTINUE,Jar=command-runner.jar,Args=[spark-submit,--master,yarn,--deploy-mode,client,--py-files,"$CODE_BASE/kkbox_common.py","$CODE_BASE/02_run_big_data_experiments.py",--raw-base,"$RAW_BASE",--run-base,"$RUN_BASE",--shuffle-partitions,400] \
  --query 'StepIds[0]' --output text)
echo "STEP2=$STEP2"
```

## 5. Submit Step 03

```bash
export STEP3=$(aws emr add-steps --cluster-id "$CLUSTER_ID" \
  --steps Type=CUSTOM_JAR,Name="FOCUSED-03-Logistic-Regression",ActionOnFailure=CONTINUE,Jar=command-runner.jar,Args=[spark-submit,--master,yarn,--deploy-mode,client,--py-files,"$CODE_BASE/kkbox_common.py","$CODE_BASE/03_train_spark_models.py",--run-base,"$RUN_BASE",--shuffle-partitions,400] \
  --query 'StepIds[0]' --output text)
echo "STEP3=$STEP3"
```

Expected Step 03 outputs are the split, class weights, threshold search,
Logistic Regression metrics and predictions, matched local inputs, the saved
Logistic Regression pipeline, and the Step 03 runtime. No baseline, Random
Forest, feature-importance, transaction-only, or coefficient output is created.

## 6. Submit Step 04

```bash
export STEP4=$(aws emr add-steps --cluster-id "$CLUSTER_ID" \
  --steps Type=CUSTOM_JAR,Name="FOCUSED-04-Result-Index",ActionOnFailure=CONTINUE,Jar=command-runner.jar,Args=[spark-submit,--master,yarn,--deploy-mode,client,--py-files,"$CODE_BASE/kkbox_common.py","$CODE_BASE/04_build_result_index.py",--run-base,"$RUN_BASE"] \
  --query 'StepIds[0]' --output text)
echo "STEP4=$STEP4"
```

## 7. Status and milestone logs

```bash
aws emr list-steps --cluster-id "$CLUSTER_ID" \
  --query 'Steps[].{Id:Id,Name:Name,State:Status.State}' --output table
```

```bash
export LOG_URI=$(aws emr describe-cluster --cluster-id "$CLUSTER_ID" \
  --query 'Cluster.LogUri' --output text)
export LOG_URI="${LOG_URI/s3n:/s3:}"
export LOG_URI="${LOG_URI%/}"
export STEP_LOG="$LOG_URI/$CLUSTER_ID/steps/$STEP3"
aws s3 cp "$STEP_LOG/stdout.gz" - | gzip -dc | grep '\[MILESTONE\]'
```

## 8. Run matched local Logistic Regression

Download `results/local_ml_input/` to Windows, then run:

```powershell
python -m pip install -r requirements-local.txt
python local\train_sklearn_models.py `
  --input-base downloaded-results\local_ml_input `
  --output-dir local-results\models `
  --seed 42
```

Insert `sklearn_logistic_metrics.csv` beside the Spark metrics. Compare runtime,
memory, ROC-AUC, PR-AUC, log loss, precision, recall, F1, and confusion counts.
