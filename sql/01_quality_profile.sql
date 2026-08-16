SELECT
  'train' AS table_name,
  COUNT(*) AS input_rows,
  COUNT(DISTINCT msno) AS distinct_users,
  SUM(CASE WHEN msno IS NULL OR TRIM(msno) = '' THEN 1 ELSE 0 END) AS invalid_msno,
  SUM(CASE WHEN is_churn NOT IN (0, 1) OR is_churn IS NULL THEN 1 ELSE 0 END) AS invalid_target,
  CAST(NULL AS BIGINT) AS invalid_primary_date,
  CAST(NULL AS BIGINT) AS invalid_secondary_date,
  CAST(NULL AS BIGINT) AS negative_total_secs,
  CAST(NULL AS BIGINT) AS negative_play_counts
FROM raw_train

UNION ALL

SELECT
  'train_v2',
  COUNT(*),
  COUNT(DISTINCT msno),
  SUM(CASE WHEN msno IS NULL OR TRIM(msno) = '' THEN 1 ELSE 0 END),
  SUM(CASE WHEN is_churn NOT IN (0, 1) OR is_churn IS NULL THEN 1 ELSE 0 END),
  NULL, NULL, NULL, NULL
FROM raw_train_v2

UNION ALL

SELECT
  'members_v3',
  COUNT(*),
  COUNT(DISTINCT msno),
  SUM(CASE WHEN msno IS NULL OR TRIM(msno) = '' THEN 1 ELSE 0 END),
  NULL,
  SUM(CASE WHEN registration_init_time IS NULL
                 OR registration_date IS NULL THEN 1 ELSE 0 END),
  NULL, NULL, NULL
FROM typed_members

UNION ALL

SELECT
  'transactions',
  COUNT(*),
  COUNT(DISTINCT msno),
  SUM(CASE WHEN msno IS NULL OR TRIM(msno) = '' THEN 1 ELSE 0 END),
  NULL,
  SUM(CASE WHEN transaction_date_raw IS NULL
                 OR transaction_date IS NULL THEN 1 ELSE 0 END),
  SUM(CASE WHEN membership_expire_date_raw IS NULL
                 OR membership_expire_date IS NULL THEN 1 ELSE 0 END),
  NULL,
  NULL
FROM typed_transactions

UNION ALL

SELECT
  'user_logs',
  COUNT(*),
  COUNT(DISTINCT msno),
  SUM(CASE WHEN msno IS NULL OR TRIM(msno) = '' THEN 1 ELSE 0 END),
  NULL,
  SUM(CASE WHEN event_date_raw IS NULL OR event_date IS NULL THEN 1 ELSE 0 END),
  NULL,
  SUM(CASE WHEN total_secs IS NULL OR total_secs < 0 THEN 1 ELSE 0 END),
  SUM(CASE WHEN COALESCE(num_25, -1) < 0
                 OR COALESCE(num_50, -1) < 0
                 OR COALESCE(num_75, -1) < 0
                 OR COALESCE(num_985, -1) < 0
                 OR COALESCE(num_100, -1) < 0
                 OR COALESCE(num_unq, -1) < 0
           THEN 1 ELSE 0 END)
FROM typed_user_logs

