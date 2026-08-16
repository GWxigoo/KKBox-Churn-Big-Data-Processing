WITH profiles AS (
  SELECT 'train' AS table_name, COUNT(*) AS rows_before,
         COUNT(DISTINCT STRUCT(msno, is_churn)) AS exact_distinct_rows
  FROM raw_train
  UNION ALL
  SELECT 'train_v2', COUNT(*), COUNT(DISTINCT STRUCT(msno, is_churn))
  FROM raw_train_v2
  UNION ALL
  SELECT 'members_v3', COUNT(*),
         COUNT(DISTINCT STRUCT(msno, city, bd, gender, registered_via,
                               registration_init_time))
  FROM raw_members
  UNION ALL
  SELECT 'transactions', COUNT(*),
         COUNT(DISTINCT STRUCT(
           msno, payment_method_id, payment_plan_days, plan_list_price,
           actual_amount_paid, is_auto_renew, transaction_date_raw,
           membership_expire_date_raw, is_cancel
         ))
  FROM typed_transactions
  UNION ALL
  SELECT 'user_logs', COUNT(*),
         COUNT(DISTINCT STRUCT(
           msno, event_date_raw, num_25, num_50, num_75, num_985,
           num_100, num_unq, total_secs
         ))
  FROM typed_user_logs
)
SELECT table_name,
       rows_before,
       exact_distinct_rows,
       rows_before - exact_distinct_rows AS exact_duplicate_rows
FROM profiles

