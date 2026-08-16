SELECT
  l.msno,
  CAST(l.is_churn AS INT) AS is_churn,
  m.membership_tenure_days,
  COALESCE(t.transaction_count_90, 0) AS transaction_count_90,
  COALESCE(t.auto_renew_rate_90, 0.0) AS auto_renew_rate_90,
  COALESCE(t.cancellation_count_90, 0) AS cancellation_count_90,
  t.days_since_last_transaction,
  t.days_until_expiration,
  CASE WHEN t.msno IS NULL THEN 1 ELSE 0 END AS no_recent_transaction,
  COALESCE(g.active_days_30, 0) AS active_days_30,
  COALESCE(g.total_secs_30, 0.0) AS total_secs_30,
  g.days_since_last_listen,
  CASE WHEN g.msno IS NULL THEN 1 ELSE 0 END AS no_recent_listening
FROM {{LABEL_VIEW}} l
LEFT JOIN {{MEMBERSHIP_VIEW}} m ON l.msno = m.msno
LEFT JOIN {{TRANSACTION_VIEW}} t ON l.msno = t.msno
LEFT JOIN {{LISTENING_VIEW}} g ON l.msno = g.msno

