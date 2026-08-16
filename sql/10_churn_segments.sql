SELECT
  CASE
    WHEN no_recent_listening = 1 THEN 'No recent listening'
    WHEN active_days_30 >= 20 THEN 'Highly active'
    WHEN active_days_30 >= 5 THEN 'Moderately active'
    ELSE 'Low activity'
  END AS engagement_group,
  CASE
    WHEN no_recent_transaction = 1 THEN 'No recent transaction'
    WHEN auto_renew_rate_90 >= 0.5 THEN 'Mostly auto-renew'
    ELSE 'Mostly manual renewal'
  END AS renewal_group,
  COUNT(*) AS subscribers,
  SUM(is_churn) AS churners,
  AVG(CAST(is_churn AS DOUBLE)) AS churn_rate
FROM march_features
GROUP BY
  CASE
    WHEN no_recent_listening = 1 THEN 'No recent listening'
    WHEN active_days_30 >= 20 THEN 'Highly active'
    WHEN active_days_30 >= 5 THEN 'Moderately active'
    ELSE 'Low activity'
  END,
  CASE
    WHEN no_recent_transaction = 1 THEN 'No recent transaction'
    WHEN auto_renew_rate_90 >= 0.5 THEN 'Mostly auto-renew'
    ELSE 'Mostly manual renewal'
  END
ORDER BY churn_rate DESC, subscribers DESC

