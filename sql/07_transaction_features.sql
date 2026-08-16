SELECT
  t.msno,
  COUNT(*) AS transaction_count_90,
  AVG(CAST(t.is_auto_renew AS DOUBLE)) AS auto_renew_rate_90,
  SUM(CASE WHEN t.is_cancel = 1 THEN 1 ELSE 0 END) AS cancellation_count_90,
  DATEDIFF(DATE '{{CUTOFF}}', MAX(t.transaction_date))
    AS days_since_last_transaction,
  DATEDIFF(MAX(t.membership_expire_date), DATE '{{CUTOFF}}')
    AS days_until_expiration
FROM transactions t
LEFT SEMI JOIN {{LABEL_VIEW}} l ON t.msno = l.msno
WHERE t.transaction_date BETWEEN DATE_SUB(DATE '{{CUTOFF}}', 89)
                             AND DATE '{{CUTOFF}}'
GROUP BY t.msno

