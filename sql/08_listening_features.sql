SELECT
  g.msno,
  COUNT(DISTINCT g.event_date) AS active_days_30,
  SUM(g.total_secs) AS total_secs_30,
  DATEDIFF(DATE '{{CUTOFF}}', MAX(g.event_date)) AS days_since_last_listen
FROM user_logs g
LEFT SEMI JOIN {{LABEL_VIEW}} l ON g.msno = l.msno
WHERE g.event_date BETWEEN DATE_SUB(DATE '{{CUTOFF}}', 29)
                       AND DATE '{{CUTOFF}}'
GROUP BY g.msno

