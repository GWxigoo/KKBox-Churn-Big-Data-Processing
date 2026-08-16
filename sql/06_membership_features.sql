SELECT
  l.msno,
  DATEDIFF(DATE '{{CUTOFF}}', m.registration_date) AS membership_tenure_days
FROM {{LABEL_VIEW}} l
LEFT JOIN members m ON l.msno = m.msno

