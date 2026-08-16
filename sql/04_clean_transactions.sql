SELECT DISTINCT
  msno,
  payment_method_id,
  payment_plan_days,
  plan_list_price,
  actual_amount_paid,
  is_auto_renew,
  transaction_date,
  membership_expire_date,
  is_cancel
FROM typed_transactions
WHERE msno IS NOT NULL
  AND TRIM(msno) <> ''
  AND transaction_date IS NOT NULL
  AND membership_expire_date IS NOT NULL
  AND is_auto_renew IN (0, 1)
  AND is_cancel IN (0, 1)

