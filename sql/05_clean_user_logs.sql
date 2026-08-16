SELECT DISTINCT
  msno,
  event_date,
  num_25,
  num_50,
  num_75,
  num_985,
  num_100,
  num_unq,
  total_secs
FROM typed_user_logs
WHERE msno IS NOT NULL
  AND TRIM(msno) <> ''
  AND event_date IS NOT NULL
  AND total_secs IS NOT NULL
  AND total_secs >= 0
  AND num_25 >= 0
  AND num_50 >= 0
  AND num_75 >= 0
  AND num_985 >= 0
  AND num_100 >= 0
  AND num_unq >= 0

