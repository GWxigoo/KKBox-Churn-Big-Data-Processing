WITH valid AS (
  SELECT msno, registration_date,
         ROW_NUMBER() OVER (
           PARTITION BY msno
           ORDER BY registration_date DESC, registration_init_time DESC
         ) AS row_number
  FROM typed_members
  WHERE msno IS NOT NULL
    AND TRIM(msno) <> ''
)
SELECT msno, registration_date
FROM valid
WHERE row_number = 1

