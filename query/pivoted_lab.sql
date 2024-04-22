WITH i AS (
  SELECT
    subject_id,
    icustay_id,
    intime,
    outtime,
    LAG(outtime) OVER (PARTITION BY subject_id ORDER BY intime) AS outtime_lag,
    LEAD(intime) OVER (PARTITION BY subject_id ORDER BY intime) AS intime_lead
  FROM `physionet-data.mimiciii_clinical.icustays`
),
iid_assign AS (
  SELECT
    subject_id,
    icustay_id,
    CASE
      WHEN outtime_lag IS NOT NULL AND outtime_lag > TIMESTAMP_SUB(intime, INTERVAL 24 HOUR)
      THEN TIMESTAMP_ADD(intime, INTERVAL CAST(TIMESTAMP_DIFF(intime, outtime_lag, MINUTE) / 2 AS INT64) MINUTE)
      ELSE TIMESTAMP_SUB(intime, INTERVAL 12 HOUR)
    END AS data_start,
    CASE
      WHEN intime_lead IS NOT NULL AND intime_lead < TIMESTAMP_ADD(outtime, INTERVAL 24 HOUR)
      THEN TIMESTAMP_ADD(outtime, INTERVAL CAST(TIMESTAMP_DIFF(intime_lead, outtime, MINUTE) / 2 AS INT64) MINUTE)
      ELSE TIMESTAMP_ADD(outtime, INTERVAL 12 HOUR)
    END AS data_end
  FROM i
),
h AS (
  SELECT
    subject_id,
    hadm_id,
    admittime,
    dischtime,
    LAG(dischtime) OVER (PARTITION BY subject_id ORDER BY admittime) AS dischtime_lag,
    LEAD(admittime) OVER (PARTITION BY subject_id ORDER BY admittime) AS admittime_lead
  FROM `physionet-data.mimiciii_clinical.admissions`
),
adm AS (
  SELECT
    subject_id,
    hadm_id,
    CASE
      WHEN dischtime_lag IS NOT NULL AND dischtime_lag > TIMESTAMP_SUB(admittime, INTERVAL 24 HOUR)
      THEN TIMESTAMP_ADD(admittime, INTERVAL CAST(TIMESTAMP_DIFF(admittime, dischtime_lag, MINUTE) / 2 AS INT64) MINUTE)
      ELSE TIMESTAMP_SUB(admittime, INTERVAL 12 HOUR)
    END AS data_start,
    CASE
      WHEN admittime_lead IS NOT NULL AND admittime_lead < TIMESTAMP_ADD(dischtime, INTERVAL 24 HOUR)
      THEN TIMESTAMP_ADD(dischtime, INTERVAL CAST(TIMESTAMP_DIFF(admittime_lead, dischtime, MINUTE) / 2 AS INT64) MINUTE)
      ELSE TIMESTAMP_ADD(dischtime, INTERVAL 12 HOUR)
    END AS data_end
  FROM h
),
le AS (
  SELECT
    subject_id,
    charttime,
    CASE
      WHEN itemid = 50868 THEN 'ANION GAP'
      WHEN itemid = 50862 THEN 'ALBUMIN'
      WHEN itemid = 51144 THEN 'BANDS'
      WHEN itemid = 50882 THEN 'BICARBONATE'
      WHEN itemid = 50885 THEN 'BILIRUBIN'
      WHEN itemid = 50912 THEN 'CREATININE'
      WHEN itemid = 50902 THEN 'CHLORIDE'
      WHEN itemid = 50931 THEN 'GLUCOSE'
      WHEN itemid = 51221 THEN 'HEMATOCRIT'
      WHEN itemid = 51222 THEN 'HEMOGLOBIN'
      WHEN itemid = 50813 THEN 'LACTATE'
      WHEN itemid = 51265 THEN 'PLATELET'
      WHEN itemid = 50971 THEN 'POTASSIUM'
      WHEN itemid = 51275 THEN 'PTT'
      WHEN itemid = 51237 THEN 'INR'
      WHEN itemid = 51274 THEN 'PT'
      WHEN itemid = 50983 THEN 'SODIUM'
      WHEN itemid = 51006 THEN 'BUN'
      WHEN itemid = 51300 THEN 'WBC'
      WHEN itemid = 51301 THEN 'WBC'
      ELSE NULL
    END AS label,
    CASE
      WHEN itemid = 50862 AND valuenum > 10 THEN NULL
      WHEN itemid = 50868 AND valuenum > 10000 THEN NULL
      WHEN itemid = 51144 AND valuenum < 0 OR itemid = 51144 AND valuenum > 100 THEN NULL
      WHEN itemid = 50882 AND valuenum > 10000 THEN NULL
      WHEN itemid = 50885 AND valuenum > 150 THEN NULL
      WHEN itemid = 50902 AND valuenum > 10000 THEN NULL
      WHEN itemid = 50912 AND valuenum > 150 THEN NULL
      WHEN itemid = 50931 AND valuenum > 10000 THEN NULL
      WHEN itemid = 51221 AND valuenum > 100 THEN NULL
      WHEN itemid = 51222 AND valuenum > 50 THEN NULL
      WHEN itemid = 50813 AND valuenum > 50 THEN NULL
      WHEN itemid = 51265 AND valuenum > 10000 THEN NULL
      WHEN itemid = 50971 AND valuenum > 30 THEN NULL
      WHEN itemid = 51275 AND valuenum > 150 THEN NULL
      WHEN itemid = 51237 AND valuenum > 50 THEN NULL
      WHEN itemid = 51274 AND valuenum > 150 THEN NULL
      WHEN itemid = 50983 AND valuenum > 200 THEN NULL
      WHEN itemid = 51006 AND valuenum > 300 THEN NULL
      WHEN itemid = 51300 AND valuenum > 1000 THEN NULL
      WHEN itemid = 51301 AND valuenum > 1000 THEN NULL
      ELSE valuenum
    END AS valuenum
  FROM `physionet-data.mimiciii_clinical.labevents`
  WHERE itemid IN (50868, 50862, 51144, 50882, 50885, 50912, 50902, 50931, 51221, 51222, 50813, 51265, 50971, 51275, 51237, 51274, 50983, 51006, 51300, 51301)
    AND valuenum IS NOT NULL
    AND valuenum > 0
),
le_avg AS (
  SELECT
    subject_id,
    charttime,
    AVG(IF(label = 'ANION GAP', valuenum, NULL)) AS ANIONGAP,
    AVG(IF(label = 'ALBUMIN', valuenum, NULL)) AS ALBUMIN,
    AVG(IF(label = 'BANDS', valuenum, NULL)) AS BANDS,
    AVG(IF(label = 'BICARBONATE', valuenum, NULL)) AS BICARBONATE,
    AVG(IF(label = 'BILIRUBIN', valuenum, NULL)) AS BILIRUBIN,
    AVG(IF(label = 'CREATININE', valuenum, NULL)) AS CREATININE,
    AVG(IF(label = 'CHLORIDE', valuenum, NULL)) AS CHLORIDE,
    AVG(IF(label = 'GLUCOSE', valuenum, NULL)) AS GLUCOSE,
    AVG(IF(label = 'HEMATOCRIT', valuenum, NULL)) AS HEMATOCRIT,
    AVG(IF(label = 'HEMOGLOBIN', valuenum, NULL)) AS HEMOGLOBIN,
    AVG(IF(label = 'LACTATE', valuenum, NULL)) AS LACTATE,
    AVG(IF(label = 'PLATELET', valuenum, NULL)) AS PLATELET,
    AVG(IF(label = 'POTASSIUM', valuenum, NULL)) AS POTASSIUM,
    AVG(IF(label = 'PTT', valuenum, NULL)) AS PTT,
    AVG(IF(label = 'INR', valuenum, NULL)) AS INR,
    AVG(IF(label = 'PT', valuenum, NULL)) AS PT,
    AVG(IF(label = 'SODIUM', valuenum, NULL)) AS SODIUM,
    AVG(IF(label = 'BUN', valuenum, NULL)) AS BUN,
    AVG(IF(label = 'WBC', valuenum, NULL)) AS WBC
  FROM le
  GROUP BY subject_id, charttime
)
SELECT
  iid.icustay_id,
  adm.hadm_id,
  le_avg.*
FROM le_avg
LEFT JOIN adm ON le_avg.subject_id = adm.subject_id
  AND le_avg.charttime >= adm.data_start
  AND le_avg.charttime < adm.data_end
LEFT JOIN iid_assign iid ON le_avg.subject_id = iid.subject_id
  AND le_avg.charttime >= iid.data_start
  AND le_avg.charttime < iid.data_end
ORDER BY le_avg.subject_id, le_avg.charttime;
