WITH ce AS (
  SELECT 
    ce.icustay_id,
    ce.charttime,
    CASE 
      WHEN itemid IN (211,220045) AND valuenum > 0 AND valuenum < 300 THEN valuenum 
      ELSE NULL 
    END AS HeartRate,
    CASE 
      WHEN itemid IN (51,442,455,6701,220179,220050) AND valuenum > 0 AND valuenum < 400 THEN valuenum 
      ELSE NULL 
    END AS SysBP,
    CASE 
      WHEN itemid IN (8368,8440,8441,8555,220180,220051) AND valuenum > 0 AND valuenum < 300 THEN valuenum 
      ELSE NULL 
    END AS DiasBP,
    CASE 
      WHEN itemid IN (456,52,6702,443,220052,220181,225312) AND valuenum > 0 AND valuenum < 300 THEN valuenum 
      ELSE NULL 
    END AS MeanBP,
    CASE 
      WHEN itemid IN (615,618,220210,224690) AND valuenum > 0 AND valuenum < 70 THEN valuenum 
      ELSE NULL 
    END AS RespRate,
    CASE 
      WHEN itemid IN (223761,678) AND valuenum > 70 AND valuenum < 120 THEN (valuenum - 32) / 1.8 
      WHEN itemid IN (223762,676) AND valuenum > 10 AND valuenum < 50 THEN valuenum 
      ELSE NULL 
    END AS TempC,
    CASE 
      WHEN itemid IN (646,220277) AND valuenum > 0 AND valuenum <= 100 THEN valuenum 
      ELSE NULL 
    END AS SpO2,
    CASE 
      WHEN itemid IN (807,811,1529,3745,3744,225664,220621,226537) AND valuenum > 0 THEN valuenum 
      ELSE NULL 
    END AS Glucose
  FROM `physionet-data.mimiciii_clinical.chartevents` ce
  WHERE ce.error != 1
    AND itemid IN (
      211, 220045, 
      51, 442, 455, 6701, 220179, 220050, 
      8368, 8440, 8441, 8555, 220180, 220051, 
      456, 52, 6702, 443, 220052, 220181, 225312, 
      618, 615, 220210, 224690, 
      646, 220277, 
      223762, 676, 223761, 678
    )
)
SELECT
  icustays.hadm_id,
  ce.charttime,
  AVG(HeartRate) AS HeartRate,
  AVG(SysBP) AS SysBP,
  AVG(DiasBP) AS DiasBP,
  AVG(MeanBP) AS MeanBP,
  AVG(RespRate) AS RespRate,
  AVG(TempC) AS TempC,
  AVG(SpO2) AS SpO2,
  AVG(Glucose) AS Glucose
FROM `physionet-data.mimiciii_clinical.icustays` icustays
LEFT JOIN ce ON ce.icustay_id = icustays.icustay_id
GROUP BY icustays.hadm_id, ce.charttime
ORDER BY icustays.hadm_id, ce.charttime;
