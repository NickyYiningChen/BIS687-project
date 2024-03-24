  -- Extract explanatory variables (run separate queries)
  -- One query for height/weight info;
  ---- gender info in 'patients', race info in 'admissions', age already extracted
  -- One query for first day vital signs;
  -- One query for fluid related measurements;
  -- Join queries together, using pat_cohort as primary table, gives us final dataset
WITH pat_cohort AS (
  SELECT 
    * 
  FROM (
    SELECT 
      icu.subject_id,
      icu.hadm_id,
      icu.icustay_id,
      icu.intime,
      icu.los,
      d.icd9_code,
      pat.dob,
      CASE 
        WHEN DATE_DIFF(icu.intime, pat.dob, YEAR) >= 300 THEN 90 
        ELSE DATE_DIFF(icu.intime, pat.dob, YEAR) 
      END AS age,
      ROW_NUMBER() OVER (PARTITION BY icu.subject_id ORDER BY icu.intime, d.seq_num) AS icu_stay_rank,
      adm.hospital_expire_flag AS mortality
    FROM 
      `physionet-data.mimiciii_clinical.icustays` icu
    JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` d ON icu.hadm_id = d.hadm_id
    JOIN `physionet-data.mimiciii_clinical.patients` pat ON icu.subject_id = pat.subject_id
    JOIN `physionet-data.mimiciii_clinical.admissions` adm ON icu.hadm_id = adm.hadm_id
    WHERE 
      (d.icd9_code = '99592' OR d.icd9_code = '78552') AND icu.los >= 1 AND DATE_DIFF(icu.intime, pat.dob, YEAR) >= 18
  ) AS subquery
  WHERE icu_stay_rank = 1
),
  -- Extract gender and ethnicity information for each icustay_ids
  demographics AS (
  SELECT
    adm.subject_id,
    adm.ethnicity,
    pat.gender,
    ROW_NUMBER() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) AS rn
  FROM
    `physionet-data.mimiciii_clinical.admissions` adm
  JOIN
    `physionet-data.mimiciii_clinical.patients` pat
  ON
    adm.subject_id = pat.subject_id ),
  notes_first_day AS (
  SELECT
    ne.subject_id,
    ne.hadm_id,
    ne.chartdate,
    ne.charttime,
    ne.text
  FROM
    `physionet-data.mimiciii_notes.noteevents` ne
  JOIN
    `physionet-data.mimiciii_clinical.icustays` icu
  ON
    ne.subject_id = icu.subject_id
    AND ne.hadm_id = icu.hadm_id
  WHERE
    ne.charttime BETWEEN icu.intime
    AND DATETIME_ADD(icu.intime, INTERVAL 1 DAY)
    AND ne.category NOT IN ('Discharge summary',
      'Echo',
      'ECG',
      'Radiology') -- Optional: Exclude certain note types
    ),
  -- Extract height and weight for ICUSTAY_IDs
  ht_stg AS (
  SELECT
    c.subject_id,
    c.icustay_id,
    c.charttime,
    CASE
      WHEN c.itemid IN (920, 1394, 4187, 3486, 226707) THEN CASE
      WHEN c.charttime <= DATETIME_ADD(pt.dob, INTERVAL 1 YEAR)
    AND (c.valuenum * 2.54) < 80 THEN c.valuenum * 2.54
      WHEN c.charttime > DATETIME_ADD(pt.dob, INTERVAL 1 YEAR) AND (c.valuenum * 2.54) > 120 AND (c.valuenum * 2.54) < 230 THEN c.valuenum * 2.54
    ELSE
    NULL
  END
    ELSE
    CASE
      WHEN c.charttime <= DATETIME_ADD(pt.dob, INTERVAL 1 YEAR) AND c.valuenum < 80 THEN c.valuenum
      WHEN c.charttime > DATETIME_ADD(pt.dob, INTERVAL 1 YEAR)
    AND c.valuenum > 120
    AND c.valuenum < 230 THEN c.valuenum
    ELSE
    NULL
  END
  END
    AS height
  FROM
    `physionet-data.mimiciii_clinical.chartevents` c
  INNER JOIN
    `physionet-data.mimiciii_clinical.patients` pt
  ON
    c.subject_id = pt.subject_id
  WHERE
    c.valuenum IS NOT NULL
    AND c.valuenum != 0
    AND COALESCE(c.error, 0) = 0
    AND c.itemid IN (920,
      1394,
      4187,
      3486,
      3485,
      4188,
      226707) ),
  -- Extract the last/first recorded heart rate and the mean heart rate for each ICU stay
  hr_stg AS (
  SELECT
    c.icustay_id,
    MIN(CASE
        WHEN rn_first = 1 THEN c.valuenum
      ELSE
      NULL
    END
      ) AS heart_rate_first,
    AVG(c.valuenum) AS heart_rate_mean,
    MAX(CASE
        WHEN rn_last = 1 THEN c.valuenum
      ELSE
      NULL
    END
      ) AS heart_rate_last
  FROM (
    SELECT
      c.icustay_id,
      c.valuenum,
      ROW_NUMBER() OVER (PARTITION BY c.icustay_id ORDER BY c.charttime) AS rn_first,
      ROW_NUMBER() OVER (PARTITION BY c.icustay_id ORDER BY c.charttime DESC) AS rn_last
    FROM
      `physionet-data.mimiciii_clinical.chartevents` c
    WHERE
      c.itemid IN (211,
        220045)  -- Item IDs for heart rate
      AND c.valuenum IS NOT NULL
      AND COALESCE(c.error, 0) = 0 ) c
  GROUP BY
    c.icustay_id ),
  -- Extract the average SBP and DBP for each ICU stay
  sbp_stg AS (
  SELECT
    c.icustay_id,
    AVG(c.valuenum) AS sbp_mean
  FROM
    `physionet-data.mimiciii_clinical.chartevents` c
  WHERE
    c.itemid IN (51,
      442,
      445,
      6701,
      220050,
      220179,
      225309,
      227243,
      224167) -- Item IDs for SBP
    AND c.valuenum IS NOT NULL
    AND COALESCE(c.error, 0) = 0
  GROUP BY
    c.icustay_id ),
  dbp_stg AS (
  SELECT
    c.icustay_id,
    AVG(c.valuenum) AS dbp_mean
  FROM
    `physionet-data.mimiciii_clinical.chartevents` c
  WHERE
    c.itemid IN (8368,
      8440,
      8441,
      8555,
      220051,
      220180) -- Item IDs for DBP
    AND c.valuenum IS NOT NULL
    AND COALESCE(c.error, 0) = 0
  GROUP BY
    c.icustay_id ),
  temp_stg AS (
  SELECT
    c.icustay_id,
    AVG(c.valuenum) AS temp_mean
  FROM
    `physionet-data.mimiciii_clinical.chartevents` c
  WHERE
    c.itemid IN (678,
      223761) -- Item IDs for body temperature
    AND c.valuenum IS NOT NULL
    AND COALESCE(c.error, 0) = 0
  GROUP BY
    c.icustay_id ),
  rr_stg AS (
  SELECT
    c.icustay_id,
    AVG(c.valuenum) AS rr_mean
  FROM
    `physionet-data.mimiciii_clinical.chartevents` c
  WHERE
    c.itemid IN (618,
      615,
      220210,
      224690) -- Item IDs for respiratory rate
    AND c.valuenum IS NOT NULL
    AND COALESCE(c.error, 0) = 0
  GROUP BY
    c.icustay_id ),
  spo2_stg AS (
  SELECT
    c.icustay_id,
    AVG(c.valuenum) AS spo2_mean
  FROM
    `physionet-data.mimiciii_clinical.chartevents` c
  WHERE
    c.itemid IN (646,
      220277) -- Item ID for SpO2
    AND c.valuenum IS NOT NULL
    AND COALESCE(c.error, 0) = 0
  GROUP BY
    c.icustay_id ),
  glucose_stg AS (
  SELECT
    c.icustay_id,
    AVG(c.valuenum) AS glucose_mean
  FROM
    `physionet-data.mimiciii_clinical.chartevents` c
  WHERE
    c.itemid IN (807,
      811,
      1529,
      3745,
      3744,
      225664,
      220621,
      226537) -- Item IDs for glucose
    AND c.valuenum IS NOT NULL
    AND COALESCE(c.error, 0) = 0
  GROUP BY
    c.icustay_id ),
  -- Extract the min, max, average sodium lab test for each ICU stay
  sodium_stg AS (
  SELECT
    icu.icustay_id,
    AVG(le.valuenum) AS sodium_mean,
    MIN(le.valuenum) AS sodium_min,
    MAX(le.valuenum) AS sodium_max
  FROM
    `physionet-data.mimiciii_clinical.labevents` le
  INNER JOIN
    `physionet-data.mimiciii_clinical.icustays` icu
  ON
    le.subject_id = icu.subject_id
    AND le.charttime BETWEEN icu.intime
    AND icu.outtime
  WHERE
    le.itemid IN (50983,
      50824)
  GROUP BY
    icu.icustay_id ),
  -- Extract the min, max, average chloride lab test for each ICU stay
  chloride_stg AS (
  SELECT
    icu.icustay_id,
    AVG(le.valuenum) AS chloride_mean,
    MIN(le.valuenum) AS chloride_min,
    MAX(le.valuenum) AS chloride_max
  FROM
    `physionet-data.mimiciii_clinical.labevents` le
  INNER JOIN
    `physionet-data.mimiciii_clinical.icustays` icu
  ON
    le.subject_id = icu.subject_id
    AND le.charttime BETWEEN icu.intime
    AND icu.outtime
  WHERE
    le.itemid = 50902
  GROUP BY
    icu.icustay_id ),
  -- Extract the min, max, average chloride lab test for each ICU stay
  potassium_stg AS (
  SELECT
    icu.icustay_id,
    AVG(le.valuenum) AS potassium_mean,
    MIN(le.valuenum) AS potassium_min,
    MAX(le.valuenum) AS potassium_max
  FROM
    `physionet-data.mimiciii_clinical.labevents` le
  INNER JOIN
    `physionet-data.mimiciii_clinical.icustays` icu
  ON
    le.subject_id = icu.subject_id
    AND le.charttime BETWEEN icu.intime
    AND icu.outtime
  WHERE
    le.itemid IN (50822,
      50971)
  GROUP BY
    icu.icustay_id ),
  -- This query extracts durations of dobutamine administration based on each icustay
  dobutamine_duration AS (
  SELECT
    icustay_id,
    SUM(DATETIME_DIFF(endtime, starttime, HOUR)) AS total_dobutamine_duration_hours
  FROM (
    WITH
      vasocv1 AS (
      SELECT
        icustay_id,
        charttime
        -- case statement determining whether the ITEMID is an instance of vasopressor usage
        ,
        MAX(CASE
            WHEN itemid IN (30042, 30306) THEN 1
          ELSE
          0
        END
          ) AS vaso -- dobutamine
        -- the 'stopped' column indicates if a vasopressor has been disconnected
        ,
        MAX(CASE
            WHEN itemid IN (30042, 30306) AND (stopped = 'Stopped' OR stopped LIKE 'D/C%') THEN 1
          ELSE
          0
        END
          ) AS vaso_stopped,
        MAX(CASE
            WHEN itemid IN (30042, 30306) AND rate IS NOT NULL THEN 1
          ELSE
          0
        END
          ) AS vaso_null,
        MAX(CASE
            WHEN itemid IN (30042, 30306) THEN rate
          ELSE
          NULL
        END
          ) AS vaso_rate,
        MAX(CASE
            WHEN itemid IN (30042, 30306) THEN amount
          ELSE
          NULL
        END
          ) AS vaso_amount
      FROM
        `physionet-data.mimiciii_clinical.inputevents_cv`
      WHERE
        itemid IN (30042,
          30306) -- dobutamine
      GROUP BY
        icustay_id,
        charttime ),
      vasocv2 AS (
      SELECT
        v.*,
        SUM(vaso_null) OVER (PARTITION BY icustay_id ORDER BY charttime) AS vaso_partition
      FROM
        vasocv1 v ),
      vasocv3 AS (
      SELECT
        v.*,
        FIRST_VALUE(vaso_rate) OVER (PARTITION BY icustay_id, vaso_partition ORDER BY charttime) AS vaso_prevrate_ifnull
      FROM
        vasocv2 v ),
      vasocv4 AS (
      SELECT
        icustay_id,
        charttime
        -- , (CHARTTIME - (LAG(CHARTTIME, 1) OVER (partition by icustay_id, vaso order by charttime))) AS delta
        ,
        vaso,
        vaso_rate,
        vaso_amount,
        vaso_stopped,
        vaso_prevrate_ifnull
        -- We define start time here
        ,
        CASE
          WHEN vaso = 0 THEN NULL
        -- if this is the first instance of the vasoactive drug
          WHEN vaso_rate > 0
        AND LAG(vaso_prevrate_ifnull,1) OVER (PARTITION BY icustay_id, vaso, vaso_null ORDER BY charttime ) IS NULL THEN 1
        -- you often get a string of 0s
        -- we decide not to set these as 1, just because it makes vasonum sequential
          WHEN vaso_rate = 0 AND LAG(vaso_prevrate_ifnull,1) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime ) = 0 THEN 0
        -- sometimes you get a string of NULL, associated with 0 volumes
        -- same reason as before, we decide not to set these as 1
        -- vaso_prevrate_ifnull is equal to the previous value *iff* the current value is null
          WHEN vaso_prevrate_ifnull = 0
        AND LAG(vaso_prevrate_ifnull,1) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime ) = 0 THEN 0
        -- If the last recorded rate was 0, newvaso = 1
          WHEN LAG(vaso_prevrate_ifnull,1) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime ) = 0 THEN 1
        -- If the last recorded vaso was D/C'd, newvaso = 1
          WHEN LAG(vaso_stopped,1) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime ) = 1 THEN 1
        -- ** not sure if the below is needed
        --when (CHARTTIME - (LAG(CHARTTIME, 1) OVER (partition by icustay_id, vaso order by charttime))) > (interval '4 hours') then 1
        ELSE
        NULL
      END
        AS vaso_start
      FROM
        vasocv3 )
      -- propagate start/stop flags forward in time
      ,
      vasocv5 AS (
      SELECT
        v.*,
        SUM(vaso_start) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime) AS vaso_first
      FROM
        vasocv4 v ),
      vasocv6 AS (
      SELECT
        v.*
        -- We define end time here
        ,
        CASE
          WHEN vaso = 0 THEN NULL
        -- If the recorded vaso was D/C'd, this is an end time
          WHEN vaso_stopped = 1 THEN vaso_first
        -- If the rate is zero, this is the end time
          WHEN vaso_rate = 0 THEN vaso_first
        -- the last row in the table is always a potential end time
        -- this captures patients who die/are discharged while on vasopressors
        -- in principle, this could add an extra end time for the vasopressor
        -- however, since we later group on vaso_start, any extra end times are ignored
          WHEN LEAD(CHARTTIME,1) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime ) IS NULL THEN vaso_first
        ELSE
        NULL
      END
        AS vaso_stop
      FROM
        vasocv5 v )
      -- -- if you want to look at the results of the table before grouping:
      -- select
      --   icustay_id, charttime, vaso, vaso_rate, vaso_amount
      --     , case when vaso_stopped = 1 then 'Y' else '' end as stopped
      --     , vaso_start
      --     , vaso_first
      --     , vaso_stop
      -- from vasocv6 order by charttime;
      ,
      vasocv AS (
        -- below groups together vasopressor administrations into groups
      SELECT
        icustay_id
        -- the first non-null rate is considered the starttime
        ,
        MIN(CASE
            WHEN vaso_rate IS NOT NULL THEN charttime
          ELSE
          NULL
        END
          ) AS starttime
        -- the *first* time the first/last flags agree is the stop time for this duration
        ,
        MIN(CASE
            WHEN vaso_first = vaso_stop THEN charttime
          ELSE
          NULL
        END
          ) AS endtime
      FROM
        vasocv6
      WHERE
        vaso_first IS NOT NULL -- bogus data
        AND vaso_first != 0 -- sometimes *only* a rate of 0 appears, i.e. the drug is never actually delivered
        AND icustay_id IS NOT NULL -- there are data for "floating" admissions, we don't worry about these
      GROUP BY
        icustay_id,
        vaso_first
      HAVING
        -- ensure start time is not the same as end time
        MIN(charttime) != MIN(CASE
            WHEN vaso_first = vaso_stop THEN charttime
          ELSE
          NULL
        END
          )
        AND MAX(vaso_rate) > 0 -- if the rate was always 0 or null, we consider it not a real drug delivery
        )
      -- now we extract the associated data for metavision patients
      ,
      vasomv AS (
      SELECT
        icustay_id,
        linkorderid,
        MIN(starttime) AS starttime,
        MAX(endtime) AS endtime
      FROM
        `physionet-data.mimiciii_clinical.inputevents_mv`
      WHERE
        itemid = 221653 -- dobutamine
        AND statusdescription != 'Rewritten' -- only valid orders
      GROUP BY
        icustay_id,
        linkorderid )
    SELECT
      icustay_id
      -- generate a sequential integer for convenience
      ,
      ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY starttime) AS vasonum,
      starttime,
      endtime,
      DATETIME_DIFF(endtime, starttime, HOUR) AS duration_hours
      -- add durations
    FROM
      vasocv
    UNION ALL
    SELECT
      icustay_id,
      ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY starttime) AS vasonum,
      starttime,
      endtime,
      DATETIME_DIFF(endtime, starttime, HOUR) AS duration_hours
    FROM
      vasomv
    ORDER BY
      icustay_id,
      vasonum )
  GROUP BY
    icustay_id ),
  -- This query extracts durations of epinephrine administration
  epinephrine_duration AS (
  SELECT
    icustay_id,
    SUM(DATETIME_DIFF(endtime, starttime, HOUR)) AS total_epinephrine_duration_hours
  FROM (
    WITH
      vasocv1 AS (
      SELECT
        icustay_id,
        charttime
        -- case statement determining whether the ITEMID is an instance of vasopressor usage
        ,
        MAX(CASE
            WHEN itemid IN (30044, 30119, 30309) THEN 1
          ELSE
          0
        END
          ) AS vaso -- epinephrine
        -- the 'stopped' column indicates if a vasopressor has been disconnected
        ,
        MAX(CASE
            WHEN itemid IN (30044, 30119, 30309) AND (stopped = 'Stopped' OR stopped LIKE 'D/C%') THEN 1
          ELSE
          0
        END
          ) AS vaso_stopped,
        MAX(CASE
            WHEN itemid IN (30044, 30119, 30309) AND rate IS NOT NULL THEN 1
          ELSE
          0
        END
          ) AS vaso_null,
        MAX(CASE
            WHEN itemid IN (30044, 30119, 30309) THEN rate
          ELSE
          NULL
        END
          ) AS vaso_rate,
        MAX(CASE
            WHEN itemid IN (30044, 30119, 30309) THEN amount
          ELSE
          NULL
        END
          ) AS vaso_amount
      FROM
        `physionet-data.mimiciii_clinical.inputevents_cv`
      WHERE
        itemid IN ( 30044,
          30119,
          30309 -- epinephrine
          )
      GROUP BY
        icustay_id,
        charttime ),
      vasocv2 AS (
      SELECT
        v.*,
        SUM(vaso_null) OVER (PARTITION BY icustay_id ORDER BY charttime) AS vaso_partition
      FROM
        vasocv1 v ),
      vasocv3 AS (
      SELECT
        v.*,
        FIRST_VALUE(vaso_rate) OVER (PARTITION BY icustay_id, vaso_partition ORDER BY charttime) AS vaso_prevrate_ifnull
      FROM
        vasocv2 v ),
      vasocv4 AS (
      SELECT
        icustay_id,
        charttime
        -- , (CHARTTIME - (LAG(CHARTTIME, 1) OVER (partition by icustay_id, vaso order by charttime))) AS delta
        ,
        vaso,
        vaso_rate,
        vaso_amount,
        vaso_stopped,
        vaso_prevrate_ifnull
        -- We define start time here
        ,
        CASE
          WHEN vaso = 0 THEN NULL
        -- if this is the first instance of the vasoactive drug
          WHEN vaso_rate > 0
        AND LAG(vaso_prevrate_ifnull,1) OVER (PARTITION BY icustay_id, vaso, vaso_null ORDER BY charttime ) IS NULL THEN 1
        -- you often get a string of 0s
        -- we decide not to set these as 1, just because it makes vasonum sequential
          WHEN vaso_rate = 0 AND LAG(vaso_prevrate_ifnull,1) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime ) = 0 THEN 0
        -- sometimes you get a string of NULL, associated with 0 volumes
        -- same reason as before, we decide not to set these as 1
        -- vaso_prevrate_ifnull is equal to the previous value *iff* the current value is null
          WHEN vaso_prevrate_ifnull = 0
        AND LAG(vaso_prevrate_ifnull,1) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime ) = 0 THEN 0
        -- If the last recorded rate was 0, newvaso = 1
          WHEN LAG(vaso_prevrate_ifnull,1) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime ) = 0 THEN 1
        -- If the last recorded vaso was D/C'd, newvaso = 1
          WHEN LAG(vaso_stopped,1) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime ) = 1 THEN 1
        -- ** not sure if the below is needed
        --when (CHARTTIME - (LAG(CHARTTIME, 1) OVER (partition by icustay_id, vaso order by charttime))) > (interval '4 hours') then 1
        ELSE
        NULL
      END
        AS vaso_start
      FROM
        vasocv3 )
      -- propagate start/stop flags forward in time
      ,
      vasocv5 AS (
      SELECT
        v.*,
        SUM(vaso_start) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime) AS vaso_first
      FROM
        vasocv4 v ),
      vasocv6 AS (
      SELECT
        v.*
        -- We define end time here
        ,
        CASE
          WHEN vaso = 0 THEN NULL
        -- If the recorded vaso was D/C'd, this is an end time
          WHEN vaso_stopped = 1 THEN vaso_first
        -- If the rate is zero, this is the end time
          WHEN vaso_rate = 0 THEN vaso_first
        -- the last row in the table is always a potential end time
        -- this captures patients who die/are discharged while on vasopressors
        -- in principle, this could add an extra end time for the vasopressor
        -- however, since we later group on vaso_start, any extra end times are ignored
          WHEN LEAD(CHARTTIME,1) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime ) IS NULL THEN vaso_first
        ELSE
        NULL
      END
        AS vaso_stop
      FROM
        vasocv5 v )
      -- -- if you want to look at the results of the table before grouping:
      -- select
      --   icustay_id, charttime, vaso, vaso_rate, vaso_amount
      --     , case when vaso_stopped = 1 then 'Y' else '' end as stopped
      --     , vaso_start
      --     , vaso_first
      --     , vaso_stop
      -- from vasocv6 order by charttime;
      ,
      vasocv AS (
        -- below groups together vasopressor administrations into groups
      SELECT
        icustay_id
        -- the first non-null rate is considered the starttime
        ,
        MIN(CASE
            WHEN vaso_rate IS NOT NULL THEN charttime
          ELSE
          NULL
        END
          ) AS starttime
        -- the *first* time the first/last flags agree is the stop time for this duration
        ,
        MIN(CASE
            WHEN vaso_first = vaso_stop THEN charttime
          ELSE
          NULL
        END
          ) AS endtime
      FROM
        vasocv6
      WHERE
        vaso_first IS NOT NULL -- bogus data
        AND vaso_first != 0 -- sometimes *only* a rate of 0 appears, i.e. the drug is never actually delivered
        AND icustay_id IS NOT NULL -- there are data for "floating" admissions, we don't worry about these
      GROUP BY
        icustay_id,
        vaso_first
      HAVING
        -- ensure start time is not the same as end time
        MIN(charttime) != MIN(CASE
            WHEN vaso_first = vaso_stop THEN charttime
          ELSE
          NULL
        END
          )
        AND MAX(vaso_rate) > 0 -- if the rate was always 0 or null, we consider it not a real drug delivery
        )
      -- now we extract the associated data for metavision patients
      ,
      vasomv AS (
      SELECT
        icustay_id,
        linkorderid,
        MIN(starttime) AS starttime,
        MAX(endtime) AS endtime
      FROM
        `physionet-data.mimiciii_clinical.inputevents_mv`
      WHERE
        itemid = 221289 -- epinephrine
        AND statusdescription != 'Rewritten' -- only valid orders
      GROUP BY
        icustay_id,
        linkorderid )
    SELECT
      icustay_id
      -- generate a sequential integer for convenience
      ,
      ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY starttime) AS vasonum,
      starttime,
      endtime,
      DATETIME_DIFF(endtime, starttime, HOUR) AS duration_hours
      -- add durations
    FROM
      vasocv
    UNION ALL
    SELECT
      icustay_id,
      ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY starttime) AS vasonum,
      starttime,
      endtime,
      DATETIME_DIFF(endtime, starttime, HOUR) AS duration_hours
      -- add durations
    FROM
      vasomv
    ORDER BY
      icustay_id,
      vasonum )
  GROUP BY
    icustay_id ),
  -- This query extracts durations of dopamine administration
  dopamine_duration AS (
  SELECT
    icustay_id,
    SUM(DATETIME_DIFF(endtime, starttime, HOUR)) AS total_dopamine_duration_hours
  FROM (
    WITH
      vasocv1 AS (
      SELECT
        icustay_id,
        charttime
        -- case statement determining whether the ITEMID is an instance of vasopressor usage
        ,
        MAX(CASE
            WHEN itemid IN (30043, 30307) THEN 1
          ELSE
          0
        END
          ) AS vaso -- dopamine
        -- the 'stopped' column indicates if a vasopressor has been disconnected
        ,
        MAX(CASE
            WHEN itemid IN (30043, 30307) AND (stopped = 'Stopped' OR stopped LIKE 'D/C%') THEN 1
          ELSE
          0
        END
          ) AS vaso_stopped,
        MAX(CASE
            WHEN itemid IN (30043, 30307) AND rate IS NOT NULL THEN 1
          ELSE
          0
        END
          ) AS vaso_null,
        MAX(CASE
            WHEN itemid IN (30043, 30307) THEN rate
          ELSE
          NULL
        END
          ) AS vaso_rate,
        MAX(CASE
            WHEN itemid IN (30043, 30307) THEN amount
          ELSE
          NULL
        END
          ) AS vaso_amount
      FROM
        `physionet-data.mimiciii_clinical.inputevents_cv`
      WHERE
        itemid IN ( 30043,
          30307 -- dopamine
          )
      GROUP BY
        icustay_id,
        charttime ),
      vasocv2 AS (
      SELECT
        v.*,
        SUM(vaso_null) OVER (PARTITION BY icustay_id ORDER BY charttime) AS vaso_partition
      FROM
        vasocv1 v ),
      vasocv3 AS (
      SELECT
        v.*,
        FIRST_VALUE(vaso_rate) OVER (PARTITION BY icustay_id, vaso_partition ORDER BY charttime) AS vaso_prevrate_ifnull
      FROM
        vasocv2 v ),
      vasocv4 AS (
      SELECT
        icustay_id,
        charttime
        -- , (CHARTTIME - (LAG(CHARTTIME, 1) OVER (partition by icustay_id, vaso order by charttime))) AS delta
        ,
        vaso,
        vaso_rate,
        vaso_amount,
        vaso_stopped,
        vaso_prevrate_ifnull
        -- We define start time here
        ,
        CASE
          WHEN vaso = 0 THEN NULL
        -- if this is the first instance of the vasoactive drug
          WHEN vaso_rate > 0
        AND LAG(vaso_prevrate_ifnull,1) OVER (PARTITION BY icustay_id, vaso, vaso_null ORDER BY charttime ) IS NULL THEN 1
        -- you often get a string of 0s
        -- we decide not to set these as 1, just because it makes vasonum sequential
          WHEN vaso_rate = 0 AND LAG(vaso_prevrate_ifnull,1) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime ) = 0 THEN 0
        -- sometimes you get a string of NULL, associated with 0 volumes
        -- same reason as before, we decide not to set these as 1
        -- vaso_prevrate_ifnull is equal to the previous value *iff* the current value is null
          WHEN vaso_prevrate_ifnull = 0
        AND LAG(vaso_prevrate_ifnull,1) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime ) = 0 THEN 0
        -- If the last recorded rate was 0, newvaso = 1
          WHEN LAG(vaso_prevrate_ifnull,1) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime ) = 0 THEN 1
        -- If the last recorded vaso was D/C'd, newvaso = 1
          WHEN LAG(vaso_stopped,1) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime ) = 1 THEN 1
        -- ** not sure if the below is needed
        --when (CHARTTIME - (LAG(CHARTTIME, 1) OVER (partition by icustay_id, vaso order by charttime))) > (interval '4 hours') then 1
        ELSE
        NULL
      END
        AS vaso_start
      FROM
        vasocv3 )
      -- propagate start/stop flags forward in time
      ,
      vasocv5 AS (
      SELECT
        v.*,
        SUM(vaso_start) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime) AS vaso_first
      FROM
        vasocv4 v ),
      vasocv6 AS (
      SELECT
        v.*
        -- We define end time here
        ,
        CASE
          WHEN vaso = 0 THEN NULL
        -- If the recorded vaso was D/C'd, this is an end time
          WHEN vaso_stopped = 1 THEN vaso_first
        -- If the rate is zero, this is the end time
          WHEN vaso_rate = 0 THEN vaso_first
          WHEN LEAD(CHARTTIME,1) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime ) IS NULL THEN vaso_first
        ELSE
        NULL
      END
        AS vaso_stop
      FROM
        vasocv5 v ),
      vasocv AS (
        -- below groups together vasopressor administrations into groups
      SELECT
        icustay_id
        -- the first non-null rate is considered the starttime
        ,
        MIN(CASE
            WHEN vaso_rate IS NOT NULL THEN charttime
          ELSE
          NULL
        END
          ) AS starttime
        -- the *first* time the first/last flags agree is the stop time for this duration
        ,
        MIN(CASE
            WHEN vaso_first = vaso_stop THEN charttime
          ELSE
          NULL
        END
          ) AS endtime
      FROM
        vasocv6
      WHERE
        vaso_first IS NOT NULL -- bogus data
        AND vaso_first != 0 -- sometimes *only* a rate of 0 appears, i.e. the drug is never actually delivered
        AND icustay_id IS NOT NULL -- there are data for "floating" admissions, we don't worry about these
      GROUP BY
        icustay_id,
        vaso_first
      HAVING
        -- ensure start time is not the same as end time
        MIN(charttime) != MIN(CASE
            WHEN vaso_first = vaso_stop THEN charttime
          ELSE
          NULL
        END
          )
        AND MAX(vaso_rate) > 0 -- if the rate was always 0 or null, we consider it not a real drug delivery
        )
      -- now we extract the associated data for metavision patients
      ,
      vasomv AS (
      SELECT
        icustay_id,
        linkorderid,
        MIN(starttime) AS starttime,
        MAX(endtime) AS endtime
      FROM
        `physionet-data.mimiciii_clinical.inputevents_mv`
      WHERE
        itemid = 221662 -- dopamine
        AND statusdescription != 'Rewritten' -- only valid orders
      GROUP BY
        icustay_id,
        linkorderid )
    SELECT
      icustay_id
      -- generate a sequential integer for convenience
      ,
      ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY starttime) AS vasonum,
      starttime,
      endtime,
      DATETIME_DIFF(endtime, starttime, HOUR) AS duration_hours
      -- add durations
    FROM
      vasocv
    UNION ALL
    SELECT
      icustay_id,
      ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY starttime) AS vasonum,
      starttime,
      endtime,
      DATETIME_DIFF(endtime, starttime, HOUR) AS duration_hours
      -- add durations
    FROM
      vasomv
    ORDER BY
      icustay_id,
      vasonum )
  GROUP BY
    icustay_id ),
  -- This query extracts durations of norepinephrine administration
  norepinephrine_duration AS (
  SELECT
    icustay_id,
    SUM(DATETIME_DIFF(endtime, starttime, HOUR)) AS total_norepinephrine_duration_hours
  FROM (
    WITH
      vasocv1 AS (
      SELECT
        icustay_id,
        charttime
        -- case statement determining whether the ITEMID is an instance of vasopressor usage
        ,
        MAX(CASE
            WHEN itemid IN (30047, 30120) THEN 1
          ELSE
          0
        END
          ) AS vaso -- norepinephrine
        -- the 'stopped' column indicates if a vasopressor has been disconnected
        ,
        MAX(CASE
            WHEN itemid IN (30047, 30120) AND (stopped = 'Stopped' OR stopped LIKE 'D/C%') THEN 1
          ELSE
          0
        END
          ) AS vaso_stopped,
        MAX(CASE
            WHEN itemid IN (30047, 30120) AND rate IS NOT NULL THEN 1
          ELSE
          0
        END
          ) AS vaso_null,
        MAX(CASE
            WHEN itemid IN (30047, 30120) THEN rate
          ELSE
          NULL
        END
          ) AS vaso_rate,
        MAX(CASE
            WHEN itemid IN (30047, 30120) THEN amount
          ELSE
          NULL
        END
          ) AS vaso_amount
      FROM
        `physionet-data.mimiciii_clinical.inputevents_cv`
      WHERE
        itemid IN (30047,
          30120) -- norepinephrine
      GROUP BY
        icustay_id,
        charttime ),
      vasocv2 AS (
      SELECT
        v.*,
        SUM(vaso_null) OVER (PARTITION BY icustay_id ORDER BY charttime) AS vaso_partition
      FROM
        vasocv1 v ),
      vasocv3 AS (
      SELECT
        v.*,
        FIRST_VALUE(vaso_rate) OVER (PARTITION BY icustay_id, vaso_partition ORDER BY charttime) AS vaso_prevrate_ifnull
      FROM
        vasocv2 v ),
      vasocv4 AS (
      SELECT
        icustay_id,
        charttime
        -- , (CHARTTIME - (LAG(CHARTTIME, 1) OVER (partition by icustay_id, vaso order by charttime))) AS delta
        ,
        vaso,
        vaso_rate,
        vaso_amount,
        vaso_stopped,
        vaso_prevrate_ifnull
        -- We define start time here
        ,
        CASE
          WHEN vaso = 0 THEN NULL
        -- if this is the first instance of the vasoactive drug
          WHEN vaso_rate > 0
        AND LAG(vaso_prevrate_ifnull,1) OVER (PARTITION BY icustay_id, vaso, vaso_null ORDER BY charttime ) IS NULL THEN 1
        -- you often get a string of 0s
        -- we decide not to set these as 1, just because it makes vasonum sequential
          WHEN vaso_rate = 0 AND LAG(vaso_prevrate_ifnull,1) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime ) = 0 THEN 0
        -- sometimes you get a string of NULL, associated with 0 volumes
        -- same reason as before, we decide not to set these as 1
        -- vaso_prevrate_ifnull is equal to the previous value *iff* the current value is null
          WHEN vaso_prevrate_ifnull = 0
        AND LAG(vaso_prevrate_ifnull,1) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime ) = 0 THEN 0
        -- If the last recorded rate was 0, newvaso = 1
          WHEN LAG(vaso_prevrate_ifnull,1) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime ) = 0 THEN 1
        -- If the last recorded vaso was D/C'd, newvaso = 1
          WHEN LAG(vaso_stopped,1) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime ) = 1 THEN 1
        -- ** not sure if the below is needed
        --when (CHARTTIME - (LAG(CHARTTIME, 1) OVER (partition by icustay_id, vaso order by charttime))) > (interval '4 hours') then 1
        ELSE
        NULL
      END
        AS vaso_start
      FROM
        vasocv3 )
      -- propagate start/stop flags forward in time
      ,
      vasocv5 AS (
      SELECT
        v.*,
        SUM(vaso_start) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime) AS vaso_first
      FROM
        vasocv4 v ),
      vasocv6 AS (
      SELECT
        v.*
        -- We define end time here
        ,
        CASE
          WHEN vaso = 0 THEN NULL
        -- If the recorded vaso was D/C'd, this is an end time
          WHEN vaso_stopped = 1 THEN vaso_first
        -- If the rate is zero, this is the end time
          WHEN vaso_rate = 0 THEN vaso_first
        -- the last row in the table is always a potential end time
        -- this captures patients who die/are discharged while on vasopressors
        -- in principle, this could add an extra end time for the vasopressor
        -- however, since we later group on vaso_start, any extra end times are ignored
          WHEN LEAD(CHARTTIME,1) OVER (PARTITION BY icustay_id, vaso ORDER BY charttime ) IS NULL THEN vaso_first
        ELSE
        NULL
      END
        AS vaso_stop
      FROM
        vasocv5 v )
      -- -- if you want to look at the results of the carevue data before grouping:
      -- select
      --   icustay_id, charttime, vaso, vaso_rate, vaso_amount
      --     , case when vaso_stopped = 1 then 'Y' else '' end as stopped
      --     , vaso_start
      --     , vaso_first
      --     , vaso_stop
      -- from vasocv6 order by charttime;
      ,
      vasocv AS (
        -- below groups together vasopressor administrations into groups
      SELECT
        icustay_id
        -- the first non-null rate is considered the starttime
        ,
        MIN(CASE
            WHEN vaso_rate IS NOT NULL THEN charttime
          ELSE
          NULL
        END
          ) AS starttime
        -- the *first* time the first/last flags agree is the stop time for this duration
        ,
        MIN(CASE
            WHEN vaso_first = vaso_stop THEN charttime
          ELSE
          NULL
        END
          ) AS endtime
      FROM
        vasocv6
      WHERE
        vaso_first IS NOT NULL -- bogus data
        AND vaso_first != 0 -- sometimes *only* a rate of 0 appears, i.e. the drug is never actually delivered
        AND icustay_id IS NOT NULL -- there are data for "floating" admissions, we don't worry about these
      GROUP BY
        icustay_id,
        vaso_first
      HAVING
        -- ensure start time is not the same as end time
        MIN(charttime) != MIN(CASE
            WHEN vaso_first = vaso_stop THEN charttime
          ELSE
          NULL
        END
          )
        AND MAX(vaso_rate) > 0 -- if the rate was always 0 or null, we consider it not a real drug delivery
        )
      -- now we extract the associated data for metavision patients
      ,
      vasomv AS (
      SELECT
        icustay_id,
        linkorderid,
        MIN(starttime) AS starttime,
        MAX(endtime) AS endtime
      FROM
        `physionet-data.mimiciii_clinical.inputevents_mv`
      WHERE
        itemid = 221906 -- norepinephrine
        AND statusdescription != 'Rewritten' -- only valid orders
      GROUP BY
        icustay_id,
        linkorderid )
    SELECT
      icustay_id
      -- generate a sequential integer for convenience
      ,
      ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY starttime) AS vasonum,
      starttime,
      endtime,
      DATETIME_DIFF(endtime, starttime, HOUR) AS duration_hours
      -- add durations
    FROM
      vasocv
    UNION ALL
    SELECT
      icustay_id,
      ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY starttime) AS vasonum,
      starttime,
      endtime,
      DATETIME_DIFF(endtime, starttime, HOUR) AS duration_hours
      -- add durations
    FROM
      vasomv
    ORDER BY
      icustay_id,
      vasonum )
  GROUP BY
    icustay_id )
SELECT
  DISTINCT pc.subject_id,
  pc.icustay_id,
  demo.ethnicity,
  demo.gender,
  pc.LOS,
  pc.icd9_code,
  pc.age,
  pc.icu_stay_rank,
  pc.hadm_id,
  pc.mortality, 
  ROUND(CAST(wt.weight_first AS NUMERIC), 2) AS weight_first,
  ROUND(CAST(ht.height_first AS NUMERIC), 2) AS height_first,
  ROUND(CAST(hr.heart_rate_first AS NUMERIC), 2) AS heart_rate_first,
  ROUND(CAST(hr.heart_rate_mean AS NUMERIC), 2) AS heart_rate_mean,
  ROUND(CAST(hr.heart_rate_last AS NUMERIC), 2) AS heart_rate_last,
  ROUND(CAST(sbp.sbp_mean AS NUMERIC), 2) AS sbp_mean,
  ROUND(CAST(dbp.dbp_mean AS NUMERIC), 2) AS dbp_mean,
  ROUND(CAST(temp.temp_mean AS NUMERIC), 2) AS temp_mean,
  ROUND(CAST(rr.rr_mean AS NUMERIC), 2) AS rr_mean,
  ROUND(CAST(spo2.spo2_mean AS NUMERIC), 2) AS spo2_mean,
  ROUND(CAST(glucose.glucose_mean AS NUMERIC), 2) AS glucose_mean,
  sodium_stg.sodium_mean,
  sodium_stg.sodium_min,
  sodium_stg.sodium_max,
  chloride_stg.chloride_mean,
  chloride_stg.chloride_min,
  chloride_stg.chloride_max,
  potassium_stg.potassium_mean,
  potassium_stg.potassium_min,
  potassium_stg.potassium_max,
  dob.total_dobutamine_duration_hours,
  epi.total_epinephrine_duration_hours,
  dop.total_dopamine_duration_hours,
  nor.total_norepinephrine_duration_hours,
  nfd.text AS first_day_notes
FROM
  pat_cohort pc
JOIN
  `physionet-data.mimiciii_clinical.icustays` ie
ON
  pc.icustay_id = ie.icustay_id
JOIN
  `physionet-data.mimiciii_clinical.admissions` adm
ON
  pc.subject_id = adm.subject_id
JOIN
  `physionet-data.mimiciii_clinical.patients` pat
ON
  pc.subject_id = pat.subject_id
LEFT JOIN (
  SELECT
    icustay_id,
    MIN(CASE
        WHEN rn = 1 THEN weight
      ELSE
      NULL
    END
      ) AS weight_first
  FROM (
    SELECT
      icustay_id,
      weight,
      ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY starttime) AS rn
    FROM
      `physionet-data.mimiciii_derived.weight_durations` ) wt_stg
  GROUP BY
    icustay_id ) wt
ON
  ie.icustay_id = wt.icustay_id
LEFT JOIN (
  SELECT
    icustay_id,
    MIN(CASE
        WHEN rn = 1 THEN height
      ELSE
      NULL
    END
      ) AS height_first
  FROM (
    SELECT
      icustay_id,
      height,
      ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY charttime) AS rn
    FROM
      ht_stg ) ht_stg2
  GROUP BY
    icustay_id ) ht
ON
  ie.icustay_id = ht.icustay_id
LEFT JOIN
  hr_stg hr
ON
  ie.icustay_id = hr.icustay_id
LEFT JOIN
  sbp_stg sbp
ON
  ie.icustay_id = sbp.icustay_id
LEFT JOIN
  dbp_stg dbp
ON
  ie.icustay_id = dbp.icustay_id
LEFT JOIN
  temp_stg TEMP
ON
  ie.icustay_id = temp.icustay_id
LEFT JOIN
  rr_stg rr
ON
  ie.icustay_id = rr.icustay_id
LEFT JOIN
  spo2_stg spo2
ON
  ie.icustay_id = spo2.icustay_id
LEFT JOIN
  glucose_stg glucose
ON
  ie.icustay_id = glucose.icustay_id
LEFT JOIN
  dobutamine_duration dob
ON
  pc.icustay_id = dob.icustay_id
LEFT JOIN
  epinephrine_duration epi
ON
  pc.icustay_id = epi.icustay_id
LEFT JOIN
  dopamine_duration dop
ON
  pc.icustay_id = dop.icustay_id
LEFT JOIN
  norepinephrine_duration nor
ON
  pc.icustay_id = nor.icustay_id
LEFT JOIN
  sodium_stg
ON
  pc.icustay_id = sodium_stg.icustay_id
LEFT JOIN
  chloride_stg
ON
  pc.icustay_id = chloride_stg.icustay_id
LEFT JOIN
  potassium_stg
ON
  pc.icustay_id = potassium_stg.icustay_id
LEFT JOIN (
  SELECT
    *
  FROM
    demographics
  WHERE
    rn = 1) demo
ON
  pc.subject_id = demo.subject_id
LEFT JOIN
  notes_first_day nfd
ON
  pc.subject_id = nfd.subject_id
  AND pc.hadm_id = nfd.hadm_id
ORDER BY
  pc.icustay_id;
