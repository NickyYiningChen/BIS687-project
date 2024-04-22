SELECT 
    p.subject_id, 
    p.gender, 
    p.dob, 
    p.dod, 
    adm.hadm_id, 
    adm.admittime, 
    adm.dischtime, 
    adm.admission_type, 
    adm.insurance, 
    adm.marital_status, 
    adm.ethnicity, 
    adm.hospital_expire_flag, 
    adm.has_chartevents_data
FROM `physionet-data.mimiciii_clinical.admissions` adm
JOIN `physionet-data.mimiciii_clinical.patients` p
    ON adm.subject_id = p.subject_id
JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` d
    ON adm.hadm_id = d.hadm_id
WHERE d.icd9_code IN ('99592', '78552')
