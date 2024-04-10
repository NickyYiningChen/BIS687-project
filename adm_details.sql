  select p.subject_id, p.gender, p.dob, p.dod, hadm_id, admittime, dischtime, admission_type, insurance, marital_status, ethnicity, hospital_expire_flag, has_chartevents_data
    from `physionet-data.mimiciii_clinical.admissions` adm
    join `physionet-data.mimiciii_clinical.patients` p
    on adm.subject_id = p.subject_id
