import pandas as pd
import numpy as np
import argparse
import warnings

from utils import bin_age, convert_icd_group, clean_text

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="Run parts of the data processing pipeline.")
    parser.add_argument('--step', choices=['cohort', 'signals', 'notes', 'merge'], required=True,
                        help="Specify which part to run: cohort, signals, notes, merge")
    parser.add_argument('--firstday', action='store_true', help="Process only the first day of notes")
    return parser.parse_args()

args = parse_args() 

def main():
    step_functions = {
        'cohort': process_cohort,
        'signals': lambda: get_signals(1, 24),
        'notes': process_notes,
        'merge': merge_ids
    }
    step_functions[args.step]()

def process_cohort():
    df_cohort = pd.read_csv('data/mimic/adm_details.csv', parse_dates=['dob', 'dod', 'admittime', 'dischtime'])
    df_cohort = df_cohort[df_cohort['has_chartevents_data'] == 1]
    df_cohort['age'] = df_cohort['admittime'].subtract(df_cohort['dob']).dt.days / 365.242
    df_cohort['los'] = (df_cohort['dischtime'] - df_cohort['admittime']).dt.total_seconds() / 86400
    df_cohort = df_cohort[df_cohort['age'] >= 18]
    df_cohort['age'] = df_cohort['age'].apply(bin_age)
    df_cohort = df_cohort[df_cohort['los'] >= 1]
    df_cohort.sort_values(['subject_id', 'admittime'], inplace=True)
    df_cohort.reset_index(drop=True, inplace=True)
    df_cohort['marital_status'] = df_cohort['marital_status'].fillna('Unknown')
    df_static = df_cohort[['hadm_id', 'age', 'gender', 'admission_type', 'insurance', 'marital_status', 'ethnicity']]
    df_static.to_csv('data/processed/demo.csv', index=False)

    process_labels(df_cohort)

def process_labels(df_cohort):
    df_icd = pd.read_csv('data/mimic/DIAGNOSES_ICD.csv')[['HADM_ID', 'ICD9_CODE']].dropna()
    df_icd.columns = map(str.lower, df_icd.columns)
    df_icd['icd9_code'] = df_icd['icd9_code'].apply(convert_icd_group)
    df_icd.drop_duplicates(inplace=True)
    df_icd = df_icd[df_icd['hadm_id'].isin(df_cohort['hadm_id'])]
    for x in range(1, 21):
        df_icd[str(x)] = (df_icd['icd9_code'] == x).astype(int)
    df_icd = df_icd.groupby('hadm_id').sum().reset_index()

    df_readmit = calculate_readmissions(df_cohort)
    df_labels = df_cohort[['hadm_id', 'los']].copy()
    df_labels['mortality'] = df_cohort['hospital_expire_flag']
    df_labels['readmit'] = df_readmit['readmit']
    df_labels.to_csv('data/processed/labels_summary.csv', index=False)
    df_icd.to_csv('data/processed/labels_icd.csv', index=False)

def calculate_readmissions(df_cohort):
    df_readmit = df_cohort.copy()
    df_readmit['next_admittime'] = df_readmit.groupby('subject_id')['admittime'].shift(-1)
    df_readmit['next_admission_type'] = df_readmit.groupby('subject_id')['admission_type'].shift(-1)
    elective_rows = df_readmit['next_admission_type'] == 'ELECTIVE'
    df_readmit.loc[elective_rows, ['next_admittime', 'next_admission_type']] = pd.NaT, np.nan
    df_readmit[['next_admittime', 'next_admission_type']].fillna(method='bfill', inplace=True)
    df_readmit['days_next_admit'] = (df_readmit['next_admittime'] - df_readmit['dischtime']).dt.total_seconds() / 86400
    df_readmit['readmit'] = (df_readmit['days_next_admit'] < 30).astype(int)
    return df_readmit

def get_signals(start_hr, end_hr):
    df_cohort = pd.read_csv('data/mimic/adm_details.csv', parse_dates=['admittime'])
    adm_ids = df_cohort.hadm_id.tolist()
    for signal in ['vital', 'lab']:
        df = pd.read_csv(f'data/mimic/pivoted_{signal}.csv', parse_dates=['charttime'])
        df = df.merge(df_cohort[['hadm_id', 'admittime']], on='hadm_id')
        df = df[df.hadm_id.isin(adm_ids)]
        df['hr'] = (df.charttime - df.admittime) / np.timedelta64(1, 'h')
        df = df[(df.hr <= end_hr) & (df.hr >= start_hr)]
        df = df.set_index('hadm_id').groupby('hadm_id').resample('H', on='charttime').mean().reset_index()
        df.to_csv(f'data/mimic/{signal}.csv', index=None)


def extract_early(df_notes, early_categories):
    df_early = df_notes[df_notes['category'].isin(early_categories)]
    df_early['hr'] = (df_early['charttime'] - df_early['admittime']).dt.total_seconds() / 3600
    df_early = df_early[df_early['hr'] <= 24]
    df_early = df_early.sort_values(['hadm_id', 'charttime'])
    df_early['text'] = df_early['text'].apply(clean_text)
    df_early[['hadm_id', 'hr', 'category', 'text']].to_csv('data/processed/earlynotes.csv', index=False)

def extract_first(df_notes, early_categories):
    df_early = df_notes[df_notes['category'].isin(early_categories)]
    df_early['hr'] = (df_early['charttime'] - df_early['admittime']).dt.total_seconds() / 3600
    df_early = df_early.sort_values(['hadm_id', 'charttime']).groupby('hadm_id').head(24)
    df_early['text'] = df_early['text'].apply(clean_text)
    df_early[['hadm_id', 'hr', 'category', 'text']].to_csv('data/processed/earlynotes.csv', index=False)

def process_notes():
    early_categories = ['Nursing', 'Nursing/other', 'Physician ', 'Radiology']
    df_notes = pd.read_csv('data/mimic/NOTEEVENTS.csv', parse_dates=['CHARTTIME'])
    df_notes.columns = df_notes.columns.str.lower()
    df_notes = df_notes[df_notes['iserror'].isnull() & df_notes['hadm_id'].notnull() & df_notes['charttime'].notnull()]
    df_cohort = pd.read_csv('data/mimic/adm_details.csv', parse_dates=['admittime'])
    df_notes = df_notes.merge(df_cohort[['hadm_id', 'admittime']], on='hadm_id')
    if args.firstday:
        print('Extracting first day notes...')
        extract_early(df_notes, early_categories)
    else:
        print('Extracting first 24 notes...')
        extract_first(df_notes, early_categories)

def merge_ids():
    df_static = pd.read_csv('data/processed/demo.csv')
    df_features = pd.read_csv('data/processed/features.csv')
    df_notes = pd.read_csv('data/processed/earlynotes.csv')
    df_icd = pd.read_csv('data/processed/labels_icd.csv')
    df_notes = df_notes[df_notes['text'].notnull()]
    adm_ids = np.intersect1d(df_static['hadm_id'], df_features['hadm_id'])
    adm_ids = np.intersect1d(adm_ids, df_notes['hadm_id'])
    adm_ids = np.intersect1d(adm_ids, df_icd['hadm_id'])
    update_datasets(adm_ids)

def update_datasets(adm_ids):
    for filename in ['demo.csv', 'features.csv', 'earlynotes.csv', 'labels_icd.csv', 'mortality.csv', 'readmit.csv', 'los.csv']:
        df = pd.read_csv(f'data/processed/{filename}')
        df[df['hadm_id'].isin(adm_ids)].to_csv(f'data/processed/{filename}', index=False)
    df_los = pd.read_csv('data/processed/los.csv')
    df_los['llos'] = (df_los['los'] > 7).astype(int)
    df_los[['hadm_id', 'llos']].to_csv('data/processed/llos.csv', index=False)

if __name__ == "__main__":
    main()
