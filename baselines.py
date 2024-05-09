import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from gensim.models.doc2vec import Doc2Vec

import argparse
import json
import os
import time
import warnings
import random

from main import cal_metric
from doc2vec import text2words

warnings.filterwarnings('ignore')


def get_ids(split_json):
    splits = list(range(10))
    adm_ids = json.load(open(split_json))
    train_ids = np.hstack([adm_ids[t] for t in splits[:7]])
    val_ids = np.hstack([adm_ids[t] for t in splits[7:8]])
    test_ids = np.hstack([adm_ids[t] for t in splits[8:]])
    train_ids = [adm_id[-10:-4] for adm_id in train_ids]
    val_ids = [adm_id[-10:-4] for adm_id in val_ids]
    test_ids = [adm_id[-10:-4] for adm_id in test_ids]
    return train_ids, val_ids, test_ids


def get_ids2(split_json, seed):
    splits = list(range(10))
    random.Random(seed).shuffle(splits)
    adm_ids = json.load(open(split_json))
    train_ids = np.hstack([adm_ids[t] for t in splits[:7]])
    val_ids = np.hstack([adm_ids[t] for t in splits[7:8]])
    test_ids = np.hstack([adm_ids[t] for t in splits[8:]])
    train_ids = [adm_id[-10:-4] for adm_id in train_ids]
    val_ids = [adm_id[-10:-4] for adm_id in val_ids]
    test_ids = [adm_id[-10:-4] for adm_id in test_ids]
    return train_ids, val_ids, test_ids


def balance_samples(df, times, task):
    df_pos = df[df[task] == 1]
    df_neg = df[df[task] == 0]
    df_neg = df_neg.sample(n=times * len(df_pos), random_state=42)
    df = pd.concat([df_pos, df_neg]).sort_values('hadm_id')
    return df

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='mortality')
    parser.add_argument('--model', type=str, default='all')
    parser.add_argument('--inputs', type=int, default=4) # 1-7
    parser.add_argument('--seed', type=int, default=42) # random seed
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    task = args.task
    model = args.model
    inputs = args.inputs
    seed = args.seed

    train_ids, val_ids, test_ids = get_ids2('data/processed/files/splits.json', seed)
    df = pd.read_csv('data/processed/%s.csv' % task).sort_values('hadm_id')
    df = df.drop_duplicates(subset='hadm_id', keep='first')

    if task != 'labels_icd' and task != 'los_bin':
        pass
    train_ids = np.intersect1d(train_ids, df['hadm_id'].tolist())

    train_ids = [int(i) for i in train_ids]

    test_ids = np.intersect1d(test_ids, df['hadm_id'].tolist())
    test_ids = [int(i) for i in test_ids]

    choices = '{0:b}'.format(inputs).rjust(3, '0')
    X_train, X_test = [], []

    df_notes = pd.read_csv('data/processed/earlynotes.csv').sort_values('hadm_id')
    doc2vec = Doc2Vec.load('models/doc2vec.model')
    df_notes['text'] = df_notes['text'].astype(str)
    df_notes['vector'] = df_notes['text'].apply(lambda note: doc2vec.infer_vector(text2words(note)))
    df_notes = df_notes.groupby('hadm_id')['vector'].apply(list).reset_index()
    df_notes['vector'] = df_notes['vector'].apply(lambda notes: np.mean(notes, axis=0))
    df_notes_col = 'vector'

    X_train_notes = df_notes[df_notes['hadm_id'].isin(train_ids)][df_notes_col].to_list()
    X_test_notes = df_notes[df_notes['hadm_id'].isin(test_ids)][df_notes_col].to_list()


    X_train.append(X_train_notes)
    X_test.append(X_test_notes)
    

    df_cols = df.columns[1:]
    X_train = np.concatenate(X_train, axis=1)
    X_test = np.concatenate(X_test, axis=1)
    y_train = df[df['hadm_id'].isin(train_ids)][df_cols].to_numpy()
    y_train = np.unique(y_train, axis=1) 
    y_test = df[df['hadm_id'].isin(test_ids)][df_cols].to_numpy()

    models = {'lr': LogisticRegression(), 'rf': RandomForestClassifier()}
    #selected_models = [models[model_type]] if model_type in models else list(models.values())
    y_train_flat = y_train.ravel()
    y_test_flat = y_test.ravel()
    for model_key, model in models.items():  # Here we iterate over items to get both key and model instance
        start_time = time.time()
        model.fit(X_train, y_train_flat)
        elapsed_time = time.time() - start_time
        probabilities = model.predict_proba(X_test)[:, 1]
        f1, auc, aupr = cal_metric(y_test_flat, probabilities)  # Assuming cal_metric returns the metrics

        output = (f"Task: {task}, Model: {model_key.upper()}, Inputs: {inputs}, Seed: {seed}, "
                f"F1 Score: {f1:.3f}, AUC: {auc:.3f}, AUPR: {aupr:.3f}, "
                f"Training Time: {elapsed_time:.2f} seconds")
        print(output)

