import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle

import json
import argparse
from text_clean import clean_text
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

def text2words(text):
    words = text.split()
    words = [w for w in words if not w in stops]
    return words

def get_cli_args():
    parser = argparse.ArgumentParser(description="Text Data Processing")
    parser.add_argument('--mode', type=str, default='infer', help='Operational mode: train or infer')
    parser.add_argument('--training_cycles', type=int, default=30, help='Number of training epochs')
    return parser.parse_args()

def prepare_text_data(file_path):
    text_df = pd.read_csv(file_path)
    text_df['processed_text'] = text_df['text'].astype(str).apply(text2words)
    return text_df

def load_data_splits(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def train_doc2vec_model(documents, epochs):
    d2v_model = Doc2Vec(dm=0, vector_size=200, negative=5, hs=0, min_count=5, sample=0, alpha=0.025, workers=16)
    d2v_model.build_vocab(documents)
    for epoch in tqdm(range(epochs), desc="Training model"):
        d2v_model.train(shuffle(documents), total_examples=d2v_model.corpus_count, epochs=1)
        d2v_model.alpha -= 0.0002
        d2v_model.min_alpha = d2v_model.alpha
    return d2v_model

def infer_vectors(model, data):
    return data.apply(lambda text: model.infer_vector(text).tolist())

if __name__ == '__main__':
    args = get_cli_args()
    notes_data = prepare_text_data('./data/processed/earlynotes.csv')

    if args.mode != 'infer':
        epochs = args.training_cycles
        data_splits = load_data_splits('./data/processed/files/splits.json')
        training_ids = [int(idx[-10:-4]) for batch in data_splits[:7] for idx in batch]
        training_docs = notes_data[notes_data['hadm_id'].isin(training_ids)]['processed_text'].tolist()

        tagged_data = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(training_docs)]
        model = train_doc2vec_model(tagged_data, epochs)
        model.save('./models/doc2vec.model')
    else:
        print("Performing inference...")
        doc2vec_model = Doc2Vec.load('./models/doc2vec.model')
        notes_data['vector'] = infer_vectors(doc2vec_model, notes_data['processed_text'])
        grouped_vectors = notes_data.groupby('hadm_id')['vector'].apply(list).reset_index()
        vectors_json = {str(int(row['hadm_id'])): row['vector'] for idx, row in grouped_vectors.iterrows()}
        json.dump(vectors_json, open('./data/processed/files/vector_dict.json', 'w'))
