import dotenv
import os
import pandas as pd
import numpy as np
import re
from scipy.sparse import csr_matrix, vstack
import pyLDAvis as vis

def allowed_file(filename):
    allowed_extensions = os.getenv("ALLOWED_EXTENSIONS", "parquet,csv,xlsx").split(",")
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def load_datasets(dataset_path: str) -> tuple:

    dataset_list = []

    datasets_name = os.listdir(dataset_path)
    shapes = np.empty((len(datasets_name), 2), dtype=int)
    print(f"Datasets found: {datasets_name}")
    for i, d in enumerate(datasets_name):
        #For each one of the datasets load in memory a short header
        #Semi harcoded, TODO: solve in future
        ds = pd.read_parquet(os.path.join(dataset_path, d, 'polylingual_df'))
        shapes[i] = ds.shape
        ds = ds.drop(columns=['index'])
        dataset_list.append(ds.head(20))
        print(f"Dataset {d} loaded with shape {ds.shape}")
    
    return dataset_list, datasets_name, shapes

def read_mallet(input_path):
    '''
    Returns a dictionary for the pyLDAvis module.
    Takes the path to the parent folder mallet_folder (path_mallet)
    and searches the parameters inside mallet_output.
    '''
    


    search_items={
        'betas_en':['betas_ES.npy','topic_term_dists_en'],
        'betas_es':['betas_ES.npy','topic_term_dists_es'],
        'thetas_en':['thetas_ES.npz', 'doc_topic_dists_en'],
        'thetas_es':['thetas_ES.npz', 'doc_topic_dists_es'],
    }
    results = {}

    for item in search_items:
        doc = np.load(os.path.join(input_path, search_items[item][0]))
        results.update({search_items[item][1]:doc})

    #in order to get the doc-topic we have to transform back from -npz
    #to a matrix format, we do it in this lines
    aux = results['doc_topic_dists_en']
    #Reshaping of the auxiliar variable
    dense_vec_en = csr_matrix((aux['data'], aux['indices'], aux['indptr']), shape=aux['shape'])
    results['doc_topic_dists_en'] = dense_vec_en.toarray()

    #Reshaping of the auxiliar variable
    dense_vec_es = csr_matrix((aux['data'], aux['indices'], aux['indptr']), shape=aux['shape'])
    results['doc_topic_dists_es'] = dense_vec_es.toarray()

    doc_topic_matrix = vstack([dense_vec_en, dense_vec_es])

    # Convert to dense
    results['doc_topic_dists'] = doc_topic_matrix.toarray()

    #Get the vocab and frequency, both stored in vocab.txt
    vocab_path = os.path.join(input_path, 'vocab_EN.txt')
    vocab_df = pd.read_csv(vocab_path, sep='\t', header = None)
    results['vocab_en'] = vocab_df[0]
    results['term_frequency_en'] = vocab_df[1] 

    vocab_path = os.path.join(input_path, 'vocab_ES.txt')
    vocab_df = pd.read_csv(vocab_path, sep='\t', header = None)
    results['vocab_es'] = vocab_df[0]
    results['term_frequency_es'] = vocab_df[1] 

    #Error, estas simplemente sumando 1 en todos los topic_lengths
    results['doc_lengths_en'] = np.round(results['doc_topic_dists_en'].sum(axis=1)).astype(int)
    results['doc_lengths_es'] = np.round(results['doc_topic_dists_es'].sum(axis=1)).astype(int)
    print(np.round(results['doc_topic_dists_en'].sum(axis=1)).astype(int))
    results['doc_lengths'] = np.round(results['doc_topic_dists'].sum(axis=1)).astype(int)

    return results

def extract_topic_id(path):
    match = re.search(r'topic_(\d+)', path)
    return int(match.group(1)) if match else None

def extract_sample_len(path):
    match = re.search(r'samples_len_(\d+)', path)
    return int(match.group(1)) if match else None


def read_ZS(path_ZS):
    '''
    Returns a dictionary for the pyLDAvis module.
    Takes the path to the parent folder ZS_results (path_ZS)
    and searches the parameters inside ZS_output.
    '''

    search_items={
        'betas':['betas.npy','topic_term_dists'],
        'thetas':['thetas.npy', 'doc_topic_dists'],
        'doc_len':['doc_len.npy', 'doc_lengths'],
        'term_freq':['term_freq.npy', 'term_frequency']
    }
    results = {}

    for item in search_items:
        doc = np.load(os.path.join(input_path, search_items[item][0]))
        results.update({search_items[item][1]:doc})


    vocab_path = os.path.join(input_path, 'vocab.txt')
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f.readlines()]

    results['vocab'] = vocab 

    return results