from src.IRQ.indexer import Indexer, Retriever
import argparse
from itertools import product
import pathlib

def main(search_mode, weight):
    parser = argparse.ArgumentParser(description="Retrieve the answers and obtain metrics")

    parser.add_argument('--input_path', required=True, help='Path to input parquet file')
    parser.add_argument('--mallet_folder', required=True, help='Path to folder where the LDA model will be stored')
    parser.add_argument('--question_folder', required=True,type=str, help='Path to folder with the questions')
    parser.add_argument('--k', required=True,type=int, help='Path to folder with the questions')


    args = parser.parse_args()

    file_path = pathlib.Path(args.input_path)
    mallet_path = pathlib.Path(args.mallet_folder)
    question_folder = pathlib.Path(args.question_folder)
    k = args.k

    mod_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' #'sentence-transformers/LaBSE'

    config = {
            "match": search_mode,
            "embedding_size": 384,
            "min_clusters": 8,
            "top_k_hits": 10,
            "batch_size": 32, #Baja batch size si es necesario por limitación de la GPU
            "thr": 'var',
            "top_k": 10,
            'storage_path': '/export/usuarios_ml4ds/ammesa/Data/4_indexed_data'
        }

    i = Indexer(file_path, mallet_path, mod_name, config)
    i.index()

    r = Retriever(file_path, mallet_path, mod_name, question_folder, config)
    r._logger.info(f'Running experiment: {search_mode} with weight: {weight}')
    r.retrieval_loop(n_tpcs=k, weight=weight)

    return

if __name__ == "__main__":
    
    search_modes = ["TB_ANN","TB_ENN"]
    weight_options = [True, False]
    for search_mode, weight in product(search_modes, weight_options):
        main(search_mode, weight)
    search_modes = ["ENN", "ANN"]
    for search_mode in search_modes:
        main(search_mode, True)

    
    
    