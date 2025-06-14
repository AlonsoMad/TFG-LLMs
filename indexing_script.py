from src.IRQ.indexer import Indexer, Retriever
import argparse
from itertools import product
import pathlib
import os
def main(search_mode, weight, nprobe:int=10):
    parser = argparse.ArgumentParser(description="Retrieve the answers and obtain metrics")

    parser.add_argument('--input_path', required=True, help='Path to input parquet file')
    parser.add_argument('--mallet_folder', required=True, help='Path to folder where the LDA model will be stored')
    parser.add_argument('--question_folder', required=True,type=str, help='Path to folder with the questions')
    parser.add_argument('--k', required=True,type=int, help='Path to folder with the questions')
    parser.add_argument('--bilingual', required=True,type=str)
    parser.add_argument('--model', required=True,type=str)
    parser.add_argument('--lang1', required=True,type=str)
    parser.add_argument('--lang2', required=True,type=str)


    args = parser.parse_args()

    file_path = pathlib.Path(args.input_path)
    mallet_path = pathlib.Path(args.mallet_folder)
    question_folder = pathlib.Path(args.question_folder)
    k = args.k
    model = args.model
    lang1 = args.lang1
    lang2 = args.lang2

    bilingual = args.bilingual == 'bilingual'
    mod_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' #'sentence-transformers/LaBSE'

    suffix = os.path.basename(mallet_path)

    config = {
            "match": search_mode,
            "embedding_size": 384,
            "min_clusters": 8,
            "k": k,
            "batch_size": 32, #Baja batch size si es necesario por limitación de la GPU
            "thr": '0.01',
            "top_k": 10,
            'storage_path': f'/export/usuarios_ml4ds/ammesa/Data/4_indexed_data/{suffix}',
            'lang1' : lang1,
            'lang2' : lang2
        }

    i = Indexer(file_path, mallet_path, mod_name, config)
    i.index(bilingual=bilingual, topic_model=model, nprobe=nprobe)

    config['thr'] = 'var'
    r = Retriever(file_path, mallet_path, mod_name, question_folder, config)
    r._logger.info(f'Running experiment: {search_mode} with weight: {weight}')
    r.retrieval_loop(bilingual=bilingual, n_tpcs=k, topic_model=model, weight=weight, evaluation_mode=True, parallel=False)
    return

if __name__ == "__main__":

    search_modes = ["TB_ENN","TB_ANN"]
    weight_options = [True, False]
    for search_mode, weight in product(search_modes, weight_options):
        main(search_mode, weight)
    main('ENN', True)
    main('ANN', True)

    

    
    
    