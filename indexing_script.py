from src.IRQ.indexer import Indexer, Retriever
from itertools import product

def main(search_mode, weight):

    file_path = '/export/usuarios_ml4ds/ammesa/Data/2_lemmatized_data/med_en'
    mallet_path = '/export/usuarios_ml4ds/ammesa/LDA_folder'
    mod_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' #'sentence-transformers/LaBSE'

    config = {
            "match": search_mode,
            "embedding_size": 384,
            "min_clusters": 8,
            "top_k_hits": 10,
            "batch_size": 32,
            "thr": 'var',
            "top_k": 10,
            'storage_path': '/export/usuarios_ml4ds/ammesa/Data/4_indexed_data'
        }

    i = Indexer(file_path, mallet_path, mod_name, config)

    r = Retriever(file_path, mallet_path, mod_name, '/export/usuarios_ml4ds/ammesa/Data/question_bank', config)
    r._logger.info(f'Running experiment: {search_mode} with weight: {weight}')
    r.retrieval_loop(n_tpcs=6, weight=weight)

    return

if __name__ == "__main__":
    search_modes = ["TB_ANN", "TB_ENN", "ENN", "ANN"]
    weight_options = [True, False]
    for search_mode, weight in product(search_modes, weight_options):
        main(search_mode, weight)