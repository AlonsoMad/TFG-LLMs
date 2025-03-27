from src.IRQ.indexer import Indexer, Retriever

def main():

    file_path = '/export/usuarios_ml4ds/ammesa/Data/3_joined_data/polylingual_df'
    mallet_path = '/export/usuarios_ml4ds/ammesa/mallet_folder'
    mod_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'#sentence-transformers/distiluse-base-multilingual-cased-v2'#sentence-transformers/quora-distilbert-multilingual' #'sentence-transformers/LaBSE'

    config = {
            "match": "TB_ANN",
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

    r.retrieval_loop(n_tpcs=6, weight=True)
    #i.index()
    r.evaluation()

    return

if __name__ == "__main__":
    main()