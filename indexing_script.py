from src.IRQ.indexer import Indexer, Retriever

def main():

    file_path = '/export/usuarios_ml4ds/ammesa/Data/3_joined_data/polylingual_df'
    mallet_path = '/export/usuarios_ml4ds/ammesa/mallet_folder'
    mod_name = 'sentence-transformers/LaBSE'#'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' 

    config = {
            "match": "ENN",
            "embedding_size": 768,
            "min_clusters": 8,
            "top_k_hits": 10,
            "batch_size": 32,
            "thr": 'var',
            "top_k": 10,
            'storage_path': '/export/usuarios_ml4ds/ammesa/Data/4_indexed_data'
        }

    i = Indexer(file_path, mallet_path, mod_name, config)

    r = Retriever(file_path, mallet_path, mod_name, '/export/usuarios_ml4ds/ammesa/Data/question_bank', config)

    r.retrieval_loop(n_tpcs=6, weight=False)
    #i.index()
    r.evaluation()

    return

if __name__ == "__main__":
    main()