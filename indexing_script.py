from src.IRQ.indexer import Indexer, Retriever

def main():

    file_path = '/export/usuarios_ml4ds/ammesa/Data/3_joined_data/polylingual_df'
    mallet_path = '/export/usuarios_ml4ds/ammesa/mallet_folder'
    mod_name = 'sentence-transformers/LaBSE' #'sentence-transformers/quora-distilbert-multilingual' #'sentence-transformers/LaBSE'
    thr = '0.01'
    top_k = 10

    config = {
            "match": "TB_ENN",
            "embedding_size": 768,
            "min_clusters": 8,
            "top_k_hits": 10,
            "batch_size": 32,
            "thr": '0.01',
            "top_k": 10
        }

    i = Indexer(file_path, mallet_path, mod_name, thr, top_k)

    r = Retriever(file_path, mallet_path, mod_name, thr, top_k, '/export/usuarios_ml4ds/ammesa/Data/question_bank', config)

    r.retrieval_loop(n_tpcs=6, weight=True)
    #i.index()

    return

if __name__ == "__main__":
    main()