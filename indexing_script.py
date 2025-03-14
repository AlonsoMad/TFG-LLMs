from src.IRQ.indexer import Indexer

def main():
    file_path = '/export/usuarios_ml4ds/ammesa/Data/3_joined_data/polylingual_df'
    mallet_path = '/export/usuarios_ml4ds/ammesa/mallet_folder'
    mod_name = 'quora-distilbert-multilingual'
    thr = '0'
    top_k = 10

    i = Indexer(file_path, mallet_path, mod_name, thr, top_k)

    i.indexing_loop()

    return

if __name__ == "__main__":
    main()