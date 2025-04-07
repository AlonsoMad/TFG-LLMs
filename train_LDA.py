from src.topic_modeling.lda_tm import *


def main():
    '''
    Executes the training of an LDA model over the specified dataset
    '''
    path = '/export/usuarios_ml4ds/ammesa/Data/2_lemmatized_data/med_en'
    k = 6

    df_aux = pd.read_parquet(path)
    df_aux.loc[:, 'lang'] = 'EN'
    df_aux['doc_id'] = df_aux.index
    df_aux.to_parquet(path)
    
    model = LDATM(
        lang1 = 'EN',
        lang2 = 'EN',
        model_folder = pathlib.Path(f"/export/usuarios_ml4ds/ammesa/LDA_folder"),
        num_topics = k
    )
    model.train(path)

    return

if __name__ == '__main__':
    main()
    