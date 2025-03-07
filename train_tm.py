import pathlib
import os
from src.topic_modeling.polylingual_tm import PolylingualTM
from src.topic_modeling.preprocessing import DataPreparer
#from src.topic_modeling.lda_tm import LDATM

def main():
    
    path_corpus_es = "/export/usuarios_ml4ds/ammesa/Data/lemmatized_data/es.parquet"
    path_corpus_en = "/export/usuarios_ml4ds/ammesa/Data/lemmatized_data/en.parquet"
    path_save_tr = "/export/usuarios_ml4ds/ammesa/Data/output_mallet"
    path_save = "define"
    
    # Generate training data
    print("-- -- Generating training data")
    # TODO: Preprocess corpus, use the methods to generate the correct input for mallet
    
    prep = DataPreparer(path_folder='/export/usuarios_ml4ds/ammesa/Data/2_lemmatized_data',
                        name_es = 'es',
                        name_en = 'en',
                        storing_path='/export/usuarios_ml4ds/ammesa/Data/3_joined_data')
    
    #Get the dataframes in the correct form
    prep.format_dataframes()

    print("-- -- Training PolyLingual Topic Model")
    # Train PolyLingual Topic Model
    for k in [5]:  #[30,5,10,15,20,50]:
        # model = LDATM(
        model = PolylingualTM(
            lang1="EN",
            lang2="ES",
            model_folder= pathlib.Path(f"/export/usuarios_ml4ds/ammesa/mallet_folder"),
            #model_folder = pathlib.Path(f"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/29_dec/LDA/lda_rosie_{str(sample)}_{k}"),
            num_topics=k
        )
        model.train(os.path.join(prep.storing_path, 'polylingual_df'))
    
if __name__ == "__main__":
    main()
