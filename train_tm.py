import pathlib
from src.topic_modeling.polylingual_tm import PolylingualTM
#from src.topic_modeling.lda_tm import LDATM

def main():
    
    path_corpus_es = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_pass_es_tr.parquet"
    path_corpus_en = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_pass_en_tr.parquet"
    path_save_tr = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/26_jan/df.parquet"
    path_save = "define"
    
    # Generate training data
    print("-- -- Generating training data")
    # TODO: Preprocess corpus
    
    print("-- -- Training PolyLingual Topic Model")
    # Train PolyLingual Topic Model
    for k in [30,5,10,15,20,50]: #,100,200,300,400,500
        # model = LDATM(
        model = PolylingualTM(
            lang1="EN",
            lang2="ES",
            model_folder= pathlib.Path(f"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/28_jan/poly_rosie_{k}"),
            #model_folder = pathlib.Path(f"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/29_dec/LDA/lda_rosie_{str(sample)}_{k}"),
            num_topics=k
        )
        model.train(path_save)
    
if __name__ == "__main__":
    main()
