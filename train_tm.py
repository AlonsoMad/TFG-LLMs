import pathlib
import os
import matplotlib.pyplot as plt
from src.topic_modeling.polylingual_tm import PolylingualTM
from src.topic_modeling.preprocessing import DataPreparer
from src.metrics.coherence import *
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

    n_topics =[6] #np.linspace(2,15,14, dtype = int)

    topic_coherences = np.empty((len(n_topics),1))

    print("-- -- Training PolyLingual Topic Model")
    # Train PolyLingual Topic Model
    for idx, k in enumerate(n_topics):  #[30,5,10,15,20,50]:
        # model = LDATM(
        model = PolylingualTM(
            lang1="EN",
            lang2="ES",
            model_folder= pathlib.Path(f"/export/usuarios_ml4ds/ammesa/mallet_folder"),
            #model_folder = pathlib.Path(f"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/29_dec/LDA/lda_rosie_{str(sample)}_{k}"),
            num_topics=k
        )
        model.train(os.path.join(prep.storing_path, 'polylingual_df'))
        topic_coherences[idx] = extract_cohr('ES')

    print(topic_coherences)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(n_topics, topic_coherences, marker='o', linestyle='-', color='b', label='Topic Coherence')
    plt.xlabel('Number of Topics (k)')
    plt.ylabel('Topic Coherence')
    plt.title('Topic Coherence vs. Number of Topics')
    plt.grid(True)
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()
