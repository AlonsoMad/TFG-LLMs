from src.topic_modeling.lda_tm import *
from src.metrics.coherence import *
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    '''
    Executes the training of an LDA model over the specified dataset
    '''

    parser = argparse.ArgumentParser(description="Train LDA model on dataset")
    parser.add_argument('--input_path', required=True, help='Path to input parquet file')
    parser.add_argument('--mallet_folder', required=True, help='Path to folder where the LDA model will be stored')
    parser.add_argument('--num_topics', type=str, default='6', help='Number of topics for the LDA model')
    parser.add_argument('--lang', type=str, help='Number of topics for the LDA model')

    args = parser.parse_args()

    path = args.input_path
    model_folder = pathlib.Path(args.mallet_folder)
    #Get the topics in the correct format for the loop
    n_topics = list(map(int, args.num_topics.split(',')))
    lang = args.lang

    df_aux = pd.read_parquet(path)
    df_aux.loc[:, 'lang'] = 'EN'
    #df_aux['doc_id'] = df_aux.index
    df_aux.to_parquet(path)
    
    topic_coherences = np.empty((len(n_topics),1))
    for idx, k in enumerate(n_topics):  #[30,5,10,15,20,50]:

        model = LDATM(
            lang1 = lang,
            lang2 = lang,
            model_folder = model_folder,
            num_topics = k
        )
        model.train(path)
        path_cohr = f'/export/usuarios_ml4ds/ammesa/LDA_folder/mallet_output/{lang}/topickeys.txt'
        topic_coherences[idx] = extract_cohr(path_cohr)

    k = n_topics[np.argmax(topic_coherences)]
    #automatically save the model with higher coherence
    final_model = LDATM(
        lang1=lang,
        lang2=lang,
        model_folder=model_folder,
        num_topics=n_topics[np.argmax(topic_coherences)]
    )
    final_model.train(path)
    df_coherences = pd.DataFrame(topic_coherences, columns=["Coherence"])

    with open("k_value.txt", "w") as f:
        f.write(str(k))

    df_coherences.to_csv("coherences.csv", index=False)

    return

if __name__ == '__main__':
    main()
    