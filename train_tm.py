import pathlib
import os
import argparse
import matplotlib.pyplot as plt
from src.topic_modeling.polylingual_tm import PolylingualTM
from src.topic_modeling.preprocessing import DataPreparer
from src.metrics.coherence import *
import pandas as pd
#from src.topic_modeling.lda_tm import LDATM

def main():
    
    
    # Generate training data
    print("-- -- Generating training data")
    # TODO: Preprocess corpus, use the methods to generate the correct input for mallet
    parser = argparse.ArgumentParser(description="Train mallet model on dataset")
    parser.add_argument('--num_topics', type=str, default='6', help='Number of topics for the ZS model')
    parser.add_argument('--lang1', type=str, default='en', help='First of the two languages for the model')
    parser.add_argument('--lang2', type=str, default='es', help='Second of the two languages for the model')
    parser.add_argument('--path_folder', type=str, help='Path for the NLpipe lemmatized files')
    parser.add_argument('--source_file', type=str, help='Original name of the file (used for path creation)')

    args = parser.parse_args()

    lang1 = args.lang1.lower()
    lang2 = args.lang2.lower()
    src_file = args.source_file
    path_folder = args.path_folder
    storing_path = os.path.join('/export/usuarios_ml4ds/ammesa/Data/3_joined_data', src_file)
    os.makedirs(storing_path,exist_ok=True)

    file_name = 'polylingual_df' 
    full_path = os.path.join(storing_path, file_name)
    import pdb; pdb.set_trace()
    if os.path.exists(full_path):
        print(f"-- -- Training data already exists at {full_path}. Skipping preprocessing.")
    else:
        prep = DataPreparer(path_folder=path_folder,
                            name_es = lang2,
                            name_en = lang1,
                            storing_path=storing_path)
        
        #Get the dataframes in the correct form
        prep.format_dataframes()

    n_topics = list(map(int, args.num_topics.split(',')))

    topic_coherences = np.empty((len(n_topics),1))
    out_path = os.path.join('/export/usuarios_ml4ds/ammesa/mallet_folder', src_file)
    path_cohr = os.path.join(out_path, 'mallet_output', 'topickeys.txt')

    print("-- -- Training PolyLingual Topic Model")
    # Train PolyLingual Topic Model
    for idx, k in enumerate(n_topics):  #[30,5,10,15,20,50]:
        # model = LDATM(
        model = PolylingualTM(
            lang1=lang1.upper(),
            lang2=lang2.upper(),
            model_folder=out_path,
            num_topics=k
        )
        model.train(os.path.join(storing_path, file_name))
        topic_coherences[idx] = extract_cohr(path_cohr)
    
    k = n_topics[np.argmax(topic_coherences)]

    df_coherences = pd.DataFrame(topic_coherences, columns=["Coherence"])
    with open("k_value.txt", "w") as f:
        f.write(str(k))

    with open("polypath.txt", "w") as f:
        f.write(str(full_path))
        
    df_coherences.to_csv("coherences.csv", index=False)

    model = PolylingualTM(
        lang1=lang1.upper(),
        lang2=lang2.upper(),
        model_folder=out_path,
        num_topics=k
    )
    model.train(os.path.join(storing_path, 'polylingual_df'))



    
if __name__ == "__main__":
    main()
