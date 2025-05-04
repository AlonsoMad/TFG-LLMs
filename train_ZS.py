'''
Script for the execution of the ZeroShotTM training
'''
import pathlib
import os
import sys
import argparse
import numpy as np
import pandas as pd
from src.metrics.coherence import *
# Ensure the CTM submodule is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "CTM"))
from src.topic_modeling.contextual_tm import ContextualTM
from src.topic_modeling.preprocessing import DataPreparer

#Avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    print("-- -- Checking for existing training data")
    parser = argparse.ArgumentParser(description="Train LDA model on dataset")
    parser.add_argument('--num_topics', type=str, default='6', help='Number of topics for the ZS model')
    parser.add_argument('--lang1', type=str, default='en', help='First of the two languages for the model')
    parser.add_argument('--lang2', type=str, default='es', help='Second of the two languages for the model')
    parser.add_argument('--path_folder', type=str, help='Path for the NLpipe lemmatized files')
    parser.add_argument('--source_file', type=str, help='Original name of the file (used for path creation)')

    args = parser.parse_args()

    src_file = args.source_file
    n_topics = list(map(int, args.num_topics.split(',')))
    path_folder = args.path_folder
    lang1 = args.lang1.lower()
    lang2 = args.lang2.lower()

    storing_path = os.path.join('/export/usuarios_ml4ds/ammesa/Data/3_joined_data', src_file)
    os.makedirs(storing_path,exist_ok=True)
    file_name = 'polylingual_df' 
    full_path = os.path.join(storing_path, file_name)

    #Ensure the storing path exists


    if os.path.exists(full_path):
        print(f"-- -- Training data already exists at {full_path}. Skipping preprocessing.")
    else:
        print("-- -- Generating training data")
        
        prep = DataPreparer(path_folder=path_folder,
                            #segmented_path='/export/usuarios_ml4ds/ammesa/Data/1_segmented_data',
                            #segmented_f_name='_2025-02-08_segmented_dataset.parquet.gzip',
                            name_es=lang2,
                            name_en=lang1,
                            storing_path=storing_path)
        
        # Get the dataframes in the correct form
        prep.format_dataframes()
        print(f"-- -- Training data saved at {full_path}")

    print("-- -- Training Polylingual Topic Model")

    out_path = os.path.join('/export/usuarios_ml4ds/ammesa/ZS_results', src_file)
    os.makedirs(out_path, exist_ok=True)

    ctm = ContextualTM(input_path=storing_path,
                       output_path=out_path,
                       input_f_name='polylingual_df',
                       lang1=lang1,
                       lang2=lang2)
    
    ctm.read_dataframes()
    
    ctm.prepare_corpus()

    topic_coherences = np.empty((len(n_topics),1))

    path_cohr = os.path.join(out_path, 'ZS_output', 'topics.txt')

    for idx, k in enumerate(n_topics):

        ctm.train(num_topics=k)
    
        ctm.save_results()

        topic_coherences[idx] = extract_cohr(path_cohr)

    k = n_topics[np.argmax(topic_coherences)]
    ctm.train(num_topics=k)
    ctm.save_results()
    df_coherences = pd.DataFrame(topic_coherences, columns=["Coherence"])

    with open("k_value.txt", "w") as f:
        f.write(str(k))

    with open("polypath.txt", "w") as f:
        f.write(str(full_path))

    df_coherences.to_csv("coherences.csv", index=False)

    return

if __name__ == "__main__":
    main()