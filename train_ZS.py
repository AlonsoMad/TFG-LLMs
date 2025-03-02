'''
Script for the execution of the ZeroShotTM training
'''
import pathlib
import os
import sys
# Ensure the CTM submodule is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "CTM"))
from src.topic_modeling.contextual_tm import ContextualTM
from src.topic_modeling.preprocessing import DataPreparer

#Avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    print("-- -- Checking for existing training data")

    storing_path = '/export/usuarios_ml4ds/ammesa/Data/3_joined_data'
    file_name = 'polylingual_df.parquet.gzip' 
    full_path = os.path.join(storing_path, file_name)

    if os.path.exists(full_path):
        print(f"-- -- Training data already exists at {full_path}. Skipping preprocessing.")
    else:
        print("-- -- Generating training data")
        
        prep = DataPreparer(path_folder='/export/usuarios_ml4ds/ammesa/Data/2_lemmatized_data',
                            segmented_path='/export/usuarios_ml4ds/ammesa/Data/1_segmented_data',
                            segmented_f_name='_2025-02-08_segmented_dataset.parquet.gzip',
                            name_es='es',
                            name_en='en',
                            storing_path=storing_path)
        
        # Get the dataframes in the correct form
        prep.format_dataframes()
        print(f"-- -- Training data saved at {full_path}")

    print("-- -- Training PolyLingual Topic Model")

    ctm = ContextualTM(input_path=storing_path,
                       output_path='/export/usuarios_ml4ds/ammesa/ZS_results',
                       input_f_name='polylingual_df',
                       lang1='es',
                       lang2='en')
    
    ctm.read_dataframes()
    
    ctm.prepare_corpus()

    ctm.train()
    
    ctm.save_results()

    return

if __name__ == "__main__":
    main()