from src.API.wiki_retrieve import *
from src.topic_modeling.preprocessing import *

import os
import time
import logging


if __name__ == "__main__":
    # Define output directory
    dir = os.path.dirname(os.path.abspath(__file__))
    father_dir = os.path.split(dir)[0]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Handles the creation of the enviroment folders
    # In case the execution is in a new computer
    dir_structure = {'Data':['0_input_data',
                             '1_segmented_data',
                             '2_lemmatized_data',
                             '3_joined_data']}
    """
    for main_dir, subdirs in dir_structure.items():

        #Create the parent directory
        main_path = os.path.join(father_dir, main_dir)

        for subdir in subdirs:
            #Populate with subdirectories
            sub_path = os.path.join(main_path, subdir)

            if not os.path.exists(sub_path):
                os.mkdir(sub_path,exist_ok=True)

                logging.info(f'Subdirectory {sub_path} correctly created!')


    output_dir = os.path.join(father_dir, 'Data/0_input_data')

    MAX_TRIES = 5
    
    # Initialize WikiRetriever with desired parameters
    retriever = WikiRetriever(
        file_path=output_dir,
        seed_lan="en",
        seed_query="George Washington",
        ndocs=2  
    )

    attempt = 0

    while attempt < MAX_TRIES:
        try:
            # Run the retrieval process
            logging.info('Starting Wikipedia retrieval... Attempt: %d', attempt+1)
            retriever.retrieval()

            # Save the dataset
            retriever.df_to_parquet()

            logging.info(f"Dataset saved in %s/dataset.parquet.gzip", output_dir)
            break
        except:
            logging.warning('Network exception ocurred')
            attempt += 1
            if attempt < MAX_TRIES:
                logging.info("Retrying in %d seconds...", 2**attempt)
                time.sleep(2**attempt)
            else:
                logging.error("Max retries reached. Exiting.")

    #I should change this so it can adapt easily

    file_name = retriever.final_file_name

    os.makedirs(segmentated_dir, exist_ok=True)

    s = Segmenter(in_directory=output_dir,
                  file_name=file_name,
                  out_directory=segmentated_dir)
    
    print('Reading data')
    s.read_dataframe()
   
    print('Segmenting data')
    s.segment()
    """
    segmentated_dir = '/export/usuarios_ml4ds/ammesa/Data/1_segmented_data'

    en_df = pd.read_parquet("/export/usuarios_ml4ds/ammesa/Data/1_segmented_data/en_2025-02-20_segmented_dataset.parquet.gzip")
    es_df = pd.read_parquet("/export/usuarios_ml4ds/ammesa/Data/1_segmented_data/es_2025-02-20_segmented_dataset.parquet.gzip")
    #Finally translating and completing the datasets

    trans = Translator(en_df, es_df)

    trans.translate()
    trans.save_dataframes(segmentated_dir)


