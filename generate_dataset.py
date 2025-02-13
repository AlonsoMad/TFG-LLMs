from src.API.wiki_retrieve import *
from src.topic_modeling.preprocessing import *

import os
import time
import logging


if __name__ == "__main__":
    # Define output directory
    dir = os.path.dirname(os.path.abspath(__file__))
    father_dir = os.path.split(dir)[0]

    #The directory for the later inputs is the generation output
    output_dir = os.path.join(father_dir, "Data/0_input_data")

    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    MAX_TRIES = 5
    
    # Initialize WikiRetriever with desired parameters
    retriever = WikiRetriever(
        file_path=output_dir,
        seed_lan="en",
        seed_query="George Washington",
        ndocs=2000  
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
    segmentated_dir = '/export/usuarios_ml4ds/ammesa/Data/1_segmented_data'

    file_name = retriever.final_file_name

    os.makedirs(segmentated_dir, exist_ok=True)

    s = Segmenter(in_directory=output_dir,
                  file_name=file_name,
                  out_directory=segmentated_dir)
    
    print('Reading data')
    s.read_dataframe()

    print('Segmenting data')
    s.segment()


