#!/bin/bash

# Define the virtual environment path
VENV_PATH="/export/usuarios_ml4ds/ammesa/TFG-LLMs/venv"

# Check if the virtual environment is already activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    echo "Virtual environment already activated."
fi

# Run the Python script

#################
# Change for ES #
#################

python3 -m src.NLPipe.nlpipe \
    --source_path /export/usuarios_ml4ds/ammesa/Data/1_segmented_data \
    --source es_2025-02-20_segmented_dataset.parquet.gzip \
    --destination_path /export/usuarios_ml4ds/ammesa/Data/2_lemmatized_data/es \
    --stw_path /export/usuarios_ml4ds/ammesa/TFG-LLMs/src/topic_modeling/stops \
    --lang es \
    --do_embeddings 
