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
    --source_path /export/usuarios_ml4ds/ammesa/Data/1.5_trans_data \
    --source dataset_PubMedQA\
    --destination_path /export/usuarios_ml4ds/ammesa/Data/2_lemmatized_data/med_en \
    --stw_path /export/usuarios_ml4ds/ammesa/TFG-LLMs/src/topic_modeling/stops \
    --lang en \
    --do_embeddings 
