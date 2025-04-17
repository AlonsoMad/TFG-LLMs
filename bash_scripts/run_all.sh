#!/bin/bash

echo "Starting pipeline execution..."

MODE="monolingual"  # Options: monolingual | bilingual
LANG1="EN"
LANG2="ES"
GENERATE_DS="NO"
SOURCE_FILE="dataset_PubMedQA"
NL_DEST_PATH="/export/usuarios_ml4ds/ammesa/Data/2_lemmatized_data/med"
QUESTION_FOLDER="/export/usuarios_ml4ds/ammesa/Data/question_bank/10_04_25_questions"
N_TOPICS='6,9,12,15,20,30,50'


echo "Starting NLPIPE execution with: "
echo "Source: $SOURCE_FILE"
echo "Destination: $NL_DEST_PATH"
source "/export/usuarios_ml4ds/ammesa/TFG-LLMs/.venv_nlpipe/bin/activate"

#Eliminar tras la ejecución del experimento
: << 'IGNORE'
if [ "$MODE" == "monolingual" ]; then
    python3 -m src.NLPipe.nlpipe \
        --source_path /export/usuarios_ml4ds/ammesa/Data/1_segmented_data/PubMed_10_04/$SOURCE_FILE \
        --source $SOURCE_FILE\
        --destination_path $NL_DEST_PATH \
        --stw_path /export/usuarios_ml4ds/ammesa/TFG-LLMs/src/topic_modeling/stops \
        --lang en \

    echo "Sucessful execution"
    # --- STEP 2: Activate virtual environment ---
    echo "Activating virtual environment for model training..."
    deactivate
    source "/export/usuarios_ml4ds/ammesa/TFG-LLMs/.venv_mallet/bin/activate"
    echo "Training monolingual LDA model..."
    MALLET_FOLDER="/export/usuarios_ml4ds/ammesa/LDA_folder"
    python3 -m train_LDA \
        --input_path $NL_DEST_PATH \
        --mallet_folder $MALLET_FOLDER \
        --num_topics $N_TOPICS \
        --lang $LANG1
    echo "Mallet files generated!"
    echo "Starting experimentation process"
    
elif [ "$MODE" == "bilingual" ]; then
    python3 -m src.NLPipe.nlpipe \
        --source_path /export/usuarios_ml4ds/ammesa/Data/1_segmented_data/$SOURCE_FILE \
        --source $SOURCE_FILE\
        --destination_path $NL_DEST_PATH \
        --stw_path /export/usuarios_ml4ds/ammesa/TFG-LLMs/src/topic_modeling/stops \
        --lang en 
    python3 -m src.NLPipe.nlpipe \
        --source_path /export/usuarios_ml4ds/ammesa/Data/1_segmented_data/$SOURCE_FILE \
        --source $SOURCE_FILE\
        --destination_path $NL_DEST_PATH \
        --stw_path /export/usuarios_ml4ds/ammesa/TFG-LLMs/src/topic_modeling/stops \
        --lang es \

else
    echo "Error: Invalid mode specified: $MODE"
    exit 1
fi
IGNORE
echo "Sucessful execution"
# --- STEP 2: Activate virtual environment ---
echo "Activating virtual environment for model training..."
deactivate
source "/export/usuarios_ml4ds/ammesa/TFG-LLMs/.venv_idx/bin/activate"
K=$(cat k_value.txt)
echo "Performing IRQ processes"
MALLET_FOLDER="/export/usuarios_ml4ds/ammesa/LDA_folder"
python3 -m indexing_script \
    --input_path $NL_DEST_PATH \
    --mallet_folder  $MALLET_FOLDER \
    --question_folder $QUESTION_FOLDER \
    --k $K