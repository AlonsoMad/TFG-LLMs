#!/bin/bash

echo "Starting pipeline execution..."

MODE="bilingual"  # Options: monolingual | bilingual
MODEL="mallet"  # Options: zeroshot | lda | mallet 
LANG1="en"
LANG2="es"
GENERATE_DS="NO"
SOURCE_FILE="en_2025-02-25_segmented_dataset.parquet.gzip"
SOURCE_FILE2="es_2025-02-25_segmented_dataset.parquet.gzip"
NL_DEST_PATH="/export/usuarios_ml4ds/ammesa/Data/2_lemmatized_data/toy_ds_folder"
QUESTION_FOLDER="/export/usuarios_ml4ds/ammesa/Data/question_bank/toy_questions"
N_TOPICS='6,9,12,15,20,30,50'

echo "Starting NLPIPE execution with: "
echo "Source: $SOURCE_FILE"
echo "Destination: $NL_DEST_PATH"
source "/export/usuarios_ml4ds/ammesa/TFG-LLMs/.venv_nlpipe/bin/activate"
#Eliminar tras la ejecución del experimento

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
    echo "Processing english files!"
    python3 -m src.NLPipe.nlpipe \
        --source_path /export/usuarios_ml4ds/ammesa/Data/1_segmented_data/toy_folder/$SOURCE_FILE \
        --source $SOURCE_FILE\
        --destination_path "$NL_DEST_PATH/$LANG1" \
        --stw_path /export/usuarios_ml4ds/ammesa/TFG-LLMs/src/topic_modeling/stops \
        --lang $LANG1
    
    echo "Processing spanish files!"
    python3 -m src.NLPipe.nlpipe \
        --source_path /export/usuarios_ml4ds/ammesa/Data/1_segmented_data/toy_folder/$SOURCE_FILE2 \
        --source $SOURCE_FILE2\
        --destination_path "$NL_DEST_PATH/$LANG2" \
        --stw_path /export/usuarios_ml4ds/ammesa/TFG-LLMs/src/topic_modeling/stops \
        --lang $LANG2
    if [ "$MODEL" == "zeroshot" ]; then
        deactivate
        source "/export/usuarios_ml4ds/ammesa/TFG-LLMs/.venv_zs/bin/activate"
        echo "Zero-Shot Topic Model selected!"
        python3 -m train_ZS\
            --path_folder $NL_DEST_PATH \
            --source_file $SOURCE_FILE \
            --num_topics $N_TOPICS \
            --lang1 $LANG1 \
            --lang2 $LANG2 

        MALLET_FOLDER="/export/usuarios_ml4ds/ammesa/ZS_results/$SOURCE_FILE"

    elif [ "$MODEL" == "mallet" ]; then
        deactivate
        source "/export/usuarios_ml4ds/ammesa/TFG-LLMs/.venv_zs/bin/activate"
        echo "Mallet selected"
        python3  -m train_tm\
            --path_folder $NL_DEST_PATH \
            --source_file $SOURCE_FILE \
            --num_topics $N_TOPICS \
            --lang1 $LANG1 \
            --lang2 $LANG2
        
        MALLET_FOLDER="/export/usuarios_ml4ds/ammesa/mallet_folder/$SOURCE_FILE"
        
    else
        echo "Error: Invalid model specified: $MODEL"
        exit 1
    fi

else
    echo "Error: Invalid mode specified: $MODE"
    exit 1
fi

echo "Sucessful execution"
# --- STEP 2: Activate virtual environment ---
echo "Activating virtual environment for model training..."

source "/export/usuarios_ml4ds/ammesa/TFG-LLMs/.venv_idx/bin/activate"
K=$(cat "/export/usuarios_ml4ds/ammesa/TFG-LLMs/k_value.txt")
POLY_PATH=$(cat polypath.txt)

echo "Performing IRQ processes"
python3 -m indexing_script \
    --input_path "$POLY_PATH"\
    --mallet_folder  $MALLET_FOLDER \
    --question_folder $QUESTION_FOLDER \
    --k "$K" \
    --bilingual $MODE \
    --model $MODEL