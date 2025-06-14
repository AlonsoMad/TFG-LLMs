#!/bin/bash
set -euo pipefail

echo "Starting pipeline execution..."

MODE="bilingual"  # Options: monolingual | bilingual
MODEL="zeroshot"  # Options: zeroshot | lda | mallet 
PREPROCESS="no" # Options: yes | no
MODEL_TRAINING="no" # Options: yes | no
EVALUATION="yes" # Options: yes | no
LANG1="en" # NOT IN CAPS
LANG2="es" # ONLY LOWERCASE
GENERATE_DS="NO"
SOURCE_FILE="en_2025-06-04_segmented_dataset.parquet.gzip" # Recently used: dataset_PubMedQA | dataset_Sports | en_2025-03-03_segm_trans | en_unaligned_dataset_75_per
SOURCE_FILE2="es_2025-06-04_segmented_dataset.parquet.gzip" # Only change if there is any
NL_DEST_PATH="/export/usuarios_ml4ds/ammesa/Data/2_lemmatized_data/f_micro_ZS_folder"
QUESTION_FOLDER="/export/usuarios_ml4ds/ammesa/Data/question_bank/sport_questions"
N_TOPICS='6,9,12,15,20,30,50'

echo "Starting NLPIPE execution with: "
echo "Source: $SOURCE_FILE"
echo "Destination: $NL_DEST_PATH"
source "/export/usuarios_ml4ds/ammesa/TFG-LLMs/.venv_nlpipe/bin/activate"
#Eliminar tras la ejecución del experimento
if [ "$MODE" == "monolingual" ]; then
    if [ "$PREPROCESS" == "yes" ]; then
    python3 -m src.NLPipe.nlpipe \
        --source_path /export/usuarios_ml4ds/ammesa/Data/1_segmented_data/sports_folder/$SOURCE_FILE \
        --source $SOURCE_FILE\
        --destination_path $NL_DEST_PATH \
        --stw_path /export/usuarios_ml4ds/ammesa/TFG-LLMs/src/topic_modeling/stops \
        --lang en \

    echo "Sucessful execution"
    fi
    # --- STEP 2: Activate virtual environment ---
    echo "Activating virtual environment for model training..."
    deactivate
    if [ "$MODEL_TRAINING" == "yes" ]; then
        echo "Model training is set to 'yes'. Proceeding with model training..."
        if [ "$MODEL" == "lda" ];then
            source "/export/usuarios_ml4ds/ammesa/TFG-LLMs/.venv_lda/bin/activate"
            echo "Training monolingual LDA model..."
            MALLET_FOLDER="/export/usuarios_ml4ds/ammesa/LDA_folder/$SOURCE_FILE"
            python3 -m train_LDA \
                --input_path $NL_DEST_PATH/$SOURCE_FILE \
                --mallet_folder $MALLET_FOLDER \
                --num_topics $N_TOPICS \
                --lang $LANG1 
            echo "Mallet files generated!"
            echo "Starting experimentation process"
        elif [ "$MODEL" == "zeroshot" ]; then
            source "/export/usuarios_ml4ds/ammesa/TFG-LLMs/.venv_zs/bin/activate"
            echo "Zero-Shot Topic Model selected!"
            python3 -m train_ZS\
                --path_folder $NL_DEST_PATH \
                --source_file $SOURCE_FILE \
                --num_topics $N_TOPICS \
                --lang1 $LANG1 \
                --lang2 $LANG2 \
                --bilingual $MODE

            MALLET_FOLDER="/export/usuarios_ml4ds/ammesa/ZS_results/$SOURCE_FILE"
        else
            echo "Invalid option"
            exit 1
        fi
    else
        echo "Model training is set to 'no'. Skipping model training."
    fi
elif [ "$MODE" == "bilingual" ]; then
    if [ "$PREPROCESS" == "yes" ]; then
    echo "Processing english files!"
    python3 -m src.NLPipe.nlpipe \
        --source_path /export/usuarios_ml4ds/ammesa/Data/1_segmented_data/micro_unaligned/$SOURCE_FILE \
        --source $SOURCE_FILE\
        --destination_path "$NL_DEST_PATH/$LANG1" \
        --stw_path /export/usuarios_ml4ds/ammesa/TFG-LLMs/src/topic_modeling/stops \
        --lang $LANG1
    
    echo "Processing spanish files!"
    python3 -m src.NLPipe.nlpipe \
        --source_path /export/usuarios_ml4ds/ammesa/Data/1_segmented_data/micro_unaligned/$SOURCE_FILE2 \
        --source $SOURCE_FILE2\
        --destination_path "$NL_DEST_PATH/$LANG2" \
        --stw_path /export/usuarios_ml4ds/ammesa/TFG-LLMs/src/topic_modeling/stops \
        --lang $LANG2
    fi
    if [ "$MODEL_TRAINING" == "yes" ]; then
        echo "Model training is set to 'yes'. Proceeding with model training..."

        if [ "$MODEL" == "zeroshot" ]; then
            deactivate
            source "/export/usuarios_ml4ds/ammesa/TFG-LLMs/.venv_zs/bin/activate"
            echo "Zero-Shot Topic Model selected!"
            python3 -m train_ZS\
                --path_folder $NL_DEST_PATH \
                --source_file $SOURCE_FILE \
                --num_topics $N_TOPICS \
                --lang1 $LANG1 \
                --lang2 $LANG2 \
                --bilingual $MODE

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
        echo "Model training is set to 'no'. Skipping model training."
    fi
else
    echo "Error: Invalid mode specified: $MODE"
    exit 1
fi
echo "Sucessful execution"
# --- STEP 2: Activate virtual environment ---
echo "Activating virtual environment for model training..."

if [ "$EVALUATION" == "yes" ]; then
    MALLET_FOLDER="/export/usuarios_ml4ds/ammesa/ZS_results/$SOURCE_FILE/n_topics_9"
    source "/export/usuarios_ml4ds/ammesa/TFG-LLMs/.venv_idx/bin/activate"
    K=$(cat "/export/usuarios_ml4ds/ammesa/TFG-LLMs/k_value.txt")
    POLY_PATH=$(cat polypath.txt)

    echo "Performing IRQ processes"
    python3 -m indexing_script \
        --input_path "$POLY_PATH" \
        --mallet_folder  $MALLET_FOLDER \
        --question_folder $QUESTION_FOLDER \
        --k "$K" \
        --bilingual $MODE \
        --model $MODEL \
        --lang1 $LANG1 \
        --lang2 $LANG2 
fi