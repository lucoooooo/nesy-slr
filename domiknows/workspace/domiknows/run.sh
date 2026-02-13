#!/bin/bash

DATA_PATH=$3
MODEL_SAVE_PATH=$4
METHOD=$1
LR=$2
BATCH_SIZE=$5
SEED=$6
NO_TRAIN=$7
models=("basic" "b0" "b3" )

for model in "${models[@]}"
do  
    case "$model" in
        ("b0" | "b3")
            LR=1e-4
            ;;
    esac
    
    if [ "$NO_TRAIN" != "no_train" ]; then
        echo "Inizio training per $model..."
        python3 ./domiknows/train.py --method ${METHOD} --num_train 500 --model $model --lr $LR --epochs 20 --modeldir ${MODEL_SAVE_PATH} --datadir ${DATA_PATH} --seed $SEED
    fi
    echo "Inizio testing per $model..."
    python3 ./domiknows/test.py --method ${METHOD} --model $model --modeldir "${MODEL_SAVE_PATH}" --datadir "${DATA_PATH}" --seed $SEED
done

echo "Training/Testing con Domiknows completati"