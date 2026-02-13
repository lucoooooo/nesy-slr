#!/bin/bash

#./run_comparison.sh 1e-3 data model/mnist_sum_2 32 123
#./run_comparison.sh 1e-3 data model/mnist_sum_2 32 123 no_train

DATA_PATH=$2
MODEL_SAVE_PATH=$3
LR=$1
BATCH_SIZE=$4
SEED=$5
NO_TRAIN=$6

#DOMIKNOWS

./domiknows/run.sh PrimalDual $LR "./domiknows/${DATA_PATH}" "./domiknows/${MODEL_SAVE_PATH}" $BATCH_SIZE $SEED $NO_TRAIN

#NEURAL BASELINE

if [ "$NO_TRAIN" != "no_train" ]; then
    echo "Avvio script baseline con training annesso"
    python3 main.py --lr $LR --epochs 20 --modeldir "${MODEL_SAVE_PATH}" --datadir "${DATA_PATH}" --batch_size $BATCH_SIZE --seed $SEED 
else
    echo "Avvio script baseline senza training"
    python3 main.py --lr $LR --epochs 20 --modeldir ${MODEL_SAVE_PATH} --datadir ${DATA_PATH} --batch_size_train $BATCH_SIZE --batch_size_test $BATCH_SIZE --seed $SEED --no_train
fi

#salvataggio risultati comparison

python3 analyze_results.py --resultdir ./results --nesy_datadir ./domiknows/data --neural_datadir ./data --method PrimalDual