#!/bin/bash

DATASET_NAME="art"
GPU=1
SEEDS=(42 43 44)
NUM_SYNS=(100 200 500)

python buffer-mtt.py --dataset_name $DATASET_NAME --gpu $GPU --num_epochs 3

for NUM_SYN in "${NUM_SYNS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    python distill-mtt.py --gpu $GPU --seed $SEED --dataset_name $DATASET_NAME --num_syn $NUM_SYN --max_start_epoch 3 --match_epoch 1
  done
done