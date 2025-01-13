#!/bin/bash

DATASET_NAME="photo"
GPU=1
SEEDS=(42 43 44)
NUM_SYNS=(100 200 500)

#for NUM_SYN in "${NUM_SYNS[@]}"; do
#  for SEED in "${SEEDS[@]}"; do
#    python coreset.py --gpu $GPU --seed $SEED --dataset_name $DATASET_NAME --num_syn $NUM_SYN --init_pair "herding"
#    python coreset.py --gpu $GPU --seed $SEED --dataset_name $DATASET_NAME --num_syn $NUM_SYN --init_pair "k_center"
#  done
#done
#
#DATASET_NAME="computer"
#for NUM_SYN in "${NUM_SYNS[@]}"; do
#  for SEED in "${SEEDS[@]}"; do
#    python coreset.py --gpu $GPU --seed $SEED --dataset_name $DATASET_NAME --num_syn $NUM_SYN --init_pair "herding"
#    python coreset.py --gpu $GPU --seed $SEED --dataset_name $DATASET_NAME --num_syn $NUM_SYN --init_pair "k_center"
#  done
#done

DATASET_NAME="arxiv"
for NUM_SYN in "${NUM_SYNS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    python coreset.py --gpu $GPU --seed $SEED --dataset_name $DATASET_NAME --num_syn $NUM_SYN --init_pair "herding"
    python coreset.py --gpu $GPU --seed $SEED --dataset_name $DATASET_NAME --num_syn $NUM_SYN --init_pair "k_center"
  done
done