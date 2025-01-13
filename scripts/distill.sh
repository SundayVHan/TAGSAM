#!/bin/bash

DATASET_NAME="art"
GPU=0
INIT_PAIR="random"
SEEDS=(42 43 44)
NUM_SYNS=(100 200 500)

# python buffer.py --dataset_name $DATASET_NAME --gpu $GPU --num_epochs 3

for NUM_SYN in "${NUM_SYNS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    python distill-sim.py --gpu $GPU --init_pair $INIT_PAIR --seed $SEED --dataset_name $DATASET_NAME --num_syn $NUM_SYN
  done
done

for NUM_SYN in "${NUM_SYNS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    python distill-sim.py --gpu $GPU --init_pair "summary" --seed $SEED --dataset_name $DATASET_NAME --num_syn $NUM_SYN
  done
done