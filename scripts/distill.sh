#!/bin/bash

DATASET_NAME="art"
GPU=6
INIT_PAIR="summary"
SEEDS=(42 43 44)
NUM_SYNS=(200)

for NUM_SYN in "${NUM_SYNS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    python distill-sim.py --gpu $GPU --init_pair $INIT_PAIR --seed $SEED --dataset_name $DATASET_NAME --num_syn $NUM_SYN
  done
done

