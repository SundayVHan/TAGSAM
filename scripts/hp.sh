#!/bin/bash

DATASET_NAME="computer"
GPU=1
INIT_PAIR="summary"
SEEDS=(42)
NUM_SYNS=(100)
NUM_SUMMARY=(2 4 8 16)
RATIOS=(40 50 60 70 80 90)

for seed in "${SEEDS[@]}"; do
    for num_syn in "${NUM_SYNS[@]}"; do
        for num_summary in "${NUM_SUMMARY[@]}"; do
            for ratio in "${RATIOS[@]}"; do
                python distill-sim.py --dataset_name ${DATASET_NAME} --gpu ${GPU} --seed ${seed} --num_syn ${num_syn} --num_summary ${num_summary} --ratio_summary ${ratio} --init_pair ${INIT_PAIR}
            done
        done
    done
done