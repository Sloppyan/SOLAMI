#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python pretrain_inference.py --part 0 &
CUDA_VISIBLE_DEVICES=1 python pretrain_inference.py --part 1 &
CUDA_VISIBLE_DEVICES=2 python pretrain_inference.py --part 2 &
CUDA_VISIBLE_DEVICES=3 python pretrain_inference.py --part 3 &

wait

echo "All processes have finished."