#!/bin/bash

DATA_DIR=(
    "SOLAMI/models/vla/infer_output/llama2-dlp-motiongpt-retrieval-final-0"
    "SOLAMI/models/vla/infer_output/llama2-dlp-motiongpt-retrieval-final-1"
    "SOLAMI/models/vla/infer_output/llama2-dlp-motiongpt-retrieval-final-2"
)

DATA_DIR2=(
    "SOLAMI/models/vla/infer_output/llama2-dlp-motiongpt-retrieval-final-3"
    "SOLAMI/models/vla/infer_output/llama2-dlp-motiongpt-retrieval-final-4"
    "SOLAMI/models/vla/infer_output/llama2-dlp-motiongpt-retrieval-final-0"
)

for i in "${!DATA_DIR[@]}"; do
    CUDA_VISIBLE_DEVICES=4 python conv_llama2_evaluation.py --data_dir ${DATA_DIR[$i]} &
    CUDA_VISIBLE_DEVICES=5 python conv_llama2_evaluation.py --data_dir ${DATA_DIR2[$i]} &
    wait
done
