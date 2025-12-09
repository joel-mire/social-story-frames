#!/bin/bash

# Check if both arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <dataset_dir> <output_dir> <adapter_path>"
    exit 1
fi

# Assign command line arguments to variables
DATASET_DIR=$1
OUTPUT_DIR=$2
ADAPTER_PATH=$3

llamafactory-cli train \
    --stage sft \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --quantization_method bnb \
    --template llama3 \
    --flash_attn auto \
    --dataset_dir "$DATASET_DIR" \
    --eval_dataset val \
    --cutoff_len 2048 \
    --max_samples 100000 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate True \
    --max_new_tokens 2048 \
    --top_p 1 \
    --temperature 0.01 \
    --output_dir "$OUTPUT_DIR" \
    --trust_remote_code True \
    --do_predict True \
    --adapter_name_or_path "$ADAPTER_PATH" \

