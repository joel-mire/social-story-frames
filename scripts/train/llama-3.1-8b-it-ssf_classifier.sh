#!/bin/bash

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <dataset_dir> <output_dir>"
    exit 1
fi

# Assign command line arguments to variables
DATASET_DIR=$1
OUTPUT_DIR=$2

# Run LLamaFactory training with the provided arguments
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template llama3 \
    --flash_attn auto \
    --dataset_dir "$DATASET_DIR" \
    --dataset train \
    --cutoff_len 1536 \
    --learning_rate 0.00005 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_ratio 0.1 \
    --packing False \
    --report_to wandb \
    --output_dir "$OUTPUT_DIR" \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj \
    --seed 25 \
    --train_on_prompt True