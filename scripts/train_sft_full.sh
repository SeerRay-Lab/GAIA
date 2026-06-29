#!/usr/bin/env bash
# Full-parameter SFT of the GUI critic with ms-swift.
# Set MODEL to a base Qwen2.5-VL and DATASET to your SFT json (see
# data/sample/critic_sft_sample.json for the record format).
set -e

MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
DATASET=${DATASET:-data/sft_train.json}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/critic-sft-full}

# 2116800 = 2700 * 28 * 28
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=2116800 \
NPROC_PER_NODE=8 \
swift sft \
    --deepspeed zero2 \
    --model "$MODEL" \
    --train_type full \
    --dataset "$DATASET" \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 16 \
    --eval_steps 1000 \
    --save_steps 500 \
    --save_total_limit 4 \
    --freeze_vit true \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir "$OUTPUT_DIR" \
    --system "" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 32
