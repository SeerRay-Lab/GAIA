#!/usr/bin/env bash
# LoRA SFT of the GUI critic with ms-swift.
# After training, merge the adapter into the base model with src/merge_lora.py.
set -e

MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
DATASET=${DATASET:-data/sft_train.json}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/critic-sft-lora}

# 2822400 = 3600 * 28 * 28
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=2822400 \
NPROC_PER_NODE=8 \
swift sft \
    --deepspeed zero2 \
    --model "$MODEL" \
    --train_type lora \
    --dataset "$DATASET" \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 8 \
    --eval_steps 2000 \
    --save_steps 500 \
    --save_total_limit 20 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir "$OUTPUT_DIR" \
    --system "" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 32
