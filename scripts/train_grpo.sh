#!/usr/bin/env bash
# GRPO (RL) training of the GUI critic with ms-swift + vLLM.
#
# This stage uses two custom reward functions (format + link) that must be
# provided via an ms-swift external plugin. Point PLUGIN at a plugin.py that
# registers `external_gui_agent_critic_format` and `external_gui_agent_critic_link`
# (these score output format validity and correct/wrong judgement agreement).
set -e

MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
DATASET=${DATASET:-data/grpo_train.json}
PLUGIN=${PLUGIN:-path/to/ms-swift/plugin/plugin.py}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-prompts/gui_critic.txt}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/critic-grpo}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=2822400 \
NPROC_PER_NODE=7 \
swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL" \
    --external_plugins "$PLUGIN" \
    --reward_funcs external_gui_agent_critic_format external_gui_agent_critic_link \
    --use_vllm true \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 8192 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset "$DATASET" \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --eval_strategy 'no' \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 7 \
    --temperature 0.9 \
    --system "$SYSTEM_PROMPT" \
    --deepspeed zero2 \
    --log_completions true
