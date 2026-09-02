#!/bin/bash

#SBATCH --job-name=grpo
#SBATCH --output=log/grpo-%j.log
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00

nvidia-smi

# source .env

source $(conda info --base)/etc/profile.d/conda.sh
conda activate vllm

# module load cudatoolkit
# module load brics/openmpi/4.1.7
# export CC=/usr/bin/gcc-12
# export CXX=/usr/bin/g++-12

# DATASET_IDS=(
#     # kdeng03/VRRPI-Diag
#     # kdeng03/Spa-500
#     kdeng03/VRRPI-Bench-train
# )
# MODEL_IDS=(
#     # Qwen/Qwen3-VL-4B-Instruct
#     Qwen/Qwen3-VL-8B-Instruct
#     # Qwen/Qwen3-VL-32B-Instruct
# )

# LR=2e-4
# EPOCH=2

# PEFT_CONFIG=(
#     --use_peft true
#     --lora_r 16
#     --lora_alpha 32
#     --lora_dropout 0.05
#     --lora_target_modules "q_proj" "k_proj" "v_proj" "o_proj" "up_proj" "down_proj" "gate_proj"
# )

# # USE_VLLM=true

# GRPO_CONFIG=(
#     --num_generations 8
#     --max_completion_length 1024
#     --shuffle_dataset true
#     --temperature 1.0
#     --beta 0.05
# )

# if [[ "${USE_VLLM:-}" == "true" ]]; then
#     GRPO_CONFIG+=(
#         --use_vllm true
#         --vllm_mode colocate
#         --vllm_max_model_length 32768
#     )
# fi

# prepare_config() {
#     local dataset_id="$1"
#     local model_id="$2"

#     # Frozen MLLM
#     TMP_TIMESTEP=$(date +%s)
#     OUTPUT_DIR="ckpt/${model_id##*/}-GRPO-${dataset_id##*/}-${TMP_TIMESTEP}"

#     # SFT MLLM
#     # OUTPUT_DIR=ckpt/Qwen3-VL-8B-Instruct-GRPO-Spa-500-1777177855

#     SCRIPT_ARG=(
#         --dataset_name "$dataset_id"
#         --dataset_train_split train
#         # --adapter_dir ckpt/Qwen3-VL-8B-Instruct-SFT-Spa-500-1776785995
#     )

#     MODEL_CONFIG=(
#         --model_name_or_path "$model_id"
#         --dtype bfloat16
#         --trust_remote_code true
#         --attn_implementation sdpa
#     )

#     TRAINING_ARGS=(
#         --output_dir "$OUTPUT_DIR"
#         --per_device_train_batch_size 2
#         --num_train_epochs "$EPOCH"
#         --learning_rate "$LR"
#         --gradient_accumulation_steps 8
#         --max_grad_norm 1
#         --bf16 true
#         --gradient_checkpointing true
#         --logging_steps 8
#         --report_to wandb
#         --run_name "${model_id##*/}-GRPO-${dataset_id##*/}-${TMP_TIMESTEP}"
#         --save_steps 64
#         --log_completions
#     )
# }

# for dataset_id in "${DATASET_IDS[@]}"; do
#     for model_id in "${MODEL_IDS[@]}"; do
#         prepare_config "$dataset_id" "$model_id"

#         accelerate launch \
#             scripts/main/grpo.py \
#             "${SCRIPT_ARG[@]}" \
#             "${MODEL_CONFIG[@]}" \
#             "${TRAINING_ARGS[@]}" \
#             "${GRPO_CONFIG[@]}" \
#             "${PEFT_CONFIG[@]}" \

#     done
# done


task_name=mol_rep_conversion
# task_name=mol_rep_ocr

# base_model_name=qwen3_4b_i
base_model_name=qwen3_vl_4b_i

# parent_training_method=base
parent_training_method=sft_lora

training_method=grpo_lora

mkdir -p logs/grpo_trl/v1.1/$task_name

### Single GPU training
PYTHONPATH="$(pwd)" \
    accelerate launch \
    scripts/grpo_trl.py \
    --task_name "$task_name" \
    --base_model_name "$base_model_name" \
    --parent_training_method "$parent_training_method" \
    --training_method "$training_method" | tee logs/grpo_trl/v1.1/$task_name/${base_model_name}_${training_method}_from_${parent_training_method}.log
