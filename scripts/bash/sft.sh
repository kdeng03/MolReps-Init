#!/bin/bash

#SBATCH --job-name=sft
#SBATCH --output=logs/sft-%j.log
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=24:00:00

nvidia-smi

# source .env

source $(conda info --base)/etc/profile.d/conda.sh
conda activate vllm
# conda activate qwen3_5

# module load cuda
# module load brics/openmpi/4.1.7
# export CC=/usr/bin/gcc-12
# export CXX=/usr/bin/g++-12

# DATASET_ID=(
#     # kdeng03/VRRPI-Diag-train
#     kdeng03/VRRPI-Diag-train-aug
#     # kdeng03/Spa-500
#     # kdeng03/VRRPI-Bench-train
#     # kdeng03/VRRPI-Bench-train-aug
# )
# MODEL_ID=(
#     # Qwen/Qwen3-VL-4B-Instruct
#     Qwen/Qwen3-VL-8B-Instruct
#     # Qwen/Qwen3-VL-32B-Instruct
# )

# PEFT_CONFIG=(
#     --use_peft true
#     --lora_r 128
#     --lora_alpha 32
#     --lora_dropout 0.05
#     --lora_target_modules "q_proj" "k_proj" "v_proj" "o_proj" "up_proj" "down_proj" "gate_proj"
# )

# export DEBUG="true"
# export WANDB_PROJECT="Spa"
# export WANDB_MODE="offline"

# LR=2e-4
# EPOCH=1

# for DATASET_ID in "${DATASET_ID[@]}"; do
#     for MODEL_ID in "${MODEL_ID[@]}"; do
#         TMP_TIMESTEP=$(date +%s)
#         OUTPUT_DIR="ckpt/${MODEL_ID##*/}-SFT-${DATASET_ID##*/}-${TMP_TIMESTEP}"
        
#         # Debug: try sft_fixed.py
#         accelerate launch \
#             --config_file config/zero3.yaml \
#             scripts/main/sft.py \
#             --dataset_name "$DATASET_ID" \
#             --model_name_or_path "$MODEL_ID" \
#             --dtype bfloat16 \
#             --attn_implementation sdpa \
#             --output_dir "$OUTPUT_DIR" \
#             --learning_rate $LR \
#             --per_device_train_batch_size 1 \
#             --num_train_epoch $EPOCH \
#             --gradient_accumulation_steps 8 \
#             --max_grad_norm 1 \
#             --gradient_checkpointing true \
#             --bf16 true \
#             --report_to wandb \
#             --run_name "${MODEL_ID##*/}-SFT-${DATASET_ID##*/}-${TMP_TIMESTEP}" \
#             --save_steps 100 \
#             --logging_steps 8 \
#             --warmup_ratio 0.03 \
#             "${PEFT_CONFIG[@]}"

#     done
# done


task_name=mol_rep_conversion
# task_name=mol_rep_ocr
# task_name=mol_rep_ocr_conversion

# model_name=qwen3_4b_i
model_name=qwen3_vl_4b_i

training_method=sft_lora

mkdir -p logs/sft/v1.1/$task_name

### Single GPU training
PYTHONPATH="$(pwd)" \
    accelerate launch \
    scripts/sft.py \
    --task_name "$task_name" \
    --model_name "$model_name" \
    --training_method "$training_method" | tee logs/sft/v1.1/$task_name/$model_name.log

# ### Distributed training with DeepSpeed
# PYTHONPATH="$(pwd)" \
#     accelerate launch \
#     --config_file configs/sft_zero3.yaml \
#     scripts/sft.py \
#     --task_name "$task_name" \
#     --model_name "$model_name" \
#     --training_method "$training_method"
