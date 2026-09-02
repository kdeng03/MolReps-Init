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


# model_name=qwen3_4b_i_sft_lora_conversion
# model_name=qwen3_vl_4b_i_sft_lora_conversion
# model_name=qwen3_vl_4b_i_sft_lora_ocr
# model_name=qwen3_vl_4b_i_sft_lora_ocr_conversion
model_name=qwen3_vl_4b_i_grpo_lora_from_sft_lora_ocr_conversion

mkdir -p logs/inference/v1.1/ablation_num_train_epoch

### Single GPU training
PYTHONPATH="$(pwd)" \
    python \
    scripts/ablation_num_train_epoch.py \
    --model_name "$model_name" | tee logs/inference/v1.1/ablation_num_train_epoch/$model_name.log
