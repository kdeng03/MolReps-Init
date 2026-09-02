#!/usr/bin/env bash
# GRPO | molecular representation | vLLM rollout | FSDP training | GPU
# VERL-based training for MolQwen3-VL-4B-Instruct
# Task: mol-rep-conversion (text-to-text molecular representation conversion)

set -xeuo pipefail

########################### user-adjustable ############################
# Device configuration
DEVICE=${DEVICE:-gpu}
MODEL_PATH=${MODEL_PATH:-kdeng03/MolQwen3-VL-4B-Instruct-SFT}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}

# Data configuration
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}  # Total prompts per global step
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}  # Must divide train_batch_size * rollout.n
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1300}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

# Training hyperparameters
ACTOR_LR=${ACTOR_LR:-5.0e-5}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}  # beta=0.0, no KL penalty
ENTROPY_COEFF=${ENTROPY_COEFF:-0.01}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
CLIP_RATIO=${CLIP_RATIO:-0.2}  # epsilon in TRL

# Rollout configuration
ROLLOUT_TP=${ROLLOUT_TP:-1}  # Tensor parallel size for vLLM
ROLLOUT_N=${ROLLOUT_N:-8}  # Number of generations per prompt
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}

# Logging and saving
SAVE_FREQ=${SAVE_FREQ:-1}
TEST_FREQ=${TEST_FREQ:-1}
PROJECT_NAME=${PROJECT_NAME:-mol-llm}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_vl_4b_i_grpo_lora_conversion_from_sft_lora_v1.1}

# Data paths (VERL expects parquet format)
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/mol-rep-conversion/train.parquet}
TEST_FILE=${TEST_FILE:-$HOME/data/mol-rep-conversion/val.parquet}

# Custom reward function
REWARD_FUNCTION=${REWARD_FUNCTION:-mol_reward.compute_score}

# Output directory
OUTPUT_DIR=${OUTPUT_DIR:-ckpt/conversion/grpo/qwen3_vl_4b_i_grpo_lora_conversion_from_sft_lora/v1.1/}
########################### end user-adjustable ########################

########################### derived defaults ###########################
n_devices_per_node=${NDEVICES_PER_NODE:-8}
case "${DEVICE}" in
    gpu)
        rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.5}
        ;;
    npu)
        export HCCL_CONNECT_TIMEOUT=1500
        export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
        export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
        export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
        rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.5}
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'."
        exit 1
        ;;
esac

# Set environment variables for HuggingFace and W&B
export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="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"
export WANDB_MODE="online"
export WANDB_API_KEY="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"
export WANDB_PROJECT="mol-llm"

########################### configuration arrays ########################
DATA=(
    data.train_files=${TRAIN_FILE}
    data.val_files=${TEST_FILE}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
)

MODEL=(
    actor_rollout_ref.actor.path=${MODEL_PATH}
    actor_rollout_ref.actor.use_remove_padding=true
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.optim.warmup_steps_ratio=${WARMUP_RATIO}
    actor_rollout_ref.actor.optim.total_epochs=${TOTAL_EPOCHS}
    actor_rollout_ref.actor.optim.lr_scheduler=constant_with_warmup
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO}
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=true
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.max_new_tokens=${MAX_RESPONSE_LENGTH}
    actor_rollout_ref.rollout.free_cache_engine=true
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=true
)

REWARD=(
    reward.reward_function=${REWARD_FUNCTION}
    reward.reward_model.enable=false
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=true
)

TRAINER=(
    trainer.balance_batch=true
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.default_local_dir=${OUTPUT_DIR}
    trainer.val_before_train=true
)

EXTRA=(
    actor_rollout_ref.actor.use_fused_kernels=true
    actor_rollout_ref.actor.fsdp_config.param_offload=false
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
    actor_rollout_ref.rollout.enforce_eager=false
    actor_rollout_ref.rollout.free_cache_engine=true
)

########################### launch ########################
# Run from the mol-llm repo root
cd "$(dirname "$0")/../.."

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${ALGORITHM[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
