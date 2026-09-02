import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "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"

os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_API_KEY"] = "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"
os.environ["WANDB_PROJECT"] = "mol-llm"
import sys
import logging
from dataclasses import dataclass, field

import yaml
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
)
from trl import (
    TrlParser, ScriptArguments, ModelConfig, 
    GRPOConfig, GRPOTrainer,
)

from src.utils import compute_mol_metrics


VERSION = "v1.1"

CONFIG_TRAINING_FILE = "configs/model_training_schema.yaml"
with open(CONFIG_TRAINING_FILE, "r") as f:
    CONFIG_TRAINING = yaml.safe_load(f)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout
)


@dataclass
class ScriptArgs(ScriptArguments):
    task_name: str = field(
        default=None,
        metadata={"help": "Path to the adapter directory for PEFT. If None, will not use PEFT."}
    )
    base_model_name: str = field(
        default=None,
        metadata={"help": "Path to the model directory for PEFT. If None, will not use PEFT."}
    )
    parent_training_method: str = field(
        default="sft_lora",
        metadata={"help": "Parent training method: sft, base, etc."}
    )
    training_method: str = field(
        default="sft_lora",
        metadata={"help": "Training method: sft, rlhf, etc."}
    )


def comprehensive_reward_func(prompts, completions, **kwargs):
    rewards = [None] * len(completions)
    output_rep = kwargs.get("output_rep", [])
    output_rep_type = kwargs.get("output_rep_type", [])
    if not output_rep or not output_rep_type:
        logger.warning("🤔🤔 No output_rep or output_rep_type provided in kwargs...")
        return rewards
    
    for i, (completion, gt_output, output_type) in enumerate(zip(completions, output_rep, output_rep_type)):
        completion: str = str(completion[0]["content"] if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict) and "content" in completion[0] else completion)
        try:
            results: dict = compute_mol_metrics(gt_output, completion, output_type)
        except Exception as e:
            logger.error(f"Error computing metrics for completion {i}: {e}")
            rewards[i] = None
            continue

        if not results["is_pred_valid"]:
            rewards[i] = -1.0  # Penalty for invalid output (rare case)
        elif results["is_inchikey_match"]:
            rewards[i] = 1.0   # Reward for correct prediction
        else:
            rewards[i] = 0.0   # Neutral for valid but wrong (most common case)

    return rewards


def get_reward_funcs() ->list:
    reward_funcs = [comprehensive_reward_func]
    return reward_funcs


def prepare_ocr_dataset(
        script_args: ScriptArguments, 
        grpo_config: GRPOConfig,
        model_config: ModelConfig,
    ) -> Dataset:
    
    raw_data_hf = load_dataset(f"kdeng03/mol-rep-ocr-{VERSION}", split="train")

    # def messages_map_func(sample):
    #     img = sample["input_rep"]
    #     prompt = sample["prompt"]
    #     completion = sample["completion"]

    #     message = [
    #         {
    #             "role": "user", 
    #             "content": [
    #                 {"type": "image"},
    #                 {"type": "text", "text": prompt},
    #             ]
    #         },
    #         {"role": "assistant", "content": completion}
    #     ]
    #     return {
    #         "image": img,
    #         "messages": message,
    #     }

    # data_hf = raw_data_hf.map(messages_map_func, remove_columns=raw_data_hf.column_names)
    
    def prompt_completion_map_func(sample):
        img = sample["input_rep"]
        prompt = sample["prompt"]
        prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        completion = sample["completion"]
        completion = [
            {
                "role": "assistant",
                "content": completion,
            },
        ]

        return {
            "image": img,
            "prompt": prompt,
            "completion": completion,
        }

    data_hf = raw_data_hf.map(prompt_completion_map_func)
    
    return data_hf


def prepare_ocr_eval_dataset(
        script_args: ScriptArguments, 
        grpo_config: GRPOConfig,
        model_config: ModelConfig,
    ) -> Dataset:
    
    raw_data_hf = load_dataset(f"kdeng03/mol-rep-ocr-{VERSION}", split="valid")
    
    def prompt_completion_map_func(sample):
        img = sample["input_rep"]
        prompt = sample["prompt"]
        prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        completion = sample["completion"]
        completion = [
            {
                "role": "assistant",
                "content": completion,
            },
        ]

        return {
            "image": img,
            "prompt": prompt,
            "completion": completion,
        }

    data_hf = raw_data_hf.map(prompt_completion_map_func)
    
    return data_hf


def prepare_conversion_dataset(
        script_args: ScriptArguments, 
        grpo_config: GRPOConfig,
        model_config: ModelConfig,
    ) -> Dataset:

    assert VERSION == "v1.1", "Only v1.1 supports GRPO training dataset preparation..."
    raw_data_hf = load_dataset(f"kdeng03/mol-rep-conversion-{VERSION}", split="train")

    # raw_data_hf = load_dataset(f"kdeng03/mol-rep-conversion-{VERSION}", split="valid")
    # sft_train_raw_data_hf = load_dataset(f"kdeng03/mol-rep-conversion-{VERSION}", split="train")
    # sft_train_raw_data_hf = sft_train_raw_data_hf.shuffle(seed=42).select(range(240))  # Use a subset of the SFT training data for GRPO training

    # ### Testing small batch
    # logger.warning(f"📢📢 Using small batch for GRPO training dataset preparation. Only 50 samples in total.")
    # raw_data_hf = raw_data_hf.shuffle(seed=42).select(range(45))  # Use a subset of the extra training data for GRPO training
    # sft_train_raw_data_hf = sft_train_raw_data_hf.shuffle(seed=42).select(range(5))  # Use a subset of the SFT training data for GRPO training

    # concat_data_hf = concatenate_datasets([raw_data_hf, sft_train_raw_data_hf])

    ### Trained on mixed difficulty subset of validset for exploring
    # logger.warning(f"📢📢 Using mixed difficulty subset of validset (🧐🧐 exploring) for GRPO training dataset preparation...")
    # INDICES_FILE = "results/pass_at_k/mol-rep-conversion/v1.1/valid/qwen3_vl_4b_i_sft_lora_ocr_conversion_pass_at_16_mixed_indices.json"
    # import json
    # with open(INDICES_FILE, "r") as f:
    #     indices: list = json.load(f)

    # logger.warning(f"📢📢 Using mixed difficulty subset of trainset (🧐🧐 exploring) for GRPO training dataset preparation...")
    # INDICES_FILE = "results/pass_at_k/mol-rep-conversion/v1.1/train/qwen3_vl_4b_i_sft_lora_ocr_conversion_pass_at_16_mixed_indices.json"
    # import json
    # with open(INDICES_FILE, "r") as f:
    #     indices: list = json.load(f)

    # midxed_data_hf = raw_data_hf.select(indices)

    # logger.warning(f"📢📢 Using mixed subset of trainset (🧐🧐 exploring) for GRPO training dataset preparation...")
    # INDICES_FILE = "results/pass_at_k/mol-rep-conversion/v1.1/train/qwen3_vl_4b_i_sft_lora_ocr_conversion_pass_at_16_indices.json"
    # import json
    # with open(INDICES_FILE, "r") as f:
    #     indices_dict: list = json.load(f)

    logger.warning(f"📢📢 Using mixed subset of trainset (🧐🧐 exploring) for GRPO training dataset preparation...")
    INDICES_FILE = "results/pass_at_k/mol-rep-conversion/v1.1/train/qwen3_vl_4b_i_sft_lora_ocr_conversion_pass_at_8_indices.json"
    import json
    with open(INDICES_FILE, "r") as f:
        indices_dict: list = json.load(f)

    easy_indices = indices_dict["easy"]
    medium_indices = indices_dict["medium"]
    hard_indices = indices_dict["hard"]
    impossible_indices = indices_dict["impossible"]

    # ### medium + hard + impossible = 283 datapoints, easy = 53 datapoints, total = 336 datapoints (evenly divided by 8)
    # import random
    # random.seed(42)

    # subset_easy_indices = random.sample(easy_indices, 53) if len(easy_indices) > 53 else easy_indices

    ### medium + hard + impossible = 283 datapoints, easy = 53 datapoints, total = 336 datapoints (evenly divided by 8)
    import random
    random.seed(42)

    subset_easy_indices = random.sample(easy_indices, 19) if len(easy_indices) > 19 else easy_indices

    final_indices = subset_easy_indices + medium_indices + hard_indices + impossible_indices

    mixed_data_hf = raw_data_hf.select(final_indices)
    logger.info(f"📢📢 Mixed subset of trainset for GRPO training dataset preparation: {len(mixed_data_hf)} samples (easy: {len(subset_easy_indices)}, medium: {len(medium_indices)}, hard: {len(hard_indices)}, impossible: {len(impossible_indices)})")

    concat_data_hf = mixed_data_hf

    def prompt_completion_map_func(sample):
        prompt = sample["prompt"]
        prompt = [{"role": "user", "content": prompt}]

        completion = sample["completion"]
        completion = [{"role": "assistant", "content": completion}]

        return {
            "prompt": prompt,
            "completion": completion,
        }

    data_hf = concat_data_hf.map(prompt_completion_map_func)

    return data_hf


def prepare_conversion_eval_dataset(
        script_args: ScriptArguments, 
        grpo_config: GRPOConfig, 
        model_config: ModelConfig,
    ) -> Dataset:

    raw_data_hf = load_dataset(f"kdeng03/mol-rep-conversion-{VERSION}", split="valid")
    # raw_data_hf = load_dataset(f"kdeng03/mol-rep-conversion-{VERSION}", split="test")

    ### Testing small batch
    small_size = 220
    logger.warning(f"📢📢 Using small batch ({small_size}) for GRPO training dataset preparation...")
    raw_data_hf = raw_data_hf.shuffle(seed=42).select(range(small_size))

    def prompt_completion_map_func(sample):
        prompt = sample["prompt"]
        prompt = [{"role": "user", "content": prompt}]

        completion = sample["completion"]
        completion = [{"role": "assistant", "content": completion}]

        return {
            "prompt": prompt,
            "completion": completion,
        }
    
    data_hf = raw_data_hf.map(prompt_completion_map_func)

    return data_hf
    

def main(
        script_args: ScriptArguments, 
        grpo_config: GRPOConfig, 
        model_config: ModelConfig,
    ) -> None:
    logger.info(f"🚀🚀 Script starts...")

    # Init configs from yaml files
    full_task_name = script_args.task_name
    base_model_name = script_args.base_model_name
    parent_training_method = script_args.parent_training_method
    training_method = script_args.training_method

    try:
        task_name = CONFIG_TRAINING[full_task_name]["code_name"]
    except KeyError:
        raise ValueError(f"Task name '{full_task_name}' not found in CONFIG_TRAINING.")
    output_model_name = f"{base_model_name}_{training_method}_{task_name}_from_{parent_training_method}"
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_model_dir = f"ckpt/{task_name}/grpo/{output_model_name}/{VERSION}/{timestamp}/"
    print(f"{output_model_dir=}")
    run_name = f"{output_model_name}_{VERSION}_{timestamp}"

    try:
        config = CONFIG_TRAINING[full_task_name]["rl"]["grpo"][output_model_name]
    except KeyError:
        raise ValueError(f"Configuration for '{output_model_name}' not found in CONFIG_TRAINING.")

    grpo_configs = config.get("grpo_configs", {})
    lora_configs = config.get("lora_configs", None)
    trainer_configs = config.get("trainer_configs", {})
    earlystoppingcallback_default_configs = {
        "early_stopping_patience": 2,
        "early_stopping_threshold": 0.0,
    }

    match task_name:
        case "ocr":
            train_dataset = prepare_ocr_dataset(script_args, grpo_config, model_config)
            eval_dataset = prepare_ocr_eval_dataset(script_args, grpo_config, model_config)
        case "conversion":
            train_dataset = prepare_conversion_dataset(script_args, grpo_config, model_config)
            eval_dataset = prepare_conversion_eval_dataset(script_args, grpo_config, model_config)
        case _:
            raise ValueError(f"Unknown task_name: {task_name}")

    # Prepare fallback lora configs
    peft_config = LoraConfig(
        r=lora_configs.get("r", 16),
        lora_alpha=lora_configs.get("lora_alpha", 32),
        lora_dropout=lora_configs.get("lora_dropout", 0.05),
        target_modules=lora_configs.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]),
    ) if lora_configs else None
    if peft_config:
        logger.info(f"Using LoRA with config: {peft_config=}")

    # Prepare fallback grpo configs
    training_args = GRPOConfig(
        # For GRPOConfig
        model_init_kwargs=grpo_configs.get("model_init_kwargs", {"dtype": "bfloat16"}),

        num_generations=grpo_configs.get("num_generations", 8),
        num_generations_eval=grpo_configs.get("num_generations_eval", 8),
        max_completion_length=grpo_configs.get("max_completion_length", 256),

        use_vllm=grpo_configs.get("use_vllm", False),
        vllm_mode=grpo_configs.get("vllm_mode", "colocate"),
        vllm_model_impl=grpo_configs.get("vllm_model_impl", "vllm"),

        ### Params for colocate
        vllm_gpu_memory_utilization=grpo_configs.get("vllm_gpu_memory_utilization", 0.3),
        vllm_max_model_length=grpo_configs.get("vllm_max_model_length", 4096),
        vllm_tensor_parallel_size=grpo_configs.get("vllm_tensor_parallel_size", 1),
        vllm_enable_sleep_mode=grpo_configs.get("vllm_enable_sleep_mode", True),

        use_transformers_continuous_batching=grpo_configs.get("use_transformers_continuous_batching", False),

        beta=grpo_configs.get("beta", 0.0),
        num_iterations=grpo_configs.get("num_iterations", 1), # mu in alg.
        epsilon=grpo_configs.get("epsilon", 0.2), # Clip range for PPO
        importance_sampling_level=grpo_configs.get("importance_sampling_level", "token"), # "token" or "sequence"
        reward_weights=grpo_configs.get("reward_weights", None),
        multi_objective_aggregation=grpo_configs.get("multi_objective_aggregation", "sum_then_normalize"), # "sum_then_normalize" or "normalize_then_sum"
        scale_rewards=grpo_configs.get("scale_rewards", "group"),
        loss_type=grpo_configs.get("loss_type", "dapo"), # "dapo" or "grpo" or "dr_grpo" or "sapo"
        mask_truncated_completions=grpo_configs.get("mask_truncated_completions", False),
        sync_ref_model=grpo_configs.get("sync_ref_model", False),
        top_entropy_quantile=grpo_configs.get("top_entropy_quantile", 0.2), ### Core param for molecule generation task
        entropy_coef=grpo_configs.get("entropy_coef", 0.01),
        use_adaptive_entropy=grpo_configs.get("use_adaptive_entropy", True),
        entropy_target=grpo_configs.get("entropy_target", 0.5),  # Short answer task: 0.5 nats is reasonable
        # entropy_coef_min=grpo_configs.get("entropy_coef_min", 0.001),
        # entropy_coef_max=grpo_configs.get("entropy_coef_max", 0.1),
        # entropy_coef_delta=grpo_configs.get("entropy_coef_delta", 0.001),  # Smaller step for stability
        vllm_importance_sampling_correction=grpo_configs.get("vllm_importance_sampling_correction", True),

        log_completions=grpo_configs.get("log_completions", True),
        num_completions_to_print=grpo_configs.get("num_completions_to_print", None),
        log_unique_prompts=grpo_configs.get("log_unique_prompts", False),

        # For TrainingArguments
        output_dir=output_model_dir,

        per_device_train_batch_size=grpo_configs.get("per_device_train_batch_size", 1),
        num_train_epochs=grpo_configs.get("num_train_epochs", 3),

        learning_rate=grpo_configs.get("learning_rate", 1e-5),
        lr_scheduler_type=grpo_configs.get("lr_scheduler_type", "linear"),
        warmup_steps=grpo_configs.get("warmup_steps", 0.03),

        gradient_accumulation_steps=grpo_configs.get("gradient_accumulation_steps", 128),
        max_grad_norm=grpo_configs.get("max_grad_norm", 1.0),

        gradient_checkpointing=grpo_configs.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs=grpo_configs.get("gradient_checkpointing_kwargs", {"use_reentrant": False}),

        use_liger_kernel=grpo_configs.get("use_liger_kernel", False),

        logging_strategy=grpo_configs.get("logging_strategy", "steps"),
        logging_steps=grpo_configs.get("logging_steps", 10),

        report_to=grpo_configs.get("report_to", "wandb"),
        run_name=grpo_configs.get("run_name", run_name),

        eval_strategy=grpo_configs.get("eval_strategy", "epoch"),
        eval_steps=grpo_configs.get("eval_steps", None),
        per_device_eval_batch_size=grpo_configs.get("per_device_eval_batch_size", 8), # To match eval num generations
        eval_on_start=grpo_configs.get("eval_on_start", True),
        eval_accumulation_steps=grpo_configs.get("eval_accumulation_steps", 1),

        save_strategy=grpo_configs.get("save_strategy", "epoch"),
        save_steps=grpo_configs.get("save_steps", None),

        load_best_model_at_end=grpo_configs.get("load_best_model_at_end", True),
        metric_for_best_model=grpo_configs.get("metric_for_best_model", "eval_reward"),
        greater_is_better=grpo_configs.get("greater_is_better", True),    
    )
    logger.info(f"Using GRPO training args: {training_args=}")

    # Prepare fallback trainer
    trainer = GRPOTrainer(
        model=trainer_configs.get("model", None),
        reward_funcs=get_reward_funcs(),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        quantization_config=None,
        peft_config=peft_config,
        callbacks=[EarlyStoppingCallback(**trainer_configs.get("callbacks", {}).get("EarlyStoppingCallback", earlystoppingcallback_default_configs))] if trainer_configs.get("callbacks", None) else None,
    )

    # logger.info(f"🕙🕙 Start training...")
    logger.info(f"🕙🕙 Start training... Resume from checkpoint-168...")
    # trainer.train(resume_from_checkpoint="ckpt/conversion/grpo/qwen3_vl_4b_i_grpo_lora_conversion_from_sft_lora/v1.1/20260822_180426/checkpoint-168")
    trainer.train()
    logger.info(f"✅✅ Training finished. Saving model to {training_args.output_dir}...")
    trainer.save_model(training_args.output_dir) # Save full model or adapter


if __name__ == "__main__":
    parser = TrlParser([ScriptArgs, GRPOConfig, ModelConfig])
    script_args, grpo_config, model_config = parser.parse_args_and_config()
    main(script_args, grpo_config, model_config)
    logger.info(f"✅ Script ends.")
    