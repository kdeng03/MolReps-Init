import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "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"

os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_API_KEY"] = "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"
os.environ["WANDB_PROJECT"] = "mol-llm"
import logging
from dataclasses import dataclass, field

import yaml
from datasets import load_dataset, Dataset
from peft import (
    LoraConfig,
)
from trl import (
    TrlParser, 
    ScriptArguments, SFTConfig, ModelConfig,
    SFTTrainer,
    get_peft_config,
)
from transformers import (
    EarlyStoppingCallback,
)


VERSION = "v1.1"

# CONFIG_INFER_FILE = "configs/model_inference_schema.yaml"
# with open(CONFIG_INFER_FILE, "r") as f:
#     CONFIG_INFER = yaml.safe_load(f)

CONFIG_TRAINING_FILE = "configs/model_training_schema.yaml"
with open(CONFIG_TRAINING_FILE, "r") as f:
    CONFIG_TRAINING = yaml.safe_load(f)


@dataclass
class ScriptArgs(ScriptArguments):
    task_name: str = field(
        default=None,
        metadata={"help": "Path to the adapter directory for PEFT. If None, will not use PEFT."}
    )
    model_name: str = field(
        default=None,
        metadata={"help": "Path to the model directory for PEFT. If None, will not use PEFT."}
    )
    training_method: str = field(
        default="sft_lora",
        metadata={"help": "Training method: sft, rlhf, etc."}
    )


def prepare_ocr_dataset(
        script_args: ScriptArguments, 
        sft_config: SFTConfig, 
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
        sft_config: SFTConfig, 
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
        sft_config: SFTConfig, 
        model_config: ModelConfig,
    ) -> Dataset:
    
    raw_data_hf = load_dataset(f"kdeng03/mol-rep-conversion-{VERSION}", split="train")

    # def messages_map_func(sample):
    #     prompt = sample["prompt"]
    #     completion = sample["completion"]

    #     message = [
    #         {"role": "user", "content": prompt},
    #         {"role": "assistant", "content": completion},
    #     ]
    #     return {
    #         "messages": message,
    #     }
    
    # data_hf = raw_data_hf.map(messages_map_func, remove_columns=raw_data_hf.column_names)

    def prompt_completion_map_func(sample):
        prompt = sample["prompt"]
        prompt = [{"role": "user", "content": prompt}]

        completion = sample["completion"]
        completion = [{"role": "assistant", "content": completion}]

        return {
            "prompt": prompt,
            "completion": completion,
        }

    ### Stage2: Trained on hard samples
    print(f"📢📢 Stage2: Using mixed subset of trainset for SFT training dataset preparation...")
    INDICES_FILE = "results/pass_at_k/mol-rep-conversion/v1.1/archive/train/qwen3_vl_4b_i_sft_lora_ocr_conversion_pass_at_16_indices.json"
    import json
    with open(INDICES_FILE, "r") as f:
        indices_dict: list = json.load(f)

    easy_indices = indices_dict["easy"]
    medium_indices = indices_dict["medium"]
    hard_indices = indices_dict["hard"]
    impossible_indices = indices_dict["impossible"]

    ### medium + hard + impossible = 283 datapoints, easy = 53 datapoints, total = 336 datapoints (evenly divided by 16)
    import random
    random.seed(42)

    subset_easy_indices = random.sample(easy_indices, 53) if len(easy_indices) > 53 else easy_indices

    final_indices = subset_easy_indices + medium_indices + hard_indices + impossible_indices

    print(f"📢📢 Final indices for SFT training dataset preparation: {len(final_indices)}")

    mixed_data_hf = raw_data_hf.select(final_indices)
    print(f"📢📢 Mixed subset of trainset for SFT training dataset preparation: {len(mixed_data_hf)} samples (easy: {len(subset_easy_indices)}, medium: {len(medium_indices)}, hard: {len(hard_indices)}, impossible: {len(impossible_indices)})")

    data_hf = mixed_data_hf.map(prompt_completion_map_func)

    return data_hf


def prepare_conversion_eval_dataset(
        script_args: ScriptArguments, 
        sft_config: SFTConfig, 
        model_config: ModelConfig,
    ) -> Dataset:
    
    raw_data_hf = load_dataset(f"kdeng03/mol-rep-conversion-{VERSION}", split="valid")

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


def main(script_args: ScriptArguments, sft_config: SFTConfig, model_config: ModelConfig) -> None:
    print(f"🚀🚀 Script starts...")
    
    ### Collect config names
    full_task_name = script_args.task_name
    parent_model_name = script_args.model_name
    training_method = script_args.training_method

    try:
        task_name = CONFIG_TRAINING[full_task_name]["code_name"]
    except KeyError:
        raise ValueError(f"Task name '{full_task_name}' not found in CONFIG_TRAINING.")
    output_model_name = f"{parent_model_name}_{training_method}_{task_name}"
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_model_dir = f"ckpt/{task_name}/{output_model_name}/{VERSION}/{timestamp}/"
    print(f"{output_model_dir=}")
    run_name = f"{output_model_name}_{VERSION}_{timestamp}"

    try:
        config = CONFIG_TRAINING[full_task_name]["sft"][output_model_name]
    except KeyError:
        raise ValueError(f"Configuration for '{output_model_name}' not found in CONFIG_TRAINING.")

    model_init_kwargs = config["model_init_kwargs"]
    sft_configs = config["sft_configs"]
    lora_configs = config.get("lora_configs", None)
    trainer_configs = config.get("trainer_configs", None)
    earlystoppingcallback_default_configs = {
        "early_stopping_patience": 2,
        "early_stopping_threshold": 0.0,
    }

    match task_name:
        case "ocr":
            training_dataset = prepare_ocr_dataset(script_args, sft_config, model_config)
            eval_dataset = prepare_ocr_eval_dataset(script_args, sft_config, model_config)

            peft_config = LoraConfig(
                r=lora_configs.get("r", 16),
                lora_alpha=lora_configs.get("lora_alpha", 32),
                lora_dropout=lora_configs.get("lora_dropout", 0.05),
                target_modules=lora_configs.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]),
            ) if lora_configs else None
            if peft_config:
                print(f"Using LoRA with config: {peft_config=}")

            training_args = SFTConfig(
                output_dir=output_model_dir,

                model_init_kwargs=sft_configs.get("model_init_kwargs", {"dtype": "bfloat16"}),

                num_train_epochs=sft_configs.get("num_train_epochs", 2),
                learning_rate=sft_configs.get("learning_rate", 1e-4),
                max_grad_norm=sft_configs.get("max_grad_norm", 1.0),

                per_device_train_batch_size=sft_configs.get("per_device_train_batch_size", 1),
                gradient_accumulation_steps=sft_configs.get("gradient_accumulation_steps", 8),

                activation_offloading=sft_configs.get("activation_offloading", False),
                gradient_checkpointing=sft_configs.get("gradient_checkpointing", True),
                gradient_checkpointing_kwargs=sft_configs.get("gradient_checkpointing_kwargs", {"use_reentrant": False}),

                max_length=sft_configs.get("max_length", None),
                packing=sft_configs.get("packing", False),
                packing_strategy=sft_configs.get("packing_strategy", "bfd"),
                # assistant_only_loss=sft_configs.get("assistant_only_loss", True), # Not supported for VLM

                report_to=sft_configs.get("report_to", "wandb"),
                run_name=sft_configs.get("run_name", run_name),

                use_liger_kernel=sft_configs.get("use_liger_kernel", False),

                logging_strategy=sft_configs.get("logging_strategy", "steps"),
                logging_steps=sft_configs.get("logging_steps", 8),

                eval_strategy=sft_configs.get("eval_strategy", "epoch"),
                eval_steps=sft_configs.get("eval_steps", None),
                per_device_eval_batch_size=sft_configs.get("per_device_eval_batch_size", 1),
                eval_accumulation_steps=sft_configs.get("eval_accumulation_steps", 1),

                save_strategy=sft_configs.get("save_strategy", "epoch"),
                save_steps=sft_configs.get("save_steps", None),

                load_best_model_at_end=sft_configs.get("load_best_model_at_end", True),
                metric_for_best_model=sft_configs.get("metric_for_best_model", "eval_loss"),
                greater_is_better=sft_configs.get("greater_is_better", False),
                
                warmup_steps=sft_configs.get("warmup_ratio", 0.03),
            )

            trainer = SFTTrainer(
                model=model_init_kwargs["model_name_or_path"],
                args=training_args,
                train_dataset=training_dataset,
                eval_dataset=eval_dataset,
                peft_config=peft_config,
                callbacks=[EarlyStoppingCallback(**trainer_configs.get("callbacks", {}).get("EarlyStoppingCallback", earlystoppingcallback_default_configs))] if trainer_configs.get("callbacks", None) else None,
            )

            print(f"🕙🕙 Start training...")
            trainer.train()
            print(f"✅✅ Training finished. Saving model to {training_args.output_dir}...")
            trainer.save_model(training_args.output_dir) # Save full model or adapter

        case "conversion":
            training_dataset = prepare_conversion_dataset(script_args, sft_config, model_config)
            eval_dataset = prepare_conversion_eval_dataset(script_args, sft_config, model_config)
            
            peft_config = LoraConfig(
                r=lora_configs.get("r", 16),
                lora_alpha=lora_configs.get("lora_alpha", 32),
                lora_dropout=lora_configs.get("lora_dropout", 0.05),
                target_modules=lora_configs.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]),
            ) if lora_configs else None
            if peft_config:
                print(f"Using LoRA with config: {peft_config=}")

            training_args = SFTConfig(
                output_dir=output_model_dir,

                model_init_kwargs=sft_configs.get("model_init_kwargs", {"dtype": "bfloat16"}),

                num_train_epochs=sft_configs.get("num_train_epochs", 2),
                max_length=sft_configs.get("max_length", None),

                per_device_train_batch_size=sft_configs.get("per_device_train_batch_size", 1),
                gradient_accumulation_steps=sft_configs.get("gradient_accumulation_steps", 8),

                learning_rate=sft_configs.get("learning_rate", 1e-4),
                max_grad_norm=sft_configs.get("max_grad_norm", 1.0),
                
                gradient_checkpointing=sft_configs.get("gradient_checkpointing", True),
                gradient_checkpointing_kwargs=sft_configs.get("gradient_checkpointing_kwargs", {"use_reentrant": False}),
                
                packing=sft_configs.get("packing", False),
                packing_strategy=sft_configs.get("packing_strategy", "bfd"),
                # assistant_only_loss=sft_configs.get("assistant_only_loss", True), # Not supported for VLM

                report_to=sft_configs.get("report_to", "wandb"),
                run_name=sft_configs.get("run_name", run_name),

                logging_strategy=sft_configs.get("logging_strategy", "steps"),
                logging_steps=sft_configs.get("logging_steps", 16),

                eval_strategy=sft_configs.get("eval_strategy", "epoch"),
                eval_steps=sft_configs.get("eval_steps", None),
                eval_on_start=sft_configs.get("eval_on_start", True),
                per_device_eval_batch_size=sft_configs.get("per_device_eval_batch_size", 1),
                eval_accumulation_steps=sft_configs.get("eval_accumulation_steps", 1),

                save_strategy=sft_configs.get("save_strategy", "epoch"),
                save_steps=sft_configs.get("save_steps", None),

                load_best_model_at_end=sft_configs.get("load_best_model_at_end", True),
                metric_for_best_model=sft_configs.get("metric_for_best_model", "eval_loss"),
                greater_is_better=sft_configs.get("greater_is_better", False),
                
                warmup_steps=sft_configs.get("warmup_ratio", 0.03),
            )
            print(f"Training args: {training_args=}")

            trainer = SFTTrainer(
                model=model_init_kwargs["model_name_or_path"],
                args=training_args,
                train_dataset=training_dataset,
                eval_dataset=eval_dataset,
                peft_config=peft_config,
                callbacks=[EarlyStoppingCallback(**trainer_configs.get("callbacks", {}).get("EarlyStoppingCallback", earlystoppingcallback_default_configs))] if trainer_configs.get("callbacks", None) else None,
            )

            print(f"🕙🕙 Start training...")
            trainer.train()
            print(f"✅✅ Training finished. Saving model to {training_args.output_dir}...")
            trainer.save_model(training_args.output_dir) # Save full model or adapter

        case "ocr_conversion":
            training_dataset = prepare_conversion_dataset(script_args, sft_config, model_config)
            eval_dataset = prepare_conversion_eval_dataset(script_args, sft_config, model_config)

            peft_config = LoraConfig(
                r=lora_configs.get("r", 16),
                lora_alpha=lora_configs.get("lora_alpha", 32),
                lora_dropout=lora_configs.get("lora_dropout", 0.05),
                target_modules=lora_configs.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]),
            ) if lora_configs else None
            if peft_config:
                print(f"Using LoRA with config: {peft_config=}")

            training_args = SFTConfig(
                output_dir=output_model_dir,

                model_init_kwargs=sft_configs.get("model_init_kwargs", {"dtype": "bfloat16"}),

                num_train_epochs=sft_configs.get("num_train_epochs", 2),
                max_length=sft_configs.get("max_length", None),

                per_device_train_batch_size=sft_configs.get("per_device_train_batch_size", 1),
                gradient_accumulation_steps=sft_configs.get("gradient_accumulation_steps", 8),

                learning_rate=sft_configs.get("learning_rate", 1e-4),
                max_grad_norm=sft_configs.get("max_grad_norm", 1.0),
                
                gradient_checkpointing=sft_configs.get("gradient_checkpointing", True),
                gradient_checkpointing_kwargs=sft_configs.get("gradient_checkpointing_kwargs", {"use_reentrant": False}),
                
                packing=sft_configs.get("packing", False),
                packing_strategy=sft_configs.get("packing_strategy", "bfd"),
                # assistant_only_loss=sft_configs.get("assistant_only_loss", True), # Not supported for VLM

                report_to=sft_configs.get("report_to", "wandb"),
                run_name=sft_configs.get("run_name", run_name),

                logging_strategy=sft_configs.get("logging_strategy", "steps"),
                logging_steps=sft_configs.get("logging_steps", 16),

                eval_strategy=sft_configs.get("eval_strategy", "epoch"),
                eval_steps=sft_configs.get("eval_steps", None),
                per_device_eval_batch_size=sft_configs.get("per_device_eval_batch_size", 1),
                eval_accumulation_steps=sft_configs.get("eval_accumulation_steps", 1),

                save_strategy=sft_configs.get("save_strategy", "epoch"),
                save_steps=sft_configs.get("save_steps", None),

                load_best_model_at_end=sft_configs.get("load_best_model_at_end", True),
                metric_for_best_model=sft_configs.get("metric_for_best_model", "eval_loss"),
                greater_is_better=sft_configs.get("greater_is_better", False),
                
                warmup_steps=sft_configs.get("warmup_ratio", 0.03),
            )
            print(f"Training args: {training_args=}")

            trainer = SFTTrainer(
                model=model_init_kwargs["model_name_or_path"],
                args=training_args,
                train_dataset=training_dataset,
                eval_dataset=eval_dataset,
                peft_config=peft_config,
                callbacks=[EarlyStoppingCallback(**trainer_configs.get("callbacks", {}).get("EarlyStoppingCallback", earlystoppingcallback_default_configs))] if trainer_configs.get("callbacks", None) else None,
            )

            print(f"🕙🕙 Start training Conversion...")
            trainer.train(resume_from_checkpoint="ckpt/ocr_conversion/qwen3_vl_4b_i_sft_lora_ocr_conversion/v1.1/20260816_195334/checkpoint-3933") # For 10-th epoch
            print(f"✅✅ Conversion training finished. Saving model to {training_args.output_dir}...")

            trainer.save_model(training_args.output_dir) # Save full model or adapter

        case _:
            raise ValueError(f"Unsupported task_name: {task_name}")

    print(f"✅✅ Script completed.")


if __name__ == "__main__":
    parser = TrlParser([ScriptArgs, SFTConfig, ModelConfig])
    script_args, sft_config, model_config = parser.parse_args_and_config()
    main(script_args, sft_config, model_config)
