#!/usr/bin/env python3
"""
Data preprocessing script for VERL GRPO training.

Converts HuggingFace datasets to VERL's expected parquet format.
VERL expects parquet files with specific columns:
- For text tasks: prompt, response, data_source, ground_truth, extra_info
- For vision tasks: prompt, response, data_source, ground_truth, extra_info, image

This script handles both:
1. mol-rep-conversion (text-to-text)
2. mol-rep-ocr (image-to-text)

Usage:
    python scripts/data_preprocess_mol_rep.py --task conversion --version v1.1
    python scripts/data_preprocess_mol_rep.py --task ocr --version v1.1
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VERSION = "v1.1"
OUTPUT_BASE_DIR = os.path.expanduser("~/data")


def prepare_conversion_dataset(version: str, output_dir: str, use_mixed_subset: bool = True):
    """
    Prepare mol-rep-conversion dataset for VERL.
    
    VERL expects columns:
    - prompt: List of message dicts (user prompt)
    - response: Ground truth completion (for reward computation)
    - data_source: Dataset identifier
    - ground_truth: The target molecular representation
    - extra_info: JSON string with additional info (output_rep_type, etc.)
    """
    logger.info(f"Loading mol-rep-conversion-{version} dataset...")
    
    # Load train split
    raw_train = load_dataset(f"kdeng03/mol-rep-conversion-{version}", split="train")
    logger.info(f"Raw train dataset: {len(raw_train)} samples")
    
    # Load valid split for evaluation
    raw_valid = load_dataset(f"kdeng03/mol-rep-conversion-{version}", split="valid")
    logger.info(f"Raw valid dataset: {len(raw_valid)} samples")
    
    if use_mixed_subset:
        # Use the same mixed difficulty subset as TRL version
        indices_file = f"results/pass_at_k/mol-rep-conversion/{version}/train/qwen3_vl_4b_i_sft_lora_ocr_conversion_pass_at_16_indices.json"
        if os.path.exists(indices_file):
            with open(indices_file, "r") as f:
                indices_dict = json.load(f)
            
            easy_indices = indices_dict["easy"]
            medium_indices = indices_dict["medium"]
            hard_indices = indices_dict["hard"]
            impossible_indices = indices_dict["impossible"]
            
            import random
            random.seed(42)
            subset_easy_indices = random.sample(easy_indices, 53) if len(easy_indices) > 53 else easy_indices
            final_indices = subset_easy_indices + medium_indices + hard_indices + impossible_indices
            
            raw_train = raw_train.select(final_indices)
            logger.info(f"Using mixed subset: {len(raw_train)} samples (easy: {len(subset_easy_indices)}, medium: {len(medium_indices)}, hard: {len(hard_indices)}, impossible: {len(impossible_indices)})")
        else:
            logger.warning(f"Indices file not found: {indices_file}. Using full train dataset.")
    
    def convert_to_verl_format(sample):
        """Convert a single sample to VERL format."""
        prompt = sample["prompt"]
        completion = sample["completion"]
        output_rep_type = sample.get("output_rep_type", "can_smiles")
        
        # VERL expects prompt as a string (will be tokenized by the trainer)
        # For text tasks, this is just the prompt text
        verl_prompt = str(prompt)
        
        # Ground truth for reward computation
        ground_truth = str(completion)
        
        # Extra info as JSON string
        extra_info = json.dumps({
            "output_rep_type": output_rep_type,
            "data_source": f"mol-rep-conversion-{version}",
        })
        
        return {
            "prompt": verl_prompt,
            "response": ground_truth,  # Used for reward computation
            "data_source": f"mol-rep-conversion-{version}",
            "ground_truth": ground_truth,
            "extra_info": extra_info,
        }
    
    # Convert train dataset
    train_data = []
    for sample in raw_train:
        train_data.append(convert_to_verl_format(sample))
    
    train_df = pd.DataFrame(train_data)
    
    # Convert valid dataset (use a subset for faster evaluation)
    valid_size = min(220, len(raw_valid))
    raw_valid_subset = raw_valid.shuffle(seed=42).select(range(valid_size))
    
    val_data = []
    for sample in raw_valid_subset:
        val_data.append(convert_to_verl_format(sample))
    
    val_df = pd.DataFrame(val_data)
    
    # Save to parquet
    task_output_dir = os.path.join(output_dir, "mol-rep-conversion")
    os.makedirs(task_output_dir, exist_ok=True)
    
    train_path = os.path.join(task_output_dir, "train.parquet")
    val_path = os.path.join(task_output_dir, "val.parquet")
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    
    logger.info(f"Saved train dataset to {train_path} ({len(train_df)} samples)")
    logger.info(f"Saved val dataset to {val_path} ({len(val_df)} samples)")
    
    return train_path, val_path


def prepare_ocr_dataset(version: str, output_dir: str):
    """
    Prepare mol-rep-ocr dataset for VERL.
    
    VERL expects columns:
    - prompt: User prompt text
    - response: Ground truth completion
    - data_source: Dataset identifier
    - ground_truth: Target molecular representation
    - extra_info: JSON string with additional info
    - image: PIL Image or image path (for vision-language models)
    """
    logger.info(f"Loading mol-rep-ocr-{version} dataset...")
    
    raw_train = load_dataset(f"kdeng03/mol-rep-ocr-{version}", split="train")
    raw_valid = load_dataset(f"kdeng03/mol-rep-ocr-{version}", split="valid")
    
    logger.info(f"Raw train dataset: {len(raw_train)} samples")
    logger.info(f"Raw valid dataset: {len(raw_valid)} samples")
    
    def convert_to_verl_format(sample):
        """Convert a single OCR sample to VERL format."""
        prompt = sample["prompt"]
        completion = sample["completion"]
        image = sample["input_rep"]  # This is the image
        output_rep_type = sample.get("output_rep_type", "can_smiles")
        
        verl_prompt = str(prompt)
        ground_truth = str(completion)
        
        extra_info = json.dumps({
            "output_rep_type": output_rep_type,
            "data_source": f"mol-rep-ocr-{version}",
        })
        
        return {
            "prompt": verl_prompt,
            "response": ground_truth,
            "data_source": f"mol-rep-ocr-{version}",
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "image": image,  # PIL Image object
        }
    
    # Convert train dataset
    train_data = []
    for sample in raw_train:
        train_data.append(convert_to_verl_format(sample))
    
    train_df = pd.DataFrame(train_data)
    
    # Convert valid dataset
    val_data = []
    for sample in raw_valid:
        val_data.append(convert_to_verl_format(sample))
    
    val_df = pd.DataFrame(val_data)
    
    # Save to parquet
    task_output_dir = os.path.join(output_dir, "mol-rep-ocr")
    os.makedirs(task_output_dir, exist_ok=True)
    
    train_path = os.path.join(task_output_dir, "train.parquet")
    val_path = os.path.join(task_output_dir, "val.parquet")
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    
    logger.info(f"Saved train dataset to {train_path} ({len(train_df)} samples)")
    logger.info(f"Saved val dataset to {val_path} ({len(val_df)} samples)")
    
    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(description="Prepare molecular representation datasets for VERL")
    parser.add_argument("--task", type=str, required=True, choices=["conversion", "ocr"], help="Task type")
    parser.add_argument("--version", type=str, default=VERSION, help="Dataset version")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_BASE_DIR, help="Output directory")
    parser.add_argument("--no-mixed-subset", action="store_true", help="Don't use mixed difficulty subset (conversion only)")
    
    args = parser.parse_args()
    
    if args.task == "conversion":
        prepare_conversion_dataset(
            version=args.version,
            output_dir=args.output_dir,
            use_mixed_subset=not args.no_mixed_subset,
        )
    elif args.task == "ocr":
        prepare_ocr_dataset(
            version=args.version,
            output_dir=args.output_dir,
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    logger.info("✅ Data preprocessing complete!")


if __name__ == "__main__":
    main()
