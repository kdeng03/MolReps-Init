# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Custom reward function for molecular representation tasks.
Used by VERL GRPO trainer to compute rewards for molecular OCR and conversion tasks.

This replaces the comprehensive_reward_func from grpo_trl.py.
"""

import sys
import os
import logging

# Add project root to path to import compute_mol_metrics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils import compute_mol_metrics

logger = logging.getLogger(__name__)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Compute reward score for molecular representation tasks.
    
    This function is called by VERL's reward manager for each sample.
    It mirrors the logic from comprehensive_reward_func in grpo_trl.py.
    
    Args:
        data_source: Dataset source identifier (e.g., "mol-rep-conversion", "mol-rep-ocr")
        solution_str: Model's generated output (the completion)
        ground_truth: Ground truth molecular representation
        extra_info: Additional information, may contain:
            - output_rep_type: Type of molecular representation (can_smiles, can_selfies, etc.)
            - prompt: Original prompt
            - Other metadata from the dataset
    
    Returns:
        float: Reward score
            - -1.0: Invalid prediction (parsing failed)
            - 1.0: Correct InChIKey match
            - 0.0: Valid but incorrect prediction
    """
    try:
        # Extract output representation type from extra_info
        output_rep_type = None
        if extra_info is not None:
            if isinstance(extra_info, dict):
                output_rep_type = extra_info.get("output_rep_type", None)
            elif isinstance(extra_info, str):
                output_rep_type = extra_info
        
        # Default to can_smiles if not specified
        if output_rep_type is None:
            output_rep_type = "can_smiles"
        
        # Clean up the solution string (remove potential formatting artifacts)
        solution_str = str(solution_str).strip()
        ground_truth = str(ground_truth).strip()
        
        # Compute molecular metrics
        results = compute_mol_metrics(
            gt_rep=ground_truth,
            pred_rep=solution_str,
            rep_type=output_rep_type
        )
        
        # Apply reward logic (matches TRL's comprehensive_reward_func)
        if not results["is_pred_valid"]:
            # Penalty for invalid output (rare case)
            reward = -1.0
        elif results["is_inchikey_match"]:
            # Reward for correct InChIKey match
            reward = 1.0
        else:
            # Neutral for valid but wrong (most common case)
            reward = 0.0
        
        return reward
        
    except Exception as e:
        logger.error(f"Error computing reward: {e}")
        # Return neutral reward on error to avoid breaking training
        return 0.0
