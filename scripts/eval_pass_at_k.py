"""
Pass@k 评估脚本
用于评估模型在不同 k 值 (1, 2, 4, 8, 16) 下的 pass@k 性能

使用方法:
    # 基础用法（使用 VLLMEngine）
    python scripts/eval_pass_at_k.py \
        --engine_name qwen3_vl_4b_i \
        --dataset_name kdeng03/mol-rep-ocr-v1.1 \
        --dataset_split valid \
        --max_samples 100
    
    # 使用自定义采样参数
    python scripts/eval_pass_at_k.py \
        --engine_name qwen3_vl_4b_i \
        --dataset_name kdeng03/mol-rep-ocr-v1.1 \
        --max_k 16 \
        --temperature 0.7 \
        --top_p 0.9 \
        --max_new_tokens 256
    
    # 保存结果
    python scripts/eval_pass_at_k.py \
        --engine_name qwen3_vl_4b_i \
        --dataset_name kdeng03/mol-rep-ocr-v1.1 \
        --output_file results/pass_at_k_results.json
"""

import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from PIL import Image
from datasets import load_dataset

# import sys
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import compute_mol_metrics
from src.inference.inference_engine_vllm import VLLMEngine

# 导入 GRPO 的 reward 函数（留接口方便后续替换）
try:
    from scripts.grpo_trl import comprehensive_reward_func as grpo_reward_func
    HAS_GRPO_REWARD = True
except ImportError:
    HAS_GRPO_REWARD = False
    grpo_reward_func = None

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)


# ==================== Reward Calculator (可插拔) ====================

class RewardCalculator:
    """
    Reward 计算器，支持从 grpo_trl 导入或自定义
    
    使用示例:
        # 使用 grpo_trl 的 reward
        calc = RewardCalculator()
        
        # 自定义 reward
        def my_reward(completion, gt_rep, rep_type):
            results = compute_mol_metrics(completion, gt_rep, rep_type)
            return 1.0 if results["is_pred_valid"] else 0.0
        
        calc = RewardCalculator(reward_func=my_reward)
    """
    
    def __init__(self, reward_func=None):
        self.reward_func = reward_func or self._default_reward_func
    
    def compute_single_reward(self, completion: str, gt_rep: str, rep_type: str) -> float:
        """计算单个 completion 的 reward"""
        return self.reward_func(completion, gt_rep, rep_type)
    
    def compute_group_rewards(self, completions: List[str], gt_rep: str, rep_type: str) -> List[float]:
        """计算一组 completions 的 rewards"""
        return [self.compute_single_reward(c, gt_rep, rep_type) for c in completions]
    
    def _default_reward_func(self, completion: str, gt_rep: str, rep_type: str) -> float:
        """
        默认 reward 函数（与 grpo_trl.py 一致）
        
        Reward = 1.0 (valid) + 1.5 (inchikey_match) + 0.5 (tanimoto_sim)
        范围: 0.0 ~ 3.0
        """
        results = compute_mol_metrics(completion, gt_rep, rep_type)
        if not results["is_pred_valid"]:
            return 0.0
        return 1.0 + 1.5 * int(results["is_inchikey_match"]) + 0.5 * float(results["tanimoto_sim"])


# ==================== Sample Difficulty Classifier (策略A: pass@k) ====================

class SampleDifficultyClassifier:
    """
    基于 pass@k 行为的样本难度分类器
    
    分类策略:
        - easy:      k=1 就 pass (模型稳定输出)
        - medium:    1 < k <= max_k/2 才 pass (需要一些采样)
        - hard:      k > max_k/2 才 pass (需要大量采样)
        - impossible: k=max_k 仍不 pass (可能需要数据清洗)
    
    使用示例:
        classifier = SampleDifficultyClassifier(max_k=16)
        difficulty = classifier.classify_sample(rewards=[0.0, 1.5, 2.0, ...])
        # -> "medium" (k=2 首次 pass)
    """
    
    def __init__(self, max_k: int = 16):
        self.max_k = max_k
    
    def classify_sample(self, sample: Dict) -> str:
        """
        根据样本结果分类难度
        
        Args:
            sample: 包含 "results" 的样本字典
            
        Returns:
            "easy", "medium", "hard", 或 "impossible"
        """
        results = sample["results"]
        for i, r in enumerate(results):
            if r.get("is_inchikey_match", False):  # InChIKey 匹配才算 pass
                first_pass_k = i + 1
                if first_pass_k == 1:
                    return "easy"
                elif first_pass_k <= self.max_k // 2:
                    return "medium"
                else:
                    return "hard"
        return "impossible"
    
    def classify_dataset(self, all_samples: List[Dict]) -> Dict[str, int]:
        """
        统计整个数据集的难度分布
        
        Args:
            all_samples: 每个样本的结果字典列表
            
        Returns:
            {"easy": 100, "medium": 50, "hard": 30, "impossible": 20}
        """
        distribution = {"easy": 0, "medium": 0, "hard": 0, "impossible": 0}
        for sample in all_samples:
            difficulty = self.classify_sample(sample)
            distribution[difficulty] += 1
        return distribution


class PassAtKVLLMEngine(VLLMEngine):
    """
    继承 VLLMEngine，添加 pass@k 评估专用方法
    
    核心功能:
        - 一次生成 n 个不同的 completions
        - 支持图文和纯文本
        - 复用父类的配置、LoRA、build 逻辑
        - 支持全量批量和分块批量
    """
    
    def generate_n_completions(
        self,
        sample: dict,
        n: int = 16,
        prompt_key: str = "prompt",
        image_key: str = "input_rep",
        **sampling_overrides,
    ) -> List[str]:
        """
        对单个样本生成 n 个不同的 completions（单样本模式）
        
        Args:
            sample: HF dataset 的单条样本
            n: 生成数量
            prompt_key: prompt 字段名
            image_key: image 字段名
            **sampling_overrides: 覆盖默认采样参数
            
        Returns:
            n 个生成的文本列表
        """
        if self.llm is None:
            self.build()
        
        # 构造 messages
        question = sample[prompt_key]
        image = sample.get(image_key)
        
        messages = []
        # 判断是否为 VLM 任务：image 必须是 PIL.Image 类型
        is_image = isinstance(image, Image.Image)
        
        if is_image:
            # VLM 图文任务
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": image},
                    {"type": "text", "text": question},
                ],
            })
        else:
            # 纯文本任务（image 是 str 或 None）
            messages.append({"role": "user", "content": question})
        
        # 设置采样参数，关键：n=k
        sampling_params = self._get_default_sampling_params(n=n, **sampling_overrides)
        
        # 调用父类 chat，但 return_text=False 获取完整输出
        outputs = self.chat(messages, sampling_params=sampling_params, return_text=False)
        
        # 提取所有 n 个 completions
        # outputs[0] 是第一个 prompt 的 RequestOutput
        # outputs[0].outputs 是 n 个 CompletionOutput
        completions = [output.text for output in outputs[0].outputs]
        
        return completions
    
    def generate_n_completions_batch(
        self,
        samples: List[dict],
        n: int = 16,
        prompt_key: str = "prompt",
        image_key: str = "input_rep",
        **sampling_overrides,
    ) -> List[List[str]]:
        """
        对多个样本批量生成 n 个不同的 completions（批量模式）
        
        Args:
            samples: HF dataset 的样本列表
            n: 每个样本的生成数量
            prompt_key: prompt 字段名
            image_key: image 字段名
            **sampling_overrides: 覆盖默认采样参数
            
        Returns:
            每个样本的 n 个生成文本列表，格式: [[comp1, comp2, ...], [comp1, comp2, ...], ...]
        """
        if self.llm is None:
            self.build()
        
        # 构造批量 messages
        all_messages = []
        for sample in samples:
            question = sample[prompt_key]
            image = sample.get(image_key)
            
            # 判断是否为 VLM 任务：image 必须是 PIL.Image 类型
            is_image = isinstance(image, Image.Image)
            
            if is_image:
                # VLM 图文任务
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image_pil", "image_pil": image},
                        {"type": "text", "text": question},
                    ],
                }]
            else:
                # 纯文本任务（image 是 str 或 None）
                messages = [{"role": "user", "content": question}]
            
            all_messages.append(messages)
        
        # 设置采样参数，关键：n=k
        sampling_params = self._get_default_sampling_params(n=n, **sampling_overrides)
        
        # 批量调用 vLLM chat
        outputs = self.chat(all_messages, sampling_params=sampling_params, return_text=False)
        
        # 提取所有 completions
        all_completions = []
        for output in outputs:
            completions = [o.text for o in output.outputs]
            all_completions.append(completions)
        
        return all_completions


@dataclass
class EvalArguments:
    """评估参数配置"""
    # 必填字段（无默认值）必须放在前面
    engine_name: str = field(
        metadata={"help": "VLLMEngine 配置名称（在 configs/model_inference_schema.yaml 中）"}
    )
    dataset_name: str = field(
        metadata={"help": "HuggingFace 数据集名称"}
    )
    # 可选字段（有默认值）放在后面
    config_path: str = field(
        default="configs/model_inference_schema.yaml",
        metadata={"help": "推理配置文件路径"}
    )
    dataset_split: str = field(
        default="valid",
        metadata={"help": "数据集分割 (train, valid, test, extra_train, extra_valid)"}
    )
    max_samples: int = field(
        default=-1,
        metadata={"help": "最大评估样本数，-1 表示使用全部"}
    )
    max_k: int = field(
        default=16,
        metadata={"help": "最大的 k 值，会自动测试 1,2,4,...,max_k"}
    )
    max_new_tokens: int = field(
        default=256,
        metadata={"help": "生成的最大 token 数"}
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "采样温度"}
    )
    top_p: float = field(
        default=0.9,
        metadata={"help": "nucleus sampling 的 top_p 参数"}
    )
    output_file: str = field(
        default=None,
        metadata={"help": "保存结果的文件路径 (.json 或 .csv)"}
    )
    prompt_key: str = field(
        default="prompt",
        metadata={"help": "数据集中 prompt 字段的 key"}
    )
    image_key: str = field(
        default="input_rep",
        metadata={"help": "数据集中 image 字段的 key，None 表示纯文本"}
    )
    infer_batch_size: int = field(
        default=-1,
        metadata={"help": "推理批次大小，-1 表示全量批量（一次传入所有样本），>0 表示按批次处理"}
    )


def get_k_values(max_k: int) -> List[int]:
    """生成 k 值列表: 1, 2, 4, 8, ..., max_k"""
    k_values = []
    k = 1
    while k <= max_k:
        k_values.append(k)
        k *= 2
    return k_values


def evaluate_pass_at_k(
    engine: PassAtKVLLMEngine,
    dataset,
    args: EvalArguments
) -> Dict[str, Any]:
    """评估 pass@k 指标"""
    k_values = get_k_values(args.max_k)
    max_k = max(k_values)
    
    logger.info(f"🎯 评估 pass@k，k 值: {k_values}")
    logger.info(f"📊 样本数: {len(dataset)}")
    logger.info(f"📦 推理批次大小: {'全量批量' if args.infer_batch_size == -1 else args.infer_batch_size}")
    
    # 初始化 reward 计算器和难度分类器
    reward_calc = RewardCalculator()
    difficulty_classifier = SampleDifficultyClassifier(max_k=max_k)
    
    # 存储每个样本的结果
    sample_results = []
    
    # 根据 infer_batch_size 决定使用单样本还是批量模式
    if args.infer_batch_size == -1:
        # 全量批量模式：一次传入所有样本
        logger.info("🚀 使用全量批量模式...")
        try:
            all_completions = engine.generate_n_completions_batch(
                samples=list(dataset),
                n=max_k,
                prompt_key=args.prompt_key,
                image_key=args.image_key,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except Exception as e:
            logger.error(f"全量批量生成失败: {e}")
            logger.info("⚠️ 回退到单样本模式...")
            all_completions = None
        
        if all_completions is not None:
            # 处理批量结果
            for idx, (sample, completions) in enumerate(tqdm(zip(dataset, all_completions), total=len(dataset), desc="评估进度")):
                gt_rep = sample.get("output_rep", sample.get("completion", ""))
                rep_type = sample.get("output_rep_type", "can_smiles")
                
                # 计算 rewards
                rewards = reward_calc.compute_group_rewards(completions, gt_rep, rep_type)
                
                # 评估每个 completion
                completion_results = []
                for comp in completions:
                    try:
                        results = compute_mol_metrics(gt_rep, comp, rep_type)
                        completion_results.append(results)
                    except Exception as e:
                        logger.error(f"样本 {idx} 评估失败: {e}")
                        completion_results.append({
                            "is_pred_valid": False,
                            "is_inchikey_match": False,
                            "tanimoto_sim": 0.0,
                        })
                
                # 样本难度分类
                sample_data = {"results": completion_results}
                difficulty = difficulty_classifier.classify_sample(sample_data)
                
                sample_results.append({
                    "idx": idx,
                    "gt_rep": gt_rep,
                    "rep_type": rep_type,
                    "completions": completions,
                    "rewards": rewards,
                    "results": completion_results,
                    "difficulty": difficulty,
                })
        else:
            # 回退到单样本模式
            for idx, sample in enumerate(tqdm(dataset, desc="评估进度")):
                gt_rep = sample.get("output_rep", sample.get("completion", ""))
                rep_type = sample.get("output_rep_type", "can_smiles")
                
                try:
                    completions = engine.generate_n_completions(
                        sample=sample,
                        n=max_k,
                        prompt_key=args.prompt_key,
                        image_key=args.image_key,
                        max_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                except Exception as e:
                    logger.error(f"样本 {idx} 生成失败: {e}")
                    completions = [""] * max_k
                
                rewards = reward_calc.compute_group_rewards(completions, gt_rep, rep_type)
                
                completion_results = []
                for comp in completions:
                    try:
                        results = compute_mol_metrics(gt_rep, comp, rep_type)
                        completion_results.append(results)
                    except Exception as e:
                        logger.error(f"样本 {idx} 评估失败: {e}")
                        completion_results.append({
                            "is_pred_valid": False,
                            "is_inchikey_match": False,
                            "tanimoto_sim": 0.0,
                        })
                
                sample_data = {"results": completion_results}
                difficulty = difficulty_classifier.classify_sample(sample_data)
                
                sample_results.append({
                    "idx": idx,
                    "gt_rep": gt_rep,
                    "rep_type": rep_type,
                    "completions": completions,
                    "rewards": rewards,
                    "results": completion_results,
                    "difficulty": difficulty,
                })
    else:
        # 分块批量模式
        batch_size = args.infer_batch_size
        logger.info(f"🔄 使用分块批量模式，batch_size={batch_size}...")
        
        for batch_start in tqdm(range(0, len(dataset), batch_size), desc="批量生成"):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch_samples = dataset[batch_start:batch_end]
            
            try:
                batch_completions = engine.generate_n_completions_batch(
                    samples=list(batch_samples),
                    n=max_k,
                    prompt_key=args.prompt_key,
                    image_key=args.image_key,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            except Exception as e:
                logger.error(f"批次 {batch_start}-{batch_end} 生成失败: {e}")
                batch_completions = [[""] * max_k] * len(batch_samples)
            
            # 处理批次结果
            for i, (sample, completions) in enumerate(zip(batch_samples, batch_completions)):
                idx = batch_start + i
                gt_rep = sample.get("output_rep", sample.get("completion", ""))
                rep_type = sample.get("output_rep_type", "can_smiles")
                
                rewards = reward_calc.compute_group_rewards(completions, gt_rep, rep_type)
                
                completion_results = []
                for comp in completions:
                    try:
                        results = compute_mol_metrics(gt_rep, comp, rep_type)
                        completion_results.append(results)
                    except Exception as e:
                        logger.error(f"样本 {idx} 评估失败: {e}")
                        completion_results.append({
                            "is_pred_valid": False,
                            "is_inchikey_match": False,
                            "tanimoto_sim": 0.0,
                        })
                
                sample_data = {"results": completion_results}
                difficulty = difficulty_classifier.classify_sample(sample_data)
                
                sample_results.append({
                    "idx": idx,
                    "gt_rep": gt_rep,
                    "rep_type": rep_type,
                    "completions": completions,
                    "rewards": rewards,
                    "results": completion_results,
                    "difficulty": difficulty,
                })
    
    # 计算 pass@k 和更多诊断指标
    pass_at_k_results = {}
    inchikey_match_at_k = {}
    can_smiles_match_at_k = {}
    avg_tanimoto_at_k = {}
    valid_rate_at_k = {}
    
    # GRPO 特定指标
    all_correct_at_k = {}
    all_wrong_at_k = {}
    mean_reward_at_k = {}
    std_reward_at_k = {}
    cv_reward_at_k = {}
    max_reward_at_k = {}
    min_reward_at_k = {}
    range_reward_at_k = {}
    
    for k in k_values:
        pass_count = 0
        inchikey_count = 0
        smiles_count = 0
        tanimoto_scores = []
        valid_completions = 0
        total_completions = 0
        
        # GRPO 统计
        all_correct_count = 0
        all_wrong_count = 0
        mean_rewards = []
        std_rewards = []
        cv_rewards = []
        max_rewards = []
        min_rewards = []
        range_rewards = []
        
        for sample in sample_results:
            # 取前 k 个 completions 的统计
            k_results = sample["results"][:k]
            k_rewards = sample["rewards"][:k]
            
            # pass@k: 至少一个 InChIKey 匹配（核心定义）
            has_pass = any(r.get("is_inchikey_match", False) for r in k_results)
            if has_pass:
                pass_count += 1
            
            # inchikey_match@k: 至少一个 InChIKey 匹配（与 pass@k 相同）
            has_inchikey = any(r.get("is_inchikey_match", False) for r in k_results)
            if has_inchikey:
                inchikey_count += 1
            
            # can_smiles_match@k: 至少一个 Canonical SMILES 匹配
            has_smiles = any(r.get("is_can_smiles_match", False) for r in k_results)
            if has_smiles:
                smiles_count += 1
            
            # avg_best_tanimoto@k: 前 k 个中最好的 Tanimoto 相似度
            best_tanimoto = max(
                [r.get("tanimoto_sim", 0.0) for r in k_results],
                default=0.0
            )
            tanimoto_scores.append(best_tanimoto)
            
            # valid_rate@k: 前 k 个中 valid 的比例（辅助指标）
            k_valid_count = sum(1 for r in k_results if r.get("is_pred_valid", False))
            valid_completions += k_valid_count
            total_completions += len(k_results)
            
            # === GRPO 指标 ===
            # all_correct@k: k 个全部 InChIKey 匹配
            if all(r.get("is_inchikey_match", False) for r in k_results):
                all_correct_count += 1
            
            # all_wrong@k: k 个全部不 InChIKey 匹配
            if not any(r.get("is_inchikey_match", False) for r in k_results):
                all_wrong_count += 1
            
            # Reward 统计
            k_rewards_values = k_rewards  # 已经是 float 列表
            mean_r = np.mean(k_rewards_values)
            std_r = np.std(k_rewards_values)
            cv_r = std_r / (mean_r + 1e-8)
            max_r = np.max(k_rewards_values)
            min_r = np.min(k_rewards_values)
            range_r = max_r - min_r
            
            mean_rewards.append(mean_r)
            std_rewards.append(std_r)
            cv_rewards.append(cv_r)
            max_rewards.append(max_r)
            min_rewards.append(min_r)
            range_rewards.append(range_r)
        
        n_samples = len(sample_results)
        pass_at_k_results[f"pass@{k}"] = pass_count / n_samples
        inchikey_match_at_k[f"inchikey_match@{k}"] = inchikey_count / n_samples
        can_smiles_match_at_k[f"can_smiles_match@{k}"] = smiles_count / n_samples
        avg_tanimoto_at_k[f"avg_best_tanimoto@{k}"] = np.mean(tanimoto_scores)
        valid_rate_at_k[f"valid_rate@{k}"] = valid_completions / total_completions if total_completions > 0 else 0.0
        
        # GRPO 指标
        all_correct_at_k[f"all_correct@{k}"] = all_correct_count / n_samples
        all_wrong_at_k[f"all_wrong@{k}"] = all_wrong_count / n_samples
        mean_reward_at_k[f"mean_reward@{k}"] = np.mean(mean_rewards)
        std_reward_at_k[f"std_reward@{k}"] = np.mean(std_rewards)
        cv_reward_at_k[f"cv_reward@{k}"] = np.mean(cv_rewards)
        max_reward_at_k[f"max_reward@{k}"] = np.mean(max_rewards)
        min_reward_at_k[f"min_reward@{k}"] = np.mean(min_rewards)
        range_reward_at_k[f"range_reward@{k}"] = np.mean(range_rewards)
        
        logger.info(f"✅ pass@{k} = {pass_at_k_results[f'pass@{k}']:.4f} ({pass_count}/{n_samples})")
        logger.info(f"🔑 inchikey_match@{k} = {inchikey_match_at_k[f'inchikey_match@{k}']:.4f} ({inchikey_count}/{n_samples})")
        logger.info(f"🧪 can_smiles_match@{k} = {can_smiles_match_at_k[f'can_smiles_match@{k}']:.4f} ({smiles_count}/{n_samples})")
        logger.info(f"📈 avg_best_tanimoto@{k} = {avg_tanimoto_at_k[f'avg_best_tanimoto@{k}']:.4f}")
        logger.info(f"📊 valid_rate@{k} = {valid_rate_at_k[f'valid_rate@{k}']:.4f} ({valid_completions}/{total_completions})")
        logger.info(f"🎯 all_correct@{k} = {all_correct_at_k[f'all_correct@{k}']:.4f} ({all_correct_count}/{n_samples})")
        logger.info(f"❌ all_wrong@{k} = {all_wrong_at_k[f'all_wrong@{k}']:.4f} ({all_wrong_count}/{n_samples})")
        logger.info(f"💰 mean_reward@{k} = {mean_reward_at_k[f'mean_reward@{k}']:.4f}")
        logger.info(f"📉 std_reward@{k} = {std_reward_at_k[f'std_reward@{k}']:.4f}")
        logger.info(f"📊 cv_reward@{k} = {cv_reward_at_k[f'cv_reward@{k}']:.4f}")
    
    # 样本难度分布统计
    difficulty_distribution = difficulty_classifier.classify_dataset(sample_results)
    logger.info(f"📋 样本难度分布: {difficulty_distribution}")
    
    # 汇总结果
    final_results = {
        "engine_name": args.engine_name,
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "num_samples": len(dataset),
        "k_values": k_values,
        "pass_at_k": pass_at_k_results,
        "inchikey_match_at_k": inchikey_match_at_k,
        "can_smiles_match_at_k": can_smiles_match_at_k,
        "avg_tanimoto_at_k": avg_tanimoto_at_k,
        "valid_rate_at_k": valid_rate_at_k,
        "all_correct_at_k": all_correct_at_k,
        "all_wrong_at_k": all_wrong_at_k,
        "mean_reward_at_k": mean_reward_at_k,
        "std_reward_at_k": std_reward_at_k,
        "cv_reward_at_k": cv_reward_at_k,
        "max_reward_at_k": max_reward_at_k,
        "min_reward_at_k": min_reward_at_k,
        "range_reward_at_k": range_reward_at_k,
        "difficulty_distribution": difficulty_distribution,
        "generation_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "sample_results": sample_results,  # 详细结果
    }
    
    return final_results


def save_difficulty_indices(results: Dict[str, Any], output_file: str):
    """保存难度索引到 JSON 文件"""
    if output_file is None:
        return
    
    # 构建索引
    indices = {"easy": [], "medium": [], "hard": [], "impossible": []}
    for sample in results["sample_results"]:
        difficulty = sample["difficulty"]
        indices[difficulty].append(sample["idx"])
    
    # 添加 metadata
    indices["metadata"] = {
        "dataset_name": results["dataset_name"],
        "dataset_split": results["dataset_split"],
        "total_samples": results["num_samples"],
        "engine_name": results["engine_name"],
        "generation_params": results["generation_params"],
    }
    
    # 保存完整索引
    indices_file = output_file.replace(".json", "_indices.json").replace(".jsonl", "_indices.json").replace(".csv", "_indices.json")
    if not indices_file.endswith("_indices.json"):
        indices_file = output_file + "_indices.json"
    
    os.makedirs(os.path.dirname(indices_file) if os.path.dirname(indices_file) else ".", exist_ok=True)
    with open(indices_file, "w", encoding="utf-8") as f:
        json.dump(indices, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 难度索引已保存到: {indices_file}")
    
    # 保存 mixed 索引 (medium + hard)
    mixed_indices = indices["medium"] + indices["hard"]
    mixed_file = indices_file.replace("_indices.json", "_mixed_indices.json")
    with open(mixed_file, "w", encoding="utf-8") as f:
        json.dump(mixed_indices, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Mixed 索引 (medium+hard) 已保存到: {mixed_file}")
    
    # 打印统计信息
    logger.info(f"📊 难度索引统计:")
    for difficulty in ["easy", "medium", "hard", "impossible"]:
        logger.info(f"  {difficulty}: {len(indices[difficulty])} 样本")
    logger.info(f"  mixed (medium+hard): {len(mixed_indices)} 样本")


def save_results(results: Dict[str, Any], output_file: str):
    """保存结果到文件"""
    if output_file is None:
        return
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    # 1. 保存统计数据为 CSV
    csv_file = output_file.replace(".json", ".csv").replace(".jsonl", ".csv")
    if not csv_file.endswith(".csv"):
        csv_file = output_file + ".csv"
    
    import pandas as pd
    
    summary_data = []
    for k in results["k_values"]:
        row = {
            "k": k,
            "pass_at_k": results["pass_at_k"].get(f"pass@{k}", 0.0),
            "inchikey_match_at_k": results["inchikey_match_at_k"].get(f"inchikey_match@{k}", 0.0),
            "can_smiles_match_at_k": results["can_smiles_match_at_k"].get(f"can_smiles_match@{k}", 0.0),
            "avg_best_tanimoto_at_k": results["avg_tanimoto_at_k"].get(f"avg_best_tanimoto@{k}", 0.0),
            "valid_rate_at_k": results["valid_rate_at_k"].get(f"valid_rate@{k}", 0.0),
            "all_correct_at_k": results["all_correct_at_k"].get(f"all_correct@{k}", 0.0),
            "all_wrong_at_k": results["all_wrong_at_k"].get(f"all_wrong@{k}", 0.0),
            "mean_reward_at_k": results["mean_reward_at_k"].get(f"mean_reward@{k}", 0.0),
            "std_reward_at_k": results["std_reward_at_k"].get(f"std_reward@{k}", 0.0),
            "cv_reward_at_k": results["cv_reward_at_k"].get(f"cv_reward@{k}", 0.0),
            "max_reward_at_k": results["max_reward_at_k"].get(f"max_reward@{k}", 0.0),
            "min_reward_at_k": results["min_reward_at_k"].get(f"min_reward@{k}", 0.0),
            "range_reward_at_k": results["range_reward_at_k"].get(f"range_reward@{k}", 0.0),
        }
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(csv_file, index=False)
    logger.info(f"💾 统计结果已保存到: {csv_file}")
    
    # 保存难度分布到单独 CSV
    difficulty_csv = csv_file.replace(".csv", "_difficulty.csv")
    difficulty_df = pd.DataFrame([results["difficulty_distribution"]])
    difficulty_df.to_csv(difficulty_csv, index=False)
    logger.info(f"💾 难度分布已保存到: {difficulty_csv}")
    
    # 2. 保存原始 LLM responses 为 JSONL
    jsonl_file = output_file.replace(".csv", ".jsonl").replace(".json", ".jsonl")
    if not jsonl_file.endswith(".jsonl"):
        jsonl_file = output_file if output_file.endswith(".jsonl") else output_file + ".jsonl"
    
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for sample in results["sample_results"]:
            record = {
                "idx": sample["idx"],
                "gt_rep": sample["gt_rep"],
                "rep_type": sample["rep_type"],
                "completions": sample["completions"],
                "rewards": sample["rewards"],
                "difficulty": sample["difficulty"],
                "results": sample["results"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"💾 原始 LLM responses 已保存到: {jsonl_file}")
    
    # 3. 保存完整结果（包含配置和元数据）为 JSON
    json_file = output_file.replace(".csv", ".json").replace(".jsonl", ".json")
    if not json_file.endswith(".json"):
        json_file = output_file + ".json"
    
    # 移除 sample_results 以减小文件大小
    compact_results = {k: v for k, v in results.items() if k != "sample_results"}
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(compact_results, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 完整结果（无原始 responses）已保存到: {json_file}")


def main():
    parser = argparse.ArgumentParser(description="Pass@k 评估脚本")
    parser.add_argument("--engine_name", type=str, required=True, help="VLLMEngine 配置名称")
    parser.add_argument("--config_path", type=str, default="configs/model_inference_schema.yaml", help="推理配置文件路径")
    parser.add_argument("--dataset_name", type=str, required=True, help="数据集名称")
    parser.add_argument("--dataset_split", type=str, default="valid", help="数据集分割")
    parser.add_argument("--max_samples", type=int, default=-1, help="最大样本数")
    parser.add_argument("--max_k", type=int, default=16, help="最大 k 值")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p 参数")
    parser.add_argument("--output_file", type=str, default=None, help="输出文件路径")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="数据集中 prompt 字段名")
    parser.add_argument("--image_key", type=str, default="input_rep", help="数据集中 image 字段名")
    parser.add_argument("--infer_batch_size", type=int, default=-1, help="推理批次大小，-1 表示全量批量，>0 表示按批次处理")
    
    args = parser.parse_args()
    eval_args = EvalArguments(**vars(args))
    
    logger.info("=" * 60)
    logger.info("🚀 Pass@k 评估开始")
    logger.info("=" * 60)
    logger.info(f"Engine: {eval_args.engine_name}")
    logger.info(f"数据集: {eval_args.dataset_name} ({eval_args.dataset_split})")
    logger.info(f"K 值: {get_k_values(eval_args.max_k)}")
    logger.info("=" * 60)
    
    # 1. 加载数据集
    logger.info(f"📚 加载数据集: {eval_args.dataset_name}")
    dataset = load_dataset(eval_args.dataset_name, split=eval_args.dataset_split)
    
    if eval_args.max_samples != -1 and eval_args.max_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(min(eval_args.max_samples, len(dataset))))
        logger.info(f"📝 使用 {len(dataset)} 个样本 (random shuffle) 进行评估")
    else:
        logger.info(f"📝 Using all samples {len(dataset)} for evaluation...")
    
    # 2. 初始化 PassAtK VLLMEngine
    logger.info(f"📦 初始化 VLLMEngine: {eval_args.engine_name}")
    engine = PassAtKVLLMEngine(eval_args.engine_name, config_path=eval_args.config_path)
    engine.build()  # 预加载模型
    logger.info(f"✅ 模型加载完成，类型: {engine.modality}")
    
    # 3. 执行评估
    results = evaluate_pass_at_k(engine, dataset, eval_args)
    
    # 4. 保存结果
    save_results(results, eval_args.output_file)
    
    # 5. 保存难度索引
    if eval_args.max_samples != -1:
        logger.warning(
            f"🔴🔴 当前 max_samples={eval_args.max_samples}，难度索引基于部分样本，可能不准确, 建议仅在 max_samples=None (全量数据) 时使用难度索引进行 GRPO 训练..."
        )
    save_difficulty_indices(results, eval_args.output_file)
    
    # 6. 打印摘要
    logger.info("=" * 60)
    logger.info("📊 Pass@k 评估结果摘要")
    logger.info("=" * 60)
    for k in results["k_values"]:
        logger.info(f"k={k}:")
        logger.info(f"  pass@{k}:              {results['pass_at_k'][f'pass@{k}']:.4f}")
        logger.info(f"  inchikey_match@{k}:    {results['inchikey_match_at_k'][f'inchikey_match@{k}']:.4f}")
        logger.info(f"  can_smiles_match@{k}:  {results['can_smiles_match_at_k'][f'can_smiles_match@{k}']:.4f}")
        logger.info(f"  avg_best_tanimoto@{k}: {results['avg_tanimoto_at_k'][f'avg_best_tanimoto@{k}']:.4f}")
        logger.info(f"  valid_rate@{k}:        {results['valid_rate_at_k'][f'valid_rate@{k}']:.4f}")
        logger.info(f"  all_correct@{k}:       {results['all_correct_at_k'][f'all_correct@{k}']:.4f}")
        logger.info(f"  all_wrong@{k}:         {results['all_wrong_at_k'][f'all_wrong@{k}']:.4f}")
        logger.info(f"  mean_reward@{k}:       {results['mean_reward_at_k'][f'mean_reward@{k}']:.4f}")
        logger.info(f"  std_reward@{k}:        {results['std_reward_at_k'][f'std_reward@{k}']:.4f}")
        logger.info(f"  cv_reward@{k}:         {results['cv_reward_at_k'][f'cv_reward@{k}']:.4f}")
    logger.info("=" * 60)
    logger.info(f"📋 样本难度分布: {results['difficulty_distribution']}")
    logger.info("=" * 60)
    logger.info("✅ 评估完成!")


if __name__ == "__main__":
    main()
