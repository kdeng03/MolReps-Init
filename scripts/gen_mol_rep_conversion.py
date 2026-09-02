import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "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"
import argparse
import random
random.seed(42)

import yaml
from datasets import load_dataset, DatasetDict

from src.dataset.mol_reps import MolReps


VERSION = "v1.1"
UPLOAD_HF = True
UPLOAD_HF_DATASET_ID = f"kdeng03/mol-rep-conversion-{VERSION}"
UPLOAD_HF_DATASET_ID_OCR = f"kdeng03/mol-rep-ocr-{VERSION}"
HF_DATASET_ID = "kdeng03/mol-reps-v0.1" # TODO: To add more molecules, please change to newer version
CONFIG_REPS_FILE = "configs/mol_rep_schema.yaml"
CONFIG_REP_CONVERSION_FILE = "configs/mol_rep_conversion_schema.yaml"

# Global parameters for data generation
NUM_RANDOM_VARIANTS = 2  # Number of random variants to sample per representation
NUM_TEMPLATES_PER_PAIR = 1  # Number of QA templates to sample per pair

with open(CONFIG_REPS_FILE, "r") as f:
    CONFIG_REPS: dict = yaml.safe_load(f)["representations"]

with open(CONFIG_REP_CONVERSION_FILE, "r") as f:
    CONFIG_REP_CONVERSION: dict = yaml.safe_load(f)["task_types"]


def load_args() -> dict:
    parser = argparse.ArgumentParser(description="Generate molecular representations from SMILES.")
    args_parser = parser.parse_args()

    # Merge
    args: dict = vars(args_parser)    
    args.update(CONFIG_REPS)
    args.update(CONFIG_REP_CONVERSION)
    return args


def generate_pairs_for_task(mol_record: dict, task_type: str, task_config: dict) -> list[tuple]:
    """统一入口，返回 list[(input_rep_type, input_rep, output_rep_type, output_rep)]"""
    match task_type:
        case "ocr":
            return _gen_ocr_pairs(mol_record, task_config)
        case "can_reps_translation":
            return _gen_translation_pairs(mol_record, task_config)
        case "iupac_understanding":
            return _gen_iupac_understanding_pairs(mol_record, task_config)
        case "cml_understanding":
            return _gen_cml_understanding_pairs(mol_record, task_config)
        case "intra_rep_normalization":
            return _gen_intra_normalization_pairs(mol_record, task_config)
        case "cross_rep_normalization":
            return _gen_cross_normalization_pairs(mol_record, task_config)
        case _:
            raise ValueError(f"Unknown task_type: {task_type}")


def _gen_ocr_pairs(mol_record: dict, task_config: dict) -> list[tuple]:
    """OCR: mol_image → 多个 canonical output"""
    pairs = []
    mol_image = mol_record.get("mol_image")
    if not mol_image:
        return pairs
    for out_type in task_config["output_rep_types"]:
        out_rep = mol_record.get(out_type)
        if out_rep:
            pairs.append(("mol_image", mol_image, out_type, out_rep))
    return pairs


def _gen_translation_pairs(mol_record: dict, task_config: dict) -> list[tuple]:
    """Translation: canonical ↔ canonical, full_cross, exclude self"""
    pairs = []
    input_types = task_config["input_rep_types"]
    output_types = task_config["output_rep_types"]
    for in_type in input_types:
        in_rep = mol_record.get(in_type)
        if not in_rep:
            continue
        for out_type in output_types:
            if not task_config.get("include_self_pairs", False) and in_type == out_type:
                continue
            out_rep = mol_record.get(out_type)
            if out_rep:
                pairs.append((in_type, in_rep, out_type, out_rep))
    return pairs


def _gen_iupac_understanding_pairs(mol_record: dict, task_config: dict) -> list[tuple]:
    """IUPAC understanding: iupac → canonical outputs"""
    pairs = []
    iupac = mol_record.get("iupac")
    if not iupac:
        return pairs
    for out_type in task_config["output_rep_types"]:
        out_rep = mol_record.get(out_type)
        if out_rep:
            pairs.append(("iupac", iupac, out_type, out_rep))
    return pairs


def _gen_cml_understanding_pairs(mol_record: dict, task_config: dict) -> list[tuple]:
    """CML understanding: cml → canonical outputs"""
    pairs = []
    cml = mol_record.get("cml")
    if not cml:
        return pairs
    for out_type in task_config["output_rep_types"]:
        out_rep = mol_record.get(out_type)
        if out_rep:
            pairs.append(("cml", cml, out_type, out_rep))
    return pairs


def _gen_intra_normalization_pairs(mol_record: dict, task_config: dict) -> list[tuple]:
    """Intra normalization: random_X → can_X, same family"""
    pairs = []
    family_map = {
        "random_smiles": "can_smiles",
        "random_selfies": "can_selfies",
        "random_deepsmiles": "can_deepsmiles",
    }
    for rand_type, can_type in family_map.items():
        rand_list = mol_record.get(rand_type, [])
        can_rep = mol_record.get(can_type)
        if not can_rep or not rand_list:
            continue
        for rand_rep in rand_list[:NUM_RANDOM_VARIANTS]:
            pairs.append((rand_type, rand_rep, can_type, can_rep))
    return pairs


def _gen_cross_normalization_pairs(mol_record: dict, task_config: dict) -> list[tuple]:
    """Cross normalization: random_X → can_Y, exclude same family"""
    pairs = []
    family_map = {
        "random_smiles": "can_smiles",
        "random_selfies": "can_selfies",
        "random_deepsmiles": "can_deepsmiles",
    }
    can_types = ["can_smiles", "can_selfies", "can_deepsmiles", "inchi"]
    for rand_type, same_family_can in family_map.items():
        rand_list = mol_record.get(rand_type, [])
        if not rand_list:
            continue
        for rand_rep in rand_list[:NUM_RANDOM_VARIANTS]:
            for out_type in can_types:
                if task_config.get("exclude_same_family_pairs", True) and out_type == same_family_can:
                    continue
                out_rep = mol_record.get(out_type)
                if out_rep:
                    pairs.append((rand_type, rand_rep, out_type, out_rep))
    return pairs


def render_qa(
    in_type: str, in_rep, out_type: str, out_rep: str, task_name: str, task_config: dict
) -> list[dict]:
    """渲染单个 pair 为 QA 格式，随机选 NUM_TEMPLATES_PER_PAIR 个 template"""
    templates = task_config.get("qa_templates", [])
    if not templates:
        return []
    
    selected_templates = random.sample(
        templates, min(NUM_TEMPLATES_PER_PAIR, len(templates))
    )
    
    in_display = CONFIG_REPS.get(in_type, {}).get("display_name", in_type)
    out_display = CONFIG_REPS.get(out_type, {}).get("display_name", out_type)

    LEGACY_CONSTRAINT_INSTRUCTIONS = "Please output the representation only. Do not output anything else."
    CONSTRAINT_INSTRUCTIONS = [
        "Please output the {output_rep_type} representation only. Do not output anything else.",
        "Output only the {output_rep_type} representation, nothing else.",
        "Just provide the {output_rep_type} without any explanation.",
        "Return only the {output_rep_type} representation.",
        "Give me only the {output_rep_type}, no additional text.",
    ]
    
    results = []
    for template in selected_templates:
        prompt = template.format(input_rep_type=in_display, input_rep=in_rep, output_rep_type=out_display)
        # Select a random constraint instruction for this prompt
        constraint = random.choice(CONSTRAINT_INSTRUCTIONS).format(output_rep_type=out_display)
        prompt += f" {constraint}"
        completion = out_rep
        task_id = f"{task_name}:{in_type}->{out_type}"

        results.append({
            "task_id": task_id,
            "task_type": task_name,
            "input_rep_type": in_type,
            "input_rep": in_rep,
            "output_rep_type": out_type,
            "output_rep": out_rep,
            "prompt": prompt,
            "completion": completion,
            "qa_template": template,
        })
    return results


def batch_map_func_mol_rep_conversion_text(batch: dict) -> dict:
    """Text 任务的 batch map: 从 per-molecule 展平成 per-QA"""
    all_records = []

    num_molecules = len(batch.get("can_smiles", []))
    for i in range(num_molecules):
        mol_record = {k: batch[k][i] for k in batch.keys()}

        for task_name, task_config in CONFIG_REP_CONVERSION.items():
            if task_name == "ocr":
                continue  # OCR 任务走单独的 batch_map_func

            pairs = generate_pairs_for_task(mol_record, task_name, task_config)
            for in_type, in_rep, out_type, out_rep in pairs:
                qa_list = render_qa(in_type, in_rep, out_type, out_rep, task_name, task_config)
                for qa in qa_list:
                    qa["can_smiles"] = mol_record.get("can_smiles")
                    qa["iupac"] = mol_record.get("iupac")
                    qa["inchi"] = mol_record.get("inchi")
                    qa["inchikey"] = mol_record.get("inchikey")
                    all_records.append(qa)

    if not all_records:
        return {
            "can_smiles": [], "iupac": [], "inchi": [], "inchikey": [],
            "prompt": [], "completion": [], "task_id": [], "task_type": [],
            "input_rep_type": [], "input_rep": [], "output_rep_type": [],
            "output_rep": [], "qa_template": [],
        }

    return {
        "can_smiles": [r["can_smiles"] for r in all_records],
        "iupac": [r["iupac"] for r in all_records],
        "inchi": [r["inchi"] for r in all_records],
        "inchikey": [r["inchikey"] for r in all_records],
        "prompt": [r["prompt"] for r in all_records],
        "completion": [r["completion"] for r in all_records],
        "task_id": [r["task_id"] for r in all_records],
        "task_type": [r["task_type"] for r in all_records],
        "input_rep_type": [r["input_rep_type"] for r in all_records],
        "input_rep": [r["input_rep"] for r in all_records],
        "output_rep_type": [r["output_rep_type"] for r in all_records],
        "output_rep": [r["output_rep"] for r in all_records],
        "qa_template": [r["qa_template"] for r in all_records],
    }


def batch_map_func_mol_rep_conversion_ocr(batch: dict) -> dict:
    """OCR 任务的 batch map: 只处理 OCR task"""
    all_records = []

    num_molecules = len(batch.get("can_smiles", []))
    for i in range(num_molecules):
        mol_record = {k: batch[k][i] for k in batch.keys()}

        ocr_config = CONFIG_REP_CONVERSION.get("ocr")
        if not ocr_config:
            continue

        pairs = generate_pairs_for_task(mol_record, "ocr", ocr_config)
        for in_type, in_rep, out_type, out_rep in pairs:
            qa_list = render_qa(in_type, in_rep, out_type, out_rep, "ocr", ocr_config)
            for qa in qa_list:
                qa["can_smiles"] = mol_record.get("can_smiles")
                qa["iupac"] = mol_record.get("iupac")
                qa["inchi"] = mol_record.get("inchi")
                qa["inchikey"] = mol_record.get("inchikey")
                all_records.append(qa)

    if not all_records:
        return {
            "can_smiles": [], "iupac": [], "inchi": [], "inchikey": [],
            "prompt": [], "completion": [], "task_id": [], "task_type": [],
            "input_rep_type": [], "input_rep": [], "output_rep_type": [],
            "output_rep": [], "qa_template": [],
        }

    return {
        "can_smiles": [r["can_smiles"] for r in all_records],
        "iupac": [r["iupac"] for r in all_records],
        "inchi": [r["inchi"] for r in all_records],
        "inchikey": [r["inchikey"] for r in all_records],
        "prompt": [r["prompt"] for r in all_records],
        "completion": [r["completion"] for r in all_records],
        "task_id": [r["task_id"] for r in all_records],
        "task_type": [r["task_type"] for r in all_records],
        "input_rep_type": [r["input_rep_type"] for r in all_records],
        "input_rep": [r["input_rep"] for r in all_records],
        "output_rep_type": [r["output_rep_type"] for r in all_records],
        "output_rep": [r["output_rep"] for r in all_records],
        "qa_template": [r["qa_template"] for r in all_records],
    }


def main(args: dict):
    # kdeng03/mol-reps-v0
    # raw_data_hf = load_dataset(HF_DATASET_ID, split="validation")
    # raw_data_hf = load_dataset(HF_DATASET_ID, split="train")
    raw_data_hf = load_dataset(HF_DATASET_ID, split="extra_train")
    
    if VERSION == "v0":
        # 8:2 train/test split with fixed seed
        split = raw_data_hf.train_test_split(test_size=0.2, seed=42)
        train_data = split["train"]
        test_data = split["test"]
        
        print(f"✅ Raw data: {len(raw_data_hf)} molecules → train: {len(train_data)}, test: {len(test_data)}")

        # Process train split
        train_text = train_data.map(
            batch_map_func_mol_rep_conversion_text,
            batched=True,
            batch_size=64,
            remove_columns=train_data.column_names,
        )
        train_ocr = train_data.map(
            batch_map_func_mol_rep_conversion_ocr,
            batched=True,
            batch_size=64,
            remove_columns=train_data.column_names,
        )

        # Process test split
        test_text = test_data.map(
            batch_map_func_mol_rep_conversion_text,
            batched=True,
            batch_size=64,
            remove_columns=test_data.column_names,
        )
        test_ocr = test_data.map(
            batch_map_func_mol_rep_conversion_ocr,
            batched=True,
            batch_size=64,
            remove_columns=test_data.column_names,
        )

        # Wrap into DatasetDict with train/test splits
        text_dataset = DatasetDict({
            "train": train_text,
            "test": test_text,
        })
        ocr_dataset = DatasetDict({
            "train": train_ocr,
            "test": test_ocr,
        })

    elif VERSION == "v1":
        full_text = raw_data_hf.map(
            batch_map_func_mol_rep_conversion_text,
            batched=True,
            batch_size=64,
            remove_columns=raw_data_hf.column_names,
        )
        full_ocr = raw_data_hf.map(
            batch_map_func_mol_rep_conversion_ocr,
            batched=True,
            batch_size=64,
            remove_columns=raw_data_hf.column_names,
        )

        print(f"✅ Raw data: {len(raw_data_hf)} molecules → text QA: {len(full_text)}, ocr QA: {len(full_ocr)}")

        text_split = full_text.train_test_split(test_size=0.2, seed=42)
        ocr_split = full_ocr.train_test_split(test_size=0.2, seed=42)

        train_text = text_split["train"]
        test_text = text_split["test"]
        train_ocr = ocr_split["train"]
        test_ocr = ocr_split["test"]

        # Wrap into DatasetDict with train/test splits
        text_dataset = DatasetDict({
            "train": train_text,
            "test": test_text,
        })
        ocr_dataset = DatasetDict({
            "train": train_ocr,
            "test": test_ocr,
        })

    elif VERSION == "v1.1":
        # full_text = raw_data_hf.map(
        #     batch_map_func_mol_rep_conversion_text,
        #     batched=True,
        #     batch_size=64,
        #     remove_columns=raw_data_hf.column_names,
        # )
        # full_ocr = raw_data_hf.map(
        #     batch_map_func_mol_rep_conversion_ocr,
        #     batched=True,
        #     batch_size=64,
        #     remove_columns=raw_data_hf.column_names,
        # )

        # print(f"✅ Raw data: {len(raw_data_hf)} molecules → text QA: {len(full_text)}, ocr QA: {len(full_ocr)}")

        # text_split = full_text.train_test_split(test_size=0.2, seed=42)
        # ocr_split = full_ocr.train_test_split(test_size=0.2, seed=42)

        # train_text = text_split["train"]
        # test_text = text_split["test"]
        # test_text_split = test_text.train_test_split(test_size=0.5, seed=42)
        # valid_text = test_text_split["train"]
        # test_text = test_text_split["test"]

        # train_ocr = ocr_split["train"]
        # test_ocr = ocr_split["test"]
        # test_ocr_split = test_ocr.train_test_split(test_size=0.5, seed=42)
        # valid_ocr = test_ocr_split["train"]
        # test_ocr = test_ocr_split["test"]

        # ### Extra test
        # extra_raw_data_hf = load_dataset(HF_DATASET_ID, split="test")

        # full_text = extra_raw_data_hf.map(
        #     batch_map_func_mol_rep_conversion_text,
        #     batched=True,
        #     batch_size=64,
        #     remove_columns=extra_raw_data_hf.column_names,
        # )
        # full_ocr = extra_raw_data_hf.map(
        #     batch_map_func_mol_rep_conversion_ocr,
        #     batched=True,
        #     batch_size=64,
        #     remove_columns=extra_raw_data_hf.column_names,
        # )

        
        conversion = load_dataset(UPLOAD_HF_DATASET_ID)
        ocr = load_dataset(UPLOAD_HF_DATASET_ID_OCR)
        
        validset = load_dataset(HF_DATASET_ID, split="extra_valid")

        ### Extra train
        full_text = raw_data_hf.map(
            batch_map_func_mol_rep_conversion_text,
            batched=True,
            batch_size=64,
            remove_columns=raw_data_hf.column_names,
        )
        full_ocr = raw_data_hf.map(
            batch_map_func_mol_rep_conversion_ocr,
            batched=True,
            batch_size=64,
            remove_columns=raw_data_hf.column_names,
        )

        ### Extra valid
        valid_text = validset.map(
            batch_map_func_mol_rep_conversion_text,
            batched=True,
            batch_size=64,
            remove_columns=validset.column_names,
        )
        valid_ocr = validset.map(
            batch_map_func_mol_rep_conversion_ocr,
            batched=True,
            batch_size=64,
            remove_columns=validset.column_names,
        )

        # Wrap into DatasetDict with train/test splits
        text_dataset = DatasetDict({
            "train": conversion["train"],
            "valid": conversion["valid"],
            "test": conversion["test"],
            "extra_train": full_text,
            "extra_valid": valid_text,
            "extra_test": conversion["extra_test"],
        })
        ocr_dataset = DatasetDict({
            "train": ocr["train"],
            "valid": ocr["valid"],
            "test": ocr["test"],
            "extra_train": full_ocr,
            "extra_valid": valid_ocr,
            "extra_test": ocr["extra_test"],
        })

    else:
        raise ValueError(f"Unsupported version: {VERSION}")
        

    print(f"✅ Text dataset: train={len(text_dataset['train'])}, test={len(text_dataset['test'])}")
    print(f"✅ OCR dataset: train={len(ocr_dataset['train'])}, test={len(ocr_dataset['test'])}")
    
    return text_dataset, ocr_dataset


if __name__ == "__main__":
    args = load_args()
    text_dataset, ocr_dataset = main(args)
    if UPLOAD_HF:
        text_dataset.push_to_hub(UPLOAD_HF_DATASET_ID, token=os.environ["HF_TOKEN"])
        print(f"✅ Uploaded text dataset to Hugging Face Hub with ID: {UPLOAD_HF_DATASET_ID}.")
        ocr_dataset.push_to_hub(UPLOAD_HF_DATASET_ID_OCR, token=os.environ["HF_TOKEN"])
        print(f"✅ Uploaded OCR dataset to Hugging Face Hub with ID: {UPLOAD_HF_DATASET_ID_OCR}.")
