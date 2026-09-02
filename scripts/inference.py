import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "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"
import argparse
from pathlib import Path

import yaml
import jsonlines
from tqdm import tqdm
from datasets import load_dataset

from src.inference import VLLMEngine


VERSION = "v1.1"

### Debug for training, default is False, when True, it runs on trainset to check training pipeline.
INFER_ON_TRAINSET = True

RAW_RESPONSES_FILENAME = "{task_name}_{model_name}.jsonl"
RAW_RESPONSES_DIR = "results/{task_name}/{VERSION}"
CONFIG_INFER_FILE = "configs/model_inference_schema.yaml"
with open(CONFIG_INFER_FILE, "r") as f:
    CONFIG_INFER: dict = yaml.safe_load(f)


def load_args() -> dict:
    parser = argparse.ArgumentParser(description="Generate molecular representations from SMILES.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Path to the configuration file defining inference schema."
    )
    args_parser = parser.parse_args()

    # Merge
    args: dict = vars(args_parser)    
    args.update(CONFIG_INFER)
    return args


def load_mol_rep_ocr_dataset():
    assert VERSION in ["v0", "v1", "v1.1"], f"VERSION {VERSION} is not supported..."
    HF_DATASET_ID = f"kdeng03/mol-rep-ocr-{VERSION}"
    SPLIT = "test" if not INFER_ON_TRAINSET else "train" # Can be train for debugging

    data_hf = load_dataset(HF_DATASET_ID, split=SPLIT)
    return data_hf


def load_mol_rep_ocr_extra_testset():
    assert VERSION in ["v1.1"], f"VERSION {VERSION} is not supported..."
    HF_DATASET_ID = f"kdeng03/mol-rep-ocr-{VERSION}"
    SPLIT = "extra_test"

    data_hf = load_dataset(HF_DATASET_ID, split=SPLIT)
    return data_hf


def load_mol_rep_ocr_messages(data_hf) -> list[list[dict]]:
    messages_list = []
    for sample in tqdm(data_hf, desc="Loading OCR messages"):
        prompt = sample["prompt"] # TODO: Need to change if another dataset is used
        image = sample["input_rep"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        messages_list.append(messages)
    return messages_list


def load_mol_rep_conversion_dataset():
    assert VERSION in ["v0", "v1", "v1.1"], f"VERSION {VERSION} is not supported..."
    HF_DATASET_ID = f"kdeng03/mol-rep-conversion-{VERSION}"
    SPLIT = "test" if not INFER_ON_TRAINSET else "train" # Can be train for debugging

    data_hf = load_dataset(HF_DATASET_ID, split=SPLIT)
    return data_hf


def load_mol_rep_conversion_extra_testset():
    assert VERSION in ["v1.1"], f"VERSION {VERSION} is not supported..."
    HF_DATASET_ID = f"kdeng03/mol-rep-conversion-{VERSION}"
    SPLIT = "extra_test"

    data_hf = load_dataset(HF_DATASET_ID, split=SPLIT)
    return data_hf


def load_mol_rep_conversion_messages(data_hf) -> list[list[dict]]:
    messages_list = []
    for sample in tqdm(data_hf, desc="Loading conversion messages"):
        prompt = sample["prompt"] # TODO: Need to change if another dataset is used
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        messages_list.append(messages)
    return messages_list


def main(args: dict):
    model_name = args["model_name"]
    llm_engine = VLLMEngine(model_name)

    if INFER_ON_TRAINSET:
        print("🔧🔧 [DEBUG] Running inference on trainset to check training pipeline...")

    ### OCR
    if llm_engine.modality == "vlm":
        task_name = "mol_rep_ocr"
        ocr_data_hf = load_mol_rep_ocr_dataset()
        ocr_messages = load_mol_rep_ocr_messages(ocr_data_hf)
        raw_responses: list[str] = llm_engine.chat(ocr_messages)
        print(f"✅ Responses for {task_name} have been generated: {len(raw_responses)} samples.")

        ### Save raw responses
        raw_responses_dir = Path(RAW_RESPONSES_DIR.format(task_name=task_name, VERSION=VERSION))
        raw_responses_dir = raw_responses_dir / "trainset" if INFER_ON_TRAINSET else raw_responses_dir # Debug
        raw_responses_filename = Path(RAW_RESPONSES_FILENAME.format(task_name=task_name, model_name=model_name))
        raw_responses_path = raw_responses_dir / "raw_responses" / raw_responses_filename
        raw_responses_path.parent.mkdir(parents=True, exist_ok=True)

        all_rows = []
        for sample, raw_response in tqdm(zip(ocr_data_hf, raw_responses), total=len(ocr_data_hf), desc="Saving OCR responses"):
            sample["input_rep"] = sample["input_rep"].size # Convert PIL image to size for saving
            row = {
                **sample,
                "task_name": task_name,
                "model_name": model_name,
                "model_modality": llm_engine.modality,
                "model_stage": llm_engine.config.get("training_stage", None),
                "model_training_method": llm_engine.config.get("training_method", None),
                "model_parent_training_method": llm_engine.config.get("parent_training_method", None),
                "raw_responses": raw_response,
            }
            all_rows.append(row)

        with jsonlines.open(str(raw_responses_path), mode='w') as writer:
            writer.write_all(all_rows)

        print(f"✅ Raw responses for {task_name} have been saved to {raw_responses_path}.")

        ### Testing on extra molecules testset
        if not INFER_ON_TRAINSET:
            print(f"⌛️⌛️ Processing extra test set for {task_name}...")
            ocr_data_hf = load_mol_rep_ocr_extra_testset()  
            ocr_messages = load_mol_rep_ocr_messages(ocr_data_hf)
            raw_responses: list[str] = llm_engine.chat(ocr_messages)
            print(f"✅ Responses for {task_name} extra testset have been generated: {len(raw_responses)} samples.")

            ### Save raw responses
            raw_responses_dir = Path(RAW_RESPONSES_DIR.format(task_name=task_name, VERSION=VERSION))
            raw_responses_dir = raw_responses_dir / "extra_test"
            raw_responses_filename = Path(RAW_RESPONSES_FILENAME.format(task_name=task_name, model_name=model_name))
            raw_responses_path = raw_responses_dir / "raw_responses" / raw_responses_filename
            raw_responses_path.parent.mkdir(parents=True, exist_ok=True)

            all_rows = []
            for sample, raw_response in tqdm(zip(ocr_data_hf, raw_responses), total=len(ocr_data_hf), desc="Saving OCR responses"):
                sample["input_rep"] = sample["input_rep"].size # Convert PIL image to size for saving
                row = {
                    **sample,
                    "task_name": task_name,
                    "model_name": model_name,
                    "model_modality": llm_engine.modality,
                    "model_stage": llm_engine.config.get("training_stage", None),
                    "model_training_method": llm_engine.config.get("training_method", None),
                    "model_parent_training_method": llm_engine.config.get("parent_training_method", None),
                    "raw_responses": raw_response,
                }
                all_rows.append(row)

            with jsonlines.open(str(raw_responses_path), mode='w') as writer:
                writer.write_all(all_rows)

            print(f"✅ Raw responses for {task_name} have been saved to {raw_responses_path}.")
    else:
        print(f"🤔🤔 Model {model_name} is not a VLM model. Skipping OCR task.")

    ### Conversion
    task_name = "mol_rep_conversion"
    conversion_data_hf = load_mol_rep_conversion_dataset()
    conversion_messages = load_mol_rep_conversion_messages(conversion_data_hf)
    raw_responses: list[str] = llm_engine.chat(conversion_messages)
    print(f"✅ Responses for {task_name} have been generated: {len(raw_responses)} samples.")

    ### Save raw responses
    raw_responses_dir = Path(RAW_RESPONSES_DIR.format(task_name=task_name, VERSION=VERSION))
    raw_responses_dir = raw_responses_dir / "trainset" if INFER_ON_TRAINSET else raw_responses_dir # Debug
    raw_responses_filename = Path(RAW_RESPONSES_FILENAME.format(task_name=task_name, model_name=model_name))
    raw_responses_path = raw_responses_dir / "raw_responses" / raw_responses_filename
    raw_responses_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for sample, raw_response in zip(conversion_data_hf, raw_responses):
        row = {
            **sample,
            "task_name": task_name,
            "model_name": model_name,
            "model_modality": llm_engine.modality,
            "model_stage": llm_engine.config.get("training_stage", None),
            "model_training_method": llm_engine.config.get("training_method", None),
            "model_parent_training_method": llm_engine.config.get("parent_training_method", None),
            "raw_responses": raw_response,
        }
        all_rows.append(row)

    with jsonlines.open(str(raw_responses_path), mode='w') as writer:
        writer.write_all(all_rows)

    print(f"✅ Raw responses for {task_name} have been saved to {raw_responses_path}.")

    ### Testing on extra molecules testset
    if not INFER_ON_TRAINSET:
        print(f"⌛️⌛️ Processing extra test set for {task_name}...")
        conversion_data_hf = load_mol_rep_conversion_extra_testset()
        conversion_messages = load_mol_rep_conversion_messages(conversion_data_hf)
        raw_responses: list[str] = llm_engine.chat(conversion_messages)
        print(f"✅ Responses for {task_name} extra testset have been generated: {len(raw_responses)} samples.")

        ### Save raw responses
        raw_responses_dir = Path(RAW_RESPONSES_DIR.format(task_name=task_name, VERSION=VERSION))
        raw_responses_dir = raw_responses_dir / "extra_test"
        raw_responses_filename = Path(RAW_RESPONSES_FILENAME.format(task_name=task_name, model_name=model_name))
        raw_responses_path = raw_responses_dir / "raw_responses" / raw_responses_filename
        raw_responses_path.parent.mkdir(parents=True, exist_ok=True)

        all_rows = []
        for sample, raw_response in zip(conversion_data_hf, raw_responses):
            row = {
                **sample,
                "task_name": task_name,
                "model_name": model_name,
                "model_modality": llm_engine.modality,
                "model_stage": llm_engine.config.get("training_stage", None),
                "model_training_method": llm_engine.config.get("training_method", None),
                "model_parent_training_method": llm_engine.config.get("parent_training_method", None),
                "raw_responses": raw_response,
            }
            all_rows.append(row)

        with jsonlines.open(str(raw_responses_path), mode='w') as writer:
            writer.write_all(all_rows)

        print(f"✅ Raw responses for {task_name} have been saved to {raw_responses_path}.")


if __name__ == "__main__":
    args = load_args()
    main(args)
    print(f"✅ Inference completed...")
