import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "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"
import argparse

import yaml
from datasets import load_dataset, DatasetDict

from src.dataset.mol_reps import MolReps


VERSION = "v0.1"
UPLOAD_HF = True
UPLOAD_HF_DATASET_ID = f"kdeng03/mol-reps-{VERSION}"
HF_DATASET_ID = "liupf/ChEBI-20-MM" # Source dataset
CONFIG_FILE = "configs/mol_rep_schema.yaml"


def load_args() -> dict:
    parser = argparse.ArgumentParser(description="Generate molecular representations from SMILES.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=False,
        help="Path to the input file containing SMILES strings (one per line).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=False,
        help="Path to the output file where molecular representations will be saved.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=CONFIG_FILE,
        help="Path to the configuration file defining molecular representation schema.",
    )
    args_parser = parser.parse_args()

    with open(CONFIG_FILE, "r") as f:
        args_yaml: dict = yaml.safe_load(f)

    # Merge
    args = vars(args_parser)    
    args.update(args_yaml)
    return args


def map_func(sample: dict) -> dict:
    smiles = sample["SMILES"]
    try:
        mol = MolReps(smiles)
        can_smiles = mol.can_smiles
        random_smiles = mol.random_smiles
        can_selfies = mol.can_selfies
        random_selfies = mol.random_selfies
        can_deepsmiles = mol.can_deepsmiles
        random_deepsmiles = mol.random_deepsmiles
        inchi = mol.inchi
        inchikey = mol.inchikey
        iupac = mol.iupac
        cml = mol.cml
        mol_image = mol.mol_image

        return {
            "can_smiles": can_smiles,
            "random_smiles": random_smiles,
            "can_selfies": can_selfies,
            "random_selfies": random_selfies,
            "can_deepsmiles": can_deepsmiles,
            "random_deepsmiles": random_deepsmiles,
            "inchi": inchi,
            "inchikey": inchikey,
            "iupac": iupac,
            "cml": cml,
            "mol_image": mol_image,

            # raw cols
            "src_dataset": HF_DATASET_ID,
            "raw_CID": sample["CID"],
            "raw_SMILES": sample["SMILES"],
            "raw_SELFIES": sample["SELFIES"],
            "raw_inchi": sample["InChI"],
            "raw_iupacname": sample["iupacname"],
            "raw_description": sample["description"],
            "raw_xlogp": sample["xlogp"],
            "raw_polararea": sample["polararea"],
        }

    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return {
            "can_smiles": None,
            "random_smiles": None,
            "can_selfies": None,
            "random_selfies": None,
            "can_deepsmiles": None,
            "random_deepsmiles": None,
            "inchi": None,
            "inchikey": None,
            "iupac": None,
            "cml": None,
            "mol_image": None,

            # raw cols
            "src_dataset": HF_DATASET_ID,
            "raw_CID": sample["CID"],
            "raw_SMILES": sample["SMILES"],
            "raw_SELFIES": sample["SELFIES"],
            "raw_inchi": sample["InChI"],
            "raw_iupacname": sample["iupacname"],
            "raw_description": sample["description"],
            "raw_xlogp": sample["xlogp"],
            "raw_polararea": sample["polararea"],
        }


def main(args: dict):
    # liupf/ChEBI-20-MM
    # raw_data_hf = load_dataset(HF_DATASET_ID, split="validation")
    raw_data_hf = load_dataset(HF_DATASET_ID, split="train")

    if VERSION == "v0":
        subset_raw_data_hf = raw_data_hf.shuffle(seed=42).select(range(100)) # 100 molecules
        sample = subset_raw_data_hf[0]
    elif VERSION == "v0.1":
        # Draw 240 to allow for some failures, then select 220 valid ones
        subset_raw_data_hf = raw_data_hf.shuffle(seed=42).select(range(65)) # TODO: 65 molecules for extra_train
        sample = subset_raw_data_hf[0]
    else:
        raise ValueError(f"Unsupported version: {VERSION}")

    ### Rename inchi col name
    subset_raw_data_hf = subset_raw_data_hf.rename_column("inchi", "InChI")
    processed_data_hf = subset_raw_data_hf.map(map_func, remove_columns=subset_raw_data_hf.column_names)
    
    # Filter out empty results from failed processing
    processed_data_hf = processed_data_hf.filter(lambda x: x["can_smiles"] is not None and len(x["random_smiles"]) == 5)
    
    # Select exactly 60 valid molecules
    if len(processed_data_hf) >= 60:
        processed_data_hf = processed_data_hf.shuffle(seed=42).select(range(60))
        print(f"✅ Selected targeted valid molecules from {len(processed_data_hf)} successfully processed.")
    else:
        print(f"⚠️ Only {len(processed_data_hf)} valid molecules available (not reaching targeted).")

    print(f"✅ Processed {len(processed_data_hf)} molecules from the dataset {HF_DATASET_ID}.")
    return processed_data_hf


if __name__ == "__main__":
    args = load_args()
    data_hf = main(args)

    dataset_hf = data_hf.train_test_split(test_size=20, seed=42)
    origin_data = load_dataset(UPLOAD_HF_DATASET_ID)

    ### TODO: Create new dataset with the same schema as the original dataset, but with the new data
    dataset_hf = DatasetDict({
        "train": origin_data["train"],
        "extra_train": dataset_hf["train"],
        "extra_valid": dataset_hf["test"],
        "test": origin_data["test"]
    })

    if UPLOAD_HF:
        dataset_hf.push_to_hub(UPLOAD_HF_DATASET_ID, token=os.environ["HF_TOKEN"])
        print(f"✅ Uploaded processed dataset to Hugging Face Hub with ID: {UPLOAD_HF_DATASET_ID}.")
