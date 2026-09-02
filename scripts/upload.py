import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "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"
# os.environ["HF_HUB_CACHE"] = "~/autodl-tmp/ckpt/tmp/"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/ckpt/tmp/"

os.environ["TMPDIR"] = "/root/autodl-tmp/ckpt/tmp/"
os.environ["TEMP"] = "/root/autodl-tmp/ckpt/tmp/"
os.environ["TMP"] = "/root/autodl-tmp/ckpt/tmp/"

from peft import AutoPeftModelForCausalLM, AutoPeftModel
from transformers import AutoProcessor


# UPLOAD_LORA_ID = "kdeng03/MolQwen3-VL-4B-Instruct-SFT-LoRA-Adapter"
# UPLOAD_MODEL_ID = "kdeng03/MolQwen3-VL-4B-Instruct-SFT"
# LORA_DIR = "ckpt/conversion/qwen3_vl_4b_i_sft_lora_conversion/v1.1/20260815_095650"

# UPLOAD_LORA_ID = "kdeng03/MolQwen3-VL-4B-Instruct-SFT-OCR-LoRA-Adapter"
# UPLOAD_MODEL_ID = "kdeng03/MolQwen3-VL-4B-Instruct-SFT-OCR"
# LORA_DIR = "ckpt/ocr/qwen3_vl_4b_i_sft_lora_ocr/v1.1/20260814_162454"

# UPLOAD_LORA_ID = "kdeng03/MolQwen3-4B-Instruct-SFT-LoRA-Adapter"
# UPLOAD_MODEL_ID = "kdeng03/MolQwen3-4B-Instruct-SFT"
# LORA_DIR = "ckpt/conversion/qwen3_4b_i_sft_lora_conversion/v1.1/20260814_235452"

# UPLOAD_LORA_ID = "kdeng03/MolQwen3-VL-4B-Instruct-SFT-LoRA-Adapter"
# UPLOAD_MODEL_ID = "kdeng03/MolQwen3-VL-4B-Instruct-SFT"
# LORA_DIR = "ckpt/ocr_conversion/qwen3_vl_4b_i_sft_lora_ocr_conversion/v1.1/20260816_195334/checkpoint-4370"

UPLOAD_LORA_ID = "kdeng03/MolQwen3-VL-4B-Instruct-SFT-Epoch4-LoRA-Adapter"
UPLOAD_MODEL_ID = "kdeng03/MolQwen3-VL-4B-Instruct-SFT-Epoch4"
LORA_DIR = "ckpt/ocr_conversion/qwen3_vl_4b_i_sft_lora_ocr_conversion/v1.1/20260816_195334/checkpoint-1748"


processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
# processor = AutoProcessor.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

model = AutoPeftModel.from_pretrained(LORA_DIR, dtype="bfloat16")
model.push_to_hub(UPLOAD_LORA_ID, token=os.environ["HF_TOKEN"])
processor.push_to_hub(UPLOAD_LORA_ID, token=os.environ["HF_TOKEN"])

model = model.merge_and_unload()
model.push_to_hub(UPLOAD_MODEL_ID, token=os.environ["HF_TOKEN"])
processor.push_to_hub(UPLOAD_MODEL_ID, token=os.environ["HF_TOKEN"])
