"""Download and verify the public Qwen base model used by QLoRA training."""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# Keep the multi-gigabyte model cache with this project rather than on C:.
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

if not torch.cuda.is_available():
    raise RuntimeError("Qwen 4-bit loading requires the CUDA-enabled project environment.")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=False,
    quantization_config=quantization_config,
    device_map={"": 0},
    dtype=torch.float16,
)

print(f"Qwen 4-bit model loaded on {next(model.parameters()).device}.")
