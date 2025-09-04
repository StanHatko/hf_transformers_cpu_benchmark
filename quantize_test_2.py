"""
Initial tests with model quantization.
"""

import json
import math
import time

import torch
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from nncf import compress_weights, CompressWeightsMode
from optimum.intel.openvino import OVModelForCausalLM

model_name = "Qwen/Qwen3-4B-Instruct-2507"

# Input data for testing.
x = torch.randint(100000, (1, 64))


# Load model, without quantization.
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
print("Model dtype:", model.dtype)

# Test model prediction.
with torch.no_grad():
    y1 = model(x)


# Quantize model.
model = OVModelForCausalLM.from_pretrained(
    model_name,
    export=True,
    load_in_8bit=False,
    compile=False,
)
model.model = compress_weights(model.model, mode=CompressWeightsMode.INT8_ASYM)
