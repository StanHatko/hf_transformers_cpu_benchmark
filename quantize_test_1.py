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


# Test quantize model with TorchAO.
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
)

quant_config = Int8DynamicActivationInt8WeightConfig()
# quant_config = Int8WeightOnlyConfig()
quantization_config = TorchAoConfig(quant_type=quant_config)

# Load and quantize the model.
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cpu",
    quantization_config=quantization_config,
)
print("Model dtype:", quantized_model.dtype)

# Test model prediction.
with torch.no_grad():
    y2 = quantized_model(x)

# Show difference.
print(torch.abs(torch.abs(y1.logits - y2.logits)))


# Compile the model.
comp_model = torch.compile(
    quantized_model,
    backend="openvino",
    fullgraph=True,
)
print("Model dtype:", comp_model.dtype)

# Test model prediction.
with torch.no_grad():
    y3 = comp_model(x)

# Show difference.
print(torch.abs(torch.abs(y1.logits - y3.logits)))
print(torch.abs(torch.abs(y2.logits - y3.logits)))
