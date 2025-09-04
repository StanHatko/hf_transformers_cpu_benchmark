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


def do_prediction(model):
    t1 = time.time()
    with torch.no_grad():
        y = model(x)
    t2 = time.time()
    print("Time:", t2 - t1)
    return y


# Load model, without quantization.
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
print("Model dtype:", model.dtype)

# Test model prediction.
y1 = do_prediction(model)


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
y2a = do_prediction(quantized_model)
y2b = do_prediction(quantized_model)

# Show difference.
print(torch.abs(torch.abs(y1.logits - y2b.logits)))
print(torch.abs(torch.abs(y2a.logits - y2b.logits)))


# Compile the model.
comp_model = torch.compile(
    quantized_model,
    backend="openvino",
)
print("Model dtype:", comp_model.dtype)

# Test model prediction.
y3a = do_prediction(comp_model)
y3b = do_prediction(comp_model)

# Show difference.
print(torch.abs(torch.abs(y1.logits - y3b.logits)))
print(torch.abs(torch.abs(y2b.logits - y3b.logits)))
print(torch.abs(torch.abs(y3a.logits - y3b.logits)))
# Did not generate speedup.
