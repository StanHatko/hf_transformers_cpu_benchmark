"""
Initial tests with model quantization.
"""

import json
import math
import time

import nncf
from nncf import compress_weights
import optimum.intel.openvino
import openvino
import openvino.torch
import torch
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig

model_name = "Qwen/Qwen3-4B-Instruct-2507"

# Input data for testing.
x = torch.randint(100000, (1, 64))


def do_prediction(model):
    t1 = time.time()
    with torch.no_grad():
        y = model(x)
    t2 = time.time()
    print("Time to predict:", t2 - t1)
    return y


# Load model, without quantization.
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
print("Model dtype:", model.dtype)

# Test model prediction.
y1 = do_prediction(model)


# Compress the weights.
quant_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    quantization_config=QuantoConfig(weights="int8"),
)

# Test model prediction.
y2a = do_prediction(quant_model)
y2b = do_prediction(quant_model)

# Show difference.
print(torch.abs(torch.abs(y1.logits - y2b.logits)))
print(torch.abs(torch.abs(y2a.logits - y2b.logits)))
# Generated significant speedup, proceed with this.


# Compile the model with openvino backend.
t1 = time.time()
comp_model = torch.compile(
    quant_model,
    backend="openvino",
    options={
        "device": "cpu",
    },
)
t2 = time.time()
print("Time to compile model:", t2 - t1)

# Test model prediction.
y3a = do_prediction(comp_model)
y3b = do_prediction(comp_model)

# Show difference.
print(torch.abs(torch.abs(y1.logits - y3b.logits)))
print(torch.abs(torch.abs(y2b.logits - y3b.logits)))
print(torch.abs(torch.abs(y3a.logits - y3b.logits)))
# No significant affect.
