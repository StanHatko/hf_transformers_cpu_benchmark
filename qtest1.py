"""
Quantization and benchmarking tests.
"""

import json
import math
import time

import torch
import nncf
import optimum.intel.openvino
import openvino
import openvino.torch
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
    print("Time to predict:", t2 - t1)
    return y


# Load model, without quantization.
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
print("Model dtype:", model.dtype)

# Test model prediction.
y1a = do_prediction(model)
y1b = do_prediction(model)
"""
Time to predict: 5.375474452972412
Time to predict: 4.652477979660034
"""


# Do computation layer-by-layer.
v1 = model.model.embed_tokens(x)
v2 = list()
v2.append(model.model.layers[0](v1))

# Test layers with and without NCCF compression.
breakpoint()
