"""
Quantization and benchmarking tests.
"""

import json
import math
import time

import torch
import torchao
import nncf
import optimum.intel.openvino
import openvino
import openvino.torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-4B-Instruct-2507"

# Input data for testing.
n = 128
x = torch.tensor([list(range(10000, 10000 + n))])


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
Time to predict: 10.654382705688477
Time to predict: 9.399781465530396
"""


# Loop over linear layers and apply nccf quantization.
breakpoint()
nncf.compress_weights(model, dataset=nncf.Dataset(x))
torchao.quantization.quantize_(
    model,
    torchao.quantization.Int8DynamicActivationInt8WeightConfig(),
)

# Test model prediction.
y2a = do_prediction(model)
y2b = do_prediction(model)
