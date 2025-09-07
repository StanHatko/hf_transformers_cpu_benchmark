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
input_messages = [
    {
        "role": "user",
        "content": "Sort the following numbers from smallest to largest: 3572, 3957, 552, 753, 6541, 4069, 9722, 3450, 4421, 9722, 857, 7876, 195, 8549, 2859, 4387, 1488, 761, 1460, 2777",
    }
]


def do_prediction(model, x):
    t1 = time.time()
    with torch.no_grad():
        y = model(**x)
    t2 = time.time()
    print("Time to predict:", t2 - t1)
    return y


# Load model, without quantization.
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
print("Model dtype:", model.dtype)

# Do tokenization.
x = tokenizer.apply_chat_template(
    input_messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    padding=True,
)

# Test model prediction.
y1a = do_prediction(model, x)
y1b = do_prediction(model, x)
"""
Time to predict: 10.654382705688477
Time to predict: 9.399781465530396
"""


# Loop over linear layers and apply nccf quantization.
nncf.compress_weights(model, dataset=nncf.Dataset(x))

# Test model prediction.
y2a = do_prediction(model, x)
y2b = do_prediction(model, x)
