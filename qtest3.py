"""
Test quantization with Intel-optimized PyTorch.
"""

import time

import torch
import intel_extension_for_pytorch as ipex
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig


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


# Load model.
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    quantization_config=QuantoConfig(weights="int8"),
)
print("Model dtype:", model.dtype)

# Test model prediction.
y1a = do_prediction(model)
y1b = do_prediction(model)

# Compile model.
model = ipex.optimize(model, inplace=True)
model = torch.compile(model, backend="ipex")

# Test model prediction.
y2a = do_prediction(model)
y2b = do_prediction(model)
