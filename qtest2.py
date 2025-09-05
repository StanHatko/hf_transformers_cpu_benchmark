"""
Test FX int8 quantization, from https://www.intel.com/content/www/us/en/developer/articles/technical/int8-quantization-for-x86-cpu-in-pytorch.html
"""

import time
import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.quantization.quantize_fx import prepare_fx, convert_fx
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
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
print("Model dtype:", model.dtype)

# Test model prediction.
y1a = do_prediction(model)
y1b = do_prediction(model)


# Do quantization.
qengine = "x86"
torch.backends.quantized.engine = qengine
qconfig_mapping = get_default_qconfig_mapping(qengine)

prepared_model = prepare_fx(model, qconfig_mapping, example_inputs=x)
quantized_model = convert_fx(prepared_model)
