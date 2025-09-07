"""
Test Intel-optimized PyTorch, with torchao int8 quantization.
"""

import time

import torch
import torchao
import intel_extension_for_pytorch as ipex
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


# Load model.
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
)
torchao.quantization.quantize_(
    model,
    torchao.quantization.Int8DynamicActivationInt8WeightConfig(),
)
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

# Compile model.
torch._dynamo.config.recompile_limit = 256
model = ipex.optimize(model, inplace=True)
model = torch.compile(model, backend="ipex")

# Test model prediction.
y2a = do_prediction(model, x)
y2b = do_prediction(model, x)
