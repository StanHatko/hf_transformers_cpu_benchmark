#!/usr/bin/env python
"""
Check speed of running LLM on CPU with huggingface transformers.
"""

import json
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_tokenizer_model(model_name: str) -> tuple:
    """
    Load the model and tokenizer.
    """

    t1 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    t2 = time.time()
    print(f"Took {round(t2 - t1, 2)} seconds to load model.")

    print("Compile the model...")
    model = torch.compile(model, mode="max-autotune", fullgraph=True)
    t3 = time.time()
    print(f"Took {round(t3 - t2, 2)} seconds to compile model.")

    return tokenizer, model


def load_inputs(input_file: str) -> list:
    """
    Load the input data.
    """
    t1 = time.time()
    with open(input_file, "r", encoding="utf-8") as f:
        inputs_raw = json.load(f)
    t2 = time.time()
    print(f"Took {round(t2 - t1, 2)} seconds to load input data.")
    return inputs_raw


def encode_inputs(tokenizer, inputs: list):
    """
    Convert list of inputs into tokenized inputs, with PyTorch tensors.
    Parameter inputs is list, each entry of which is a list of messages.
    Runs on CPU here so OK for not to move to GPU device.
    """

    t1 = time.time()
    y = [
        tokenizer.apply_chat_template(
            x,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        for x in inputs
    ]
    t2 = time.time()

    print(f"Took {round(t2 - t1, 2)} seconds to encode inputs.")
    return y


def generate_llm(model, input_data: list, max_tokens: int):
    """
    Run the LLM on the input data.
    """

    t1 = time.time()
    outputs = [
        model.generate(**x, max_new_tokens=max_tokens, do_sample=False)
        for x in input_data
    ]
    t2 = time.time()

    print(f"Took {round(t2 - t1, 2)} seconds to run LLM.")
    return outputs


def decode_outputs(tokenizer, input_data: list, output_data: list):
    """
    Convert numeric outputs back into tokens.
    """

    t1 = time.time()
    r = []
    breakpoint()
    for x, y in zip(input_data, output_data):
        nx = x["input_ids"].shape[-1]
        w = tokenizer.decode(y[0, nx:])
        r.append(w)
    t2 = time.time()

    print(f"Took {round(t2 - t1, 2)} seconds to decode LLM output.")
    return r


def benchmark_llm(model_name: str, input_file: str, num_cpu: int, max_tokens: int):
    """
    Benchmark speed of running specified LLM on some queries.
    """

    print("Benchmark speed of model on CPU:", model_name)
    print("Input file:", input_file)
    print("Number of CPU cores to use:", num_cpu)
    print("Maximum number of output tokens:", max_tokens)

    torch.set_num_threads(num_cpu)
    tokenizer, model = load_tokenizer_model(model_name)

    input_raw = load_inputs(input_file)
    input_encoded = encode_inputs(tokenizer, input_raw)

    output_encoded = generate_llm(model, input_encoded, max_tokens)
    breakpoint()
    output_decoded = decode_outputs(tokenizer, input_encoded, output_encoded)
    print("Contents of final outputs:", output_decoded)
