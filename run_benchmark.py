#!/usr/bin/env python
"""
Check speed of running LLM on CPU with huggingface transformers.
"""

import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_tokenizer_model(model_name: str) -> tuple:
    """
    Load the model and tokenizer.
    """

    t1 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    t2 = time.time()

    print(f"Took {round(t2 - t1, 2)} seconds to load model.")
    return tokenizer, model


def encode_inputs(tokenizer, inputs: list):
    """
    Convert list of inputs into tokenized inputs, with PyTorch tensors.
    Parameter inputs is list, each entry of which is a list of messages.
    Runs on CPU here so OK for not to move to GPU device.
    """

    t1 = time.time()
    x = [
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        for messages in inputs
    ]
    t2 = time.time()

    print(f"Took {round(t2 - t1, 2)} seconds to encode inputs.")
    return x


def generate_llm(model, input_data: list, max_tokens: int):
    """
    Run the LLM on the input data.
    """

    t1 = time.time()
    outputs = [
        model.generate(
            **x,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
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
    for x, y in zip(input_data, output_data):
        nx = x["input_ids"].shape[-1]
        w = tokenizer.decode(y[0, nx:])
        r.append(w)
    t2 = time.time()

    print(f"Took {round(t2 - t1, 2)} seconds to decode LLM output.")
    return r


def benchmark_llm(model_name: str, num_cpu: int, max_tokens: int):
    """
    Benchmark speed of running specified LLM on some queries.
    """

    print("Benchmark speed of model on CPU:", model_name)
    print("Number of CPU cores to use:", num_cpu)
    os.environ["OMP_NUM_THREADS"] = str(num_cpu)
    torch.set_num_threads(num_cpu)

    tokenizer, model = load_tokenizer_model(model_name)

    # TODO: replace hardcoded input!
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    inputs = encode_inputs(
        tokenizer,
        [
            messages,
        ],
    )

    num_outputs = generate_llm(model, inputs, max_tokens)
    final_outputs = decode_outputs(tokenizer, inputs, num_outputs)
    print("Contents of final outputs:", final_outputs)
