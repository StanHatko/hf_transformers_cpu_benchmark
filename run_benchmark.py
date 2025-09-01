#!/usr/bin/env python
"""
Check speed of running LLM on CPU with huggingface transformers.
"""

import json
import math
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Benchmark:
    """
    Benchmark LLM on CPU.
    Methods add data to object during run.
    """

    def __init__(self, model_name: str, input_file: str, num_cpu: int, max_tokens: int):
        print("Benchmark speed of model:", model_name)
        print("Input file:", input_file)
        print("Number of CPU cores to use:", num_cpu)
        print("Maximum number of output tokens:", max_tokens)

        self.model_name = model_name
        self.num_cpu = num_cpu
        self.input_file = input_file
        self.max_tokens = max_tokens
        self.times = {}

        torch.set_num_threads(num_cpu)
        self.tokenizer, self.model = self.load_tokenizer_model()
        self.inputs_raw = self.load_inputs()
        self.inputs_encoded = self.encode_inputs()
        self.outputs_llm = self.generate_llm()
        self.outputs_decoded = self.decode_outputs()

    def time_benchmark(self, name: str, t1: float, t2: float):
        """
        Print and add time benchmark.
        """
        t = t2 - t1
        dt = round(t, 2)
        print(f"Operation {name} took {dt} seconds.")
        self.times[name] = t

    def load_tokenizer_model(self) -> tuple:
        """
        Load the model and tokenizer.
        """

        t1 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        t2 = time.time()
        self.time_benchmark("load_model", t1, t2)

        print("Compile the model...")
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
        t3 = time.time()
        self.time_benchmark("compile_model", t2, t3)

        return tokenizer, model

    def load_inputs(self):
        """
        Load the input data.
        """

        t1 = time.time()
        with open(self.input_file, "r", encoding="utf-8") as f:
            inputs_raw = json.load(f)
        t2 = time.time()

        self.time_benchmark("load_input", t1, t2)
        return inputs_raw

    def encode_inputs(self):
        """
        Convert list of inputs into tokenized inputs, with PyTorch tensors.
        Parameter inputs is list, each entry of which is a list of messages.
        Runs on CPU here so OK for not to move to GPU device.
        """

        t1 = time.time()
        inputs_encoded = [
            self.tokenizer.apply_chat_template(
                x,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            for x in self.inputs_raw
        ]
        t2 = time.time()

        self.time_benchmark("encode_input", t1, t2)
        return inputs_encoded

    def generate_llm(self):
        """
        Run the LLM on the input data.
        """

        t1 = time.time()
        outputs = [
            self.model.generate(**x, max_new_tokens=self.max_tokens, do_sample=False)
            for x in self.inputs_encoded
        ]
        t2 = time.time()

        self.time_benchmark("generate_llm", t1, t2)
        return outputs

    def decode_outputs(self):
        """
        Convert numeric outputs back into tokens.
        """

        t1 = time.time()
        r = []
        for x, y in zip(self.inputs_encoded, self.outputs_llm):
            nx = x["input_ids"].shape[-1]
            for i in range(y.shape[0]):
                w = self.tokenizer.decode(y[i, nx:])
                r.append(w)
        t2 = time.time()

        self.time_benchmark("decode_outputs", t1, t2)
        return r


def benchmark_llm(
    model_name: str,
    input_file: str,
    num_cpu: int,
    max_tokens: int,
    output_file: str,
):
    """
    Benchmark speed of running specified LLM on some queries.
    """

    print(f"Benchmark speed of model on CPU with {num_cpu} cores...")
    b = Benchmark(model_name, input_file, num_cpu, max_tokens)

    n_input_tokens = sum([math.prod(x["input_ids"].shape) for x in b.inputs_encoded])
    n_total_tokens = sum([math.prod(x.shape) for x in b.outputs_llm])
    n_output_tokens = n_total_tokens - n_input_tokens

    r = b.times.copy()
    r["n_input_tokens"] = n_input_tokens
    r["n_total_tokens"] = n_total_tokens
    r["n_output_tokens"] = n_output_tokens
    r["output"] = b.outputs_decoded

    print("Save to output file:", output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(r, f, indent="")
    print("Done saving to output file.")
