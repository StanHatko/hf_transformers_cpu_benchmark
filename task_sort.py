#!/usr/bin/env python
"""
Generate LLM task of random integers to sort, for benchmarking.
"""

import json

import numpy as np


def _gen_problem(gen, num_ints: int):
    sort_list = gen.integers(0, 10000, num_ints)
    return [
        {
            "role": "user",
            "content": f"Sort the following numbers from smallest to largest: {sort_list}",
        }
    ]


def _gen_batch(gen, num_per_batch: int, num_ints: int):
    return [_gen_problem(gen, num_ints) for _ in range(num_per_batch)]


def generate_task_sort(
    num_batches: int,
    num_per_batch: int,
    num_ints: int,
    seed: int,
    out_file: str,
):
    """
    Generate the sorting task, in integer format.
    """

    print("Generate sorting task, with parameters:")
    print("num_batches:", num_batches)
    print("num_per_batch:", num_per_batch)
    print("num_ints:", num_ints)
    print("seed:", seed)
    print("out_file:", out_file)

    gen = np.random.Generator(np.random.PCG64(seed=seed))
    x = [_gen_batch(gen, num_per_batch, num_ints) for _ in range(num_batches)]

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(x, f)
    print("Done saving generated problems to output file.")
