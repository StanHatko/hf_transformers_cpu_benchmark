# Justfile for running the LLM benchmarking on CPU.
export PYTHONPATH := "./"


# Generate sorting integers task.
generate_task_sort num_batches num_per_batch num_ints seed out_file:
    #!/usr/bin/env python
    from task_sort import generate_task_sort

    generate_task_sort(
        {{num_batches}},
        {{num_per_batch}},
        {{num_ints}},
        {{seed}},
        "{{out_file}}",
    )


# Run benchmarking program, add data params as this develops.
run_benchmark model_name num_queries num_cpu max_tokens:
    #!/usr/bin/env python
    from run_benchmark import benchmark_llm

    benchmark_llm(
        "{{model_name}}",
        {{num_queries}},
        {{num_cpu}},
        {{max_tokens}}
    )
