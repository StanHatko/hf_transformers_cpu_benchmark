# Justfile for running the LLM benchmarking on CPU.
export PYTHONPATH := "./"


# Run benchmarking program, add data params as this develops.
run_benchmark model_name max_tokens:
    #!/usr/bin/env python
    from run_benchmark import benchmark_llm

    benchmark_llm("{{model_name}}", {{max_tokens}})
