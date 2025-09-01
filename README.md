# Benchmark Speed of Running LLM on CPU

Initial test runs:

```bash
just generate_task_sort 5 10 20 2025001 /tmp/task1.json

just run_benchmark Qwen/Qwen3-4B-Instruct-2507 10 1 10
just run_benchmark Qwen/Qwen3-4B-Instruct-2507 10 2 40
just run_benchmark Qwen/Qwen3-4B-Instruct-2507 10 4 40
just run_benchmark Qwen/Qwen3-4B-Instruct-2507 10 6 40
just run_benchmark Qwen/Qwen3-4B-Instruct-2507 10 8 40
```
