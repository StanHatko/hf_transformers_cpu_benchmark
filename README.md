# Benchmark Speed of Running LLM on CPU

Initial test runs:

```bash
just generate_task_sort 5 10 20 2025001 /tmp/task1.json

just run_benchmark Qwen/Qwen3-4B-Instruct-2507 /tmp/task1.json 1 10 /tmp/out1-1.json
just run_benchmark Qwen/Qwen3-4B-Instruct-2507 /tmp/task1.json 2 40 /tmp/out1-2.json
just run_benchmark Qwen/Qwen3-4B-Instruct-2507 /tmp/task1.json 4 40 /tmp/out1-4.json
```
