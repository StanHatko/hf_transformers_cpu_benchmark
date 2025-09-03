# Benchmark Speed of Running LLM on CPU

Initial test runs:

```bash
just generate_task_sort 5 10 20 2025001 /tmp/task1.json

just run_benchmark Qwen/Qwen3-4B-Instruct-2507 /tmp/task1.json 1 10 /tmp/out1-1.json
just run_benchmark Qwen/Qwen3-4B-Instruct-2507 /tmp/task1.json 2 40 /tmp/out1-2.json
just run_benchmark Qwen/Qwen3-4B-Instruct-2507 /tmp/task1.json 4 40 /tmp/out1-4.json
```

Install dependencies and software on EC2 instance:

```bash
sudo apt update
sudo apt install python3-pip
sudo apt install python3.12-venv
python3 -m venv env
source env/bin/activate
pip install uv
uv pip install torch torchvision transformers numpy pandas rust-just
git clone https://github.com/StanHatko/hf_transformers_cpu_benchmark
```

Run test on AWS EC2 US Ohio region, with instance type `c7a.8xlarge` (32 vCPU, 64 GiB RAM):

```bash
just generate_task_sort 10 20 20 2025002 run1/task.json

just run_benchmark Qwen/Qwen3-4B-Instruct-2507 run1/task.json 4 50 run1/out-4.json
just run_benchmark Qwen/Qwen3-4B-Instruct-2507 run1/task.json 8 50 run1/out-8.json
just run_benchmark Qwen/Qwen3-4B-Instruct-2507 run1/task.json 16 50 run1/out-16.json
just run_benchmark Qwen/Qwen3-4B-Instruct-2507 run1/task.json 32 50 run1/out-32.json
just run_benchmark Qwen/Qwen3-4B-Instruct-2507 run1/task.json 64 50 run1/out-64.json
```

Another run on AWS EC2 US Ohio region, with instance type `c7a.8xlarge` (32 vCPU, 64 GiB RAM):

```bash
just generate_task_sort 1 32 20 2025002 run2/task.json

just run_benchmark Qwen/Qwen3-4B-Instruct-2507 run2/task.json 8 150 run2/out-8.json
just run_benchmark Qwen/Qwen3-4B-Instruct-2507 run2/task.json 16 150 run2/out-16.json
just run_benchmark Qwen/Qwen3-4B-Instruct-2507 run2/task.json 32 150 run2/out-32.json
just run_benchmark Qwen/Qwen3-4B-Instruct-2507 run2/task.json 64 150 run2/out-64.json
```

Same data as previous, but now use different model:

```bash
just generate_task_sort 1 32 20 2025002 run3/task.json

just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run3/task.json 8 150 run3/out-8.json
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run3/task.json 16 150 run3/out-16.json
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run3/task.json 32 150 run3/out-32.json
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run3/task.json 64 150 run3/out-64.json
```
