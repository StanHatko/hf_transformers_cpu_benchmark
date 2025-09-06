# Benchmark Speed of Running LLM on CPU

Initial test runs:

```bash
just generate_task_sort 5 10 20 2025001 /tmp/task1.json

just run_benchmark Qwen/Qwen3-4B-Instruct-2507 /tmp/task1.json 1 10 /tmp/out1-1.json
just run_benchmark Qwen/Qwen3-4B-Instruct-2507 /tmp/task1.json 2 40 /tmp/out1-2.json
just run_benchmark Qwen/Qwen3-4B-Instruct-2507 /tmp/task1.json 4 40 /tmp/out1-4.json
```

## Tests on EC2, Original Settings

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

Same data as previous, but now use different model (30 billion parameter Qwen3 MoE model)
and instance type `r8i.8xlarge` (32 vCPU, 256 GiB RAM):

```bash
just generate_task_sort 1 32 20 2025002 run3/task.json

just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run3/task.json 8 150 run3/out-8.json
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run3/task.json 16 150 run3/out-16.json
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run3/task.json 32 150 run3/out-32.json
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run3/task.json 64 150 run3/out-64.json
```

In last test, process had 116 GB of RAM in resident set (monitor using top command).

With 16 cores, last test took 242.91 seconds for batch with 4352 input tokens and
4800 output tokens (including padding for both).
Total number of tokens is 9152, so speed per total tokens is 37.68 tokens / second.
Speed per generated output token (including padding) is 19.76 tokens / second.

All this is before any CPU-optimization discussed in https://huggingface.co/docs/transformers/perf_infer_cpu,
like using optimum and BetterTransformer.
Intel bfloat16 support would also be good to test.
The choice of parallelism (`torch.set_num_threads` vs. having each entry in its own thread)
also needs to be looked into.

## Tests on EC2, Set `dtype=auto`

This should use more optimized dtype instead of default float32 on CPU.

Rerun previous on same `r8i.8xlarge`, but now with `dtype=auto` set:

```bash
just generate_task_sort 1 32 20 2025002 run4/task.json

just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run4/task.json 8 150 run4/out-8.json
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run4/task.json 16 150 run4/out-16.json
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run4/task.json 32 150 run4/out-32.json
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run4/task.json 64 150 run4/out-64.json
```

After first run (due to initialization overhead I suspect), next runs for run4 were faster than run3.
Fastest run was with 16 cores taking 166.14 seconds for generation, which means a speed of
55.09 total tokens / second or 28.89 output tokens / seconds (including padding for both).

Rerun with AWQ-quantized model on same `r8i.8xlarge` instance:

```bash
just generate_task_sort 1 32 20 2025002 run5/task.json

just run_benchmark cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit run5/task.json 8 150 run5/out-8.json
just run_benchmark cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit run5/task.json 16 150 run5/out-16.json
just run_benchmark cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit run5/task.json 32 150 run5/out-32.json
just run_benchmark cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit run5/task.json 64 150 run5/out-64.json
```

Higher memory usage than run4 and much slower than run4.

Run with int8 GPTQ quantized model `QuantTrio/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int8` on same same instance
gave error that needs GPU.

## Tests on EC2, Enable Quantization

Test on same `r8i.8xlarge` instance with `int8` quantization done using `optimum-quanto`:

```bash
just generate_task_sort 1 32 20 2025002 run6/task.json

just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run6/task.json 8 150 run6/out-8.json quanto_int8
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run6/task.json 16 150 run6/out-16.json quanto_int8
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run6/task.json 32 150 run6/out-32.json quanto_int8
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run6/task.json 64 150 run6/out-64.json quanto_int8

just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run6/task.json 8 150 run6/out2-8.json quanto_int8
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run6/task.json 16 150 run6/out2-16.json quanto_int8
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run6/task.json 32 150 run6/out2-32.json quanto_int8
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run6/task.json 64 150 run6/out2-64.json quanto_int8
```

Need to benchmark in more depth with smaller things before coming back to full runs.

## Setup New Intel-Optimized PyTorch Environment

Use commands:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch
pip install optimum optimum-intel optimum-quanto accelerate pandas pyarrow transformers openvino
```

## Check Memory Transfer Speed under Various Conditions

Check speed of memory access in series and in parallel using large files in `/dev/shm`.

Use program `./memory_benchmark.sh file_size log_file` to do this,
with `file_size` being size (like `1G`) and `log_file` path of log file to save results to.

Check memory speed on AWS `r8i.8xlarge` instance:

```bash
# Test at various data sizes.
./memory_benchmark.sh 1G ~/memtest_1G.txt
./memory_benchmark.sh 2G ~/memtest_2G.txt
./memory_benchmark.sh 4G ~/memtest_4G.txt
./memory_benchmark.sh 8G ~/memtest_8G.txt
./memory_benchmark.sh 12G ~/memtest_12G.txt

# Rerun 12G test a few times.
./memory_benchmark.sh 12G ~/memtest_12G-1.txt
./memory_benchmark.sh 12G ~/memtest_12G-2.txt
./memory_benchmark.sh 12G ~/memtest_12G-3.txt
./memory_benchmark.sh 12G ~/memtest_12G-4.txt
./memory_benchmark.sh 12G ~/memtest_12G-5.txt
```
