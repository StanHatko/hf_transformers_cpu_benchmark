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
pip install optimum optimum-intel optimum-quanto accelerate pandas pyarrow transformers openvino torchao nncf rust-just
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

## Test `bfloat16` with Intel Extensions for PyTorch

Run on `r8i.8xlarge` instance type:

```bash
just generate_task_sort 10 8 20 2025002 run7/task.json

just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run7/task.json 8 150 run7/out-1-8.json float32 2>&1 | tee -a run7.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run7/task.json 16 150 run7/out-1-16.json float32 2>&1 | tee -a run7.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run7/task.json 32 150 run7/out-1-32.json float32 2>&1 | tee -a run7.log

just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run7/task.json 8 150 run7/out-2-8.json none 2>&1 | tee -a run7.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run7/task.json 16 150 run7/out-2-16.json none 2>&1 | tee -a run7.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run7/task.json 32 150 run7/out-2-32.json none 2>&1 | tee -a run7.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run7/task.json 64 150 run7/out-2-64.json none 2>&1 | tee -a run7.log

just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run7/task.json 8 150 run7/out-3-8.json intel_optimize 2>&1 | tee -a run7.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run7/task.json 16 150 run7/out-3-16.json intel_optimize 2>&1 | tee -a run7.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run7/task.json 32 150 run7/out-3-32.json intel_optimize 2>&1 | tee -a run7.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run7/task.json 64 150 run7/out-3-64.json intel_optimize 2>&1 | tee -a run7.log
```

## Test Various Batch Sizes

Again test on `r8i.8xlarge` instance type:

```bash
just generate_task_sort 5 1 20 2025002 run8/task-b1.json
just generate_task_sort 5 2 20 2025002 run8/task-b2.json
just generate_task_sort 5 4 20 2025002 run8/task-b4.json
just generate_task_sort 5 6 20 2025002 run8/task-b6.json
just generate_task_sort 5 8 20 2025002 run8/task-b8.json
just generate_task_sort 5 12 20 2025002 run8/task-b12.json
just generate_task_sort 5 16 20 2025002 run8/task-b16.json
just generate_task_sort 5 24 20 2025002 run8/task-b24.json
just generate_task_sort 5 32 20 2025002 run8/task-b32.json
just generate_task_sort 5 48 20 2025002 run8/task-b48.json
just generate_task_sort 5 64 20 2025002 run8/task-b64.json
just generate_task_sort 5 96 20 2025002 run8/task-b96.json
just generate_task_sort 5 128 20 2025002 run8/task-b128.json
just generate_task_sort 5 192 20 2025002 run8/task-b192.json
just generate_task_sort 5 256 20 2025002 run8/task-b256.json
just generate_task_sort 5 384 20 2025002 run8/task-b384.json
just generate_task_sort 5 512 20 2025002 run8/task-b512.json
just generate_task_sort 5 768 20 2025002 run8/task-b768.json
just generate_task_sort 5 1024 20 2025002 run8/task-b1024.json
just generate_task_sort 5 1536 20 2025002 run8/task-b1536.json
just generate_task_sort 5 2048 20 2025002 run8/task-b2048.json

just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run8/task-b1.json 16 150 run8/out-b1.json intel_optimize 2>&1 | tee -a run8.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run8/task-b2.json 16 150 run8/out-b2.json intel_optimize 2>&1 | tee -a run8.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run8/task-b4.json 16 150 run8/out-b4.json intel_optimize 2>&1 | tee -a run8.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run8/task-b6.json 16 150 run8/out-b6.json intel_optimize 2>&1 | tee -a run8.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run8/task-b8.json 16 150 run8/out-b8.json intel_optimize 2>&1 | tee -a run8.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run8/task-b16.json 16 150 run8/out-b16.json intel_optimize 2>&1 | tee -a run8.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run8/task-b24.json 16 150 run8/out-b24.json intel_optimize 2>&1 | tee -a run8.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run8/task-b32.json 16 150 run8/out-b32.json intel_optimize 2>&1 | tee -a run8.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run8/task-b48.json 16 150 run8/out-b48.json intel_optimize 2>&1 | tee -a run8.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run8/task-b64.json 16 150 run8/out-b64.json intel_optimize 2>&1 | tee -a run8.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run8/task-b96.json 16 150 run8/out-b96.json intel_optimize 2>&1 | tee -a run8.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run8/task-b128.json 16 150 run8/out-b128.json intel_optimize 2>&1 | tee -a run8.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run8/task-b192.json 16 150 run8/out-b192.json intel_optimize 2>&1 | tee -a run8.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run8/task-b256.json 16 150 run8/out-b256.json intel_optimize 2>&1 | tee -a run8.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run8/task-b384.json 16 150 run8/out-b384.json intel_optimize 2>&1 | tee -a run8.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run8/task-b512.json 16 150 run8/out-b512.json intel_optimize 2>&1 | tee -a run8.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run8/task-b768.json 16 150 run8/out-b768.json intel_optimize 2>&1 | tee -a run8.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run8/task-b1024.json 16 150 run8/out-b1024.json intel_optimize 2>&1 | tee -a run8.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run8/task-b1536.json 16 150 run8/out-b1536.json intel_optimize 2>&1 | tee -a run8.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run8/task-b2048.json 16 150 run8/out-b2048.json intel_optimize 2>&1 | tee -a run8.log
```

## Test Various Generation Lengths

Again test on `r8i.8xlarge` instance type:

```bash
just generate_task_sort 3 64 250 2025003 run9/task.json

just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run9/task.json 16 50 run9/out-50.json intel_optimize 2>&1 | tee -a run9.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run9/task.json 16 100 run9/out-100.json intel_optimize 2>&1 | tee -a run9.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run9/task.json 16 150 run9/out-150.json intel_optimize 2>&1 | tee -a run9.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run9/task.json 16 200 run9/out-200.json intel_optimize 2>&1 | tee -a run9.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run9/task.json 16 250 run9/out-250.json intel_optimize 2>&1 | tee -a run9.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run9/task.json 16 300 run9/out-300.json intel_optimize 2>&1 | tee -a run9.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run9/task.json 16 350 run9/out-350.json intel_optimize 2>&1 | tee -a run9.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run9/task.json 16 400 run9/out-400.json intel_optimize 2>&1 | tee -a run9.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run9/task.json 16 450 run9/out-450.json intel_optimize 2>&1 | tee -a run9.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run9/task.json 16 500 run9/out-500.json intel_optimize 2>&1 | tee -a run9.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run9/task.json 16 550 run9/out-550.json intel_optimize 2>&1 | tee -a run9.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run9/task.json 16 600 run9/out-600.json intel_optimize 2>&1 | tee -a run9.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run9/task.json 16 650 run9/out-650.json intel_optimize 2>&1 | tee -a run9.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run9/task.json 16 700 run9/out-700.json intel_optimize 2>&1 | tee -a run9.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run9/task.json 16 750 run9/out-750.json intel_optimize 2>&1 | tee -a run9.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run9/task.json 16 800 run9/out-800.json intel_optimize 2>&1 | tee -a run9.log
```

## Test Various Encoding Lengths

Again test on `r8i.8xlarge` instance type:

```bash
just generate_task_sort 3 64 5 2025004 run10/task-5.json
just generate_task_sort 3 64 10 2025004 run10/task-10.json
just generate_task_sort 3 64 15 2025004 run10/task-15.json
just generate_task_sort 3 64 20 2025004 run10/task-20.json
just generate_task_sort 3 64 25 2025004 run10/task-25.json
just generate_task_sort 3 64 30 2025004 run10/task-30.json
just generate_task_sort 3 64 35 2025004 run10/task-35.json
just generate_task_sort 3 64 40 2025004 run10/task-40.json
just generate_task_sort 3 64 45 2025004 run10/task-45.json
just generate_task_sort 3 64 50 2025004 run10/task-50.json

just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run10/task-5.json 16 20 run10/out-5.json intel_optimize 2>&1 | tee -a run10.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run10/task-10.json 16 20 run10/out-10.json intel_optimize 2>&1 | tee -a run10.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run10/task-15.json 16 20 run10/out-15.json intel_optimize 2>&1 | tee -a run10.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run10/task-20.json 16 20 run10/out-20.json intel_optimize 2>&1 | tee -a run10.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run10/task-25.json 16 20 run10/out-25.json intel_optimize 2>&1 | tee -a run10.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run10/task-30.json 16 20 run10/out-30.json intel_optimize 2>&1 | tee -a run10.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run10/task-35.json 16 20 run10/out-35.json intel_optimize 2>&1 | tee -a run10.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run10/task-40.json 16 20 run10/out-40.json intel_optimize 2>&1 | tee -a run10.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run10/task-45.json 16 20 run10/out-45.json intel_optimize 2>&1 | tee -a run10.log
just run_benchmark Qwen/Qwen3-30B-A3B-Instruct-2507 run10/task-50.json 16 20 run10/out-50.json intel_optimize 2>&1 | tee -a run10.log
```
