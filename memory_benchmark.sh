#!/bin/bash
# Benchmark speed of various operations in RAM.
# Use up to 8 parallel processes.

file_size=$1
out_file=$2

echo "Benchmark memory access speed..."
echo "File size: $file_size"
echo "Output logs: $out_file"
mkdir -p /dev/shm/benchmark

make_files_threads_1() {
    echo "Measure time write files, 1 thread"
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf1
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf2
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf3
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf4
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf5
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf6
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf7
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf8
}

make_files_threads_2() {
    echo "Measure time write files, 2 threads"
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf1 &
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf2 &
    wait
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf3 &
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf4 &
    wait
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf5 &
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf6 &
    wait
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf7 &
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf8 &
    wait
}

make_files_threads_4() {
    echo "Measure time write files, 4 threads"
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf1 &
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf2 &
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf3 &
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf4 &
    wait
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf5 &
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf6 &
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf7 &
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf8 &
    wait
}

make_files_threads_8() {
    echo "Measure time write files, 8 threads"
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf1 &
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf2 &
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf3 &
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf4 &
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf5 &
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf6 &
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf7 &
    cat /dev/zero | head -c $file_size >/dev/shm/benchmark/zf8 &
    wait
}

echo >"$out_file"
time make_files_threads_1 2>&1 | tee -a "$out_file"
time make_files_threads_2 2>&1 | tee -a "$out_file"
time make_files_threads_4 2>&1 | tee -a "$out_file"
time make_files_threads_8 2>&1 | tee -a "$out_file"

rm -rf /dev/shm/benchmark
