#!/bin/bash
# Benchmark speed of various operations in RAM.
# Use up to 8 parallel processes.

file_size=$1
out_file=$2

echo "Benchmark memory access speed..." | tee -a "$out_file"
echo "File size: $file_size" | tee -a "$out_file"
echo "Output logs: $out_file" | tee -a "$out_file"
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

get_files_threads_1() {
    echo "Measure time read files, 1 thread"
    cat /dev/shm/benchmark/zf1 >/dev/zero
    cat /dev/shm/benchmark/zf2 >/dev/zero
    cat /dev/shm/benchmark/zf3 >/dev/zero
    cat /dev/shm/benchmark/zf4 >/dev/zero
    cat /dev/shm/benchmark/zf5 >/dev/zero
    cat /dev/shm/benchmark/zf6 >/dev/zero
    cat /dev/shm/benchmark/zf7 >/dev/zero
    cat /dev/shm/benchmark/zf8 >/dev/zero
}

get_files_threads_2() {
    echo "Measure time read files, 2 threads"
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    cat /dev/shm/benchmark/zf2 >/dev/zero &
    wait
    cat /dev/shm/benchmark/zf3 >/dev/zero &
    cat /dev/shm/benchmark/zf4 >/dev/zero &
    wait
    cat /dev/shm/benchmark/zf5 >/dev/zero &
    cat /dev/shm/benchmark/zf6 >/dev/zero &
    wait
    cat /dev/shm/benchmark/zf7 >/dev/zero &
    cat /dev/shm/benchmark/zf8 >/dev/zero &
    wait
}

get_files_threads_4() {
    echo "Measure time read files, 4 threads"
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    cat /dev/shm/benchmark/zf2 >/dev/zero &
    cat /dev/shm/benchmark/zf3 >/dev/zero &
    cat /dev/shm/benchmark/zf4 >/dev/zero &
    wait
    cat /dev/shm/benchmark/zf5 >/dev/zero &
    cat /dev/shm/benchmark/zf6 >/dev/zero &
    cat /dev/shm/benchmark/zf7 >/dev/zero &
    cat /dev/shm/benchmark/zf8 >/dev/zero &
    wait
}

get_files_threads_8() {
    echo "Measure time read files, 8 threads"
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    cat /dev/shm/benchmark/zf2 >/dev/zero &
    cat /dev/shm/benchmark/zf3 >/dev/zero &
    cat /dev/shm/benchmark/zf4 >/dev/zero &
    cat /dev/shm/benchmark/zf5 >/dev/zero &
    cat /dev/shm/benchmark/zf6 >/dev/zero &
    cat /dev/shm/benchmark/zf7 >/dev/zero &
    cat /dev/shm/benchmark/zf8 >/dev/zero &
    wait
}

get_repeat_threads_1() {
    echo "Measure time read single repeat, 1 thread"
    cat /dev/shm/benchmark/zf1 >/dev/zero
    cat /dev/shm/benchmark/zf1 >/dev/zero
    cat /dev/shm/benchmark/zf1 >/dev/zero
    cat /dev/shm/benchmark/zf1 >/dev/zero
    cat /dev/shm/benchmark/zf1 >/dev/zero
    cat /dev/shm/benchmark/zf1 >/dev/zero
    cat /dev/shm/benchmark/zf1 >/dev/zero
    cat /dev/shm/benchmark/zf1 >/dev/zero
}

get_repeat_threads_2() {
    echo "Measure time read single repeat, 2 threads"
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    wait
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    wait
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    wait
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    wait
}

get_repeat_threads_4() {
    echo "Measure time read single repeat, 4 threads"
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    wait
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    wait
}

get_repeat_threads_8() {
    echo "Measure time read single repeat, 8 threads"
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    cat /dev/shm/benchmark/zf1 >/dev/zero &
    wait
}


echo >"$out_file"

time make_files_threads_1 2>&1 | tee -a "$out_file"
time make_files_threads_1 2>&1 | tee -a "$out_file"
time make_files_threads_2 2>&1 | tee -a "$out_file"
time make_files_threads_4 2>&1 | tee -a "$out_file"
time make_files_threads_8 2>&1 | tee -a "$out_file"

time get_files_threads_1 2>&1 | tee -a "$out_file"
time get_files_threads_1 2>&1 | tee -a "$out_file"
time get_files_threads_2 2>&1 | tee -a "$out_file"
time get_files_threads_4 2>&1 | tee -a "$out_file"
time get_files_threads_8 2>&1 | tee -a "$out_file"

time get_repeat_threads_1 2>&1 | tee -a "$out_file"
time get_repeat_threads_1 2>&1 | tee -a "$out_file"
time get_repeat_threads_2 2>&1 | tee -a "$out_file"
time get_repeat_threads_4 2>&1 | tee -a "$out_file"
time get_repeat_threads_8 2>&1 | tee -a "$out_file"

rm -rf /dev/shm/benchmark
