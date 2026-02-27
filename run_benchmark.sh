#!/bin/bash
QUESTIONS="data/all_questions.json"
ANSWERS="data/all_answers.json"
IMAGES="Images_LR"

# Get Current Max GPU Temp
get_gpu_temp() {
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -nr | head -1
}

# Wait GPU Temp <= Target (40 C: Base Temp of DGX)
wait_for_temp() {
    local target=$1
    local temp
    while true; do
        temp=$(get_gpu_temp)
        if [ "$temp" -le "$target" ]; then
            echo "Temperature is $temp°C (<= $target°C). Proceeding."
            break
        else
            echo "Temperature is $temp°C (> $target°C). Waiting 30 seconds..."
            sleep 30
        fi
    done
}

# Ollama models
OLLAMA_MODELS=("llava:7b" "llava-llama3:latest")

for RUN in {1..5}; do
    RUN_DIR="results/1000/Run${RUN}"
    mkdir -p "$RUN_DIR"

    for MODEL in "${OLLAMA_MODELS[@]}"; do
        # Replace ":" and "/" with "_" for safe filenames
        SAFE_MODEL=$(echo "$MODEL" | tr ':/' '_')
        OUTPUT="${RUN_DIR}/benchmark_results_ollama_${SAFE_MODEL}.csv"

        # Wait for cool-down before starting this run
        echo "=== Preparing Run $RUN for $MODEL (OLLAMA) ==="
        wait_for_temp 40

        # Clear old results for this run (if any)
        rm -f "$OUTPUT"

        echo "=== Benchmarking $MODEL via OLLAMA (Run $RUN) ==="
        for ((i=0; i<1000; i++)); do
            echo ">>> Run $RUN, question $i / 999"
            python3 benchmarking.py \
                --engine ollama \
                --questions "$QUESTIONS" \
                --answers "$ANSWERS" \
                --image-dir "$IMAGES" \
                --model "$MODEL" \
                --output "$OUTPUT" \
                --index "$i"
        done
    done
done

# VLLM models
VLLM_MODELS=("llava-hf/llava-1.5-7b-hf")

for RUN in {1..5}; do
    RUN_DIR="results/1000/Run${RUN}"
    mkdir -p "$RUN_DIR"

    for MODEL in "${VLLM_MODELS[@]}"; do
        SAFE_MODEL=$(echo "$MODEL" | tr ':/' '_')
        OUTPUT="${RUN_DIR}/benchmark_results_vllm_${SAFE_MODEL}.csv"

        echo "=== Preparing Run $RUN for $MODEL (VLLM) ==="
        wait_for_temp 40

        rm -f "$OUTPUT"

        echo "=== Benchmarking $MODEL via VLLM (Run $RUN) ==="
        for ((i=0; i<1000; i++)); do
            echo ">>> Run $RUN, question $i / 999"
            python3 benchmarking.py \
                --engine vllm \
                --questions "$QUESTIONS" \
                --answers "$ANSWERS" \
                --image-dir "$IMAGES" \
                --model "$MODEL" \
                --output "$OUTPUT" \
                --index "$i"
        done
    done
done
