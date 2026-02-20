#!/bin/bash
QUESTIONS="data/all_questions.json"
ANSWERS="data/all_answers.json"
IMAGES="Images_LR"

# Ollama
OLLAMA_MODELS=("llava:7b" "llava-llama3:latest")

for MODEL in "${OLLAMA_MODELS[@]}"; do
    # replace ":" and "/" with "_" for safe filenames
    SAFE_MODEL=$(echo $MODEL | tr ':/' '_')
    OUTPUT="results/benchmark_results_ollama_${SAFE_MODEL}.csv"

    # clear old results
    rm -f $OUTPUT

    echo "=== Benchmarking $MODEL via OLLAMA ==="

    for ((i=0; i < 1000; i++)); do
        echo ">>> Running question $i / 999"
        python3 benchmarking.py \
            --engine ollama \
            --questions $QUESTIONS \
            --answers $ANSWERS \
            --image-dir $IMAGES \
            --model $MODEL \
            --output $OUTPUT \
            --index $i
    done
done

# VLLM
VLLM_MODELS=("llava-hf/llava-1.5-7b-hf")
for MODEL in "${VLLM_MODELS[@]}"; do
    # replace ":" and "/" with "_" for safe filenames
    SAFE_MODEL=$(echo $MODEL | tr ':/' '_')
    OUTPUT="results/benchmark_results_vllm_${SAFE_MODEL}.csv"

    # clear old results
    rm -f "$OUTPUT"

    echo "=== Benchmarking $MODEL via VLLM ==="
    for ((i=0; i < 1000; i++)); do
        echo ">>> Running question $i / 999"
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
