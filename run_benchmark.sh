#!/bin/bash
QUESTIONS="data/all_questions.json"
ANSWERS="data/all_answers.json"
IMAGES="Images_LR"

# List of models
MODELS=("llava:7b", "llava-llama3:latest")

for MODEL in "${MODELS[@]}"; do
    # replace ":" and "/" with "_" for safe filenames
    SAFE_MODEL=$(echo $MODEL | tr ':/' '_')
    OUTPUT="results/benchmark_results_${SAFE_MODEL}.csv"

    # clear old results
    rm -f $OUTPUT

    # how many questions in the dataset
    TOTAL=$(jq ".questions | length" $QUESTIONS)

    echo "=== Benchmarking $MODEL on 1000 questions ==="

    for ((i=0; i < 1000; i++)); do
        echo ">>> Running question $i / $TOTAL"
        python3 benchmarking.py \
            --backend ollama \
            --questions $QUESTIONS \
            --answers $ANSWERS \
            --image-dir $IMAGES \
            --model $MODEL \
            --output $OUTPUT \
            --index $i
    done
done
