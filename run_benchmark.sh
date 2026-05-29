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

# Wait Min
wait_for_minutes() {
	local minutes=$1
	local seconds=$((minutes * 60))
	echo "Waiting for ${minutes} minute(s) (${seconds} seconds)..."
	sleep $seconds
}

# Ollama models
OLLAMA_MODELS=("llava-llama3:latest")

for RUN in {1..5}; do
	RUN_DIR="results/1000/Run${RUN}"
	mkdir -p "$RUN_DIR"

	for MODEL in "${OLLAMA_MODELS[@]}"; do
		# Replace ":" and "/" with "_" for safe filenames
		SAFE_MODEL=$(echo "$MODEL" | tr ':/' '_')
		OUTPUT="${RUN_DIR}/benchmark_results_ollama_${SAFE_MODEL}.csv"

				# Check is test is ran
				if [[ -f $OUTPUT ]]; then
					LINES=$(wc -l < $OUTPUT)
					# Check if test is completed 1001 lines including header. Restart if not
					if [ $LINES -eq 1001 ]; then
						echo "=== Run $RUN already completed for $MODEL (OLLAMA) ==="
						continue
					else
						echo "=== Run $RUN not completed. Restarting Run $RUN for $MODEL from Index 0 (OLLAMA) ==="
					fi
				else
					echo "=== Preparing Run $RUN for $MODEL (OLLAMA) ==="
				fi

				# Wait for cool-down before starting this run
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

for RUN in {1..6}; do
	RUN_DIR="results/1000/Run${RUN}"
	mkdir -p "$RUN_DIR"

	for MODEL in "${VLLM_MODELS[@]}"; do
		SAFE_MODEL=$(echo "$MODEL" | tr ':/' '_')
		OUTPUT="${RUN_DIR}/benchmark_results_vllm_${SAFE_MODEL}.csv"

		echo "=== Preparing Run $RUN for $MODEL (VLLM) ==="

		wait_for_minutes 10

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
