#!/bin/bash
QUESTIONS="data/all_questions.json"
ANSWERS="data/all_answers.json"
IMAGES="Images_LR"

# Configurable batch size (number of questions per Python invocation)
BATCH_SIZE=1

# Configurable total number of questions to process
NUM_QUESTIONS=1000

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
OLLAMA_MODELS=()

for RUN in {1..5}; do
	RUN_DIR="results/${NUM_QUESTIONS}/Run${RUN}"
	mkdir -p "$RUN_DIR"

	for MODEL in "${OLLAMA_MODELS[@]}"; do
		# Replace ":" and "/" with "_" for safe filenames
		SAFE_MODEL=$(echo "$MODEL" | tr ':/' '_')
		OUTPUT="${RUN_DIR}/benchmark_results_ollama_${SAFE_MODEL}.csv"

		# Check is test is ran
		if [[ -f $OUTPUT ]]; then
			LINES=$(wc -l < $OUTPUT)
			EXPECTED_LINES=$((NUM_QUESTIONS + 1))
			# Check if test is completed (NUM_QUESTIONS+1 lines including header). Restart if not
			if [ $LINES -eq $EXPECTED_LINES ]; then
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
		for ((i=0; i<NUM_QUESTIONS; i+=BATCH_SIZE)); do
			end=$((i + BATCH_SIZE - 1))
			[ $end -ge $((NUM_QUESTIONS - 1)) ] && end=$((NUM_QUESTIONS - 1))
			echo ">>> Run $RUN, questions $i to $end / $((NUM_QUESTIONS - 1))"
			python3 benchmarking.py \
				--engine ollama \
				--questions "$QUESTIONS" \
				--answers "$ANSWERS" \
				--image-dir "$IMAGES" \
				--model "$MODEL" \
				--output "$OUTPUT" \
				--index "$i" \
				--batch-size "$BATCH_SIZE"
		done
	done
done

# VLLM models
VLLM_MODELS=("Qwen/Qwen2.5-VL-32B-Instruct")
# Extra vLLM server args per model (empty string = none)
VLLM_ARGS=("--max-model-len 8192")

# ---------------- VLLM Server Management ---------------- #

VLLM_PID=""

start_vllm_server() {
	local model=$1
	local extra_args=$2
	echo "=== Starting vLLM server for $model ==="

	# Kill any existing vLLM server first
	stop_vllm_server

	# Start fresh vLLM server in background
	# shellcheck disable=SC2086
	vllm serve "$model" --port 8000 $extra_args &
	VLLM_PID=$!

	# Wait for server to be ready (health endpoint)
	echo "Waiting for vLLM server to be ready..."
	local retries=0
	while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do
		sleep 5
		retries=$((retries + 1))
	done
	echo "vLLM server is ready (PID $VLLM_PID)"
}

stop_vllm_server() {
	if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
		echo "Stopping vLLM server (PID $VLLM_PID)..."
		kill "$VLLM_PID" 2>/dev/null
		wait "$VLLM_PID" 2>/dev/null
		VLLM_PID=""
	fi
	# Also catch any leftover vllm processes
	pkill -f "vllm serve" 2>/dev/null || true
	sleep 2
}

# ---------------- Benchmark Loop ---------------- #

for RUN in {1..6}; do
	RUN_DIR="results/${NUM_QUESTIONS}/Run${RUN}"
	mkdir -p "$RUN_DIR"

	for i in "${!VLLM_MODELS[@]}"; do
		MODEL="${VLLM_MODELS[$i]}"
		EXTRA="${VLLM_ARGS[$i]:-}"

		# Replace ":" and "/" with "_" for safe filenames
		SAFE_MODEL=$(echo "$MODEL" | tr ':/' '_')
		OUTPUT="${RUN_DIR}/benchmark_results_vllm_${SAFE_MODEL}.csv"

		echo "=== Preparing Run $RUN for $MODEL (VLLM) ==="

		wait_for_minutes 10

		rm -f "$OUTPUT"

		# Start fresh vLLM server for this run (clears KV cache)
		start_vllm_server "$MODEL" "$EXTRA"

		echo "=== Benchmarking $MODEL via VLLM (Run $RUN) ==="
		for ((q=0; q<NUM_QUESTIONS; q+=BATCH_SIZE)); do
			end=$((q + BATCH_SIZE - 1))
			[ $end -ge $((NUM_QUESTIONS - 1)) ] && end=$((NUM_QUESTIONS - 1))
			echo ">>> Run $RUN, questions $q to $end / $((NUM_QUESTIONS - 1))"
			python3 benchmarking.py \
				--engine vllm \
				--questions "$QUESTIONS" \
				--answers "$ANSWERS" \
				--image-dir "$IMAGES" \
				--model "$MODEL" \
				--output "$OUTPUT" \
				--index "$q" \
				--batch-size "$BATCH_SIZE"
		done

		# Stop server to clear KV cache after the run
		stop_vllm_server
	done
done