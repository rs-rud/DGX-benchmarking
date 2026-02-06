# Temporary README to run llvm going to be moved in README later

## Follow steps in Install Python Env from README

## Need to still add extra parameter (Planned)
## Run Benchmark
python benchmark.py \
  --questions questions.json \
  --answers answers.json \
  --model llava-hf/llava-1.5-7b-hf \
  --index 0 \
  --backend vllm \

## Ollama is now run with an extra parameter
ollama pull llava:7b
python benchmark.py \
  --questions questions.json \
  --answers answers.json \
  --model llava:7b \
  --index 0 \
  --backend ollama
