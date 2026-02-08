# Temporary README to run llvm going to be moved in README later

## Follow steps in Install Python Env from README

> [!NOTE]
> References: https://vllm.ai/
> Models: https://docs.vllm.ai/en/latest/models/supported_models/?h=models#text-generation
> Going to add more models based on what is run on the Jetson. So far only got llava:7b which is below

## Run Benchmark
python benchmark.py \
  --questions questions.json \
  --answers answers.json \
  --model llava-hf/llava-1.5-7b-hf \
  --index 0 \
  --engine vllm \

## Ollama is now run with an extra parameter
ollama pull llava:7b
python benchmark.py \
  --questions questions.json \
  --answers answers.json \
  --model llava:7b \
  --index 0 \
  --engine ollama
