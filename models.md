| Ollama Model | vLLM Model (Hugging Face ID) |
| ------------- | ------------- |
| gemma3:4b | google/gemma-3-4b-it |
| gemma3:27b | 
| llava:7b | llava-hf/llava-1.5-7b-hf |
| llava-llama3:latest | xtuner/llava-llama-3-8b-v1.1 |
| ministral-3:latest |
| ministral-3:3b |
| ministral-3:14b |
| qwen2.5vl:3b |
| qwen2.5vl:7b |

> [!NOTE]
> Ollama uses 4-bit quantization, while vLLM uses 16-bit by default. 
> There should be vLLM models that run using the same memory usage as Ollama.
> Additionally, all models are ran at 40 degrees C which is roughly base temp for DGX
> Ollama: Latency graphs have question 0 scrapped due to starting up the model which adds 2-3 sec of latency
