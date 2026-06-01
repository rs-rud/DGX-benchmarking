| Ollama Model | vLLM Model (Hugging Face ID) |
| ------------- | ------------- |
| gemma3:4b | google/gemma-3-4b-it |
| gemma3: 12b | google/gemma-3-12b-it |
| gemma3:27b | google/gemma-3-27b-it |
| llava:7b | llava-hf/llava-1.5-7b-hf |
| llava-llama3:latest | xtuner/llava-llama-3-8b-v1_1-hf |
| ministral-3:latest (8b) | mistralai/Ministral-3-8B-Instruct-2512 |
| ministral-3:3b | mistralai/Ministral-3-3B-Instruct-2512 |
| ministral-3:14b | mistralai/Ministral-3-14B-Instruct-2512 |
| qwen2.5vl:3b | Qwen/Qwen2.5-VL-3B-Instruct |
| qwen2.5vl:7b | Qwen/Qwen2.5-VL-7B-Instruct |

> [!NOTE]
> Ollama uses 4-bit quantization, while vLLM uses 16-bit by default. 
> There should be vLLM models that run using the same memory usage as Ollama.
> Additionally, all Ollama models are ran at 40 degrees C which is roughly base temp for DGX
> vLLM models are run after a 10 minutes wait to give it time to cool down back to "base temperature"
> Ollama: Latency graphs have question 0 scrapped due to starting up the model which adds 2-3 sec of latency
