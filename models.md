| Ollama Model | vLLM Model (Hugging Face ID) |
| ------------- | ------------- |
| gemma3:4b | google/gemma-3-4b-it |
| gemma3:1b | google/gemma-3-1b-it |
| gemma3n:e2b | google/gemma-3n-e2b-it |
| llava:7b | llava-hf/llava-1.5-7b-hf |
| llava-llama3:latest | xtuner/llava-llama-3-8b-v1.1 |

> [!NOTE]
> Ollama uses 4-bit quantization, while vLLM uses 16-bit by default. 
> There should be vLLM models that run using the same memory usage as Ollama.


## LOGS

llava:7b is ran at normal temp: ~40 C
llava-llama3:latest is ran directly after llava:7b without cooling

llava-hf/llava-1.5-7b-hf id ran at normal temp: ~40 C

gemma3:4b is ran at normal temp: ~40 C
gemma3:1b is ran at normal temp: ~40 C
