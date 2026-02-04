# DGX-benchmarking
##  Requirements

### System Requirements

- Linux (tested on DGX systems)
- NVIDIA GPU(s)
- NVIDIA driver with NVML support
- Python **3.10+** (3.12 tested)

### Software Dependencies

- Ollama installed and available in `$PATH`
- Python packages (installed via `requirements.txt`):
  - pandas
  - matplotlib
  - pynvml

---

## 🛠 Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone <your-repo-url>
cd DGX-benchmarking
```

### 2️⃣ Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install Python dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ `pynvml` requires NVIDIA drivers to already be installed on the system.

### 4️⃣ Verify Ollama

Ensure Ollama is installed and working:

```bash
ollama list
```

And that your desired model (e.g. `llava:7b`) is available:

```bash
ollama pull llava:7b
```

---

##  Running Benchmarks

Benchmarks are run one question per process invocation.

### Example: Running a full benchmark loop

The provided script `run_benchmark1.sh` loops over all questions:

```bash
bash run_benchmark1.sh
```

Internally this calls:

```bash
python benchmarking.py \
  --questions data/yesno_questions.json \
  --answers data/yesno_answers.json \
  --image-dir Images_LR \
  --model llava:7b \
  --output results/benchmark_results_llava_7b.csv \
  --index <question_index>
```

### Important Notes

- Output CSVs are **append-only**
- GPU power is sampled at ~100 ms using NVML

---

##  Generating Plots

Once you have benchmark CSVs in `results/`, you can generate plots.

### Configuration (`config.json`)

```json
{
  "results_dir": "results",
  "plots_dir": "plots",
  "models": {
    "LLaVA-7B": "benchmark_results_llava_7b.csv"
  }
}
```

You can add multiple models to compare them.

### Generate all plots

```bash
python plots.py \
  --config config.json \
  --accuracy \
  --accuracy-by-type \
  --latency \
  --power
```

### Output Plots

- `accuracy_table.png` — overall accuracy per model
- `accuracy_by_type.png` — accuracy broken down by question type
- `latency.png` — per-prompt latency
- `power_gpu.png` — smoothed average GPU power

Plots are saved to the `plots/` directory.

---

## Power & Energy Measurement Details

- Power is measured via **pynvml**
- Values represent total GPU power across all GPUs
- Energy is computed via trapezoidal integration
- CPU power is not measured (yet)


---

## Accuracy Evaluation

- Model outputs are normalized (lowercase, punctuation stripped)
- Exact-match comparison against ground-truth answers
- Each question may have multiple valid answers