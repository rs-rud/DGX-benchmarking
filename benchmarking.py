import json
import time
import os
import string
import csv
import argparse
import subprocess
import threading
import pynvml
from vllm import LLM
from PIL import Image

# ---------------- Utilities ---------------- #

def normalize(text):
    return text.strip().lower().translate(str.maketrans("", "", string.punctuation))

# ---------------- Ollama ------------------ #
def run_ollama_cli(model, prompt, image_path):
    """Run Ollama CLI for one multimodal query and return output + latency."""
    start = time.time()
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt, image_path],
            capture_output=True,
            text=True,
            check=True
        )
        latency = time.time() - start
        output = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        enforced = output.strip().split()[0].lower() if output else ""
        return enforced, latency
    except subprocess.CalledProcessError as e:
        print(f"Error running ollama: {e.stderr}", flush=True)
        return "", 0.0

# ---------------- VLLM ------------------ #
def run_vllm_cli(model, prompt, image_path):
    start = time.time()
    try:
        llm = LLM(model=model)
        image = Image.open(image_path)
        result = llm.generate({
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        })
        latency = time.time() - start
        output = result[0].outputs[0].text.strip() if result else ""
        enforced = output.split()[0].lower() if output else ""
        return enforced, latency
    except Exception as e:
        print(f"Error running vLLM: {e}", flush=True)
        return "", 0.0

# ---------------- NVML Power Sampling ---------------- #

class NVMLPowerSampler:
    def __init__(self):
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        self.handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i)
            for i in range(self.device_count)
        ]

    def read_power_watts(self):
        """
        Returns total GPU power draw in watts across all GPUs.
        """
        total_w = 0.0
        for h in self.handles:
            try:
                # milliwatts → watts
                total_w += pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except pynvml.NVMLError:
                pass
        return total_w

    def shutdown(self):
        pynvml.nvmlShutdown()


def sample_power_nvml(sampler, samples, stop_event, interval=0.1):
    """Background sampler using NVML."""
    while not stop_event.is_set():
        p = sampler.read_power_watts()
        samples.append({
            "t": time.time(),
            "tot": p,
            "cpu_gpu": p  # NVML exposes GPU only
        })
        time.sleep(interval)


def integrate_energy(samples):
    """Compute average power in watts via trapezoidal integration."""
    if len(samples) < 2:
        return 0.0

    E = 0.0
    for a, b in zip(samples[:-1], samples[1:]):
        dt = b["t"] - a["t"]
        E += 0.5 * (a["tot"] + b["tot"]) * dt

    duration = samples[-1]["t"] - samples[0]["t"]
    return E / duration if duration > 0 else 0.0


# ---------------- Main ---------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--image-dir", default="Images_LR")
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--engine", required=True)
    args = parser.parse_args()

    if args.output is None:
        safe_model = args.model.replace(":", "_").replace("/", "_")
        args.output = f"benchmark_results_{safe_model}.csv"

    with open(args.questions) as f:
        all_questions = json.load(f)["questions"]

    with open(args.answers) as f:
        all_answers = {a["id"]: a for a in json.load(f)["answers"]}

    if args.index >= len(all_questions):
        print(f"❌ Index {args.index} out of range", flush=True)
        return

    q = all_questions[args.index]
    qid = q["id"]
    img_id = q["img_id"]
    image_path = os.path.join(args.image_dir, f"{img_id}.tif")

    if not os.path.exists(image_path):
        print(f"Missing image {image_path}, skipping.", flush=True)
        return

    question_text = q["question"] + "\nAnswer with exactly one word or number only. Do not explain."
    gt_answers = [
        normalize(all_answers[aid]["answer"])
        for aid in q.get("answers_ids", [])
        if aid in all_answers
    ]

    if not gt_answers:
        print(f"No ground truth for qid {qid}, skipping.", flush=True)
        return

    # -------- NVML Power Measurement -------- #

    power_samples = []
    sampler = NVMLPowerSampler()

    stop_event = threading.Event()
    sampler_thread = threading.Thread(
        target=sample_power_nvml,
        args=(sampler, power_samples, stop_event),
        daemon=True
    )

    sampler_thread.start()
    start_time = time.time()
    if args.engine == "vllm":
        response, latency = run_vllm_cli(args.model, question_text, image_path)
    else:
        response, latency = run_ollama_cli(args.model, question_text, image_path)
    
    end_time = time.time()
    stop_event.set()
    sampler_thread.join(timeout=2)
    sampler.shutdown()

    # -------- Stats -------- #

    samples_in_window = [s for s in power_samples if start_time <= s["t"] <= end_time]

    if samples_in_window:
        avg_tot = sum(s["tot"] for s in samples_in_window) / len(samples_in_window)
        max_tot = max(s["tot"] for s in samples_in_window)
        avg_cpu_gpu = avg_tot
        max_cpu_gpu = max_tot
        avg_power_integrated_w = integrate_energy(samples_in_window)
    else:
        avg_tot = max_tot = avg_cpu_gpu = max_cpu_gpu = avg_power_integrated_w = 0.0

    is_correct = normalize(response) in gt_answers

    # -------- CSV Output -------- #

    file_exists = os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "question_id", "latency_sec", "correct",
                "model_response", "ground_truth", "question_text",
                "avg_gpu_w", "max_gpu_w",
                "avg_power_integrated_w"
            ])
        writer.writerow([
            qid, f"{latency:.3f}", int(is_correct),
            response, "|".join(gt_answers), q["question"],
            f"{avg_tot:.2f}", f"{max_tot:.2f}",
            f"{avg_power_integrated_w:.2f}"
        ])

    # -------- Console -------- #

    print(f"[Q{qid}] Engine: {args.engine}", flush=True)
    print(f"Model: {response}", flush=True)
    print(f"GT: {gt_answers}", flush=True)
    print(f"Correct: {is_correct}, Time: {latency:.2f}s", flush=True)
    print(f"GPU Power: avg {avg_tot:.2f} W, max {max_tot:.2f} W", flush=True)
    print(f"Energy (avg): {avg_power_integrated_w:.2f} W", flush=True)


if __name__ == "__main__":
    main()
