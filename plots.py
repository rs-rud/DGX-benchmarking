import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Style (matplotlib-only, stable) ---------------- #
plt.style.use("ggplot")

# ---------------- Helpers ---------------- #

def load_config(path):
    with open(path) as f:
        return json.load(f)

def load_csv(results_dir, filename):
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved → {path}")

# ---------------- Plot: Accuracy (table) ---------------- #

def plot_accuracy(cfg):
    rows = []

    for model, file in cfg["models"].items():
        df = load_csv(cfg["results_dir"], file)
        acc = df["correct"].mean() * 100
        rows.append((model, acc))

    df = pd.DataFrame(rows, columns=["Model", "Accuracy"])
    df = df.sort_values("Accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")

    table = ax.table(
        cellText=[[m, f"{a:.2f}%"] for m, a in rows],
        colLabels=["Model", "Accuracy"],
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    ax.set_title("Overall Accuracy", fontsize=14, weight="bold")
    save(fig, cfg["plots_dir"], "accuracy_table.png")

# ---------------- Plot: Accuracy by question type ---------------- #

def categorize_question(text):
    t = text.lower()
    if "equal to" in t:
        return "Equality"
    if "more" in t or "less" in t:
        return "Comparison"
    if any(w in t for w in ["many", "number", "amount"]):
        return "Counting"
    if any(w in t for w in ["small", "large", "square", "circle"]):
        return "Attribute"
    if " at the " in t:
        return "Location"
    if t.startswith(("is", "are", "does", "do")):
        return "Yes/No"
    return "Other"

def plot_accuracy_by_type(cfg):
    frames = []

    for model, file in cfg["models"].items():
        df = load_csv(cfg["results_dir"], file)
        df["qtype"] = df["question_text"].apply(categorize_question)
        g = df.groupby("qtype")["correct"].mean().reset_index()
        g["Accuracy"] = g["correct"] * 100
        g["Model"] = model
        frames.append(g[["Model", "qtype", "Accuracy"]])

    df = pd.concat(frames)

    fig, ax = plt.subplots(figsize=(12, 6))
    for model in df["Model"].unique():
        sub = df[df["Model"] == model]
        ax.plot(sub["qtype"], sub["Accuracy"], marker="o", label=model)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Question Type", weight="bold")
    ax.legend()
    save(fig, cfg["plots_dir"], "accuracy_by_type.png")

# ---------------- Plot: Latency ---------------- #

def plot_latency(cfg):
    fig, ax = plt.subplots(figsize=(10, 6))

    for model, file in cfg["models"].items():
        df = load_csv(cfg["results_dir"], file)
        ax.plot(df["latency_sec"], label=model, linewidth=2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency per Prompt", weight="bold")
    ax.legend()
    save(fig, cfg["plots_dir"], "latency.png")

# ---------------- Plot: Power ---------------- #

def plot_power(cfg):
    fig, ax = plt.subplots(figsize=(10, 6))

    for model, file in cfg["models"].items():
        df = load_csv(cfg["results_dir"], file)
        smoothed = df["avg_gpu_w"].rolling(10, min_periods=1).mean()
        ax.plot(smoothed, label=model, linewidth=2)

    ax.set_ylabel("GPU Power (W)")
    ax.set_xlabel("Iteration")
    ax.set_title("Average GPU Power (Smoothed)", weight="bold")
    ax.legend()
    save(fig, cfg["plots_dir"], "power_gpu.png")

# ---------------- Main ---------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--accuracy", action="store_true")
    p.add_argument("--accuracy-by-type", action="store_true")
    p.add_argument("--latency", action="store_true")
    p.add_argument("--power", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)

    if args.accuracy:
        plot_accuracy(cfg)
    if args.accuracy_by_type:
        plot_accuracy_by_type(cfg)
    if args.latency:
        plot_latency(cfg)
    if args.power:
        plot_power(cfg)

if __name__ == "__main__":
    main()
