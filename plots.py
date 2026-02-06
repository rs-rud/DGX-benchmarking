import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

    df_table = pd.DataFrame(rows, columns=["Model", "Accuracy"])
    df_table = df_table.sort_values("Accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")

    table = ax.table(
        cellText=[[m, f"{a:.2f}%"] for m, a in df_table.values],
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
    t = str(text).lower()
    if "equal to" in t:
        return "Equality"
    elif "more" in t or "less" in t:
        return "Comparison"
    elif any(w in t for w in ["many", "number", "amount"]):
        return "Counting"
    elif any(w in t for w in ["small", "large", "square", "circle"]):
        return "Attribute"
    elif " at the " in t:
        return "Location"
    elif t.startswith(("is", "are", "does", "do")):
        return "Yes/No"
    return "Other"

def plot_accuracy_by_type(cfg):
    all_data = []

    for model, filename in cfg["models"].items():
        file_path = os.path.join(cfg["results_dir"], filename)
        df = pd.read_csv(file_path)
        
        df["qtype"] = df["question_text"].apply(categorize_question)
        grouped = df.groupby("qtype")["correct"].mean() * 100
        grouped.name = model
        all_data.append(grouped)

    combined_df = pd.concat(all_data, axis=1).fillna(0)
    categories = combined_df.index
    models = combined_df.columns
    
    x = np.arange(len(categories))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * width
        rects = ax.bar(x + offset, combined_df[model], width, label=model)
        
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, weight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Question Type', fontsize=12)
    ax.set_title('Model Accuracy by Question Type', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15)
    ax.set_ylim(0, 115)
    ax.legend()

    plt.tight_layout()
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
        if "avg_gpu_w" in df.columns:
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
