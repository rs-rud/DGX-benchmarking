import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

# ---------------- Style ---------------- #
plt.style.use("ggplot")

# ---------------- Helpers ---------------- #

def load_config(path):
    with open(path) as f:
        return json.load(f)

def load_csv(results_dir, filename):
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close(fig)

def get_run_dirs(base_results):
    pattern = os.path.join(base_results, "Run*")
    dirs = glob.glob(pattern)
    dirs = [d for d in dirs if os.path.isdir(d) and os.path.basename(d).startswith("Run")]
    return sorted(dirs)

def load_all_runs(base_results, filename):
    run_dirs = get_run_dirs(base_results)
    dfs = []
    for rd in run_dirs:
        df = load_csv(rd, filename)
        if df is not None:
            dfs.append(df)
    return dfs

def categorize_question(text):
    t = str(text).lower()
    if "equal to" in t: return "Equality"
    elif "more" in t or "less" in t: return "Comparison"
    elif any(w in t for w in ["many", "number", "amount"]): return "Counting"
    elif any(w in t for w in ["small", "large", "square", "circle"]): return "Attribute"
    elif " at the " in t: return "Location"
    elif t.startswith(("is", "are", "does", "do")): return "Yes/No"
    return "Other"

# ---------------- Single‑run plots ---------------- #

def plot_accuracy(cfg, results_dir, plots_dir):
    for engine, models in cfg["models"].items():
        rows = []
        for model, file in models.items():
            df = load_csv(results_dir, file)
            if df is not None:
                acc = df["correct"].mean() * 100
                rows.append((model, acc))

        if not rows: continue
        df_table = pd.DataFrame(rows, columns=["Model", "Accuracy"]).sort_values("Accuracy", ascending=False)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        table = ax.table(cellText=[[m, f"{a:.2f}%"] for m, a in df_table.values],
                         colLabels=["Model", "Accuracy"], cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        ax.set_title(f"{engine} - Overall Accuracy", fontsize=14, weight="bold")
        save(fig, plots_dir, f"accuracy_table_{engine}.png")

def plot_accuracy_by_type(cfg, results_dir, plots_dir):
    for engine, models in cfg["models"].items():
        all_data = []
        for model, filename in models.items():
            df = load_csv(results_dir, filename)
            if df is not None:
                df["qtype"] = df["question_text"].apply(categorize_question)
                grouped = df.groupby("qtype")["correct"].mean() * 100
                grouped.name = model
                all_data.append(grouped)

        if not all_data: continue
        combined_df = pd.concat(all_data, axis=1).fillna(0)
        x = np.arange(len(combined_df.index))
        width = 0.8 / len(combined_df.columns)
        fig, ax = plt.subplots(figsize=(12, 7))

        for i, model in enumerate(combined_df.columns):
            offset = (i - (len(combined_df.columns) - 1) / 2) * width
            ax.bar(x + offset, combined_df[model], width, label=model)

        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{engine} - Accuracy by Question Type', weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(combined_df.index, rotation=15)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
        save(fig, plots_dir, f"accuracy_by_type_{engine}.png")

def plot_latency(cfg, results_dir, plots_dir):
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, file in models.items():
            df = load_csv(results_dir, file)
            if df is not None:
                smoothed = df["latency_sec"].iloc[1:].rolling(window=10).mean()
                ax.plot(smoothed, label=model, linewidth=2)
        ax.set_ylabel("Latency (s)")
        ax.set_title(f"{engine} - Latency per Prompt", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"latency_{engine}.png")

def plot_power(cfg, results_dir, plots_dir):
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, file in models.items():
            df = load_csv(results_dir, file)
            if df is not None and "avg_gpu_w" in df.columns:
                smoothed = df["avg_gpu_w"].rolling(10, min_periods=1).mean()
                ax.plot(smoothed, label=model, linewidth=2)
        ax.set_ylabel("GPU Power (W)")
        ax.set_title(f"{engine} - Average GPU Power", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"power_gpu_{engine}.png")

def plot_temperature(cfg, results_dir, plots_dir):
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, file in models.items():
            df = load_csv(results_dir, file)
            if df is not None and "max_gpu_temp_c" in df.columns:
                smoothed = df["max_gpu_temp_c"].rolling(10, min_periods=1).mean()
                ax.plot(smoothed, label=model, linewidth=2)
        ax.set_ylabel("Temperature (°C)")
        ax.set_title(f"{engine} - Max Temperature", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"temperature_{engine}.png")

def plot_accuracy_vs_temp(cfg, results_dir, plots_dir):
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, file in models.items():
            df = load_csv(results_dir, file)
            if df is not None and "max_gpu_temp_c" in df.columns:
                df["temp_bin"] = df["max_gpu_temp_c"].round()
                temp_acc = df.groupby("temp_bin")["correct"].mean() * 100
                ax.plot(temp_acc.index, temp_acc.values, marker='o', label=model)
        ax.set_xlabel("GPU Temperature (°C)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{engine} - Accuracy Trend as Temperature Rises", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"acc_vs_temp_{engine}.png")

# ---------------- Aggregated plots (across runs) ---------------- #

def plot_accuracy_aggregated(cfg, results_dir, plots_dir):
    for engine, models in cfg["models"].items():
        rows = []
        for model, filename in models.items():
            dfs = load_all_runs(results_dir, filename)
            if not dfs: continue
            run_accs = [df["correct"].mean() * 100 for df in dfs]
            rows.append((model, np.mean(run_accs), np.std(run_accs)))

        if not rows: continue
        rows.sort(key=lambda x: x[1], reverse=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        cell_text = [[m, rf"{mu:.2f}% $\pm$ {sd:.2f}"] for m, mu, sd in rows]
        table = ax.table(cell_text, colLabels=["Model", r"Accuracy (Mean $\pm$ Std)"], cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        ax.set_title(f"{engine} - Overall Aggregated Accuracy", fontsize=14, weight="bold")
        save(fig, plots_dir, f"accuracy_overall_{engine}.png")

def plot_accuracy_by_type_aggregated(cfg, results_dir, plots_dir):
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(12, 7))
        model_names = list(models.keys())
        all_types = sorted(["Equality", "Comparison", "Counting", "Attribute", "Location", "Yes/No"])
        x = np.arange(len(all_types))
        width = 0.8 / len(model_names)

        for i, model in enumerate(model_names):
            dfs = load_all_runs(results_dir, models[model])
            type_accs = {t: [] for t in all_types}
            for df in dfs:
                df["qtype"] = df["question_text"].apply(categorize_question)
                grouped = df.groupby("qtype")["correct"].mean() * 100
                for t in all_types:
                    if t in grouped: type_accs[t].append(grouped[t])
            
            means = [np.mean(type_accs[t]) if type_accs[t] else 0 for t in all_types]
            stds = [np.std(type_accs[t]) if type_accs[t] else 0 for t in all_types]
            offset = (i - (len(model_names) - 1) / 2) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=model)

        ax.set_xticks(x)
        ax.set_xticklabels(all_types, rotation=15)
        ax.set_title(f"{engine} - Accuracy by Question Type (Mean ± Std)", weight="bold")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
        save(fig, plots_dir, f"accuracy_by_type_{engine}.png")

def plot_latency_aggregated(cfg, results_dir, plots_dir):
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, filename in models.items():
            dfs = load_all_runs(results_dir, filename)
            if not dfs: continue
            processed = [df["latency_sec"].iloc[1:].values for df in dfs]
            min_len = min(len(p) for p in processed)
            arr = np.array([p[:min_len] for p in processed])
            mean_lat = pd.Series(np.mean(arr, axis=0)).rolling(10, min_periods=1).mean()
            std_lat = pd.Series(np.std(arr, axis=0)).rolling(10, min_periods=1).mean()
            iters = np.arange(1, min_len + 1)
            line, = ax.plot(iters, mean_lat, label=model, linewidth=2)
            ax.fill_between(iters, mean_lat - std_lat, mean_lat + std_lat, alpha=0.2, color=line.get_color())
        ax.set_title(f"{engine} - Latency (Mean ± Std)", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"latency_{engine}.png")

def plot_power_aggregated(cfg, results_dir, plots_dir):
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, filename in models.items():
            dfs = load_all_runs(results_dir, filename)
            if not dfs or "avg_gpu_w" not in dfs[0].columns: continue
            processed = [df["avg_gpu_w"].values for df in dfs]
            min_len = min(len(p) for p in processed)
            arr = np.array([p[:min_len] for p in processed])
            mean_p = pd.Series(np.mean(arr, axis=0)).rolling(10, min_periods=1).mean()
            std_p = pd.Series(np.std(arr, axis=0)).rolling(10, min_periods=1).mean()
            line, = ax.plot(mean_p, label=model, linewidth=2)
            ax.fill_between(range(min_len), mean_p - std_p, mean_p + std_p, alpha=0.2, color=line.get_color())
        ax.set_title(f"{engine} - GPU Power (Mean ± Std)", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"power_gpu_{engine}.png")

def plot_temperature_aggregated(cfg, results_dir, plots_dir):
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, filename in models.items():
            dfs = load_all_runs(results_dir, filename)
            if not dfs or "max_gpu_temp_c" not in dfs[0].columns: continue
            processed = [df["max_gpu_temp_c"].values for df in dfs]
            min_len = min(len(p) for p in processed)
            arr = np.array([p[:min_len] for p in processed])
            mean_t = pd.Series(np.mean(arr, axis=0)).rolling(10, min_periods=1).mean()
            std_t = pd.Series(np.std(arr, axis=0)).rolling(10, min_periods=1).mean()
            line, = ax.plot(mean_t, label=model, linewidth=2)
            ax.fill_between(range(min_len), mean_t - std_t, mean_t + std_t, alpha=0.2, color=line.get_color())
        ax.set_title(f"{engine} - Max Temperature (Mean ± Std)", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"temperature_{engine}.png")

def plot_accuracy_vs_temp_aggregated(cfg, results_dir, plots_dir):
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, filename in models.items():
            dfs = load_all_runs(results_dir, filename)
            if not dfs or "max_gpu_temp_c" not in dfs[0].columns: continue
            all_df = pd.concat(dfs, ignore_index=True)
            all_df["temp_bin"] = all_df["max_gpu_temp_c"].round()
            # Mean and std for accuracy per temp bin
            grouped = all_df.groupby("temp_bin")["correct"].agg(['mean', 'std']) * 100
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], marker='o', capsize=3, label=model)
        ax.set_title(f"{engine} - Accuracy vs Temperature (Mean ± Std)", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"acc_vs_temp_{engine}.png")

# ---------------- Main ---------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run", help="Run subdirectory (e.g., Run1)")
    p.add_argument("--aggregate", action="store_true", help="Aggregate across all runs")
    p.add_argument("--accuracy", action="store_true")
    p.add_argument("--accuracy-by-type", action="store_true")
    p.add_argument("--latency", action="store_true")
    p.add_argument("--power", action="store_true")
    p.add_argument("--temp", action="store_true")
    p.add_argument("--acc-vs-temp", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    base_results = cfg["results_dir"]
    base_plots = cfg["plots_dir"]

    if args.aggregate:
        results_dir = base_results
        plots_dir = os.path.join(base_plots, "aggregated")
        if args.accuracy or args.all: plot_accuracy_aggregated(cfg, results_dir, plots_dir)
        if args.accuracy_by_type or args.all: plot_accuracy_by_type_aggregated(cfg, results_dir, plots_dir)
        if args.latency or args.all: plot_latency_aggregated(cfg, results_dir, plots_dir)
        if args.power or args.all: plot_power_aggregated(cfg, results_dir, plots_dir)
        if args.temp or args.all: plot_temperature_aggregated(cfg, results_dir, plots_dir)
        if args.acc_vs_temp or args.all: plot_accuracy_vs_temp_aggregated(cfg, results_dir, plots_dir)
    else:
        results_dir = os.path.join(base_results, args.run) if args.run else base_results
        plots_dir = os.path.join(base_plots, args.run) if args.run else base_plots
        if args.accuracy or args.all: plot_accuracy(cfg, results_dir, plots_dir)
        if args.accuracy_by_type or args.all: plot_accuracy_by_type(cfg, results_dir, plots_dir)
        if args.latency or args.all: plot_latency(cfg, results_dir, plots_dir)
        if args.power or args.all: plot_power(cfg, results_dir, plots_dir)
        if args.temp or args.all: plot_temperature(cfg, results_dir, plots_dir)
        if args.acc_vs_temp or args.all: plot_accuracy_vs_temp(cfg, results_dir, plots_dir)

if __name__ == "__main__":
    main()
