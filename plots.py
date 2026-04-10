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
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved → {path}")

def get_run_dirs(base_results):
    """Return sorted list of run subdirectories (Run1, Run2, ...)."""
    pattern = os.path.join(base_results, "Run*")
    dirs = glob.glob(pattern)
    # Filter only directories that match "Run" followed by a number
    dirs = [d for d in dirs if os.path.isdir(d) and os.path.basename(d).startswith("Run")]
    return sorted(dirs)  # natural order (glob may already be sorted)

def load_all_runs(base_results, filename):
    """Load the same CSV file from all run subdirectories.
    Returns a list of DataFrames, one per run.
    """
    run_dirs = get_run_dirs(base_results)
    dfs = []
    for rd in run_dirs:
        path = os.path.join(rd, filename)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
        else:
            print(f"Warning: {path} not found, skipping this run.")
    return dfs

# ---------------- Single‑run plots (original, with table for accuracy) ---------------- #

def plot_accuracy(cfg, results_dir, plots_dir):
    """Original: accuracy table per model."""
    for engine, models in cfg["models"].items():
        rows = []
        for model, file in models.items():
            df = load_csv(results_dir, file)
            acc = df["correct"].mean() * 100
            rows.append((model, acc))

        df = pd.DataFrame(rows, columns=["Model", "Accuracy"])
        df = df.sort_values("Accuracy", ascending=False)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")

        table = ax.table(
            cellText=[[m, f"{a:.2f}%"] for m, a in df.values],
            colLabels=["Model", "Accuracy"],
            cellLoc="center",
            loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)

        ax.set_title(f"{engine} - Overall Accuracy", fontsize=14, weight="bold")
        save(fig, plots_dir, f"accuracy_table_{engine}.png")

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

def plot_accuracy_by_type(cfg, results_dir, plots_dir):
    """Original: grouped bar chart without error bars."""
    for engine, models in cfg["models"].items():
        all_data = []

        for model, filename in models.items():
            file_path = os.path.join(results_dir, filename)
            df = pd.read_csv(file_path)
            
            df["qtype"] = df["question_text"].apply(categorize_question)
            grouped = df.groupby("qtype")["correct"].mean() * 100
            grouped.name = model
            all_data.append(grouped)

        combined_df = pd.concat(all_data, axis=1).fillna(0)
        categories = combined_df.index
        models_list = combined_df.columns
        
        x = np.arange(len(categories))
        width = 0.8 / len(models_list)

        fig, ax = plt.subplots(figsize=(12, 7))
        for i, model in enumerate(models_list):
            offset = (i - (len(models_list) - 1) / 2) * width
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
        ax.set_title(f'{engine} - Model Accuracy by Question Type', fontsize=14, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=15)
        ax.set_ylim(0, 115)
        ax.legend()

        plt.tight_layout()
        save(fig, plots_dir, f"accuracy_by_type_{engine}.png")

def plot_latency(cfg, results_dir, plots_dir):
    """Original: latency line plot (smoothed)."""
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, file in models.items():
            df = load_csv(results_dir, file)
            df = df.iloc[1:]
            smoothed = df["latency_sec"].rolling(window=10).mean()
            ax.plot(smoothed, label=model, linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Latency (s)")
        ax.set_title(f"{engine} - Latency per Prompt", weight="bold")
        ax.legend()
        save(fig, plots_dir, f"latency_{engine}.png")

def plot_power(cfg, results_dir, plots_dir):
    """Original: GPU power line plot (smoothed)."""
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, file in models.items():
            df = load_csv(results_dir, file)
            if "avg_gpu_w" in df.columns:
                smoothed = df["avg_gpu_w"].rolling(10, min_periods=1).mean()
                ax.plot(smoothed, label=model, linewidth=2)
        ax.set_ylabel("GPU Power (W)")
        ax.set_xlabel("Iteration")
        ax.set_title(f"{engine} - Average GPU Power (Smoothed)", weight="bold")
        ax.legend()
        save(fig, plots_dir, f"power_gpu_{engine}.png")

def plot_temperature(cfg, results_dir, plots_dir):
    """Original: temperature line plot (smoothed)."""
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, file in models.items():
            df = load_csv(results_dir, file)
            if "max_gpu_temp_c" in df.columns:
                smoothed = df["max_gpu_temp_c"].rolling(10, min_periods=1).mean()
                ax.plot(smoothed, label=model, linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title(f"{engine} - Max Temperature per Prompt (Smoothed)", weight="bold")
        ax.legend()
        save(fig, plots_dir, f"temperature_{engine}.png")

def plot_accuracy_vs_temp(cfg, results_dir, plots_dir):
    """Original: accuracy vs temperature scatter/line."""
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, file in models.items():
            df = load_csv(results_dir, file)
            if "max_gpu_temp_c" in df.columns and "correct" in df.columns:
                df["temp_bin"] = df["max_gpu_temp_c"].round()
                temp_acc = df.groupby("temp_bin")["correct"].mean() * 100
                ax.plot(temp_acc.index, temp_acc.values, marker='o', label=model, linewidth=2)

        ax.set_xlabel("GPU Temperature (°C)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{engine} - Accuracy Trend as Temperature Rises", weight="bold")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        save(fig, plots_dir, f"acc_vs_temp_{engine}.png")

# ---------------- Aggregated plots (across runs) ---------------- #
def plot_accuracy_aggregated(cfg, results_dir, plots_dir):
    """
    Aggregated accuracy table per model (mean ± std across runs).
    """
    for engine, models in cfg["models"].items():
        rows = []
        for model, filename in models.items():
            dfs = load_all_runs(results_dir, filename)
            if not dfs:
                continue
            
            run_accs = [df["correct"].mean() * 100 for df in dfs]
            mean_acc = np.mean(run_accs)
            std_acc = np.std(run_accs)
            rows.append((model, mean_acc, std_acc))

        if not rows:
            continue

        rows.sort(key=lambda x: x[1], reverse=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")

        cell_text = [[m, rf"{mu:.2f}% $\pm$ {sd:.2f}"] for m, mu, sd in rows]
        
        table = ax.table(
            cellText=cell_text,
            colLabels=["Model", r"Accuracy (Mean $\pm$ Std)"],
            cellLoc="center",
            loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)  # Scaled to be more readable as a table

        ax.set_title(f"{engine} - Overall Aggregated Accuracy", fontsize=14, weight="bold")
        save(fig, plots_dir, f"accuracy_overall_{engine}.png")

def plot_accuracy_by_type_aggregated(cfg, results_dir, plots_dir):
    """Grouped bar chart with error bars for accuracy per question type."""
    for engine, models in cfg["models"].items():
        all_dfs = []
        for filename in models.values():
            all_dfs.extend(load_all_runs(results_dir, filename))
        if not all_dfs:
            continue

        # Sort
        all_types = set()
        for df in all_dfs:
            df["qtype"] = df["question_text"].apply(categorize_question)
            all_types.update(df["qtype"].unique())
        all_types = sorted(all_types)

        model_data = {model: {t: [] for t in all_types} for model in models.keys()}

        for model, filename in models.items():
            dfs = load_all_runs(results_dir, filename)
            for df in dfs:
                df["qtype"] = df["question_text"].apply(categorize_question)
                grouped = df.groupby("qtype")["correct"].mean() * 100
                for t in all_types:
                    val = grouped.get(t, np.nan)
                    model_data[model][t].append(val)

        # Compute mean and std 
        x = np.arange(len(all_types))
        width = 0.8 / len(models)
        fig, ax = plt.subplots(figsize=(12, 7))

        for i, (model_name, type_dict) in enumerate(model_data.items()):
            means = []
            stds = []
            for t in all_types:
                vals = [v for v in type_dict[t] if not np.isnan(v)]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                else:
                    means.append(0)
                    stds.append(0)
            offset = (i - (len(models) - 1) / 2) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=model_name)

        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("Question Type")
        ax.set_title(f"{engine} - Accuracy by Question Type (mean ± std across 6 runs)")
        ax.set_xticks(x)
        ax.set_xticklabels(all_types, rotation=15)
        ax.legend()
        ax.set_ylim(0, 115)
        fig.tight_layout()
        save(fig, plots_dir, f"accuracy_by_type_{engine}.png")

def plot_latency_aggregated(cfg, results_dir, plots_dir):
    """Line plot with shaded standard deviation region, skipping the first iteration."""
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, filename in models.items():
            dfs = load_all_runs(results_dir, filename)
            if not dfs:
                continue
            
            processed_dfs = [df.iloc[1:] for df in dfs]
            max_len = max(len(df) for df in processed_dfs)
            latencies = np.array([df["latency_sec"].values[:max_len] for df in processed_dfs])
            
            mean_lat = np.nanmean(latencies, axis=0)
            std_lat = np.nanstd(latencies, axis=0)

            iters = np.arange(1, len(mean_lat) + 1)
            
            window = 10
            if len(mean_lat) > window:
                mean_smooth = pd.Series(mean_lat).rolling(window, min_periods=1).mean()
                std_smooth = pd.Series(std_lat).rolling(window, min_periods=1).mean()
            else:
                mean_smooth = mean_lat
                std_smooth = std_lat

            ax.plot(iters, mean_smooth, label=model, linewidth=2)
            ax.fill_between(iters, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3)

        ax.set_xlabel("Iteration (starting from 1)")
        ax.set_ylabel("Latency (s)")
        ax.set_title(f"{engine} - Latency per Prompt (mean ± std across runs, skipping first)")
        ax.legend()
        fig.tight_layout()
        save(fig, plots_dir, f"latency_{engine}.png")

def plot_power_aggregated(cfg, results_dir, plots_dir):
    """Line plot with shaded std for GPU power."""
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, filename in models.items():
            dfs = load_all_runs(results_dir, filename)
            if not dfs or "avg_gpu_w" not in dfs[0].columns:
                continue
            max_len = max(len(df) for df in dfs)
            power = np.array([df["avg_gpu_w"].values[:max_len] for df in dfs])
            mean_pow = np.nanmean(power, axis=0)
            std_pow = np.nanstd(power, axis=0)

            iters = np.arange(len(mean_pow))
            window = 10
            if len(mean_pow) > window:
                mean_smooth = pd.Series(mean_pow).rolling(window, min_periods=1).mean()
                std_smooth = pd.Series(std_pow).rolling(window, min_periods=1).mean()
            else:
                mean_smooth = mean_pow
                std_smooth = std_pow

            ax.plot(iters, mean_smooth, label=model, linewidth=2)
            ax.fill_between(iters, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("GPU Power (W)")
        ax.set_title(f"{engine} - Average GPU Power (mean ± std across 6 runs)")
        ax.legend()
        fig.tight_layout()
        save(fig, plots_dir, f"power_gpu_{engine}.png")

def plot_temperature_aggregated(cfg, results_dir, plots_dir):
    """Line plot with shaded std for temperature."""
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, filename in models.items():
            dfs = load_all_runs(results_dir, filename)
            if not dfs or "max_gpu_temp_c" not in dfs[0].columns:
                continue
            max_len = max(len(df) for df in dfs)
            temp = np.array([df["max_gpu_temp_c"].values[:max_len] for df in dfs])
            mean_temp = np.nanmean(temp, axis=0)
            std_temp = np.nanstd(temp, axis=0)

            iters = np.arange(len(mean_temp))
            window = 10
            if len(mean_temp) > window:
                mean_smooth = pd.Series(mean_temp).rolling(window, min_periods=1).mean()
                std_smooth = pd.Series(std_temp).rolling(window, min_periods=1).mean()
            else:
                mean_smooth = mean_temp
                std_smooth = std_temp

            ax.plot(iters, mean_smooth, label=model, linewidth=2)
            ax.fill_between(iters, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title(f"{engine} - Max Temperature per Prompt (mean ± std across 6 runs)")
        ax.legend()
        fig.tight_layout()
        save(fig, plots_dir, f"temperature_{engine}.png")

def plot_accuracy_vs_temp_aggregated(cfg, results_dir, plots_dir):
    """Accuracy vs temperature with error bars (std)."""
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, filename in models.items():
            dfs = load_all_runs(results_dir, filename)
            if not dfs or "max_gpu_temp_c" not in dfs[0].columns:
                continue
            # Collect all data points across runs, group by temperature
            all_df = pd.concat(dfs, ignore_index=True)
            all_df["temp_bin"] = all_df["max_gpu_temp_c"].round()

            # Compute mean accuracy across all runs and std
            grouped = all_df.groupby("temp_bin")["correct"].agg(['mean', 'std']) * 100
            grouped = grouped.sort_index()
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                        marker='o', capsize=3, label=model, linewidth=2)

        ax.set_xlabel("GPU Temperature (°C)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{engine} - Accuracy Trend as Temperature Rises (mean ± std across 6 runs)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()
        save(fig, plots_dir, f"acc_vs_temp_{engine}.png")

# ---------------- Main ---------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run", help="Run subdirectory (e.g., Run1). If omitted and --aggregate not used, uses base results_dir directly.")
    p.add_argument("--aggregate", action="store_true", help="Aggregate across all runs (Run1..Run6). Overrides --run.")
    p.add_argument("--accuracy", action="store_true")
    p.add_argument("--accuracy-by-type", action="store_true")
    p.add_argument("--latency", action="store_true")
    p.add_argument("--power", action="store_true")
    p.add_argument("--temp", action="store_true")
    p.add_argument("--acc-vs-temp", action="store_true")
    p.add_argument("--all", action="store_true", help="Run all plots")
    args = p.parse_args()

    cfg = load_config(args.config)

    base_results = cfg["results_dir"]
    base_plots = cfg["plots_dir"]

    if args.aggregate:
        results_dir = base_results
        plots_dir = os.path.join(base_plots, "aggregated")
        if args.accuracy or args.all:
            plot_accuracy_aggregated(cfg, results_dir, plots_dir)
        if args.accuracy_by_type or args.all:
            plot_accuracy_by_type_aggregated(cfg, results_dir, plots_dir)
        if args.latency or args.all:
            plot_latency_aggregated(cfg, results_dir, plots_dir)
        if args.power or args.all:
            plot_power_aggregated(cfg, results_dir, plots_dir)
        if args.temp or args.all:
            plot_temperature_aggregated(cfg, results_dir, plots_dir)
        if args.acc_vs_temp or args.all:
            plot_accuracy_vs_temp_aggregated(cfg, results_dir, plots_dir)
    else:
        if args.run:
            results_dir = os.path.join(base_results, args.run)
            plots_dir = os.path.join(base_plots, args.run)
        else:
            results_dir = base_results
            plots_dir = base_plots

        if not os.path.isdir(results_dir):
            raise FileNotFoundError(f"Results directory not found: {results_dir}")

        if args.accuracy or args.all:
            plot_accuracy(cfg, results_dir, plots_dir)
        if args.accuracy_by_type or args.all:
            plot_accuracy_by_type(cfg, results_dir, plots_dir)
        if args.latency or args.all:
            plot_latency(cfg, results_dir, plots_dir)
        if args.power or args.all:
            plot_power(cfg, results_dir, plots_dir)
        if args.temp or args.all:
            plot_temperature(cfg, results_dir, plots_dir)
        if args.acc_vs_temp or args.all:
            plot_accuracy_vs_temp(cfg, results_dir, plots_dir)

if __name__ == "__main__":
    main()

