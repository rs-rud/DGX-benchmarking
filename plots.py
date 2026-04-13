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

def extract_family_and_model(filename):
    """
    Parse filename like benchmark_results_ollama_gemma3_2b.csv
    Returns (family, model_variant)
    """
    name = os.path.splitext(filename)[0]
    lower = name.lower()
    if "gemma3" in lower:
        family = "Gemma3"
    elif "ministral" in lower:
        family = "Ministral3"
    elif "qwen2" in lower:
        family = "Qwen2"
    elif "llava" in lower:
        family = "Llava"
    else:
        family = "Other"
    model_variant = name
    return family, model_variant

# ---------------- Single‑run plots (engine‑based) ---------------- #

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

# ---------------- Aggregated plots (engine‑based, across runs) ---------------- #

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
            grouped = all_df.groupby("temp_bin")["correct"].agg(['mean', 'std']) * 100
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], marker='o', capsize=3, label=model)
        ax.set_title(f"{engine} - Accuracy vs Temperature (Mean ± Std)", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"acc_vs_temp_{engine}.png")

def plot_energy_cdf_aggregated(cfg, results_dir, plots_dir):
    """
    For each engine, plot CDF of energy per query (Joules) for each model variant.
    Energy = avg_gpu_w * latency_sec (if both columns exist).
    """
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        has_any = False
        for model_name, filename in models.items():
            dfs = load_all_runs(results_dir, filename)
            if not dfs or "avg_gpu_w" not in dfs[0].columns or "latency_sec" not in dfs[0].columns:
                continue
            all_energy = []
            for df in dfs:
                energy_joules = df["avg_gpu_w"] * df["latency_sec"]
                all_energy.extend(energy_joules.dropna().values)
            if not all_energy:
                continue
            has_any = True
            sorted_e = np.sort(all_energy)
            cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
            ax.step(sorted_e, cdf, where='post', label=model_name, linewidth=2)
        if not has_any:
            continue
        ax.set_title(f"{engine} - Energy per Query CDF", weight="bold")
        ax.set_xlabel("Energy (Joules)")
        ax.set_ylabel("Cumulative Probability")
        ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.6)
        save(fig, plots_dir, f"energy_cdf_{engine}.png")

# ---------------- Family‑based aggregated plots (new --family flag) ---------------- #

def load_all_data_by_family(base_results, config):
    """
    Returns dict: family -> model_variant -> list of DataFrames (one per run)
    """
    family_data = {}
    run_dirs = get_run_dirs(base_results)
    for run_dir in run_dirs:
        for engine, models in config["models"].items():
            for model_name, filename in models.items():
                df = load_csv(run_dir, filename)
                if df is None:
                    continue
                family, model_var = extract_family_and_model(filename)
                if family not in family_data:
                    family_data[family] = {}
                if model_var not in family_data[family]:
                    family_data[family][model_var] = []
                family_data[family][model_var].append(df)
    return family_data

def plot_accuracy_family(family_data, plots_dir):
    for family, models in family_data.items():
        rows = []
        for model_var, dfs in models.items():
            if not dfs:
                continue
            all_acc = []
            for df in dfs:
                all_acc.extend(df["correct"].values)
            mean_acc = np.mean(all_acc) * 100
            std_acc = np.std(all_acc) * 100
            rows.append((model_var, mean_acc, std_acc))
        if not rows:
            continue
        rows.sort(key=lambda x: x[1], reverse=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        cell_text = [[m, f"{mu:.2f}% ± {sd:.2f}"] for m, mu, sd in rows]
        table = ax.table(cell_text, colLabels=["Model", "Accuracy (Mean ± Std)"],
                         cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        ax.set_title(f"{family} - Overall Aggregated Accuracy", fontsize=14, weight="bold")
        save(fig, plots_dir, f"accuracy_overall_{family}.png")

def plot_accuracy_by_type_family(family_data, plots_dir):
    all_types = ["Equality", "Comparison", "Counting", "Attribute", "Location", "Yes/No"]
    for family, models in family_data.items():
        fig, ax = plt.subplots(figsize=(12, 7))
        model_vars = list(models.keys())
        x = np.arange(len(all_types))
        width = 0.8 / len(model_vars) if model_vars else 0.8
        for i, model_var in enumerate(model_vars):
            dfs = models[model_var]
            type_accs = {t: [] for t in all_types}
            for df in dfs:
                df = df.copy()
                df["qtype"] = df["question_text"].apply(categorize_question)
                grouped = df.groupby("qtype")["correct"].mean() * 100
                for t in all_types:
                    if t in grouped:
                        type_accs[t].append(grouped[t])
            means = [np.mean(type_accs[t]) if type_accs[t] else 0 for t in all_types]
            stds = [np.std(type_accs[t]) if type_accs[t] else 0 for t in all_types]
            offset = (i - (len(model_vars) - 1) / 2) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=model_var)
        ax.set_xticks(x)
        ax.set_xticklabels(all_types, rotation=15)
        ax.set_title(f"{family} - Accuracy by Question Type (Mean ± Std)", weight="bold")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
        save(fig, plots_dir, f"accuracy_by_type_{family}.png")

def plot_latency_family(family_data, plots_dir):
    for family, models in family_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_var, dfs in models.items():
            lat_series = [df["latency_sec"].iloc[1:].values for df in dfs]  # skip first outlier
            if not lat_series:
                continue
            min_len = min(len(ls) for ls in lat_series)
            arr = np.array([ls[:min_len] for ls in lat_series])
            mean_lat = pd.Series(np.mean(arr, axis=0)).rolling(10, min_periods=1).mean()
            std_lat = pd.Series(np.std(arr, axis=0)).rolling(10, min_periods=1).mean()
            iters = np.arange(1, min_len + 1)
            line, = ax.plot(iters, mean_lat, label=model_var, linewidth=2)
            ax.fill_between(iters, mean_lat - std_lat, mean_lat + std_lat, alpha=0.2, color=line.get_color())
        ax.set_title(f"{family} - Latency (Mean ± Std)", weight="bold")
        ax.set_xlabel("Prompt number")
        ax.set_ylabel("Latency (s)")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"latency_{family}.png")

def plot_power_family(family_data, plots_dir):
    for family, models in family_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_var, dfs in models.items():
            if "avg_gpu_w" not in dfs[0].columns:
                continue
            power_series = [df["avg_gpu_w"].values for df in dfs]
            min_len = min(len(ps) for ps in power_series)
            arr = np.array([ps[:min_len] for ps in power_series])
            mean_p = pd.Series(np.mean(arr, axis=0)).rolling(10, min_periods=1).mean()
            std_p = pd.Series(np.std(arr, axis=0)).rolling(10, min_periods=1).mean()
            line, = ax.plot(mean_p, label=model_var, linewidth=2)
            ax.fill_between(range(min_len), mean_p - std_p, mean_p + std_p, alpha=0.2, color=line.get_color())
        ax.set_title(f"{family} - GPU Power (Mean ± Std)", weight="bold")
        ax.set_xlabel("Prompt number")
        ax.set_ylabel("Power (W)")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"power_gpu_{family}.png")

def plot_temperature_family(family_data, plots_dir):
    for family, models in family_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_var, dfs in models.items():
            if "max_gpu_temp_c" not in dfs[0].columns:
                continue
            temp_series = [df["max_gpu_temp_c"].values for df in dfs]
            min_len = min(len(ts) for ts in temp_series)
            arr = np.array([ts[:min_len] for ts in temp_series])
            mean_t = pd.Series(np.mean(arr, axis=0)).rolling(10, min_periods=1).mean()
            std_t = pd.Series(np.std(arr, axis=0)).rolling(10, min_periods=1).mean()
            line, = ax.plot(mean_t, label=model_var, linewidth=2)
            ax.fill_between(range(min_len), mean_t - std_t, mean_t + std_t, alpha=0.2, color=line.get_color())
        ax.set_title(f"{family} - Max Temperature (Mean ± Std)", weight="bold")
        ax.set_xlabel("Prompt number")
        ax.set_ylabel("Temperature (°C)")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"temperature_{family}.png")

def plot_accuracy_vs_temp_family(family_data, plots_dir):
    for family, models in family_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_var, dfs in models.items():
            if "max_gpu_temp_c" not in dfs[0].columns:
                continue
            all_df = pd.concat(dfs, ignore_index=True)
            all_df["temp_bin"] = all_df["max_gpu_temp_c"].round()
            grouped = all_df.groupby("temp_bin")["correct"].agg(['mean', 'std']) * 100
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                        marker='o', capsize=3, label=model_var)
        ax.set_title(f"{family} - Accuracy vs Temperature (Mean ± Std)", weight="bold")
        ax.set_xlabel("GPU Temperature (°C)")
        ax.set_ylabel("Accuracy (%)")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"acc_vs_temp_{family}.png")

def plot_energy_cdf_family(family_data, plots_dir):
    """
    For each family, plot CDF of energy per query (Joules) for each model variant.
    Energy = avg_gpu_w * latency_sec (if both columns exist).
    """
    for family, models in family_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        has_any = False
        for model_var, dfs in models.items():
            if not dfs or "avg_gpu_w" not in dfs[0].columns or "latency_sec" not in dfs[0].columns:
                continue
            all_energy = []
            for df in dfs:
                energy_joules = df["avg_gpu_w"] * df["latency_sec"]
                all_energy.extend(energy_joules.dropna().values)
            if not all_energy:
                continue
            has_any = True
            sorted_e = np.sort(all_energy)
            cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
            ax.step(sorted_e, cdf, where='post', label=model_var, linewidth=2)
        if not has_any:
            continue
        ax.set_title(f"{family} - Energy per Query CDF", weight="bold")
        ax.set_xlabel("Energy (Joules)")
        ax.set_ylabel("Cumulative Probability")
        ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.6)
        save(fig, plots_dir, f"energy_cdf_{family}.png")

# ---------------- Main ---------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run", help="Run subdirectory (e.g., Run1) – used for single‑run plots")
    p.add_argument("--aggregate", action="store_true", help="Aggregate across all runs (engine‑based) → plots_dir/aggregated/")
    p.add_argument("--family", action="store_true", help="Family‑based aggregation (Gemma3, Ministral3, Qwen2, Llava) → plots_dir/family/")
    p.add_argument("--accuracy", action="store_true")
    p.add_argument("--accuracy-by-type", action="store_true")
    p.add_argument("--latency", action="store_true")
    p.add_argument("--power", action="store_true")
    p.add_argument("--temp", action="store_true")
    p.add_argument("--acc-vs-temp", action="store_true")
    p.add_argument("--energy-cdf", action="store_true", help="Plot CDF of energy per query (available with --aggregate or --family)")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    base_results = cfg["results_dir"]
    base_plots = cfg["plots_dir"]

    # Single run (no --aggregate, no --family)
    if not args.aggregate and not args.family:
        if args.run:
            results_dir = os.path.join(base_results, args.run)
            plots_dir = os.path.join(base_plots, args.run)
        else:
            results_dir = base_results
            plots_dir = base_plots
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
        if args.energy_cdf:
            print("Warning: --energy-cdf is only available with --aggregate or --family. Ignoring.")
        return

    # Engine‑based aggregation (original --aggregate behavior)
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
        if args.energy_cdf or args.all:
            plot_energy_cdf_aggregated(cfg, results_dir, plots_dir)
        return

    # Family‑based aggregation (new --family flag)
    if args.family:
        results_dir = base_results
        plots_dir = os.path.join(base_plots, "family")
        family_data = load_all_data_by_family(results_dir, cfg)
        if args.accuracy or args.all:
            plot_accuracy_family(family_data, plots_dir)
        if args.accuracy_by_type or args.all:
            plot_accuracy_by_type_family(family_data, plots_dir)
        if args.latency or args.all:
            plot_latency_family(family_data, plots_dir)
        if args.power or args.all:
            plot_power_family(family_data, plots_dir)
        if args.temp or args.all:
            plot_temperature_family(family_data, plots_dir)
        if args.acc_vs_temp or args.all:
            plot_accuracy_vs_temp_family(family_data, plots_dir)
        if args.energy_cdf or args.all:
            plot_energy_cdf_family(family_data, plots_dir)
        return

if __name__ == "__main__":
    main()
