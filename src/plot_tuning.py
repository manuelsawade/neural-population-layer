import os
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from library import get_display_name

def smooth_curve(values, weight=0.9):
    """EMA smoothing (TensorBoard-like)"""
    last = values[0]
    smoothed = []
    for v in values:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return smoothed

def get_latest_run_folder():
    folder = Path('/Users/manuelsawade/ray_results')
    for dir in sorted(folder.glob("run*"), reverse=True):
        return dir

def plot_ray_tune_metric(run_path, metric_names, smoothing=0.0):
    trials = []
    print(f"Loading Ray Tune logs from: {run_path}")

    # --- Load all trials ---
    for root, dirs, files in os.walk(run_path):
        for file in files:
            if file == "progress.csv":
                df = pd.read_csv(os.path.join(root, file))
                df["trial"] = os.path.basename(root)
                trials.append(df)

    if not trials:
        raise ValueError("No Ray Tune progress.csv files found in given path!")

    # Combine all trials
    df_all = pd.concat(trials, ignore_index=True)

    # --- Plot ---
    fig, axes = plt.subplots(len(metric_names), len(metric_names[0]),figsize=(12,5), sharex=True)
    cmap = plt.get_cmap("tab10")

    for ax_row, metric_row in zip(axes, metric_names):
        for ax, metric_name in zip(ax_row, metric_row):
            for i, (trial_name, group) in enumerate(df_all.groupby("trial")):
                steps = group["training_iteration"] if "training_iteration" in group else np.arange(len(group))
                values = group[metric_name].values

                i_norm = (i - 0) / (len(df_all.groupby("trial")) - 0)

                ax.plot(
                    steps, values,
                    color=cmap(i_norm), linewidth=2
                )

                ax.set_title(f"{get_display_name(metric_name)}", fontsize=14)
                if "fsa" in metric_name:
                    ax.set_xlabel("Training Iteration", fontsize=12)

                #ax.set_ylabel(get_display_name(metric_name), fontsize=14)
                ax.grid(True, linestyle="--", alpha=0.3)

    # Save figure
    out_file = f"./images/tuning_trials.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close(fig)

    print(f"Saved plot → {out_file}")


# Example usage:
plot_ray_tune_metric(get_latest_run_folder(), [["loss_norm", "loss"], ["fsa_inf_mean_norm", "fsa_inf_mean_diff"]])
