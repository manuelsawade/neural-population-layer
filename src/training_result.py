#!/usr/bin/env python3
"""
plot_experiment_results_all_metrics.py

Usage:
    python plot_experiment_results_all_metrics.py /path/to/json/folder --outdir ./plots

This version automatically plots ALL numeric metrics found in the data.
"""

import json
from pathlib import Path
from typing import Any, Dict
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_csv_files(folder: str) -> pd.DataFrame:
    folder_path = Path(folder)

    dataframes = []

    for p, csv in zip(sorted(folder_path.glob("*.json")), sorted(folder_path.glob("*.csv"))):
        print(csv)
        try:
            with open(p, "r") as f:
                raw = json.loads(f.read())
        except Exception as e:
            print(f"Skipping {p} (could not read): {e}")
            continue

        stack = raw["network"]
        noise = raw["hyper_parameter"]["training_noise"]

        df = pd.read_csv(csv)

        df["stack"] = stack
        df["noise"] = noise

        dataframes.append(df)

    if not dataframes:
        raise RuntimeError(f"No JSON files loaded from {folder}")
    return pd.concat(dataframes)

def main():
    identifier = "mnist_evaluation_preferred_value"
    folder = f"./experiments/{identifier}/"
    print("Loading JSON files...")
    df = load_csv_files(folder)
    print(f"Loaded {len(df)} records.")
    #print(df)

    metrics = ["accuracy", "loss", "fsa_inf_mean", "fsa_inf_mean_smoothed"]

    # Prepare a figure with 3 subplots in a row
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharex=True)
    print("train")
    
    df["fsa_inf_mean_smoothed"] = (df
        .groupby(["stack", "noise"])["fsa_inf_mean"]
        .transform(lambda x: x
                   .rolling(window=20, center=True, min_periods=1)
                   .mean())
    )

    color_map = {
        "linear": {
            "1.0": "darkorange",
            "0.5": "#FFA447",
            "0.0": "#FFE0B2",
        },
        "population": {
            "0.0": "#E0AAFF",
            "0.5": "#9A58D0",
            "1.0": "purple",
        },
    }

    # Loop through each metric and create a plot
    for ax, metric in zip(axes, metrics):
        # Group by stack and noise for each metric

        # Plot each stack as a line with error bars
        for stack_name, group in df.groupby("stack"):
            for noise_level, group in group.groupby("noise"):
                print(stack_name, noise_level)
                color = color_map[stack_name][str(noise_level)]
                ax.plot(
                    group["epoch"], 
                    group[metric], 
                    linewidth=2,
                    color=color,
                    label=f"{stack_name} (noise={noise_level})"
                )

        # Axis settings
        ax.set_title(metric.capitalize())
        if metric == "accuracy":
            ax.set_ylabel("Training")

        if metric == "fsa_inf_mean":
            ax.set_ylim(0.35, 0.56)
            ax.set_title("FSA Inf")

        if metric == "fsa_inf_mean_smoothed":
            ax.set_ylim(0.35, 0.56)
            ax.set_title("FSA Inf Smoothed (Window = 20)")

        ax.grid(True, linestyle=":", alpha=0.6)

    # # Add legend to the first subplot only (to avoid clutter)
    axes[0].legend(title="Stack", loc="best")
    fig.suptitle("CIFAR10 Training")

    # Layout and save
    plt.tight_layout()
    plt.savefig(f"{folder}{identifier}_training_performance.png", dpi=300)
    plt.close(fig)

    print("Figure saved as performance_vs_noise_all_metrics.png")
    


if __name__ == "__main__":
    main()
