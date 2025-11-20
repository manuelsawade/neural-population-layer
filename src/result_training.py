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


def load_csv_files(folder: str, linear_folder: str) -> pd.DataFrame:
    folder_path = Path(folder)
    linear_path = Path(linear_folder)

    iter_folders = [
        zip(sorted(folder_path.glob("*.json")), sorted(folder_path.glob("*.csv"))),
        zip(sorted(linear_path.glob("*.json")), sorted(linear_path.glob("*.csv")))
    ]

    dataframes = []
       
    for iter_folder in iter_folders:
        for p, csv in iter_folder:
            try:
                with open(p, "r") as f:
                    raw = json.loads(f.read())
            except Exception as e:
                print(f"Skipping {p} (could not read): {e}")
                continue
            print(raw["network"])
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
    dataset = "cifar10"
    identifier = f"{dataset}_evaluation_population"
    folder = f"./experiments/{identifier}/"
    linear_folder = f"./experiments/{dataset}_evaluation_linear/"
    print("Loading JSON files...")
    df = load_csv_files(folder, linear_folder)
    print(f"Loaded {len(df)} records.")

    metrics = ["accuracy", "loss", "fsa_inf_mean"]

    # Prepare a figure with 3 subplots in a row
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharex=True)
    
    df["fsa_inf_mean_smoothed"] = (df
        .groupby(["stack", "noise"])["fsa_inf_mean"]
        .transform(lambda x: x
                   .rolling(window=10, center=True, min_periods=1)
                   .mean())
    )

    enum = iter("abcdefghijklmnopqrstuvwxyz")

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
        ax.text(
            0.02,        # a little left of the axes
            0.95,               # same vertical height as the title
            f"{next(enum)})",
            fontsize=11, fontweight="bold",
            transform=ax.transAxes
        )

        label: str = ""

        # Plot each stack as a line with error bars
        for stack_name, group in df.groupby("stack"):
            for noise_level, group in group.groupby("noise"):
                if metric == "accuracy":
                    label = f"{stack_name} (noise={noise_level})"

                grouped_metric = group.groupby("epoch")
                color = color_map[stack_name][str(noise_level)]
                ax.plot(
                    grouped_metric.groups.keys(), 
                    grouped_metric.mean(numeric_only=True)[metric], 
                    linewidth=2,
                    color=color,
                    label=label
                )

        # Axis settings
        ax.set_title(metric.capitalize())
        if metric == "accuracy":
            ax.set_ylabel("Metric Value")

        if metric == "fsa_inf_mean":
            #ax.set_ylim(0.35, 0.56)
            ax.set_title("FSA Inf")

        if metric == "fsa_inf_mean_smoothed":
            #ax.set_ylim(0.35, 0.56)
            ax.set_title("FSA Inf Smoothed (Window = 20)")

        ax.grid(True, linestyle=":", alpha=0.6)
        # ax.set_ylim(-0.1, 1.1)
        # ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])


    # # Add legend to the first subplot only (to avoid clutter)
    axes[0].legend(loc="best", prop={'size': 10}, ncol=1)
    fig.supxlabel('Epoch', y=0.05)

    # Layout and save
    plt.tight_layout()
    plt.savefig(f"{folder}{identifier}_training_performance.png", dpi=300)
    plt.close(fig)

    print("Figure saved as performance_vs_noise_all_metrics.png")
    


if __name__ == "__main__":
    main()
