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


def load_json_files(folder: str, ignore: list[str]) -> pd.DataFrame:
    folder_path = Path(folder)
    records = []
    for p in sorted(folder_path.glob("*.json")):
        try:
            with open(p, "r") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"Skipping {p} (could not read): {e}")
            continue
        
        for r in raw:
            record: Dict[str, Any] = {}
            for k, v in r.items():
                if k in ignore: continue

                if isinstance(v, list):
                    record[k] = "_".join(map(str, v))
                else:
                    record[k] = v

            records.append(record)

    if not records:
        raise RuntimeError(f"No JSON files loaded from {folder}")
    return pd.DataFrame.from_records(records)

def main():
    folder = f"./tuning/"

    ignore = ["noise_probability", "lr", "sigma", "weight_decay", "batch_size", "orientation", "neurons", "stimulus", "metric"]

    print("Loading JSON files...")
    df = load_json_files(folder, ignore)
    print(f"Loaded {len(df)} records.")

    df = df.loc[df['target_metric'] == 'fsa_inf_std']

    global_loss = df[['loss', 'test_loss']].values.flatten()
    print(global_loss)
    global_min = global_loss.min()
    global_max = global_loss.max()

    #global_mean = global_loss.mean()
    #global_std = global_loss.std()

    df["loss"]=(df["loss"] - global_min)/(global_max - global_min)
    df["test_loss"]=(df["test_loss"] - global_min)/(global_max - global_min)

    #df["loss"]=(df["loss"] - global_mean)/global_std
    #df["test_loss"]=(df["test_loss"] - global_mean)/global_std

    print(df)

    df["test_accuracy"]=df["test_accuracy"] / 100

    metrics = ["accuracy", "loss", "fsa_inf_mean"]
    test_metrics = ["test_accuracy", "test_loss", "test_fsa_inf_mean"]

    # Prepare a figure with 3 subplots in a row
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    print("train")

    # Loop through each metric and create a plot
    for ax, metric in zip(axes[0], metrics):
        # Group by stack and noise for each metric
        summary = (
            df.groupby(["stack", "noise"])[metric]
            .agg(["mean", "std"])
            .reset_index()
        )

        # Plot each stack as a line with error bars
        for stack_name, group in summary.groupby("stack"):
            ax.errorbar(
                group["noise"],
                group["mean"],
                yerr=group["std"],
                fmt="o-",
                capsize=5,
                linewidth=2,
                markersize=6,
                label=f"Stack {stack_name}"
            )

        # Axis settings
        ax.set_title(metric.capitalize())
        ax.set_ylim(0, 1)
        if metric == "accuracy":
            ax.set_ylabel("Train")

        ax.set_xticks([0, 0.5, 1.0])
        ax.grid(True, linestyle=":", alpha=0.6)

    print("test")

    for ax, metric in zip(axes[1], test_metrics):
        summary = (
            df.groupby(["stack", "noise"])[metric]
            .agg(["mean", "std"])
            .reset_index()
        )


        # Plot each stack as a line with error bars
        for stack_name, group in summary.groupby("stack"):
            ax.errorbar(
                group["noise"],
                group["mean"],
                yerr=group["std"],
                fmt="o-",
                capsize=5,
                linewidth=2,
                markersize=6,
                label=f"Stack {stack_name}"
            )

        # Axis settings
        ax.set_xlabel("Noise Level")
        ax.set_ylim(0, 1)

        if metric == "test_accuracy":
            ax.set_ylim(0, 1)

        ax.set_xticks([0, 0.5, 1.0])
        ax.grid(True, linestyle=":", alpha=0.6)

    # Add legend to the first subplot only (to avoid clutter)
    axes[0][0].legend(title="Stack", loc="best")

    # Layout and save
    plt.tight_layout()
    plt.savefig(f"{folder}performance_vs_noise_all_metrics.png", dpi=300)
    plt.close(fig)

    print("Figure saved as performance_vs_noise_all_metrics.png")
    


if __name__ == "__main__":
    main()
