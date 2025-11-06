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

def flatten(d: Dict[str, Any], ignore: list[str], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Recursively flatten nested dicts. Convert lists -> tuples for hashability."""
    items: Dict[str, Any] = {}
    for k, v in d.items():
        if k in ignore: continue

        new_key = f"{parent_key}{sep}{k}" if parent_key else k[:3]
        if isinstance(v, dict):
            items.update(flatten(v, ignore, new_key, sep=sep))
        elif isinstance(v, list):
            items[new_key] = "_".join(map(str, v))
        else:
            items[new_key] = v
    return items


def load_json_files(folder: str, ignore: list[str]) -> pd.DataFrame:
    folder_path = Path(folder)
    records = []
    for p, csv in zip(sorted(folder_path.glob("*.json")), sorted(folder_path.glob("*.csv"))):
        try:
            with open(p, "r") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"Skipping {p} (could not read): {e}")
            continue
        
        df = pd.read_csv(csv)
        df = df.loc[df['epoch'] == 99].iloc[0]

        record: Dict[str, Any] = {}

        for name, col in df.items():
            if name == "epoch": continue

            record[f"train_{name}"] = col

        for k, v in raw.items():
            if isinstance(v, dict):
                record.update(flatten({k: v}, ignore))
            elif isinstance(v, list):
                record[k] = "_".join(map(str, v))
            else:
                record[k] = v

        #record["_source_file"] = str(p.name)
        records.append(record)

    if not records:
        raise RuntimeError(f"No JSON files loaded from {folder}")
    return pd.DataFrame.from_records(records)

def main():
    folder = "./experiments/cifar10_evaluation/"
    ignore = ["neurons", "orientation", "activation", "stimulus", "sigma", "sharpness_scores", "activation_scores", "dataset", "noise_sensitivity", "seed", "created_on", "fsd_2", "fsa_2", "fsd_inf", "hidden_dim", "test_noise", "batch_size", "learning_rate", "weight_decay", "epochs", "subset"]
    print("Loading JSON files...")
    df = load_json_files(folder, ignore)
    print(f"Loaded {len(df)} records.")
    print(df)

    df["accuracy"]=df["accuracy"] / 100

    metric_map = [["train_accuracy", "accuracy"], ["train_loss", "avg_loss"], ["train_fsa_inf_mean", "rub.fsa_inf.mean"]]

    # Prepare a figure with 3 subplots in a row
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    fig.supxlabel('Noise Level')
    print("train")
    

    # Loop through each metric and create a plot
    for ax, metrics in zip(axes, metric_map):
        # Group by stack and noise for each metric

        # Plot each stack as a line with error bars
        for stack_name, group in df.groupby("network"):
            for i, metric in enumerate(metrics):
                stage = "train" if i == 0 else "test"

                print(group)
                ax.plot(
                    group["hyp.training_noise"], 
                    group[metric], 
                    linewidth=2,
                    label=f"{stack_name} ({stage})"
                )

        # Axis settings
        ax.set_title(metrics[0].capitalize())

        if metrics[0] == "fsa_inf_mean":
            ax.set_ylim(0.35, 0.56)
            ax.set_title("FSA Inf")

        if metrics[0] == "fsa_inf_mean_smoothed":
            ax.set_ylim(0.35, 0.56)
            ax.set_title("FSA Inf Smoothed (Window = 20)")

        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xticks([0, 0.5, 1.0])
        #ax.set_xlabel("Noise Level")
        #ax.set_ylim(0, 1)

    # # Add legend to the first subplot only (to avoid clutter)
    axes[0].legend(title="Stack", loc="best")
    fig.suptitle("CIFAR10 Training")

    # Layout and save
    plt.tight_layout()
    plt.savefig(f"{folder}training_test_performance.png", dpi=300)
    plt.close(fig)

    print("Figure saved as performance_vs_noise_all_metrics.png")
    


if __name__ == "__main__":
    main()
