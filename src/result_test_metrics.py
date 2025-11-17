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

from library import get_display_name

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
    for p in sorted(folder_path.glob("*.json")):
        try:
            with open(p, "r") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"Skipping {p} (could not read): {e}")
            continue

        record: Dict[str, Any] = {}

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

def normalize(df, columns: list[str]):
    print(columns) 
    global_val = df[columns].values.flatten()
    global_min = global_val.min()
    global_max = global_val.max()

    for col in columns:
        df[col]=(df[col] - global_min)/(global_max - global_min)

    return df

def main():
    population_stack = "population"
    #population_stack = "preferred_value"
    #population_stack = "softmax_gaussian"
    dataset = "mnist"
    
    
    identifier = f"{dataset}_evaluation_{population_stack}_1"
    #identifier = "mnist_evaluation"
    folder = f"./experiments/{identifier}/"

    linear_stack = "linear"

    ignore = ["neurons", "orientation", "activation", "stimulus", "sigma", "activation_scores", "dataset", "seed", "created_on", "hidden_dim", "test_noise", "batch_size", "learning_rate", "weight_decay", "epochs", "subset"]
    print("Loading JSON files...")
    df = load_json_files(folder, ignore)
    print(f"Loaded {len(df)} records.")

    df["accuracy"]=df["accuracy"] / 100
    df["avg_loss"]=(df["avg_loss"] - df["avg_loss"].min())/(df["avg_loss"].max() - df["avg_loss"].min())
    df = normalize(df, ["sha.layers.0.weight", "sha.layers.2.weight"])
    df = normalize(df, ["sha.layers.0.bias", "sha.layers.2.bias"])
    df = normalize(df, ["noi.fgsm.mean"])
    df = normalize(df, ["noi.fgsm.std"])

    stacks_list = ["linear", "population"]
    metric_map = [
        ["accuracy", "avg_loss", "noi.fgsm.mean", "noi.fgsm.std"], 
        ["rub.fsa_inf.mean", "rub.fsa_2.mean", "rub.fsd_inf.mean", "rub.fsd_2.mean"],
        ["rub.fsa_inf.std", "rub.fsa_2.std", "rub.fsd_inf.std", "rub.fsd_2.std"],
        ["sha.layers.0.weight",  "sha.layers.0.bias",  "sha.layers.2.weight", "sha.layers.2.bias"]
    ]

    target_map = [
        ["max", "min", "min", "min"],
        ["max", "max", "max", "max"],
        ["min", "min", "min", "min"],
        ["min", "min", "min", "min"],
    ]

    fig, axes = plt.subplots(len(metric_map), len(metric_map[0]), figsize=(11, 7.5), sharex=True, sharey=True)
    fig.supxlabel('Noise Level')

    color_map = {
        "linear": {

                "stage":"train",
                "color":"orange"
            
        },
        "population": {
                "stage":"test",
                "color":"purple"
            },
        
    }

    label: str | None = None

    cmap = plt.get_cmap("RdYlGn")
    enum = iter("abcdefghijklmnopqrstuvwxyz")

    for ax_row, metrics_row, target_row in zip(axes, metric_map, target_map):
        for ax, metrics, target in zip(ax_row, metrics_row, target_row):
            ax.text(
                0.02,        # a little left of the axes
                0.88,               # same vertical height as the title
                f"{next(enum)})",
                fontsize=11, fontweight="bold",
                transform=ax.transAxes
            )

            print(ax, metric_map)
            ax.set_title(get_display_name(metrics), fontsize=11)

            df = df.sort_values(by='hyp.training_noise', ascending=True)

            for network in stacks_list:     
                if metrics == "accuracy":                
                    label = get_display_name(network)
                color = color_map[network]["color"]

                subset = df.loc[df['network'] == network]
                print(network)
                ax.plot(subset["hyp.training_noise"], 
                        subset[metrics], 
                        marker="o",
                        linewidth=0,
                        color=color,
                        label=label)  

                label = ""       

            min = -0.5
            max = 0.5

            for i, noise in enumerate([0.0, 0.5, 1.0]):
                train_val = df.loc[df['network'] == "linear"][metrics].values              
                test_val = df.loc[df['network'] == "population"][metrics].values
                print(df)
                print(train_val, test_val)

                difference_label = ""

                if i == 0 and "rub.fsa_inf.std" in metrics:
                    difference_label = "difference"

                
                if len(train_val) and len(test_val):
                    diff_abs = abs(train_val[i] - test_val[i])
                    y_mid = (train_val[i] + test_val[i]) / 2

                    size = diff_abs * 1000 * ((1 + diff_abs) ** 1.5)

                    if target == "min":
                        diff = train_val[i] - test_val[i]
                    else:
                        diff = test_val[i] - train_val[i]

                    diff = (diff - min) / (max - min)
                    color=cmap(diff)

                    ax.scatter(noise, y_mid, s=size, color=color, alpha=0.6, zorder=5, label=difference_label)
                    difference_label = ""

            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_xlim(-0.2, 1.2)
            
            if metrics == "accuracy":
                ax.set_title("Accuracy")

            if metrics == "loss":
                ax.set_title("Loss (Normalized)")


            ax.grid(True, linestyle=":", alpha=0.6)

    # # Add legend to the first subplot only (to avoid clutter)
    leg = fig.legend(loc="outside lower right", prop={'size': 10}, ncol=2)
    fig.suptitle("")

    leg.legend_handles[2].set_color('black')

    # Layout and save
    plt.tight_layout()
    plt.savefig(f"{folder}{identifier}_test_metrics.png", dpi=600)
    plt.close(fig)

    print("Figure saved as performance_vs_noise_all_metrics.png")
    


if __name__ == "__main__":
    main()
