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
    identifier = "cifar10_evaluation"
    folder = f"./experiments/{identifier}/"

    ignore = ["neurons", "orientation", "activation", "stimulus", "sigma", "sharpness_scores", "activation_scores", "dataset", "noise_sensitivity", "seed", "created_on", "fsd_2", "fsa_2", "fsd_inf", "hidden_dim", "test_noise", "batch_size", "learning_rate", "weight_decay", "epochs", "subset"]
    print("Loading JSON files...")
    df = load_json_files(folder, ignore)
    print(f"Loaded {len(df)} records.")

    df["accuracy"]=df["accuracy"] / 100

    global_loss = df[['train_loss', 'avg_loss']].values.flatten()
    global_min = global_loss.min()
    global_max = global_loss.max()

    df["train_loss"]=(df["train_loss"] - global_min)/(global_max - global_min)
    df["avg_loss"]=(df["avg_loss"] - global_min)/(global_max - global_min)

    stacks = ["linear", "population"]
    metric_map = [["train_accuracy", "accuracy"], ["train_loss", "avg_loss"], ["train_fsa_inf_mean", "rub.fsa_inf.mean"]]

    fig, axes = plt.subplots(2, 3, figsize=(10, 4), sharex=True)
    fig.supxlabel('Noise Level')

    color_map = {
        "linear": {
            1: {
                "stage":"test",
                "color":"purple"
            },
            0: {
                "stage":"train",
                "color":"orange"
            }
        },
        "population": {
            1: {
                "stage":"test",
                "color":"purple"
            },
            0: {
                "stage":"train",
                "color":"orange"
            }
        }
    }
    

    train_label: str | None = None
    test_label: str | None = None

    for row_axes, stacks in zip(axes, stacks):
        for ax, metrics in zip(row_axes, metric_map):
            subset = df.loc[df['network'] == stacks]
            subset = subset.sort_values(by='hyp.training_noise', ascending=True)
            for i, split_df in enumerate(subset.groupby(metrics)):

                for i, metric in enumerate(metrics):                    
                    label = ""

                    stage = color_map[stacks][i]["stage"]
                    color = color_map[stacks][i]["color"]

                    if i == 0 and train_label is None:
                        train_label =f"{stage}"
                        label = train_label

                    if i == 1 and test_label is None:
                        test_label =f"{stage}"
                        label = test_label
                  
                    ax.plot(split_df[1]["hyp.training_noise"], 
                            split_df[1][metric], 
                            marker="o", 
                            linewidth=10, 
                            color=color,
                            label=label)          

            for i, noise in enumerate([0.0, 0.5, 1.0]):
                train_val = subset[metrics[0]].values              
                test_val = subset[metrics[1]].values
                
                if len(train_val) and len(test_val):
                    print(f"line for {stacks} {metrics}", subset)
                    ax.plot([noise, noise], [train_val[i], test_val[i]],
                            color='black', linestyle=':', alpha=0.8, linewidth=3)

            ax.grid(True, linestyle=":", alpha=0.6)

            if stacks in "linear":            
                if metrics[0] == "train_accuracy":
                    ax.set_title("Accuracy")
                    ax.set_ylim(0.0, 1.0)

                if metrics[0] == "train_loss":
                    ax.set_ylim(0.0, 1.0)
                    ax.set_title("Loss (Normalized)")
                
                if metrics[0] == "train_fsa_inf_mean":
                    ax.set_ylim(0.35, 0.56)
                    ax.set_title("FSA Inf")
            else:
                ax.set_xticks([0, 0.5, 1.0])
                #ax.set_ylim(0, 1)
                
                if metrics[0] == "train_accuracy":
                    ax.set_ylim(0.0, 1.0)

                if metrics[0] == "train_loss":
                    ax.set_ylim(0.0, 1.0)
                
                if metrics[0] == "train_fsa_inf_mean":
                    ax.set_ylim(0.35, 0.56)


            ax.grid(True, linestyle=":", alpha=0.6)

    # # Add legend to the first subplot only (to avoid clutter)
    axes[0][0].set_ylabel("Linear Network")
    axes[1][0].set_ylabel("Population Network")
    fig.legend(loc="outside upper left", prop={'size': 10})
    fig.suptitle("CIFAR10 Train-Test-Difference")

    # Layout and save
    plt.tight_layout()
    plt.savefig(f"{folder}{identifier}_training_test_performance.png", dpi=300)
    plt.close(fig)

    print("Figure saved as performance_vs_noise_all_metrics.png")
    


if __name__ == "__main__":
    main()
