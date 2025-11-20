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

from library import get_display_name, normalize_columns

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


def load_json_files(folder: str, linear_folder:str, ignore: list[str]) -> pd.DataFrame:
    folder_path = Path(folder)
    linear_path = Path(linear_folder)

    iter_folders = [
        zip(sorted(folder_path.glob("*.json")), sorted(folder_path.glob("*.csv"))),
        zip(sorted(linear_path.glob("*.json")), sorted(linear_path.glob("*.csv")))]

    records = []

        
    for iter_folder in iter_folders:
        for p, csv in iter_folder:
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
    population_stack = "population"
    #population_stack = "preferred_value"
    #population_stack = "softmax_gaussian"
    dataset = "cifar10"
    
    
    identifier = f"{dataset}_evaluation_{population_stack}"
    #identifier = "mnist_evaluation"
    folder = f"./experiments/{identifier}/"

    linear_stack = "linear"
    linear_folder = f"./experiments/cifar10_evaluation_linear/"

    ignore = ["neurons", "orientation", "activation", "stimulus", "sigma", "sharpness_scores", "activation_scores", "dataset", "noise_sensitivity", "seed", "created_on", "fsd_2", "fsa_2", "fsd_inf", "hidden_dim", "test_noise", "batch_size", "learning_rate", "weight_decay", "epochs", "subset"]
    print("Loading JSON files...")
    df = load_json_files(folder, linear_folder, ignore)
    print(f"Loaded {len(df)} records.")

    df["accuracy"]=df["accuracy"] / 100

    stacks = ["linear", "population"]
    metric_map = [["train_accuracy", "accuracy"], ["train_loss", "avg_loss"], ["train_fsa_inf_mean", "rub.fsa_inf.mean"]]
    target_map = ["max", "center", "max"]
    fig, axes = plt.subplots(2, 3, figsize=(8, 4), sharex=True, sharey=True)
    fig.supxlabel('Noise Level', y=0.05)

    df = normalize_columns(df, ["train_loss", "avg_loss"])

    axes[0][0].set_ylabel(get_display_name(linear_stack))
    axes[1][0].set_ylabel(get_display_name(population_stack))

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

    cmap = plt.get_cmap("RdYlGn")
    enum = iter("abcdefghijklmnopqrstuvwxyz")

    for row_axes, stacks in zip(axes, stacks):
        for ax, metrics, target in zip(row_axes, metric_map, target_map):
            ax.text(
                0.02,        # a little left of the axes
                0.88,               # same vertical height as the title
                f"{next(enum)})",
                fontsize=11, fontweight="bold",
                transform=ax.transAxes
            )

            subset = df.loc[df['network'] == stacks]
            subset = subset.sort_values(by='hyp.training_noise', ascending=True)


            #subset = normalize_columns(subset, ["train_loss", "avg_loss"])

            for x, (col, split_df) in enumerate(subset.groupby("hyp.training_noise")):
                #print(split_df[1])
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

                    #grouped_metric = split_df[1].groupby("")
                    #print(grouped_metric)
                  
                    ax.plot(col, 
                            split_df[metric].mean(numeric_only=True), 
                            marker="o", 
                            linewidth=0, 
                            color=color,
                            label=label)   

                min = -0.5
                max = 0.5      

                difference_label = ""
                print(x, metrics, stacks)
                if x == 1 and "train_loss" in metrics and stacks == linear_stack:
                    print(difference_label)
                    difference_label = "difference"

                #for i, noise in enumerate([0.0, 0.5, 1.0]):
                #print(split_df)
                train_val = split_df[metrics[0]].mean()              
                test_val = split_df[metrics[1]].mean()
                #print(train_val, test_val)
                

                diff_abs = abs(train_val - test_val)
                y_mid = (train_val + test_val) / 2

                size = diff_abs * 1000 * ((1 + diff_abs) ** 1.5)

                if target == "center":
                    diff = (diff_abs - 0) / (max - 0)
                    if test_val > train_val:
                        diff = diff * -1
                elif target == "min":
                    diff = train_val - test_val
                    diff = (diff - min) / (max - min)
                else:
                    diff = test_val - train_val
                    diff = (diff - min) / (max - min)

                print(stacks,metrics[0], diff)
                color=cmap(diff)

                ax.scatter(col, y_mid, s=size, color=color, alpha=0.6, zorder=5, label=difference_label)
                difference_label = ""

            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_xlim(-0.2, 1.2)
            print(stacks)
            if stacks in linear_stack:            
                if metrics[0] == "train_accuracy":
                    ax.set_title("Accuracy")

                if metrics[0] == "train_loss":
                    ax.set_title("Loss (Normalized)")
                
                if metrics[0] == "train_fsa_inf_mean":
                    ax.set_title("FSA Inf")


            ax.grid(True, linestyle=":", alpha=0.6)

    # # Add legend to the first subplot only (to avoid clutter)
    leg = fig.legend(
        loc="outside lower right", 
        prop={'size': 8}, 
        markerscale=0.7,
        ncol=2)
    
    leg.legend_handles[2].set_color('black')
    #fig.suptitle("CIFAR10 Train-Test-Difference")

    # Layout and save
    plt.tight_layout()
    plt.savefig(f"{folder}{identifier}_training_test_performance.png", dpi=300)
    plt.close(fig)

    print("Figure saved as performance_vs_noise_all_metrics.png")
    


if __name__ == "__main__":
    main()
