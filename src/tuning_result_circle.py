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

from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from display_names import get_display_name


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
    population_stack = "population"
    #population_stack = "preferred_value"
    #population_stack = "softmax_gaussian"
    dataset = "mnist"
    
    
    identifier = f"{dataset}_evaluation_{population_stack}"
    #identifier = "mnist_evaluation"
    folder = f"./experiments/{identifier}/tuning/"

    linear_stack = "linear"

    ignore = ["initializer", "requires_grad", "noise_probability", "lr", "freq", "weight_decay", "batch_size", "phase", "amp", "distribution", "metric"]

    print("Loading JSON files...")
    df = load_json_files(folder, ignore)
    print(f"Loaded {len(df)} records.")

    df = df.loc[df['target_metric'] == 'fsa_inf_mean_diff']

    global_loss = df[['loss', 'test_loss']].values.flatten()
    global_min = global_loss.min()
    global_max = global_loss.max()


    df["loss"]=(df["loss"] - global_min)/(global_max - global_min)
    df["test_loss"]=(df["test_loss"] - global_min)/(global_max - global_min)

    df["test_accuracy"]=df["test_accuracy"] / 100

    stacks = [linear_stack, population_stack]
    metric_map = [["accuracy", "test_accuracy"], ["loss", "test_loss"], ["fsa_inf_mean", "test_fsa_inf_mean"]]

    fig, axes = plt.subplots(2, 3, figsize=(8, 4), sharex=True, sharey=True)
    fig.supxlabel('Noise Level', y=0.05)
    fig.suptitle(f"Tuning Evaluation of {get_display_name(population_stack)} Stack on {get_display_name(dataset)} Dataset", y=0.97)

    axes[0][0].set_ylabel(get_display_name(linear_stack))
    axes[1][0].set_ylabel(get_display_name(population_stack))

    color_map = {
        linear_stack: {
            1: {
                "stage":"test",
                "color":"purple"
            },
            0: {
                "stage":"train",
                "color":"orange"
            }
        },
        population_stack: {
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

    difference_label = "difference"

    for row_axes, stacks in zip(axes, stacks):
        for ax, metrics in zip(row_axes, metric_map):
            subset = df.loc[df['stack'] == stacks]
            subset = subset.sort_values(by='noise', ascending=True)
            for i, split_df in enumerate(subset.groupby(metrics)):
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
                  
                    ax.plot(split_df[1]["noise"], 
                            split_df[1][metric], 
                            marker="o", 
                            linewidth=10, 
                            color=color,
                            label=label)         

            for i, noise in enumerate([0.0, 0.5, 1.0]):
                train_val = subset[metrics[0]].values              
                test_val = subset[metrics[1]].values

                
                if len(train_val) and len(test_val):
                    diff = abs(train_val[i] - test_val[i])
                    y_mid = (train_val[i] + test_val[i]) / 2
                    print(diff, y_mid)

                    # Circle size scales with difference magnitude
                    base_size = 10
                    size = diff * 1000 * ((1 + diff) ** 1.5)
                    cmap = plt.cm.get_cmap("Greys")
                    #color = cmap(diff * 1.5) # normalized to [0,1]

                    ax.scatter(noise, y_mid, s=size, color="black", alpha=0.6, zorder=5, label=difference_label)
                    difference_label = ""

            ax.set_ylim(0.0, 1.0)
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_xlim(-0.2, 1.2)
            if stacks in linear_stack:            
                if metrics[0] == "accuracy":
                    ax.set_title("Accuracy")

                if metrics[0] == "loss":
                    ax.set_title("Loss (Normalized)")
                
                if metrics[0] == "fsa_inf_mean":
                    ax.set_title("FSA Inf")


            ax.grid(True, linestyle=":", alpha=0.6)


    # # Add legend to the first subplot only (to avoid clutter)
    fig.legend(loc="outside lower right", prop={'size': 9}, ncol=2)
    # Layout and save
    plt.tight_layout()
    plt.savefig(f"{folder}{identifier}_training_test_performance.png", dpi=300)
    plt.close(fig)
    


if __name__ == "__main__":
    main()
