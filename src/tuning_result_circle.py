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

from library import get_display_name, get_evaluation_folder, get_evaluation_identifier, normalize_columns


def load_json_files(target_folder: str, compare_folder: str, ignore: list[str]) -> pd.DataFrame:
    records = []
    
    for folder in [compare_folder, target_folder]:  
        print(folder)
        folder_path = Path(folder)

        for p in sorted(folder_path.glob("*.json")):
            try:
                with open(p, "r") as f:
                    raw = json.load(f)
            except Exception as e:
                print(f"Skipping {p} (could not read): {e}")
                continue
            

            record: Dict[str, Any] = {}
            for k, v in raw.items():
                if k not in ignore: continue

                if isinstance(v, list):
                    record[k] = "_".join(map(str, v))
                else:
                    record[k] = v

            records.append(record)

    if not records:
        raise RuntimeError(f"No JSON files loaded from {folder}")
    return pd.DataFrame.from_records(records)

def main():
    linear_stack = "linear"
    population_stack = "population"
    #population_stack = "population_encoding"
    #population_stack = "population_circular"
    #population_stack = "preferred_value"
    #population_stack = "softmax_gaussian"
    dataset = "mnist"
    
    
    identifier = get_evaluation_identifier(dataset, population_stack)
    folder = get_evaluation_folder(identifier)

    linear_folder = get_evaluation_folder(get_evaluation_identifier(dataset, linear_stack))


    ignore = ["stack","noise", "accuracy", "test_accuracy", "loss", "test_loss", "fsa_inf_mean", "test_fsa_inf_mean"]

    print("Loading JSON files...")
    df = load_json_files(folder, linear_folder, ignore)
    print(f"Loaded {len(df)} records.")
    df = normalize_columns(df, ["loss", "test_loss"])


    df["test_accuracy"]=df["test_accuracy"] / 100

    stacks = [linear_stack, population_stack]
    metric_map = [["accuracy", "test_accuracy", "max"], ["loss", "test_loss", "min"], ["fsa_inf_mean", "test_fsa_inf_mean", "max"]]

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

    cmap = plt.get_cmap("RdYlGn")

    enum = iter("abcdefg")

    for row_axes, stacks in zip(axes, stacks):
        for ax, metrics in zip(row_axes, metric_map):
            ax.text(
                0.02,        # a little left of the axes
                0.88,               # same vertical height as the title
                f"{next(enum)})",
                fontsize=9, fontweight="bold",
                transform=ax.transAxes
            )


            subset = df.loc[df['stack'] == stacks]
            subset = subset.sort_values(by='noise', ascending=True)
            for i, split_df in enumerate(subset.groupby(metrics)):
                for i, metric in enumerate(metrics[:2]):                    
                    label = ""

                    stage = color_map[stacks][i]["stage"]
                    color = color_map[stacks][i]["color"]

                    if i == 0 and train_label is None:
                        train_label =f"{stage}"
                        label = train_label

                    if i == 1 and test_label is None:
                        test_label =f"{stage}"
                        label = test_label
                    #print(split_df[1][metric])
                    ax.plot(split_df[1]["noise"], 
                            split_df[1][metric], 
                            marker="o", 
                            linewidth=10, 
                            color=color,
                            label=label)      

            min = -0.5
            max = 0.5   

            for i, noise in enumerate([0.0, 0.5, 1.0]):
                train_val = subset[metrics[0]].values              
                test_val = subset[metrics[1]].values

                difference_label = ""

                if i == 2 and "accuracy" in metrics and stacks == linear_stack:
                    difference_label = "difference"
              
                if len(train_val) and len(test_val):
                    diff_abs = abs(train_val[i] - test_val[i])
                    y_mid = (train_val[i] + test_val[i]) / 2

                    size = diff_abs * 1000 * ((1 + diff_abs) ** 1.5)

                    if metrics[2] == "min":
                        diff = train_val[i] - test_val[i]
                    else:
                        diff = test_val[i] - train_val[i]

                    diff = (diff - min) / (max - min)
                    color=cmap(diff)

                    ax.scatter(noise, y_mid, s=size, color=color, alpha=0.6, zorder=5, label=difference_label)
                    difference_label = ""

            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_ylim(-0.1, 1.1)
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
    leg = fig.legend(
    loc="outside lower right", 
    prop={'size': 8}, 
    markerscale=0.4,
    ncol=2)
    
    leg.legend_handles[2].set_color('black')
    # Layout and save
    plt.tight_layout()
    plt.savefig(f"{folder}{identifier}_tuning_performance.png", dpi=300)
    plt.close(fig)
    


if __name__ == "__main__":
    main()
