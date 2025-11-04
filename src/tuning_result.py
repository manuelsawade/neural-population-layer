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
    for p in sorted(folder_path.glob("*.json")):
        try:
            with open(p, "r") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"Skipping {p} (could not read): {e}")
            continue

        print(p)
        
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


def aggregate_specs(df: pd.DataFrame,
                    class_cols: list[str]) -> pd.DataFrame:
    group_cols = []

    for col in df.columns:
        for column in class_cols:
            if col.startswith(column):
                group_cols.append(col)

    numeric_cols = df.drop(group_cols, axis=1).select_dtypes([np.number]).columns.tolist()

    agg = df.groupby(group_cols, dropna=False, as_index=False)[numeric_cols].max()
    agg = agg.reset_index(drop=False)

    return agg, group_cols, numeric_cols

def get_max(df: pd.DataFrame,
            class_cols: list[str]) -> pd.DataFrame:
    group_cols = []

    for col in df.columns:
        for column in class_cols:
            if col.startswith(column):
                group_cols.append(col)

    numeric_cols = df.drop(group_cols, axis=1).select_dtypes([np.number]).columns.tolist()

    agg = df.groupby(group_cols, dropna=False, as_index=False)[numeric_cols].mean()
    agg = agg.reset_index(drop=True)

    return agg


def plot_per_class(agg: pd.DataFrame,
                   outdir: str = "./plots",
                   group_cols:list[str]=[],
                   metric_cols:list[str]=[],
                   class_cols:list[str]=[]):
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    group_cols.append('index')
    print("group")
    print(group_cols)
    print("agg")
    print(agg)

    for class_vals, df in agg.groupby(['hyp.training_noise'], dropna=False):
        print(df)
        max_rows_idx = []
        for metric_col in metric_cols:
            if metric_col[:3] in ["sha", "avg"]:
                idx = df[metric_col].idxmin()
            else:
                idx = df[metric_col].idxmax()
            if idx in max_rows_idx: continue

            max_rows_idx.append(idx)

        print(max_rows_idx)

        max = df[df.index.isin(max_rows_idx)]

        #metrics = max.drop(group_cols, axis=1)
        metric_groups: dict[str, list] = {}

        for key, col in [(metric_col[:3], metric_col) for metric_col in metric_cols]:
            if key not in metric_groups:
                metric_groups[key] = []

            metric_groups[key].append(col)

        fig, axs = plt.subplots(len(metric_groups), figsize=(12, 24))
        print(max)

        for metric_group, ax in zip(metric_groups, axs):
            metrics = max[metric_groups[metric_group]]
            sub_axs = metrics.plot(kind="bar", ax=ax, sharex=True)

            group_cols = []
            for idx, row in max.iterrows():
                print("row")
                label = "_".join(map(str, row[["network", "hyp.dataset", "hyp.hidden_dim", "hyp.neurons", "hyp.orientation", "hyp.stimulus"]].values))
                group_cols.append(label)


            sub_axs.set_xticklabels(group_cols)

        # print(max) 
        # print(metrics)

        # fig, ax = plt.subplots(figsize=(12, 6))
        # metrics.plot(kind="bar", ax=ax)

        ax.set_title(class_vals)
        ax.set_ylabel("Value")
        ax.set_xlabel("Metrics")
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        safe_class_name = "_".join([str(v) for v in max[['hyp.dataset', 'hyp.hidden_dim']].columns])
        out_path = f"{outdir}/{class_vals}_{safe_class_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {out_path}")
        plt.close(fig)



def main():
    folder = f"./tuning/"

    ignore = ["lr", "sigma", "weight_decay", "batch_size", "orientation", "neurons", "stimulus", "metric"]
    class_cols = ["stack", "noise", "noise_probability"]

    print("Loading JSON files...")
    df = load_json_files(folder, ignore)
    print(f"Loaded {len(df)} records.")

    #loss = df[["loss","test_loss"]]
    # print(loss)
    # global_mean = loss.mean(axis=None, skipna=True)
    # global_std = loss.std(axis=None, skipna=True)
    # print(global_mean)
    # print(global_std)
    df["loss"]=(df["loss"]-df["loss"].min())/(df["loss"].max()-df["loss"].min())
    df["test_loss"]=(df["test_loss"]-df["test_loss"].min())/(df["test_loss"].max()-df["test_loss"].min())

    #print(df)

    df["test_accuracy"]=df["test_accuracy"] / 100

    metrics = ["accuracy", "loss", "fsa_inf"]
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
        if metric == "accuracy":
            ax.set_ylim(0, 1)
            ax.set_ylabel("Train")
        elif metric == "fsa_inf":
            ax.set_ylim(0, 1)
            #ax.set_ylabel("FSA Inf")

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
        if metric == "test_accuracy":
            ax.set_ylim(0, 1)
            ax.set_ylabel("Test")
        elif metric == "test_fsa_inf_mean":
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
