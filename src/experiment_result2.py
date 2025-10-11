#!/usr/bin/env python3
"""
plot_experiment_results.py

Usage:
    python plot_experiment_results.py /path/to/json/folder --metrics accuracy avg_loss --outdir ./plots

By default it plots 'accuracy' vs sigma for each network class. If you provide multiple metrics
it will create one subplot per metric within each network-class figure.
"""

import json
from pathlib import Path
from typing import Any, Dict
import argparse
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def flatten(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Recursively flatten a nested dict (dot-separated keys). Convert lists -> tuples for hashability."""
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten(v, new_key, sep=sep))
        elif isinstance(v, list):
            # convert list to tuple (hashable) for keys like 'orientation'
            # but if list contains dicts, stringify them (rare in this dataset)
            if all(not isinstance(e, dict) for e in v):
                items[new_key] = tuple(v)
            else:
                items[new_key] = tuple(json.dumps(e, sort_keys=True) if isinstance(e, dict) else e for e in v)
        else:
            items[new_key] = v
    return items

def load_json_files(folder: str) -> pd.DataFrame:
    folder_path = Path(folder)
    records = []
    for p in sorted(folder_path.glob("*.json")):
        try:
            with open(p, "r") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"Skipping {p} (could not read): {e}")
            continue

        # Bring hyper_parameter keys to top-level (no prefix) and flatten other nested dicts
        record: Dict[str, Any] = {}
        # copy top-level keys except hyper_parameter
        for k, v in raw.items():
            if k == "hyper_parameter":
                # flatten hyper_parameter without prefix
                for hk, hv in v.items():
                    if isinstance(hv, dict):
                        # flatten further with hk as prefix
                        nested = flatten({hk: hv})
                        # nested keys start with hk. ...
                        # we want top-level names where possible; keep as nested keys
                        record.update(nested)
                    elif isinstance(hv, list):
                        record[hk] = tuple(hv)
                    else:
                        record[hk] = hv
            else:
                if isinstance(v, dict):
                    nested = flatten({k: v})
                    record.update(nested)
                elif isinstance(v, list):
                    record[k] = tuple(v)
                else:
                    record[k] = v

        # Ensure orientation (and any other lists) are hashable tuples already
        for kk, vv in list(record.items()):
            if isinstance(vv, list):
                record[kk] = tuple(vv)

        record["_source_file"] = str(p.name)
        records.append(record)

    if not records:
        raise RuntimeError(f"No JSON files loaded from {folder}")

    df = pd.DataFrame.from_records(records)
    return df

def aggregate_specs(df: pd.DataFrame,
                    class_cols = ["network", "hidden_dim", "training_noise"],
                    spec_cols = ["sigma", "neurons", "orientation", "activation", "stimulus"]) -> pd.DataFrame:
    # Ensure spec and class columns exist (fill missing with NaN)
    for c in class_cols + spec_cols:
        if c not in df.columns:
            df[c] = pd.NA

    # Convert orientation tuples to a stable, hashable representation (they already are tuples from loader).
    # Also create a human-readable string for legends
    if "orientation" in df.columns:
        df["_orientation_str"] = df["orientation"].apply(lambda x: str(tuple(x)) if (not pd.isna(x) and isinstance(x, (list,tuple))) else str(x))

    # Select numeric columns to average
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # If your numeric metrics are stored nested strings like 'ruby_scores.fsa_2.mean' they will be numeric and included.

    group_cols = class_cols + spec_cols
    # Do the aggregation (mean of numeric columns)
    agg = df.groupby(group_cols, dropna=False, as_index=False)[numeric_cols].mean()
    # Bring back a readable orientation string if available
    if "_orientation_str" in df.columns:
        # Merge the readable orientation string back (take first non-null)
        orient_lookup = df.groupby(group_cols, dropna=False, as_index=False)["_orientation_str"].agg(lambda s: next((x for x in s if pd.notna(x)), None))
        # join
        agg = agg.merge(orient_lookup, on=group_cols, how="left")
    return agg

def plot_per_class(agg: pd.DataFrame,
                   metric: str = "accuracy",
                   outdir: str = "./plots",
                   class_cols = ["network", "hidden_dim", "training_noise"],
                   spec_cols_for_lines = ["neurons", "orientation", "activation", "stimulus"]):
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    required = class_cols + ["sigma"] + spec_cols_for_lines
    missing = [c for c in required if c not in agg.columns]
    if missing:
        raise RuntimeError(f"Missing required columns for plotting: {missing}")

    # For each network class, plot
    for class_vals, class_df in agg.groupby(class_cols, dropna=False):
        # class_vals is a tuple
        network_label = ", ".join(f"{col}={val}" for col, val in zip(class_cols, class_vals))
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted_any = False

        # group by all spec fields except sigma, so we can draw a line across sigma values
        line_group_cols = spec_cols_for_lines
        # If orientation is tuple, it is preserved in grouping keys; make legend-friendly label later
        for spec_vals, spec_df in class_df.groupby(line_group_cols, dropna=False):
            # spec_vals is tuple of (neurons, orientation, activation, stimulus)
            # sort by sigma to have meaningful line
            spec_df = spec_df.sort_values("sigma")
            if metric not in spec_df.columns:
                continue
            x = spec_df["sigma"].to_numpy(dtype=float)
            y = spec_df[metric].to_numpy(dtype=float)

            # Skip all-NaN y
            if np.all(np.isnan(y)):
                continue

            neurons, orientation, activation, stimulus = spec_vals
            # readable orientation
            orient_str = str(tuple(orientation)) if isinstance(orientation, (list, tuple)) else str(orientation)

            label = f"n={neurons}, orient={orient_str}, act={activation}, stim={stimulus}"
            # If only one point available, plot marker; else plot line
            if len(x) >= 2:
                ax.plot(x, y, marker="o", label=label)
            else:
                ax.scatter(x, y, marker="o", label=label)
            plotted_any = True

        if not plotted_any:
            print(f"No usable data to plot for class {network_label}; skipping.")
            plt.close(fig)
            continue

        ax.set_title(f"{metric} vs sigma — {network_label}")
        ax.set_xlabel("sigma")
        ax.set_ylabel(metric)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize="small", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # safe filename
        safe_class_name = "_".join([str(v) for v in class_vals])
        out_path = outdir_p / f"{safe_class_name}__{metric}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {out_path}")
        plt.close(fig)

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("folder", help="Folder containing JSON experiment result files")
    # parser.add_argument("--metrics", nargs="+", default=["accuracy"], help="Metric(s) to plot (default: accuracy)")
    # parser.add_argument("--outdir", default="./plots", help="Output folder for plots")
    # args = parser.parse_args()
    args = object()
    args.folder = "./experiments/mnist_2025_09_10_17_53_03"
    args.metrics = ["accuracy", "avg_loss", sharpness_scores]

    print("Loading JSON files...")
    df = load_json_files(args.folder)
    print(f"Loaded {len(df)} records; columns: {list(df.columns)}")

    print("Aggregating specs...")
    agg = aggregate_specs(df)
    print(f"Aggregated dataframe has {len(agg)} rows (spec groups). Columns: {list(agg.columns)}")


    plot_per_class(agg, outdir=args.outdir)

if __name__ == "__main__":
    main()
