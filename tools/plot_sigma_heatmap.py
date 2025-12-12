#!/usr/bin/env python

# tools/plot_sigma_heatmap.py
#
# Usage:
#   python -m tools.plot_sigma_heatmap \
#     --csv results/fqe_sigma_gated_bounds_sweep_antmaze.csv \
#     --env antmaze-umaze-v2 \
#     --method td3bc
#
# or change --method to iql_u_conf_weighted, etc.

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str,
                    default="results/fqe_sigma_gated_bounds_sweep_antmaze.csv")
    ap.add_argument("--env", type=str, default="antmaze-umaze-v2")
    ap.add_argument("--method", type=str, default="td3bc",
                    help="Method name as stored in the CSV (e.g., 'td3bc' or 'iql_u_conf_weighted').")
    ap.add_argument("--agg", type=str, default="mean",
                    choices=["mean", "median"],
                    help="How to aggregate over seeds.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Filter to env + method
    sub = df[(df["env"] == args.env) & (df["method"] == args.method)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for env={args.env}, method={args.method} in {args.csv}")

    # Ensure numeric
    sub["sigma_low"] = sub["sigma_low"].astype(float)
    sub["sigma_high"] = sub["sigma_high"].astype(float)
    sub["fqe_return"] = sub["fqe_return"].astype(float)

    # Aggregate across seeds
    if args.agg == "mean":
        grouped = sub.groupby(["sigma_low", "sigma_high"])["fqe_return"].mean().reset_index()
    else:
        grouped = sub.groupby(["sigma_low", "sigma_high"])["fqe_return"].median().reset_index()

    # Pivot into matrix form
    pivot = grouped.pivot(index="sigma_low", columns="sigma_high", values="fqe_return")

    sigma_lows = pivot.index.values
    sigma_highs = pivot.columns.values
    Z = pivot.values

    plt.figure(figsize=(6, 4))
    im = plt.imshow(Z, origin="lower", aspect="auto")

    plt.xticks(
        np.arange(len(sigma_highs)),
        [f"{v:.3f}" for v in sigma_highs],
        rotation=45
    )
    plt.yticks(
        np.arange(len(sigma_lows)),
        [f"{v:.3f}" for v in sigma_lows]
    )

    plt.xlabel(r"$\sigma_{\text{high}}$")
    plt.ylabel(r"$\sigma_{\text{low}}$")
    plt.title(f"{args.env} – {args.method} – FQE ({args.agg} over seeds)")
    cbar = plt.colorbar(im)
    cbar.set_label("FQE return")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()