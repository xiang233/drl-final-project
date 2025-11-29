#!/usr/bin/env python
"""
Collect FQE and OOD stats for all checkpoints and aggregate to CSV + Markdown.

Usage (from repo root):
  python -m tools.collect_results

This will:
  - Scan generated_data/*.pt
  - For each ckpt:
      * infer env/method/seed
      * load OOD npz and compute mean/p50/p90/p95
      * recompute FQE via scripts.estimate_return
  - Write:
      results/suite_results.csv
      results/suite_results_summary.csv
      results/suite_results_summary.md
"""

import argparse
import glob
import os
import os.path as osp
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ua.datasets import load_d4rl
from ua.utils import set_seed
from scripts.estimate_return import load_policy, fqe_evaluate


# -------------------- helpers to infer method/seed -------------------- #

def infer_method_from_name(stem: str) -> str:
    """
    Infer algorithm name from checkpoint filename stem.
    E.g. 'bc_hopper_medium_replay_v2_seed0' -> 'bc'
         'td3bc_u_walker2d_medium_v2_seed1' -> 'td3bc_u'
    """
    if stem.startswith("bc_"):
        return "bc"
    if stem.startswith("td3bc_u_"):
        return "td3bc_u"
    if stem.startswith("iql_u_"):
        return "iql_u"
    if stem.startswith("iql_"):
        return "iql"
    return "unknown"


def parse_seed_from_name(stem: str):
    """
    Parse ..._seedX from filename stem, returns int or None.
    """
    m = re.search(r"_seed(\d+)", stem)
    if m:
        return int(m.group(1))
    return None


def find_ood_npz(stem: str, env_name: str, pt_dir: str, ood_subdir: str | None = None):
    """
    Try to find OOD .npz for this checkpoint.
    Looks for: <stem>.ood_<env_name>.npz in:
      - pt_dir
      - pt_dir/ood_subdir (if given)
    Returns path or None.
    """
    base_name = f"{stem}.ood_{env_name}.npz"
    candidates = [osp.join(pt_dir, base_name)]
    if ood_subdir is not None:
        candidates.append(osp.join(pt_dir, ood_subdir, base_name))

    for p in candidates:
        if osp.exists(p):
            return p
    return None


def ood_stats_from_npz(path: str):
    """
    Load distances from .npz (expects key 'd') and compute mean / percentiles.
    Returns dict with keys: ood_mean, ood_p50, ood_p90, ood_p95.
    If anything goes wrong, returns NaNs.
    """
    if path is None or not osp.exists(path):
        return dict(
            ood_mean=np.nan,
            ood_p50=np.nan,
            ood_p90=np.nan,
            ood_p95=np.nan,
        )
    try:
        npz = np.load(path)
        d = npz["d"].astype(np.float32)
        return dict(
            ood_mean=float(d.mean()),
            ood_p50=float(np.percentile(d, 50)),
            ood_p90=float(np.percentile(d, 90)),
            ood_p95=float(np.percentile(d, 95)),
        )
    except Exception as e:
        print(f"[WARN] Failed to read OOD npz {path}: {e}")
        return dict(
            ood_mean=np.nan,
            ood_p50=np.nan,
            ood_p90=np.nan,
            ood_p95=np.nan,
        )


# -------------------- aggregation helpers (same idea as summarize_csv) -------------------- #

AGG_COLS = ["fqe", "ood_mean", "ood_p50", "ood_p90", "ood_p95"]


def agg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-(env, method, seed) rows into per-(env, method) stats.
    """
    # Drop exact duplicates so repeated runs don't double-count
    df = df.drop_duplicates(subset=["env", "seed", "method", "ckpt"])
    g = (
        df.groupby(["env", "method"], dropna=False)[AGG_COLS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Flatten MultiIndex columns e.g. ('fqe','mean') -> 'fqe_mean'
    g.columns = [
        c if isinstance(c, str) else "_".join([x for x in c if x])
        for c in g.columns.to_list()
    ]
    return g


def fmt_cell(m, s, n):
    """
    Pretty string like: '96.73 ± 3.60 (n=3)' or '285.61 ± nan (n=1)'
    """
    try:
        s_str = f"{s:.2f}" if pd.notna(s) else "nan"
        return f"{m:.2f} ± {s_str} (n={int(n)})"
    except Exception:
        return "—"


def to_markdown(out_df: pd.DataFrame) -> str:
    lines = [
        "| env | method | FQE | OOD mean | OOD p50 | OOD p90 | OOD p95 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in out_df.iterrows():
        FQE = fmt_cell(r["fqe_mean"], r["fqe_std"], r["fqe_count"])
        M = fmt_cell(r["ood_mean_mean"], r["ood_mean_std"], r["ood_mean_count"])
        P50 = fmt_cell(r["ood_p50_mean"], r["ood_p50_std"], r["ood_p50_count"])
        P90 = fmt_cell(r["ood_p90_mean"], r["ood_p90_std"], r["ood_p90_count"])
        P95 = fmt_cell(r["ood_p95_mean"], r["ood_p95_std"], r["ood_p95_count"])
        lines.append(
            f"| {r['env']} | {r['method']} | {FQE} | {M} | {P50} | {P90} | {P95} |"
        )
    return "\n".join(lines)


# -------------------- main collection logic -------------------- #

def collect_for_ckpt(
    ckpt_path: str,
    fqe_iters: int,
    pt_dir: str,
    ood_subdir: str | None = None,
):
    """
    For a single checkpoint:
      - load metadata (env_name, seed, algo/method)
      - compute FQE
      - load OOD npz stats
    Returns a dict row for the CSV.
    """
    stem = osp.splitext(osp.basename(ckpt_path))[0]

    # Load checkpoint metadata
    try:
        state = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"[ERROR] Failed to load {ckpt_path}: {e}")
        return None

    env_name = state.get("env_name", None)
    seed_ckpt = state.get("seed", None)
    algo = state.get("algo", None)

    if env_name is None:
        print(f"[WARN] {ckpt_path} missing 'env_name'; skipping.")
        return None

    # Method: prefer algo field, else infer from filename
    if algo is not None:
        method = algo
    else:
        method = infer_method_from_name(stem)

    # Seed: prefer ckpt seed; if missing, fallback to name
    seed_name = parse_seed_from_name(stem)
    if seed_ckpt is None:
        seed = seed_name
    else:
        seed = seed_ckpt

    if seed is None:
        print(f"[WARN] Could not infer seed for {ckpt_path}; skipping.")
        return None

    # --- FQE: recompute using estimate_return utilities ---
    set_seed(seed)
    _, data = load_d4rl(env_name, seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    s_dim = data["S"].shape[1]
    a_dim = data["A"].shape[1]

    pi = load_policy(ckpt_path, s_dim, a_dim, device)
    fqe = fqe_evaluate(pi, data, iters=fqe_iters, device=device)

    # --- OOD: read npz and compute stats ---
    ood_npz_path = find_ood_npz(stem, env_name, pt_dir, ood_subdir=ood_subdir)
    ood_stats = ood_stats_from_npz(ood_npz_path)

    row = {
        "env": env_name,
        "method": method,
        "seed": int(seed),
        "ckpt": ckpt_path,
        "fqe": float(fqe),
        **ood_stats,
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt_dir", type=str, default="generated_data",
                    help="Directory containing .pt checkpoints and OOD npz.")
    ap.add_argument("--ood_subdir", type=str, default=None,
                    help="Optional subdir under pt_dir where OOD .npz live (e.g. 'oods').")
    ap.add_argument("--results_dir", type=str, default="results",
                    help="Where to write suite_results.csv and summaries.")
    ap.add_argument("--fqe_iters", type=int, default=20000,
                    help="Number of FQE iterations per checkpoint.")
    args = ap.parse_args()

    pt_dir = args.pt_dir
    ood_subdir = args.ood_subdir
    results_dir = args.results_dir
    fqe_iters = args.fqe_iters

    ckpts = sorted(glob.glob(osp.join(pt_dir, "*.pt")))
    if not ckpts:
        print(f"[WARN] No .pt files found in {pt_dir}")
        return

    print(f"[INFO] Found {len(ckpts)} checkpoints in {pt_dir}")
    rows = []

    for ck in ckpts:
        print(f"[INFO] Processing {ck} ...")
        row = collect_for_ckpt(ck, fqe_iters=fqe_iters, pt_dir=pt_dir, ood_subdir=ood_subdir)
        if row is not None:
            rows.append(row)

    if not rows:
        print("[WARN] No valid rows collected; exiting.")
        return

    df = pd.DataFrame(rows)
    os.makedirs(results_dir, exist_ok=True)

    # 1) Write raw per-seed CSV
    base_csv = Path(results_dir) / "suite_results.csv"
    df.to_csv(base_csv, index=False)
    print(f"[INFO] Wrote per-seed results to {base_csv}")

    # 2) Aggregate and write summary CSV + Markdown
    out = agg(df)
    summary_csv = base_csv.with_name("suite_results_summary.csv")
    out.to_csv(summary_csv, index=False)
    print(f"[INFO] Wrote summary CSV to {summary_csv}")

    md = to_markdown(out)
    summary_md = base_csv.with_name("suite_results_summary.md")
    summary_md.write_text(md)
    print(f"[INFO] Wrote summary Markdown to {summary_md}")


if __name__ == "__main__":
    main()
