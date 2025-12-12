
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


# helpers 

def infer_method_from_name(stem: str) -> str:
    # BC
    if stem.startswith("bc_"):
        return "bc"

    # IQL family
    if stem.startswith("iql_u_mcdo_"):
        return "iql_u_mcdo"
    if stem.startswith("iql_u_"):
        return "iql_u"
    if stem.startswith("iql_"):
        return "iql"

    # TD3BC family
    if stem.startswith("td3bc_u_mcdo_"):
        return "td3bc_u_mcdo"

    if stem.startswith("td3bc_u_"):
        return "td3bc_u"

    if stem.startswith("td3bc_"):
        return "td3bc"

    return "unknown"


def parse_seed_from_name(stem: str):
    m = re.search(r"_seed(\d+)", stem)
    if m:
        return int(m.group(1))
    return None


def find_ood_npz(stem: str, env_name: str, pt_dir: str, ood_subdir: str | None = None):
    base_name = f"{stem}.ood_{env_name}.npz"
    candidates = [osp.join(pt_dir, base_name)]
    if ood_subdir is not None:
        candidates.append(osp.join(pt_dir, ood_subdir, base_name))

    for p in candidates:
        if osp.exists(p):
            return p
    return None


def ood_stats_from_npz(path: str):
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



AGG_COLS = ["fqe", "ood_mean", "ood_p50", "ood_p90", "ood_p95"]


def agg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["env", "seed", "method", "ckpt"])
    g = (
        df.groupby(["env", "method"], dropna=False)[AGG_COLS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    g.columns = [
        c if isinstance(c, str) else "_".join([x for x in c if x])
        for c in g.columns.to_list()
    ]
    return g


def fmt_cell(m, s, n):
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



BASELINE_METHODS = {
    "bc",
    "iql",
    "iql_u",
    "iql_u_mcdo",
    "td3bc",
    "td3bc_u",
    "td3bc_u_mcdo",
}

U_FAMILY = {"iql_u", "iql_u_mcdo", "td3bc_u", "td3bc_u_mcdo"}


def collect_for_ckpt(
    ckpt_path: str,
    fqe_iters: int,
    pt_dir: str,
    ood_subdir: str | None = None,
):
    stem = osp.splitext(osp.basename(ckpt_path))[0]

    try:
        state = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"[ERROR] Failed to load {ckpt_path}: {e}")
        return None

    env_name = state.get("env_name", None)
    seed_ckpt = state.get("seed", None)
    algo = state.get("algo", None)
    cfg = state.get("cfg", {})

    if env_name is None:
        print(f"[WARN] {ckpt_path} missing 'env_name'; skipping.")
        return None

    method = infer_method_from_name(stem)
    if method == "unknown" and algo is not None:
        method = algo

    seed_name = parse_seed_from_name(stem)
    if seed_ckpt is None:
        seed = seed_name
    else:
        seed = seed_ckpt

    if seed is None:
        print(f"[WARN] Could not infer seed for {ckpt_path}; skipping.")
        return None

    has_alpha_in_name = ("_alpha" in stem)
    unc_alpha = None
    if isinstance(cfg, dict) and "unc_alpha" in cfg:
        try:
            unc_alpha = float(cfg["unc_alpha"])
        except Exception:
            unc_alpha = None

    set_seed(seed)
    _, data = load_d4rl(env_name, seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    s_dim = data["S"].shape[1]
    a_dim = data["A"].shape[1]

    pi = load_policy(ckpt_path, s_dim, a_dim, device)
    fqe = fqe_evaluate(pi, data, iters=fqe_iters, device=device)

    ood_npz_path = find_ood_npz(stem, env_name, pt_dir, ood_subdir=ood_subdir)
    ood_stats = ood_stats_from_npz(ood_npz_path)

    row = {
        "env": env_name,
        "method": method,
        "seed": int(seed),
        "ckpt": ckpt_path,
        "fqe": float(fqe),
        "has_alpha_in_name": bool(has_alpha_in_name),
        "unc_alpha": np.nan if unc_alpha is None else float(unc_alpha),
        **ood_stats,
    }
    return row


def apply_u_family_selection(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for env in df["env"].unique():
        for m in U_FAMILY:
            mask = (df["env"] == env) & (df["method"] == m)
            if not mask.any():
                continue

            sub = df[mask]
            plain_mask = mask & (~df["has_alpha_in_name"])

            if plain_mask.any():
                drop_mask = mask & (~plain_mask)
                if drop_mask.any():
                    print(
                        f"[INFO] For env='{env}', method='{m}': "
                        f"found plain -u ckpts; dropping {drop_mask.sum()} *_alpha* rows."
                    )
                df = df[~drop_mask]
            else:
                alpha_vals = sub["unc_alpha"]
                keep_mask = mask & (alpha_vals.sub(1.0).abs() <= 1e-6)
                drop_mask = mask & (~keep_mask)

                if keep_mask.any():
                    print(
                        f"[INFO] For env='{env}', method='{m}': "
                        f"no plain -u ckpts; keeping {keep_mask.sum()} rows "
                        f"with unc_alpha≈1.0 and dropping {drop_mask.sum()} others."
                    )
                    df = df[~drop_mask]
                else:
                    print(
                        f"[WARN] For env='{env}', method='{m}': "
                        f"no plain -u and no unc_alpha≈1.0; keeping all {sub.shape[0]} rows."
                    )

    return df


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
    ap.add_argument("--env_filter", type=str, default=None,
                    help="If set, only keep rows whose env contains this substring "
                         "(e.g. 'antmaze-umaze-v2').")
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

    if args.env_filter is not None:
        before = len(df)
        df = df[df["env"].str.contains(args.env_filter)]
        print(f"[INFO] Filtered env by '{args.env_filter}': {before} -> {len(df)} rows")

    before = len(df)
    df = df[df["method"].isin(BASELINE_METHODS)]
    print(f"[INFO] Baseline-method filter: {before} -> {len(df)} rows")

    df = apply_u_family_selection(df)

    os.makedirs(results_dir, exist_ok=True)

    base_csv = Path(results_dir) / "suite_results.csv"
    df.to_csv(base_csv, index=False)
    print(f"[INFO] Wrote per-seed results to {base_csv}")

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
