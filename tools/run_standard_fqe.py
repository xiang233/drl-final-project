#!/usr/bin/env python
"""
tools/run_standard_fqe.py

Thin wrapper around scripts/estimate_return_old.py to run *standard* FQE
on a single checkpoint and log the result to a CSV, so it parallels
tools/run_sigma_gated_fqe.py.

Usage:
  python -m tools.run_standard_fqe \
    --ckpt generated_data/iql_u_hopper_medium_replay_v2_seed0.pt

Optional flags:
  --iters  : number of FQE gradient steps (default: 20000)
  --csv    : CSV path to append into (default: results/fqe_standard.csv)
"""

import argparse
import os
import torch

from ua.datasets import load_d4rl
from ua.utils import set_seed

from scripts.estimate_return_old import (
    load_policy,
    fqe_evaluate,
    infer_method_from_state_or_name,
    append_fqe_csv,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint .pt file (e.g. generated_data/iql_u_hopper_medium_replay_v2_seed0.pt)",
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=20000,
        help="Number of FQE updates (same meaning as in estimate_return_old.py)",
    )
    ap.add_argument(
        "--csv",
        type=str,
        default="results/fqe_standard.csv",
        help="CSV path to append standard FQE result",
    )
    args = ap.parse_args()

    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load checkpoint metadata so we can infer env/method/seed
    state = torch.load(ckpt_path, map_location="cpu")
    env_name = state.get("env_name", None)
    if env_name is None:
        raise ValueError(
            f"Checkpoint {ckpt_path} missing 'env_name'; "
            "please retrain with scripts that store env_name."
        )

    method = infer_method_from_state_or_name(state, ckpt_path)
    seed = state.get("seed", 0)

    print(f"[INFO] Running STANDARD FQE on {ckpt_path}")
    print(f"[INFO] env={env_name}, method={method}, seed={seed}, iters={args.iters}")

    set_seed(seed)
    _, data = load_d4rl(env_name, seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    s_dim = data["S"].shape[1]
    a_dim = data["A"].shape[1]

    # Reuse the same policy loader as estimate_return_old
    pi = load_policy(ckpt_path, s_dim, a_dim, device)

    # Standard FQE
    ret = fqe_evaluate(pi, data, iters=args.iters, device=device)
    print(f"[RESULT] standard FQE return = {ret:.3f}")

    # Append to CSV
    csv_path = args.csv
    row = {
        "env": env_name,
        "method": method,
        "seed": seed,
        "ckpt": os.path.basename(ckpt_path),
        "fqe_return": f"{ret:.6f}",
    }
    append_fqe_csv(csv_path, row)
    print(f"[INFO] Appended standard FQE result to {csv_path}")


if __name__ == "__main__":
    main()