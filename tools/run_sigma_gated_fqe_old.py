#!/usr/bin/env python

#
# python -m tools.run_sigma_gated_fqe \
#   --ckpt generated_data/td3bc_antmaze_umaze_v2_seed0.pt \
#   --sigma_low 0.001 --sigma_high 0.015 \
#   --iters 20000 \
#   --csv results/fqe_sigma_gated_bounds_antmaze.csv
#

import argparse
import os
import csv
import torch
from scripts.estimate_return import (
    load_policy,
    fqe_evaluate,
    infer_method_from_state_or_name,
)
from ua.datasets import load_d4rl


def append_fqe_csv(csv_path, row, header):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sigma_low", type=float, default=0.3)
    p.add_argument("--sigma_high", type=float, default=1.0)
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument(
        "--csv",
        type=str,
        default="",
        help="Optional CSV to append gated FQE result (with bounds).",
    )
    args = p.parse_args()

    # Load checkpoint
    state = torch.load(args.ckpt, map_location="cpu")
    env_name = state["env_name"]
    seed = state.get("seed", 0)
    method = infer_method_from_state_or_name(state, args.ckpt)

    print(f"[INFO] Running post-hoc σ-gated FQE on {args.ckpt}")
    print(f"[INFO] env={env_name}, method={method}, seed={seed}")
    print(f"[INFO] Gating thresholds: low={args.sigma_low}, high={args.sigma_high}")

    # Load dataset
    _, data = load_d4rl(env_name, seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Load π and critics ensemble
    pi = load_policy(
        args.ckpt,
        data["S"].shape[1],
        data["A"].shape[1],
        device,
        return_critics=True,  # your modified estimate_return.py supports this
    )

    if isinstance(pi, tuple):
        pi, _ = pi

    # Evaluate with σ-gated safe policy
    ret = fqe_evaluate(
        pi,
        data,
        iters=args.iters,
        device=device,
        use_sigma_gate=True,
        sigma_low=args.sigma_low,
        sigma_high=args.sigma_high,
    )

    print(f"\n[RESULT] σ-gated FQE return = {ret:.3f}\n")

    # Optional CSV logging
    if args.csv:
        header = [
            "env",
            "method",
            "seed",
            "ckpt",
            "sigma_low",
            "sigma_high",
            "fqe_return",
        ]
        row = {
            "env": env_name,
            "method": method,
            "seed": seed,
            "ckpt": os.path.basename(args.ckpt),
            "sigma_low": f"{args.sigma_low:.6f}",
            "sigma_high": f"{args.sigma_high:.6f}",
            "fqe_return": f"{ret:.6f}",
        }
        append_fqe_csv(args.csv, row, header)
        print(f"[INFO] Appended σ-gated FQE result to {args.csv}")


if __name__ == "__main__":
    main()