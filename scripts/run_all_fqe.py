# scripts/run_all_fqe.py
import argparse
import glob
import os
import torch

from ua.datasets import load_d4rl
from ua.utils import set_seed
from scripts.estimate_return import (
    load_policy,
    fqe_evaluate,
    infer_method_from_state_or_name,
    append_fqe_csv,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob", type=str, default="*.pt",
        help="Glob pattern to select checkpoints (default: '*.pt')"
    )
    ap.add_argument(
        "--csv", type=str, default="results/fqe_results.csv",
        help="CSV file to append results to"
    )
    ap.add_argument(
        "--iters", type=int, default=20000,
        help="FQE training iterations"
    )
    ap.add_argument(
        "--device", type=str, default="auto",
        help="'auto', 'cpu', or 'cuda'"
    )
    args = ap.parse_args()

    ckpts = sorted(glob.glob(args.glob))
    if not ckpts:
        print(f"[WARN] No checkpoints matched pattern: {args.glob}")
        return

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"[INFO] Running FQE for {len(ckpts)} checkpoints on device={device}")

    for path in ckpts:
        try:
            state = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            continue

        env_name = state.get("env_name", None)
        seed = state.get("seed", 0)

        if env_name is None:
            print(f"[WARN] Skipping {path}: no 'env_name' in checkpoint")
            continue

        method = infer_method_from_state_or_name(state, path)

        print(f"[FQE] {os.path.basename(path)} | env={env_name} | method={method} | seed={seed}")

        try:
            set_seed(seed)
            _, data = load_d4rl(env_name, seed)
            s_dim = data["S"].shape[1]
            a_dim = data["A"].shape[1]

            pi = load_policy(path, s_dim, a_dim, device)
            ret = fqe_evaluate(pi, data, iters=args.iters, device=device)
        except Exception as e:
            print(f"[ERROR] FQE failed for {path}: {e}")
            continue

        row = {
            "env": env_name,
            "method": method,
            "seed": seed,
            "ckpt": os.path.basename(path),
            "fqe_return": f"{ret:.6f}",
        }
        append_fqe_csv(args.csv, row)

        print(f"   -> FQE return = {ret:.3f}, appended to {args.csv}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
