# tools/run_sigma_gated_fqe.py
import argparse
import os
import csv
import torch
import numpy as np

from ua.datasets import load_d4rl
from scripts.estimate_return import (
    load_policy,
    fqe_evaluate,
    infer_method_from_state_or_name,
)

def append_csv(path: str, row: dict, header: list[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--iters", type=int, default=20000)

    # EITHER explicit bounds...
    p.add_argument("--sigma_low", type=float, default=None)
    p.add_argument("--sigma_high", type=float, default=None)

    # ...OR quantile bounds (recommended)
    p.add_argument("--q_low", type=float, default=None,
                   help="e.g. 0.10 to set sigma_low=quantile(sigma,0.10)")
    p.add_argument("--q_high", type=float, default=None,
                   help="e.g. 0.90 to set sigma_high=quantile(sigma,0.90)")
    p.add_argument("--sigma_source", type=str, default="policy",
                   choices=["policy", "dataset"],
                   help="Compute sigma from π(s) (policy) or from dataset actions a (dataset).")
    p.add_argument("--sigma_batch", type=int, default=50000,
                   help="How many transitions to sample when estimating sigma quantiles.")

    p.add_argument("--csv", type=str, default="", help="Optional CSV path")
    args = p.parse_args()

    # ----- load ckpt metadata -----
    state = torch.load(args.ckpt, map_location="cpu")
    env_name = state.get("env_name", None)
    if env_name is None:
        raise ValueError("Checkpoint missing 'env_name'.")
    seed = state.get("seed", 0)
    method = infer_method_from_state_or_name(state, args.ckpt)

    # ----- load dataset -----
    _, data = load_d4rl(env_name, seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # load_policy may return (pi, critics) when return_critics=True
    pi, critics = load_policy(
        args.ckpt,
        data["S"].shape[1],
        data["A"].shape[1],
        device,
        return_critics=True,
    )

    # ----- compute sigma_low/high from quantiles if requested -----
    sigma_low = args.sigma_low
    sigma_high = args.sigma_high

    used_q_low = None
    used_q_high = None
    used_sigma_source = ""

    if args.q_low is not None or args.q_high is not None:
        if critics is None:
            raise ValueError(
                "Quantile gating needs critics/ensemble to compute sigma. "
                "This ckpt didn't return critics. (BC ckpts won't work.)"
            )

        ql = 0.10 if args.q_low is None else float(args.q_low)
        qh = 0.90 if args.q_high is None else float(args.q_high)
        if not (0.0 < ql < qh < 1.0):
            raise ValueError("Require 0 < q_low < q_high < 1.")

        used_q_low, used_q_high = ql, qh
        used_sigma_source = args.sigma_source

        # subsample indices to estimate quantiles
        N = data["S"].shape[0]
        B = min(int(args.sigma_batch), int(N))
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, N, size=B, dtype=np.int64)

        # critics were trained on normalized states in your TD3BC/IQL-U scripts
        s_mean = data["s_mean"]
        s_std = data["s_std"]
        S_norm = ((data["S"][idx] - s_mean) / (s_std + 1e-6)).astype(np.float32)
        S_t = torch.from_numpy(S_norm).to(device)

        with torch.no_grad():
            if args.sigma_source == "policy":
                A_ref = pi(S_t)
            else:
                A_ref = torch.from_numpy(data["A"][idx].astype(np.float32)).to(device)

            Q_all = critics.forward(S_t, A_ref, keepdim=False)          # [B, K]
            sigma = Q_all.std(dim=1, unbiased=False).detach().cpu().numpy()  # [B]

        sigma_low = float(np.quantile(sigma, ql))
        sigma_high = float(np.quantile(sigma, qh))

    if sigma_low is None or sigma_high is None:
        raise ValueError("Provide either (--sigma_low & --sigma_high) or (--q_low/--q_high).")

    if not (sigma_low < sigma_high):
        raise ValueError(f"Need sigma_low < sigma_high, got {sigma_low} >= {sigma_high}")

    # ----- run gated FQE -----
    ret = fqe_evaluate(
        pi, data,
        iters=args.iters,
        device=device,
        use_sigma_gate=True,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
    )

    print(
        f"[RESULT] σ-gated FQE return = {ret:.6f}  "
        f"(sigma_low={sigma_low:.6g}, sigma_high={sigma_high:.6g}, "
        f"source={used_sigma_source or 'manual'})"
    )

    # ----- optional CSV -----
    if args.csv:
        header = [
            "env", "method", "seed", "ckpt",
            "sigma_source", "q_low", "q_high",
            "sigma_low", "sigma_high",
            "iters", "sigma_batch",
            "fqe_return",
        ]
        row = {
            "env": env_name,
            "method": method,
            "seed": seed,
            "ckpt": os.path.basename(args.ckpt),
            "sigma_source": used_sigma_source,
            "q_low": "" if used_q_low is None else f"{used_q_low:.3f}",
            "q_high": "" if used_q_high is None else f"{used_q_high:.3f}",
            "sigma_low": f"{sigma_low:.6g}",
            "sigma_high": f"{sigma_high:.6g}",
            "iters": str(args.iters),
            "sigma_batch": str(args.sigma_batch),
            "fqe_return": f"{ret:.6f}",
        }
        append_csv(args.csv, row, header)
        print(f"[INFO] Appended to {args.csv}")

if __name__ == "__main__":
    main()