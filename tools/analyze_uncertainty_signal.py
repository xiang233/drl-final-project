#!/usr/bin/env python
"""
tools/analyze_uncertainty_signal.py

Check how strong ensemble std σ_Q is as a signal of "badness":
  - Correlate σ_Q with TD residual (Bellman error) on dataset transitions.
  - Do this both for:
      * dataset actions: (s, a)
      * policy actions:  (s, π(s))

Usage (from repo root):
  python -m tools.analyze_uncertainty_signal \
    --ckpt generated_data/td3bc_hopper_medium_replay_v2_seed0.pt \
    --batch_size 50000

You can point it at any ckpt: td3bc, td3bc_u, iql, iql_u, etc.
"""

import argparse
import os.path as osp

import numpy as np
import torch

from ua.datasets import load_d4rl
from ua.utils import set_seed
from ua.nets import CriticEnsemble, Actor  # Actor may not be strictly needed but harmless
from scripts.estimate_return import load_policy


# ----------------- small helpers ----------------- #

def load_critics_from_ckpt(state, critics_module):
    """
    Loads critics weights into `critics_module`.

    Supports:
      - ensemble checkpoints: state["critics"]
      - single-critic checkpoints: state["q"] (K=1 fallback)

    Returns:
      K_loaded (int), has_ensemble (bool)
    """
    if "critics" in state:
        critics_module.load_state_dict(state["critics"])
        return int(getattr(critics_module, "K", 0) or 0), True

    if "q" in state:
        # K=1 fallback: load into the first critic
        if hasattr(critics_module, "qs"):
            critics_module.qs[0].load_state_dict(state["q"])
            return 1, False
        if hasattr(critics_module, "critics"):
            critics_module.critics[0].load_state_dict(state["q"])
            return 1, False

        # If CriticEnsemble is actually a single module (unlikely here)
        critics_module.load_state_dict(state["q"])
        return 1, False

    raise KeyError("Checkpoint has neither 'critics' (ensemble) nor 'q' (single critic).")


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation using numpy only."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    x = x[mask]
    y = y[mask]
    if x.std() < 1e-8 or y.std() < 1e-8:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (no scipy)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    x = x[mask]
    y = y[mask]
    n = x.shape[0]

    def rank(a):
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        return ranks

    rx = rank(x)
    ry = rank(y)
    if rx.std() < 1e-8 or ry.std() < 1e-8:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def bin_stats(x: np.ndarray, y: np.ndarray, nbins: int = 10):
    """
    Bin y by quantiles of x. Returns (bin_centers, mean_y_in_bin, count_in_bin).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return np.array([]), np.array([]), np.array([])
    qs = np.quantile(x, np.linspace(0.0, 1.0, nbins + 1))
    centers = []
    means = []
    counts = []
    for i in range(nbins):
        lo, hi = qs[i], qs[i + 1]
        if i == nbins - 1:
            m = (x >= lo) & (x <= hi)
        else:
            m = (x >= lo) & (x < hi)
        c = int(m.sum())
        if c == 0:
            continue
        centers.append(x[m].mean())
        means.append(y[m].mean())
        counts.append(c)
    return np.array(centers), np.array(means), np.array(counts)


# ----------------- main logic ----------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to checkpoint .pt file (e.g. generated_data/td3bc_hopper_medium_replay_v2_seed0.pt)")
    ap.add_argument("--batch_size", type=int, default=50000,
                    help="Number of dataset transitions to sample for analysis.")
    ap.add_argument("--nbins", type=int, default=10,
                    help="Number of bins for binned stats.")
    args = ap.parse_args()

    ckpt_path = args.ckpt
    if not osp.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")

    env_name = state.get("env_name", None)
    seed = state.get("seed", 0)
    K = state.get("K", 4)
    cfg = state.get("cfg", {})
    gamma = float(cfg.get("gamma", 0.99))

    if env_name is None:
        raise ValueError("Checkpoint missing 'env_name' field; cannot load dataset.")

    print(f"[INFO] env_name={env_name}, seed={seed}, K={K}, gamma={gamma}")

    set_seed(seed)
    _, data = load_d4rl(env_name, seed)

    # dataset
    S = data["S"].astype(np.float32)
    A = data["A"].astype(np.float32)
    Sn = data.get("S_next", None)
    Rp = data.get("rewards", None)
    terminals = data.get("terminals", None)
    timeouts = data.get("timeouts", None)

    if Sn is None or Rp is None:
        raise ValueError("Dataset missing S_next or rewards; needed for TD error analysis.")

    if terminals is None:
        terminals = np.zeros((S.shape[0],), dtype=np.float32)
    if timeouts is None:
        timeouts = np.zeros((S.shape[0],), dtype=np.float32)

    s_mean, s_std = data["s_mean"], data["s_std"]
    S  = (S  - s_mean) / (s_std + 1e-6)
    Sn = (Sn - s_mean) / (s_std + 1e-6)

    N, s_dim = S.shape
    a_dim = A.shape[1]
    B = min(args.batch_size, N)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[INFO] Using device: {device}")

    # ---- rebuild critics then load weights (supports TD3BC ensembles + IQL single Q) ----
    critics = CriticEnsemble(s_dim, a_dim, K=K).to(device)

    # ✅ THIS IS WHERE YOUR SNIPPET GOES
    K_loaded, has_ensemble = load_critics_from_ckpt(state, critics)
    print(f"[INFO] Loaded critics. has_ensemble={has_ensemble}, K={K_loaded}")

    # policy loader (works for td3bc, td3bc_u, iql, iql_u)
    pi = load_policy(ckpt_path, s_dim, a_dim, device)

    # sample a big batch
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, size=B)

    S_b  = torch.from_numpy(S[idx]).to(device)
    A_b  = torch.from_numpy(A[idx]).to(device)
    Sn_b = torch.from_numpy(Sn[idx]).to(device)
    R_b  = torch.from_numpy(Rp[idx].squeeze().astype(np.float32)).to(device)
    D_b  = torch.from_numpy(
        np.clip(terminals[idx] + timeouts[idx], 0, 1).astype(np.float32)
    ).to(device)

    # ---------- compute σ_Q and TD error ---------- #
    with torch.no_grad():
        # uncertainty on dataset actions
        Q_all_ds = critics.forward(S_b, A_b, keepdim=False)  # [B, K] (or [B,1] effectively)
        q_mean_ds = Q_all_ds.mean(dim=1)                     # [B]
        q_std_ds  = Q_all_ds.std(dim=1)                      # [B]

        # ✅ use the fallback for IQL (no ensemble)
        sigma_ds = (q_std_ds if has_ensemble else torch.zeros_like(q_mean_ds)).cpu().numpy()

        # uncertainty on policy actions
        A_pi = pi(S_b)
        Q_all_pi = critics.forward(S_b, A_pi, keepdim=False)
        q_mean_pi = Q_all_pi.mean(dim=1)
        q_std_pi  = Q_all_pi.std(dim=1)

        sigma_pi = (q_std_pi if has_ensemble else torch.zeros_like(q_mean_pi)).cpu().numpy()

        # TD target using pessimistic ensemble min at next state, policy action
        A_next = pi(Sn_b)
        Q_next_all = critics.forward(Sn_b, A_next, keepdim=False)  # [B, K] (or [B,1])

        # If no ensemble, min==mean==that single value; this is safe either way
        Q_next_min = Q_next_all.min(dim=1).values                  # [B]

        target = R_b + gamma * (1.0 - D_b) * Q_next_min            # [B]
        td_error = (q_mean_ds - target).abs().cpu().numpy()        # [B]

    # ---------- correlations ---------- #
    print("\n===== Correlation: σ_Q vs TD error (dataset actions) =====")
    r_p_ds = pearson_corr(sigma_ds, td_error)
    r_s_ds = spearman_corr(sigma_ds, td_error)
    print(f"Pearson r(sigma_ds, td_error)   = {r_p_ds:.4f}")
    print(f"Spearman ρ(sigma_ds, td_error)  = {r_s_ds:.4f}")

    print("\n===== Correlation: σ_Q vs TD error (policy actions) =====")
    r_p_pi = pearson_corr(sigma_pi, td_error)
    r_s_pi = spearman_corr(sigma_pi, td_error)
    print(f"Pearson r(sigma_pi, td_error)   = {r_p_pi:.4f}")
    print(f"Spearman ρ(sigma_pi, td_error)  = {r_s_pi:.4f}")

    # ---------- binned stats ---------- #
    nbins = args.nbins

    print(f"\n===== Binned TD error vs σ_Q (dataset actions, {nbins} bins) =====")
    centers_ds, td_means_ds, counts_ds = bin_stats(sigma_ds, td_error, nbins=nbins)
    for c, m, n in zip(centers_ds, td_means_ds, counts_ds):
        print(f"sigma_ds ~ {c:.4f}  ->  mean TD error = {m:.4f}  (n={n})")

    print(f"\n===== Binned TD error vs σ_Q (policy actions, {nbins} bins) =====")
    centers_pi, td_means_pi, counts_pi = bin_stats(sigma_pi, td_error, nbins=nbins)
    for c, m, n in zip(centers_pi, td_means_pi, counts_pi):
        print(f"sigma_pi ~ {c:.4f}  ->  mean TD error = {m:.4f}  (n={n})")

    print("\n[INFO] Done. If correlations and binned means are flat/near-zero, σ_Q is a weak signal.")


if __name__ == "__main__":
    main()