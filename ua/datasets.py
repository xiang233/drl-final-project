#ua/datasets.py
import os
import numpy as np
import h5py

def z_norm_states(S, s_mean=None, s_std=None, eps=1e-6):
    """
    Dual-mode helper:
      - If s_mean/s_std are None: return (mean, std) computed from S.
      - If s_mean/s_std are provided: return normalized S.
    Always uses float32.
    """
    S = S.astype(np.float32)
    if s_mean is None or s_std is None:
        s_mean = S.mean(axis=0).astype(np.float32)
        s_std  = (S.std(axis=0) + eps).astype(np.float32)
        return s_mean, s_std
    else:
        return (S - s_mean.astype(np.float32)) / (s_std.astype(np.float32) + eps)

def _hdf5_path(env_name: str) -> str:
    # it expect files named exactly like D4RL: <env>.hdf5 under ~/.d4rl
    return os.path.join(os.path.expanduser("~"), ".d4rl", f"{env_name}.hdf5")

def load_d4rl(env_name: str, seed: int = 123, require_env: bool = False):
    """
    Offline-only loader: returns (env, data) where env is always None on Mac.
    Data contains S, A, S_next (if we can construct it), rewards, terminals,
    timeouts, and z-norm stats (s_mean, s_std).

    Works with:
      - Classic D4RL HDF5: observations, actions, next_observations, ...
      - Farama-style HDF5 where some fields may sit under 'infos'.
    """
    path = _hdf5_path(env_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing dataset: {path}\n"
            f"Place the D4RL HDF5 there (copy from a Linux/x86 box or a mirror)."
        )

    with h5py.File(path, "r") as f:
        # --- Core S, A ---
        S = f["observations"][:].astype(np.float32)
        A = f["actions"][:].astype(np.float32)
        data = {"S": S, "A": A}

        # --- Try to get S_next in the best possible way ---
        S_next = None

        # 1) Standard D4RL field
        if "next_observations" in f:
            S_next = f["next_observations"][:].astype(np.float32)

        # 2) Maybe under infos/next_observations (Farama-style variant)
        elif "infos" in f and "next_observations" in f["infos"]:
            S_next = f["infos"]["next_observations"][:].astype(np.float32)
            S_next = S_next.astype(np.float32)

        # 3) Fallback: reconstruct by shifting, respecting terminals/timeouts if present
        if S_next is None:
            # optional terminal/timeouts to avoid crossing episodes
            terminals = None
            timeouts = None

            if "terminals" in f:
                terminals = f["terminals"][:].astype(bool)
            elif "infos" in f and "terminals" in f["infos"]:
                terminals = f["infos"]["terminals"][:].astype(bool)

            if "timeouts" in f:
                timeouts = f["timeouts"][:].astype(bool)
            elif "infos" in f and "timeouts" in f["infos"]:
                timeouts = f["infos"]["timeouts"][:].astype(bool)

            done_mask = None
            if terminals is not None and timeouts is not None:
                done_mask = np.logical_or(terminals, timeouts)
            elif terminals is not None:
                done_mask = terminals
            elif timeouts is not None:
                done_mask = timeouts

            S_next = S.copy()  # default: self-loop
            # shift by one for all but last transition
            S_next[:-1] = S[1:]

            # if we know which steps are terminal/timeout, don't cross episodes
            if done_mask is not None:
                done_mask = done_mask.astype(bool)
                # for done steps, keep S_next = S at that index
                S_next[done_mask] = S[done_mask]

            print(f"[INFO] Reconstructed S_next by shifting for {env_name}: shape={S_next.shape}")

        data["S_next"] = S_next

        # --- Rewards / terminals / timeouts ---
        for k in ["rewards", "terminals", "timeouts"]:
            if k in f:
                data[k] = f[k][:].astype(np.float32)
            elif "infos" in f and k in f["infos"]:
                data[k] = f["infos"][k][:].astype(np.float32)

    # --- Z-normalization stats ---
    s_mean, s_std = z_norm_states(S)  # compute stats once
    data["s_mean"], data["s_std"] = s_mean, s_std

    return None, data  # env=None in offline-only mode
