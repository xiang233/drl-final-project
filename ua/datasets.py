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
    Data contains S, A, optional S_next/rewards/terminals/timeouts, and z-norm stats.
    """
    path = _hdf5_path(env_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing dataset: {path}\n"
            f"Place the D4RL HDF5 there (copy from a Linux/x86 box or a mirror)."
        )

    with h5py.File(path, "r") as f:
        S = f["observations"][:].astype(np.float32)
        A = f["actions"][:].astype(np.float32)
        data = {"S": S, "A": A}

        if "next_observations" in f:
            data["S_next"] = f["next_observations"][:].astype(np.float32)
        for k in ["rewards", "terminals", "timeouts"]:
            if k in f:
                data[k] = f[k][:].astype(np.float32)

    s_mean, s_std = z_norm_states(S)  # compute stats once
    data["s_mean"], data["s_std"] = s_mean, s_std
    return None, data  # env=None in offline-only mode
