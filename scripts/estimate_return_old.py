# scripts/estimate_return.py
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ua.datasets import load_d4rl
from ua.utils import set_seed


# --------- Basic networks (match training architectures) ---------

class MLP(nn.Module):
    """
    Simple 2-layer MLP, same as in train_bc.py:
      Linear(din, 256) -> ReLU -> Linear(256, 256) -> ReLU -> Linear(256, dout)
    """
    def __init__(self, din, dout, hid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, dout)
        )

    def forward(self, x):
        return self.net(x)


class PolicyNetwork(nn.Module):
    """
    Deterministic tanh policy: a = tanh(backbone(s)).
    This matches the PolicyNetwork we used for IQL / IQL-U.
    """
    def __init__(self, state_dim, action_dim, hid=256):
        super().__init__()
        self.backbone = MLP(state_dim, action_dim, hid=hid)

    def forward(self, s):
        return torch.tanh(self.backbone(s))


# --------- Policy loader that handles all algos ---------

def load_policy(path, s_dim, a_dim, device):
    """
    Load a policy from a checkpoint, handling:
      - BC:                  {'model': MLP.state_dict(), 'algo': 'bc' or no 'algo'}
      - TD3BC-U (+ variants):{'actor': Actor.state_dict(), 'algo': 'td3bc_u' / 'td3bc_u_pessimistic' / 'td3bc_u_pessimistic_mcdo'}
      - IQL:                 {'model': PolicyNetwork.state_dict(), 'algo': 'iql'}
      - IQL-U (+ mcdo):      {'actor': PolicyNetwork.state_dict(), 'algo': 'iql_u' / 'iql_u_mcdo'}
    Falls back to MLP if algo is unknown but we find 'model' or 'actor'.
    """
    state = torch.load(path, map_location=device)
    algo = state.get("algo", None)

    # ---- IQL / IQL-U / IQL-U-MCDO ----
    # Unify all these as the same deterministic PolicyNetwork architecture.
    if algo in ("iql", "iql_u", "iql_u_mcdo"):
        pi = PolicyNetwork(s_dim, a_dim).to(device)
        # In our code:
        #   - plain IQL saves under 'model'
        #   - IQL-U / IQL-U-MCDO save under 'actor'
        actor_sd = state.get("actor") or state.get("model")
        if actor_sd is None:
            raise KeyError(
                f"Expected 'actor' or 'model' key in {algo} checkpoint {path}, "
                f"found keys: {list(state.keys())}"
            )
        pi.load_state_dict(actor_sd, strict=True)
        pi.eval()
        return pi

    # ---- TD3BC-U (+ pessimistic + mcdo variants) ----
    # TD3BC-U checkpoints look like:
    #   {"actor": ..., "critics": ..., "K": ..., "env_name": ..., "seed": ..., "cfg": {...}}
    if algo in ("td3bc_u", "td3bc_u_pessimistic", "td3bc_u_pessimistic_mcdo") or (
        "actor" in state and "critics" in state and "K" in state and "v" not in state
    ):
        try:
            from ua.nets import Actor
        except ImportError as e:
            raise ImportError(
                "Could not import Actor from ua.nets; "
                "this is required to load TD3BC-U policies."
            ) from e

        pi = Actor(s_dim, a_dim).to(device)
        actor_sd = state.get("actor")
        if actor_sd is None:
            raise KeyError(
                f"Expected 'actor' key in TD3BC-U-like checkpoint {path}, "
                f"found keys: {list(state.keys())}"
            )
        pi.load_state_dict(actor_sd, strict=True)
        pi.eval()
        return pi

    # ---- BC (behavior cloning) ----
    # BC checkpoints: {"model": MLP.state_dict(), "env_name": ..., "seed": ..., "val_mse": ...}
    if "model" in state and "q" not in state and "v" not in state:
        pi = MLP(s_dim, a_dim).to(device)
        pi.load_state_dict(state["model"], strict=True)
        pi.eval()
        return pi

    # ---- Fallback ----
    key = None
    if "model" in state:
        key = "model"
    elif "actor" in state:
        key = "actor"

    if key is None:
        raise ValueError(
            f"Could not determine how to load policy from {path}. "
            f"algo={algo}, keys={list(state.keys())}"
        )

    print(
        f"[WARN] Unknown algo '{algo}' in {path}; "
        f"falling back to MLP using state['{key}']."
    )

    pi = MLP(s_dim, a_dim).to(device)
    pi.load_state_dict(state[key], strict=False)
    pi.eval()
    return pi


# --------- FQE implementation ---------

class QMLP(MLP):
    """Q-network: same MLP, but scalar output."""
    def __init__(self, din, hid=256):
        super().__init__(din, 1, hid)


def fqe_evaluate(pi, data, gamma=0.99, iters=20000, bs=1024, lr=3e-4, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    needed = ["S", "A", "S_next", "rewards", "terminals", "s_mean", "s_std"]
    missing = [k for k in needed if k not in data]
    if missing:
        raise ValueError(f"FQE requires keys {needed}, missing: {missing}")

    S       = data["S"]
    A       = data["A"]
    S_next  = data["S_next"]
    R       = data["rewards"].reshape(-1, 1)
    D       = data["terminals"].reshape(-1, 1)
    s_mean  = data["s_mean"]
    s_std   = data["s_std"]

    def z(x):
        return (x - s_mean) / (s_std + 1e-6)

    Sz      = z(S).astype(np.float32)
    Snext_z = z(S_next).astype(np.float32)
    A       = A.astype(np.float32)
    R       = R.astype(np.float32)
    D       = D.astype(np.float32)

    q = QMLP(Sz.shape[1] + A.shape[1]).to(device)
    opt = optim.Adam(q.parameters(), lr=lr)
    mse = nn.MSELoss()

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Sz),
        torch.from_numpy(A),
        torch.from_numpy(Snext_z),
        torch.from_numpy(R),
        torch.from_numpy(D),
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)

    q.train()
    step = 0
    for (s, a, sp, r, d) in dl:
        s, a, sp, r, d = (
            s.to(device),
            a.to(device),
            sp.to(device),
            r.to(device),
            d.to(device),
        )

        with torch.no_grad():
            a_pi_sp = pi(sp)
            q_sp = q(torch.cat([sp, a_pi_sp], dim=-1))
            tgt = r + gamma * (1.0 - d) * q_sp

        q_sa = q(torch.cat([s, a], dim=-1))
        loss = mse(q_sa, tgt)

        opt.zero_grad()
        loss.backward()
        opt.step()

        step += 1
        if step >= iters:
            break

    q.eval()
    with torch.no_grad():
        s_all = torch.from_numpy(Sz).to(device)
        a_all = pi(s_all)
        v_hat = q(torch.cat([s_all, a_all], dim=-1)).mean().item()
    return v_hat


def infer_method_from_state_or_name(state, ckpt_path):
    """
    Infer a human-readable method name:
      - Prefer explicit 'algo' field if present (e.g., 'iql_u_mcdo').
      - Otherwise fall back to filename stem prefixes.
    """
    algo = state.get("algo", None)
    if algo is not None:
        return algo

    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    if stem.startswith("bc_"):
        return "bc"
    if stem.startswith("td3bc_u_"):
        return "td3bc_u"
    if stem.startswith("iql_u_mcdo_"):
        return "iql_u_mcdo"
    if stem.startswith("iql_u_"):
        return "iql_u"
    if stem.startswith("iql_"):
        return "iql"
    return "unknown"


def append_fqe_csv(csv_path, row):
    """
    Append a row dict to a CSV file. Create file + header if needed.
    Expected columns: env, method, seed, ckpt, fqe_return
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header = ["env", "method", "seed", "ckpt", "fqe_return"]
    file_exists = os.path.exists(csv_path)

    import csv
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# --------- CLI entrypoint ---------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="hopper-medium-replay-v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--csv", type=str, default="", help="Optional CSV path to append FQE result")
    args = ap.parse_args()

    # Load checkpoint metadata first so we can infer env/method/seed
    state = torch.load(args.ckpt, map_location="cpu")
    env_name = state.get("env_name", args.env)
    method = infer_method_from_state_or_name(state, args.ckpt)
    seed = state.get("seed", args.seed)

    set_seed(seed)
    _, data = load_d4rl(env_name, seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    s_dim = data["S"].shape[1]
    a_dim = data["A"].shape[1]

    pi = load_policy(args.ckpt, s_dim, a_dim, device)
    ret = fqe_evaluate(pi, data, iters=args.iters, device=device)

    print(f"FQE estimated return for {args.ckpt} on {env_name}: {ret:.3f}")

    if args.csv:
        row = {
            "env": env_name,
            "method": method,
            "seed": seed,
            "ckpt": os.path.basename(args.ckpt),
            "fqe_return": f"{ret:.6f}",
        }
        append_fqe_csv(args.csv, row)
        print(f"[INFO] Appended FQE result to {args.csv}")