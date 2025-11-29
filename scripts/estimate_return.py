# scripts/estimate_return.py
import argparse
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
      - BC:          {'model': MLP.state_dict(), 'algo': 'bc' or no 'algo'}
      - TD3BC-U:     {'actor': Actor.state_dict(), 'algo': 'td3bc_u'} (or no 'algo')
      - IQL:         {'model': PolicyNetwork.state_dict(), 'algo': 'iql'}
      - IQL-U:       {'actor': PolicyNetwork.state_dict(), 'algo': 'iql_u'}
    Falls back to MLP if algo is unknown but we find 'model' or 'actor'.
    """
    state = torch.load(path, map_location=device)
    algo = state.get("algo", None)

    # ---- IQL (explicit) ----
    if algo == "iql":
        pi = PolicyNetwork(s_dim, a_dim).to(device)
        actor_sd = state.get("model")
        if actor_sd is None:
            raise KeyError(
                f"Expected 'model' key in IQL checkpoint {path}, "
                f"found keys: {list(state.keys())}"
            )
        pi.load_state_dict(actor_sd, strict=True)
        pi.eval()
        return pi

    # ---- IQL-U (explicit) ----
    if algo == "iql_u":
        pi = PolicyNetwork(s_dim, a_dim).to(device)
        actor_sd = state.get("actor") or state.get("model")
        if actor_sd is None:
            raise KeyError(
                f"Expected 'actor' or 'model' key in IQL-U checkpoint {path}, "
                f"found keys: {list(state.keys())}"
            )
        pi.load_state_dict(actor_sd, strict=True)
        pi.eval()
        return pi

    # ---- TD3BC-U ----
    # Your TD3BC-U checkpoints look like:
    #   {"actor": ..., "critics": ..., "K": ..., "env_name": ..., "seed": ..., "cfg": {...}}
    # with no 'algo' initially, or algo == 'td3bc_u' if you added it.
    if algo == "td3bc_u" or (
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
                f"Expected 'actor' key in TD3BC-U checkpoint {path}, "
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
    # Unknown algo: try to interpret as MLP with either 'model' or 'actor'
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

    print(f"[WARN] Unknown algo '{algo}' in {path}; "
          f"falling back to MLP using state['{key}'].")

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


# --------- CLI entrypoint ---------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="hopper-medium-replay-v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--iters", type=int, default=20000)
    args = ap.parse_args()

    set_seed(args.seed)
    _, data = load_d4rl(args.env, args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    s_dim = data["S"].shape[1]
    a_dim = data["A"].shape[1]

    pi = load_policy(args.ckpt, s_dim, a_dim, device)
    ret = fqe_evaluate(pi, data, iters=args.iters, device=device)

    print(f"FQE estimated return for {args.ckpt} on {args.env}: {ret:.3f}")
