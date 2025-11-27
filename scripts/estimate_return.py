# scripts/estimate_return.py
import argparse, torch
from ua.datasets import load_d4rl
from ua.utils import set_seed

# use the same MLP you used for the actor
try:
    from ua.nets import MLP
except Exception:
    # fallback if you didn't make ua/nets.py
    import torch.nn as nn
    class MLP(nn.Module):
        def __init__(self, din, dout, hid=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(din, hid), nn.ReLU(),
                nn.Linear(hid, hid), nn.ReLU(),
                nn.Linear(hid, dout)
            )
        def forward(self, x): return self.net(x)

def _extract_actor_state(obj):
    """
    Accepts:
      {"model": sd} (BC)
      {"actor": sd, ...} (TD3+BC-U)
      {"policy": sd}
      or a raw state_dict itself.
    """
    if isinstance(obj, dict):
        for k in ("model", "actor", "policy"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        # maybe it's already a state dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise KeyError("Could not find actor weights in checkpoint (looked for 'model'/'actor'/'policy').")

def load_policy(ckpt_path, s_dim, a_dim, device):
    # safer load (get rid of future warning)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_sd = _extract_actor_state(state)
    pi = MLP(s_dim, a_dim).to(device)
    pi.load_state_dict(actor_sd, strict=True)
    pi.eval()
    return pi

# --------- minimal FQE ----------
import numpy as np, torch.nn as nn, torch.optim as optim
class QMLP(MLP):  # same shape helper
    def __init__(self, din, hid=256): super().__init__(din, 1, hid)

def fqe_evaluate(pi, data, gamma=0.99, iters=20000, bs=1024, lr=3e-4, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    needed = ["S","A","S_next","rewards","terminals","s_mean","s_std"]
    for k in needed:
        if k not in data:
            raise ValueError(f"FQE requires '{k}' in dataset; missing keys: {set(needed)-set(data.keys())}")

    S, A = data["S"], data["A"]
    S_next, R, D = data["S_next"], data["rewards"].reshape(-1,1), data["terminals"].reshape(-1,1)
    s_mean, s_std = data["s_mean"], data["s_std"]
    def z(x): return (x - s_mean) / (s_std + 1e-6)
    Sz, Snext_z = z(S), z(S_next)

    q = QMLP(Sz.shape[1] + A.shape[1]).to(device)
    opt, mse = optim.Adam(q.parameters(), lr=lr), nn.MSELoss()

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Sz), torch.from_numpy(A),
        torch.from_numpy(Snext_z), torch.from_numpy(R), torch.from_numpy(D)
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)

    q.train()
    for t, (s, a, sp, r, d) in enumerate(dl, start=1):
        s, a, sp, r, d = s.to(device), a.to(device), sp.to(device), r.to(device), d.to(device)
        with torch.no_grad():
            a_pi_sp = pi(sp)
            tgt = r + gamma * (1.0 - d) * q(torch.cat([sp, a_pi_sp], dim=-1))
        pred = q(torch.cat([s, a], dim=-1))
        loss = mse(pred, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if t >= iters: break

    q.eval()
    with torch.no_grad():
        s = torch.from_numpy(Sz).to(device)
        v = q(torch.cat([s, pi(s)], dim=-1)).mean().item()
    return v

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

    s_dim, a_dim = data["S"].shape[1], data["A"].shape[1]
    pi = load_policy(args.ckpt, s_dim, a_dim, device)

    ret = fqe_evaluate(pi, data, iters=args.iters, device=device)
    print(f"FQE estimated return for {args.ckpt} on {args.env}: {ret:.3f}")
