import argparse
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim

from ua.datasets import load_d4rl
from ua.utils import set_seed


class MLP(nn.Module):
    def __init__(self, din, dout, hid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, dout)
        )

    def forward(self, x):
        return self.net(x)


def main(env_name="hopper-medium-replay-v2", seed=0, steps=20000, bs=1024):
    set_seed(seed)
    _, data = load_d4rl(env_name, seed)
    S, A = data["S"], data["A"]
    s_mean, s_std = data["s_mean"], data["s_std"]

    # Normalize states with provided stats (avoid signature mismatches)
    Sn = (S - s_mean) / (s_std + 1e-6)

    # 90/10 train/val split (deterministic by seed)
    import numpy as np
    N = Sn.shape[0]
    idx = np.random.RandomState(seed).permutation(N)
    n_tr = int(0.9 * N)
    tr, va = idx[:n_tr], idx[n_tr:]

    S_tr, A_tr = Sn[tr], A[tr]
    S_va, A_va = Sn[va], A[va]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pi = MLP(Sn.shape[1], A.shape[1]).to(device)
    opt = optim.Adam(pi.parameters(), lr=3e-4)
    mse = nn.MSELoss()

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(S_tr), torch.from_numpy(A_tr)
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(S_va), torch.from_numpy(A_va)
    )

    dl = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True, drop_last=True
    )

    pi.train()
    updates = 0
    for s, a in cycle(dl):
        s, a = s.to(device), a.to(device)
        loss = mse(pi(s), a)
        opt.zero_grad()
        loss.backward()
        opt.step()

        updates += 1
        if updates % 100 == 0:
            print(f"step {updates}: BC loss {loss.item():.4f}")
        if updates >= steps:
            break

    # Validation MSE on held-out split
    pi.eval()
    with torch.no_grad():
        s_va = torch.from_numpy(S_va).to(device)
        a_va = torch.from_numpy(A_va).to(device)
        val_mse = mse(pi(s_va), a_va).item()
    print(f"Validation BC MSE: {val_mse:.4f}")

    out_path = f"bc_{env_name.replace('-', '_')}.pt"
    torch.save({"model": pi.state_dict(),
                "env_name": env_name,
                "seed": seed,
                "val_mse": val_mse}, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hopper-medium-replay-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--bs", type=int, default=1024)
    args = parser.parse_args()
    main(env_name=args.env, seed=args.seed, steps=args.steps, bs=args.bs)
