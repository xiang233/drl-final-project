import argparse
import copy

import numpy as np
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


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hid=256):
        super().__init__()
        self.net = MLP(state_dim + action_dim, 1, hid=hid)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x).squeeze(-1)  


class VNetwork(nn.Module):
    def __init__(self, state_dim, hid=256):
        super().__init__()
        self.net = MLP(state_dim, 1, hid=hid)

    def forward(self, s):
        return self.net(s).squeeze(-1)  


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hid=256):
        super().__init__()
        self.backbone = MLP(state_dim, action_dim, hid=hid)

    def forward(self, s):
        return torch.tanh(self.backbone(s))


def expectile_loss(diff, tau):
    weight = torch.where(diff >= 0, tau, 1.0 - tau)
    return (weight * diff.pow(2)).mean()


def main(
    env_name="hopper-medium-replay-v2",
    seed=0,
    steps=200000,
    bs=1024,
    gamma=0.99,
    tau=0.7,
    beta=3.0,
):
    set_seed(seed)
    _, data = load_d4rl(env_name, seed)

    S = data["S"]          
    A = data["A"]          

    if "S_next" in data:
        S2 = data["S_next"]
    elif "next_observations" in data:
        S2 = data["next_observations"]
    else:
        raise KeyError("No next-state key (S_next/next_observations) in data.")

    R = data["rewards"].astype(np.float32).reshape(-1)  
    terminals = data.get("terminals", None)
    timeouts = data.get("timeouts", None)

    if terminals is not None and timeouts is not None:
        D = np.logical_or(terminals > 0.5, timeouts > 0.5).astype(np.float32)
    elif terminals is not None:
        D = (terminals > 0.5).astype(np.float32)
    elif timeouts is not None:
        D = (timeouts > 0.5).astype(np.float32)
    else:
        D = np.zeros_like(R, dtype=np.float32)

    s_mean, s_std = data["s_mean"], data["s_std"]

    Sn = (S - s_mean) / (s_std + 1e-6)
    S2n = (S2 - s_mean) / (s_std + 1e-6)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    state_dim = Sn.shape[1]
    action_dim = A.shape[1]

    q = QNetwork(state_dim, action_dim).to(device)
    v = VNetwork(state_dim).to(device)
    pi = PolicyNetwork(state_dim, action_dim).to(device)

    v_target = copy.deepcopy(v).to(device)
    for p in v_target.parameters():
        p.requires_grad = False

    q_opt = optim.Adam(q.parameters(), lr=3e-4)
    v_opt = optim.Adam(v.parameters(), lr=3e-4)
    pi_opt = optim.Adam(pi.parameters(), lr=3e-4)

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(Sn).float(),
        torch.from_numpy(A).float(),
        torch.from_numpy(R).float(),
        torch.from_numpy(S2n).float(),
        torch.from_numpy(D).float(),
    )
    dl = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)

    def soft_update_v_target(tau_ema=0.005):
        with torch.no_grad():
            for p, p_t in zip(v.parameters(), v_target.parameters()):
                p_t.data.mul_(1.0 - tau_ema)
                p_t.data.add_(tau_ema * p.data)

    q.train()
    v.train()
    pi.train()
    updates = 0
    dl_iter = iter(dl)

    while updates < steps:
        try:
            s_b, a_b, r_b, s2_b, d_b = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            s_b, a_b, r_b, s2_b, d_b = next(dl_iter)

        s_b = s_b.to(device)
        a_b = a_b.to(device)
        r_b = r_b.to(device)
        s2_b = s2_b.to(device)
        d_b = d_b.to(device)

        with torch.no_grad():
            q_sa = q(s_b, a_b)
        v_s = v(s_b)
        v_loss = expectile_loss(q_sa - v_s, tau)

        v_opt.zero_grad()
        v_loss.backward()
        v_opt.step()

        with torch.no_grad():
            v_s2 = v_target(s2_b)
            target = r_b + gamma * (1.0 - d_b) * v_s2

        q_sa = q(s_b, a_b)
        q_loss = 0.5 * (q_sa - target).pow(2).mean()

        q_opt.zero_grad()
        q_loss.backward()
        q_opt.step()

        with torch.no_grad():
            q_sa = q(s_b, a_b)
            v_s = v(s_b)
            adv = q_sa - v_s
            adv_mean = adv.mean()
            adv_std = adv.std() + 1e-6
            norm_adv = (adv - adv_mean) / adv_std
            weights = torch.exp(beta * norm_adv).clamp(max=100.0)

        pi_a = pi(s_b)
        mse = (pi_a - a_b).pow(2).sum(-1)  # [B]
        pi_loss = (weights * mse).mean()

        pi_opt.zero_grad()
        pi_loss.backward()
        pi_opt.step()

        soft_update_v_target()

        updates += 1
        if updates % 1000 == 0:
            print(
                f"step {updates}: "
                f"Q loss {q_loss.item():.4f}, "
                f"V loss {v_loss.item():.4f}, "
                f"Pi loss {pi_loss.item():.4f}"
            )

    out_path = f"iql_{env_name.replace('-', '_')}_seed{seed}.pt"
    torch.save(
        {
            "model": pi.state_dict(),   
            "q": q.state_dict(),
            "v": v.state_dict(),
            "env_name": env_name,
            "seed": seed,
            "s_mean": s_mean,
            "s_std": s_std,
            "algo": "iql",
            "tau": tau,
            "beta": beta,
        },
        out_path,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hopper-medium-replay-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--bs", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=3.0)
    args = parser.parse_args()

    main(
        env_name=args.env,
        seed=args.seed,
        steps=args.steps,
        bs=args.bs,
        gamma=args.gamma,
        tau=args.tau,
        beta=args.beta,
    )
