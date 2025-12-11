# scripts/train_iql_u.py
#
# IQL-UA with sigma_Q-based scaling of the advantage temperature.
#
# Policy extraction:
#   weights(s,a) = exp( beta_eff(s) * norm_adv(s,a) )
#
# where:
#   norm_adv is the z-scored advantage Q(s,a) - V(s),
#   beta_eff(s) = beta * g(sigma_rel(s)),
#   sigma_rel(s) = sigma_Q(s, pi(s)) / E[sigma_Q],
#   g is decreasing: high sigma_rel -> g < 1 (more conservative),
#                    low  sigma_rel -> g > 1 (more optimistic).
#
# This mirrors the TD3+BC sigma_Q scaling of lambda, but for the
# IQL policy temperature instead of the TD3+BC Q-weight.

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from ua.datasets import load_d4rl
from ua.utils import set_seed
from ua.nets import CriticEnsemble  # reuse your ensemble implementation


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


class VNetwork(nn.Module):
    def __init__(self, state_dim, hid=256):
        super().__init__()
        self.net = MLP(state_dim, 1, hid=hid)

    def forward(self, s):
        return self.net(s).squeeze(-1)  # [B]


class PolicyNetwork(nn.Module):
    """
    Deterministic tanh policy: a = tanh(f(s)).
    We train this via advantage-weighted regression (IQL style).
    """
    def __init__(self, state_dim, action_dim, hid=256):
        super().__init__()
        self.backbone = MLP(state_dim, action_dim, hid=hid)

    def forward(self, s):
        return torch.tanh(self.backbone(s))


def expectile_loss(diff, tau):
    """
    diff = target - value
    L_tau(u) = |tau - 1(u < 0)| * u^2
    """
    weight = torch.where(diff >= 0, tau, 1.0 - tau)
    return (weight * diff.pow(2)).mean()


def make_batch_indices(N, bs, rng):
    return rng.integers(0, N, size=bs, dtype=np.int64)


def main(env_name="hopper-medium-replay-v2",
         seed=0,
         steps=200000,
         bs=1024,
         K=4,
         gamma=0.99,
         tau=0.7,
         beta=3.0,
         v_tau_ema=0.005,
         # uncertainty scaling hyperparams (sigma_Q-based)
         unc_alpha=1.0,      # α_uq: how strongly uncertainty changes beta
         w_min=0.5,          # clamp range for g(σ_rel)
         w_max=2.0,
         critic_lr=3e-4,
         v_lr=3e-4,
         actor_lr=3e-4):

    """
    IQL-UA (sigma_Q version):
      - Q: CriticEnsemble with K heads
      - V: expectile regression to ensemble-mean Q(s,a)
      - Policy: advantage-weighted regression where the temperature beta is
        scaled per-state by a monotone function of ensemble std Q(s, pi(s)).

      sigma_Q(s) = std_k Q_k(s, pi(s))
      sigma_rel(s) = sigma_Q(s) / E[sigma_Q]
      g(sigma_rel) = exp(-unc_alpha * (sigma_rel - 1)) in [w_min, w_max]
      beta_eff(s) = beta * g(sigma_rel)
      weights(s,a) = exp(beta_eff(s) * norm_adv(s,a))
    """

    set_seed(seed)
    _, data = load_d4rl(env_name, seed)

    # ----- load + normalize data -----
    S = data["S"].astype(np.float32)
    A = data["A"].astype(np.float32)
    Rp = data.get("rewards", None)
    Sn = data.get("S_next", None)
    terminals = data.get("terminals", None)
    timeouts = data.get("timeouts", None)

    if Rp is None or Sn is None:
        raise ValueError("Dataset missing next_observations or rewards; required for IQL training.")
    if terminals is None:
        terminals = np.zeros((S.shape[0],), dtype=np.float32)
    if timeouts is None:
        timeouts = np.zeros((S.shape[0],), dtype=np.float32)

    s_mean, s_std = data["s_mean"], data["s_std"]
    S  = (S  - s_mean) / (s_std + 1e-6)
    Sn = (Sn - s_mean) / (s_std + 1e-6)

    N, obs_dim = S.shape
    act_dim = A.shape[1]

    done_mask = np.clip(terminals + timeouts, 0, 1).astype(np.float32)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # ----- networks -----
    critics = CriticEnsemble(obs_dim, act_dim, K=K).to(device)
    v = VNetwork(obs_dim).to(device)
    pi = PolicyNetwork(obs_dim, act_dim).to(device)

    # target V for stable TD
    v_targ = copy.deepcopy(v).to(device)
    for p in v_targ.parameters():
        p.requires_grad = False

    crt_opt = optim.Adam(critics.parameters(), lr=critic_lr)
    v_opt = optim.Adam(v.parameters(), lr=v_lr)
    pi_opt = optim.Adam(pi.parameters(), lr=actor_lr)

    # pre-torch tensors for fast sampling (index in torch)
    S_t  = torch.from_numpy(S).to(device)
    A_t  = torch.from_numpy(A).to(device)
    R_t  = torch.from_numpy(Rp.squeeze().astype(np.float32)).to(device)
    Sn_t = torch.from_numpy(Sn).to(device)
    D_t  = torch.from_numpy(done_mask.squeeze().astype(np.float32)).to(device)

    rng = np.random.default_rng(seed)

    def soft_update_v_target():
        with torch.no_grad():
            for p, p_t in zip(v.parameters(), v_targ.parameters()):
                p_t.data.mul_(1.0 - v_tau_ema)
                p_t.data.add_(v_tau_ema * p.data)

    # logging helpers
    beta_eff_mean_val = torch.tensor(beta, device=device)
    g_mean_val = torch.tensor(1.0, device=device)
    sigma_q_mean_val = torch.tensor(0.0, device=device)

    # ----- training loop -----
    for t in range(1, steps + 1):
        idx = make_batch_indices(N, bs, rng)
        s  = S_t[idx]
        a  = A_t[idx]
        r  = R_t[idx]
        s2 = Sn_t[idx]
        d  = D_t[idx]

        # 1) V update: expectile regression to ensemble-mean Q(s,a)
        with torch.no_grad():
            Q_all = critics.forward(s, a, keepdim=False)  # [B, K]
            Q_mean = Q_all.mean(dim=1)                   # [B]
        v_s = v(s)
        v_loss = expectile_loss(Q_mean - v_s, tau)

        v_opt.zero_grad()
        v_loss.backward()
        v_opt.step()

        # 2) Q update: TD to r + gamma * V_target(s')
        with torch.no_grad():
            v_s2 = v_targ(s2)
            target = r + gamma * (1.0 - d) * v_s2  # [B]

        Q_all = critics.forward(s, a, keepdim=True).squeeze(-1)  # [K, B]
        td_err = Q_all.transpose(0, 1) - target.unsqueeze(-1)    # [B, K]
        critic_loss = 0.5 * (td_err.pow(2).mean())

        crt_opt.zero_grad()
        critic_loss.backward()
        crt_opt.step()

        # 3) Policy update: advantage-weighted regression with sigma_Q-scaled beta
        with torch.no_grad():
            pi_s = pi(s)
            Q_pi_all = critics.forward(s, pi_s, keepdim=False)  # [B, K]
            Q_pi_mean = Q_pi_all.mean(dim=1)                    # [B]
            Q_pi_std  = Q_pi_all.std(dim=1)                     # [B]

            v_s = v(s)
            adv = Q_pi_mean - v_s

            adv_mean = adv.mean()
            adv_std  = adv.std() + 1e-6
            norm_adv = (adv - adv_mean) / adv_std

            # relative uncertainty (dimensionless)
            sigma_q_mean = Q_pi_std.mean()
            sigma_rel = Q_pi_std / (sigma_q_mean + 1e-8)

            # g(σ_rel): >1 when σ_rel < 1, <1 when σ_rel > 1
            g = torch.exp(-unc_alpha * (sigma_rel - 1.0))
            g = torch.clamp(g, min=w_min, max=w_max)

            beta_eff = beta * g   # [B]

            # IQL-style advantage weights with sigma_Q-scaled beta
            weights = torch.exp(beta_eff * norm_adv).clamp(max=100.0)

        pi_s = pi(s)
        mse = (pi_s - a).pow(2).sum(dim=1)   # [B]
        pi_loss = (weights * mse).mean()

        pi_opt.zero_grad()
        pi_loss.backward()
        pi_opt.step()

        # 4) soft-update V target
        soft_update_v_target()

        # update logging stats
        beta_eff_mean_val = beta_eff.mean().detach()
        g_mean_val = g.mean().detach()
        sigma_q_mean_val = sigma_q_mean.detach()

        if t % 1000 == 0:
            with torch.no_grad():
                pi_train = pi(S_t[:4096])
                Q_mean_log = critics.forward(S_t[:4096], pi_train, keepdim=False).mean().item()
            print(
                f"[{t}/{steps}] "
                f"critic_loss={critic_loss.item():.4f} "
                f"v_loss={v_loss.item():.4f} "
                f"pi_loss={pi_loss.item():.4f} "
                f"Q_mean@pi={Q_mean_log:.3f} "
                f"sigma_Q_mean={sigma_q_mean_val.item():.3f} "
                f"beta_eff_mean={beta_eff_mean_val.item():.3f} "
                f"g_mean={g_mean_val.item():.3f}"
            )

    # ----- save checkpoint -----
    out = {
        "env_name": env_name,
        "seed": seed,
        "K": K,
        "actor": pi.state_dict(),
        "critics": critics.state_dict(),
        "v": v.state_dict(),
        "s_mean": s_mean,
        "s_std": s_std,
        "cfg": dict(
            gamma=gamma,
            tau=tau,
            beta=beta,
            v_tau_ema=v_tau_ema,
            unc_alpha=unc_alpha,
            w_min=w_min,
            w_max=w_max,
            critic_lr=critic_lr,
            v_lr=v_lr,
            actor_lr=actor_lr,
            steps=steps,
            bs=bs,
        ),
        "algo": "iql_u_sigmaQ",   # distinguish in metadata; filename still starts with iql_u_
    }

    # encode unc_alpha in the filename so different alphas don't overwrite each other
    alpha_str = f"{unc_alpha}".rstrip("0").rstrip(".")  # 1.0 -> "1", 0.30 -> "0.3"
    env_slug = env_name.replace("-", "_")
    out_stem = f"iql_u_alpha{alpha_str}_{env_slug}_seed{seed}"
    out_path = f"{out_stem}.pt"

    torch.save(out, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="hopper-medium-replay-v2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--bs", type=int, default=1024)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.7)
    p.add_argument("--beta", type=float, default=3.0)
    p.add_argument("--v_tau_ema", type=float, default=0.005)
    p.add_argument("--unc_alpha", type=float, default=1.0)
    p.add_argument("--w_min", type=float, default=0.5)
    p.add_argument("--w_max", type=float, default=2.0)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--v_lr", type=float, default=3e-4)
    p.add_argument("--actor_lr", type=float, default=3e-4)
    args = p.parse_args()

    main(
        env_name=args.env,
        seed=args.seed,
        steps=args.steps,
        bs=args.bs,
        K=args.K,
        gamma=args.gamma,
        tau=args.tau,
        beta=args.beta,
        v_tau_ema=args.v_tau_ema,
        unc_alpha=args.unc_alpha,
        w_min=args.w_min,
        w_max=args.w_max,
        critic_lr=args.critic_lr,
        v_lr=args.v_lr,
        actor_lr=args.actor_lr,
    )