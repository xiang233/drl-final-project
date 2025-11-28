# scripts/train_td3bc_u.py
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ua.datasets import load_d4rl
from ua.utils import set_seed
from ua.nets import Actor, CriticEnsemble, soft_update_

def make_batch_indices(N, bs, rng):
    idx = rng.integers(0, N, size=bs, dtype=np.int64)
    return idx

def main(env_name="hopper-medium-replay-v2",
         seed=0,
         steps=50000,
         bs=1024,
         K=4,
         gamma=0.99,
         tau=0.005,
         base_w=1.0,
         w_min=0.1,
         w_max=5.0,
         actor_lr=3e-4,
         critic_lr=3e-4,
         policy_delay=2):
    """
    TD3+BC-UA (uncertainty-adaptive BC weight).
    Uncertainty = std over Q-ensemble at Q(s, pi(s)); per-batch z-score -> per-sample weight.
    """
    set_seed(seed)
    _, data = load_d4rl(env_name, seed)
    S  = data["S"].astype(np.float32)
    A  = data["A"].astype(np.float32)
    Rp = data.get("rewards", None)
    Sn = data.get("S_next", None)
    terminals = data.get("terminals", None)
    timeouts  = data.get("timeouts", None)

    if Rp is None or Sn is None:
        raise ValueError("Dataset missing next_observations or rewards; required for TD3-style training.")
    if terminals is None:
        terminals = np.zeros((S.shape[0],), dtype=np.float32)
    if timeouts is None:
        timeouts = np.zeros((S.shape[0],), dtype=np.float32)

    # normalize states
    s_mean, s_std = data["s_mean"], data["s_std"]
    S  = (S  - s_mean) / (s_std + 1e-6)
    Sn = (Sn - s_mean) / (s_std + 1e-6)

    N, obs_dim = S.shape
    act_dim = A.shape[1]

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    actor = Actor(obs_dim, act_dim).to(device)
    actor_targ = Actor(obs_dim, act_dim).to(device)
    actor_targ.load_state_dict(actor.state_dict())

    critics = CriticEnsemble(obs_dim, act_dim, K=K).to(device)
    critics_t = critics.clone_targets().to(device)
    critics_t.load_state_dict(critics.state_dict())

    act_opt = optim.Adam(actor.parameters(), lr=actor_lr)
    crt_opt = optim.Adam(critics.parameters(), lr=critic_lr)
    mse = nn.MSELoss(reduction='mean')

    # numpy views for fast sampling
    rng = np.random.default_rng(seed)
    done_mask = np.clip(terminals + timeouts, 0, 1).astype(np.float32)

    # pre-torch tensors to avoid repeated transfers (we'll index in torch)
    S_t  = torch.from_numpy(S).to(device)
    A_t  = torch.from_numpy(A).to(device)
    R_t  = torch.from_numpy(Rp.squeeze().astype(np.float32)).to(device)
    Sn_t = torch.from_numpy(Sn).to(device)
    D_t  = torch.from_numpy(done_mask.squeeze()).to(device)

    # training loop
    for t in range(1, steps + 1):
        idx = make_batch_indices(N, bs, rng)
        s  = S_t[idx]; a = A_t[idx]; r = R_t[idx]; s2 = Sn_t[idx]; d = D_t[idx]

        with torch.no_grad():
            # target policy smoothing (small noise, TD3-style)
            a2 = actor_targ(s2)
            # compute conservative target with min over ensemble
            Qt = critics_t.forward(s2, a2, keepdim=True)  # [K,B,1]
            Qt_min = torch.min(Qt, dim=0).values.squeeze(-1)  # [B]
            y = r + gamma * (1.0 - d) * Qt_min

        # ----- critic update -----
        Qs = critics.forward(s, a, keepdim=True).squeeze(-1)  # [K,B]
        critic_loss = torch.mean((Qs.transpose(0,1) - y.unsqueeze(-1))**2)  # average over ensemble and batch
        crt_opt.zero_grad()
        critic_loss.backward()
        crt_opt.step()

        # ----- delayed actor (TD3) -----
        if t % policy_delay == 0:
            s_detach = s  # same batch
            pi_s = actor(s_detach)

            # uncertainty = std over ensemble Q(s, pi(s)) per sample
            with torch.no_grad():
                Q_pi = critics.forward(s_detach, pi_s, keepdim=False)  # [B,K]
                uq = torch.std(Q_pi, dim=1)  # [B]
                # batch z-score
                mu = torch.mean(uq)
                sd = torch.std(uq) + 1e-8
                z = (uq - mu) / sd
                w = torch.clamp(base_w * (1.0 + z), min=w_min, max=w_max)  # higher w when uncertainty higher

            # TD3+BC actor loss: -Q_mean(s, pi(s)) + w * ||pi(s) - a||^2 (per-sample)
            Q_pi_mean = torch.mean(Q_pi, dim=1)  # [B]
            imitation_err = torch.mean(w * torch.sum((pi_s - a)**2, dim=1))
            actor_loss = -Q_pi_mean.mean() + imitation_err

            act_opt.zero_grad()
            actor_loss.backward()
            act_opt.step()

            # soft target updates
            soft_update_(critics, critics_t, tau)
            soft_update_(actor, actor_targ, tau)

        if t % 1000 == 0:
            # lightweight logging
            with torch.no_grad():
                pi_train = actor(S_t[:4096])
                Q_mean = critics.forward(S_t[:4096], pi_train, keepdim=False).mean().item()
            print(f"[{t}/{steps}] critic_loss={critic_loss.item():.4f} "
                  f"Q_mean@pi={Q_mean:.3f} "
                  f"(w: mean={w.mean().item():.3f}, min={w.min().item():.3f}, max={w.max().item():.3f})")

    # save checkpoint
    out = {
        "env_name": env_name,
        "seed": seed,
        "K": K,
        "actor": actor.state_dict(),
        "critics": critics.state_dict(),
        "s_mean": s_mean, "s_std": s_std,
        "cfg": dict(gamma=gamma, tau=tau, base_w=base_w, w_min=w_min, w_max=w_max,
                    actor_lr=actor_lr, critic_lr=critic_lr, policy_delay=policy_delay, steps=steps, bs=bs)
    }
    out_path = f"td3bc_u_{env_name.replace('-','_')}_seed{seed}.pt"
    torch.save(out, out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="hopper-medium-replay-v2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--bs", type=int, default=1024)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--base_w", type=float, default=1.0)
    p.add_argument("--w_min", type=float, default=0.1)
    p.add_argument("--w_max", type=float, default=5.0)
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--policy_delay", type=int, default=2)
    args = p.parse_args()
    main(env_name=args.env, seed=args.seed, steps=args.steps, bs=args.bs, K=args.K,
         gamma=args.gamma, tau=args.tau, base_w=args.base_w, w_min=args.w_min, w_max=args.w_max,
         actor_lr=args.actor_lr, critic_lr=args.critic_lr, policy_delay=args.policy_delay)
