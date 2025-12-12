# scripts/train_td3bc_u_conf.py
#
# Confidence-Weighted TD3+BC.
#
# Actor loss:
#   L_pi = - E[ λ_conf(s) * Q(s, π(s)) - ||π(s) - a||^2 ]
#
# where:
#   λ_base = base_w / E_s[ |Q(s, a_dataset)| ]   (Standard TD3+BC normalization)
#   w_conf(σ) = 1 / (1 + β * σ_Q)                 (Inverse Variance Weighting)
#   λ_conf(s) = λ_base * w_conf(σ)
#
# Mechanism:
#   - High Uncertainty (AntMaze) -> w_conf -> 0 -> λ_conf -> 0 -> Pure BC (Safety)
#   - Low Uncertainty (Walker2d) -> w_conf -> 1 -> λ_conf -> λ_base -> Full Exploitation

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ua.datasets import load_d4rl
from ua.utils import set_seed
from ua.nets import Actor, CriticEnsemble, soft_update_


def make_batch_indices(N, bs, rng):
    return rng.integers(0, N, size=bs, dtype=np.int64)


def main(env_name="hopper-medium-replay-v2",
         seed=0,
         steps=50000,
         bs=1024,
         K=4,
         gamma=0.99,
         tau=0.005,
         base_w=2.5,        # TD3+BC base weight
         beta_uq=5.0,       # Sensitivity to uncertainty (Higher = Stronger filtering)
         w_min=0.05,        # Minimum trust (prevents total collapse of Q-signal)
         actor_lr=3e-4,
         critic_lr=3e-4,
         policy_delay=2,
         target_noise=0.2,
         noise_clip=0.5):

    set_seed(seed)
    _, data = load_d4rl(env_name, seed)

    S  = data["S"].astype(np.float32)
    A  = data["A"].astype(np.float32)
    Rp = data.get("rewards", None)
    Sn = data.get("S_next", None)
    terminals = data.get("terminals", None)
    timeouts  = data.get("timeouts", None)

    if Rp is None or Sn is None:
        raise ValueError("Dataset missing rewards or next states; required for training.")
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

    rng = np.random.default_rng(seed)
    done_mask = np.clip(terminals + timeouts, 0, 1).astype(np.float32)

    # pre-torch tensors
    S_t  = torch.from_numpy(S).to(device)
    A_t  = torch.from_numpy(A).to(device)
    R_t  = torch.from_numpy(Rp.squeeze().astype(np.float32)).to(device)
    Sn_t = torch.from_numpy(Sn).to(device)
    D_t  = torch.from_numpy(done_mask.squeeze().astype(np.float32)).to(device)

    # logging placeholders
    lam_base_val = torch.tensor(base_w, device=device)
    lam_conf_mean_val = torch.tensor(base_w, device=device)
    w_conf_mean_val = torch.tensor(1.0, device=device)

    for t in range(1, steps + 1):
        idx = make_batch_indices(N, bs, rng)
        s  = S_t[idx]
        a  = A_t[idx]
        r  = R_t[idx]
        s2 = Sn_t[idx]
        d  = D_t[idx]

        # ---------- Critic update ----------
        with torch.no_grad():
            a2 = actor_targ(s2)
            noise = torch.randn_like(a2) * target_noise
            noise = torch.clamp(noise, -noise_clip, noise_clip)
            a2_noisy = torch.clamp(a2 + noise, -1.0, 1.0)

            Qt = critics_t.forward(s2, a2_noisy, keepdim=True)  # [K,B,1]
            Qt_min = torch.min(Qt, dim=0).values.squeeze(-1)    # [B]
            y = r + gamma * (1.0 - d) * Qt_min                  # [B]

        Qs = critics.forward(s, a, keepdim=True).squeeze(-1)    # [K,B]
        critic_loss = ((Qs.transpose(0, 1) - y.unsqueeze(-1)) ** 2).mean()

        crt_opt.zero_grad()
        critic_loss.backward()
        crt_opt.step()

        # ---------- Actor update (delayed) ----------
        if t % policy_delay == 0:
            s_detach = s
            pi_s = actor(s_detach)  # [B, act_dim]

            # 1. Get Q-values and absolute uncertainty
            Q_pi_all = critics.forward(s_detach, pi_s, keepdim=False)  # [B,K]
            Q_pi_mean = Q_pi_all.mean(dim=1)                           # [B]
            sigma_q = Q_pi_all.std(dim=1, unbiased=False)              # [B]

            # 2. Calculate Standard TD3+BC Lambda (Base Trust)
            with torch.no_grad():
                Q_bc = critics.forward(s_detach, a, keepdim=False).mean(dim=1)
                Q_mean_abs = Q_bc.abs().mean()
                lam_base = base_w / (Q_mean_abs + 1e-8)

                # 3. Calculate Confidence Weight (Inverse Variance)
                # w_conf = 1 / (1 + beta * sigma)
                # This has NO dependency on batch means. High sigma -> Low weight always.
                w_conf = 1.0 / (1.0 + beta_uq * sigma_q)
                
                # Optional: Clamp to ensure minimal Q-signal remains (e.g. 5%)
                w_conf = torch.clamp(w_conf, min=w_min, max=1.0)

                # 4. Effective Lambda
                lam_conf = lam_base * w_conf  # [B]

            # 5. Actor Loss
            # Maximize (lam_conf * Q) - BC
            bc_term = ((pi_s - a) ** 2).sum(dim=1)
            actor_loss = -(lam_conf * Q_pi_mean - bc_term).mean()

            act_opt.zero_grad()
            actor_loss.backward()
            act_opt.step()

            # soft targets
            soft_update_(critics, critics_t, tau)
            soft_update_(actor, actor_targ, tau)

            # logging
            lam_base_val = lam_base.detach()
            lam_conf_mean_val = lam_conf.mean().detach()
            w_conf_mean_val = w_conf.mean().detach()

        # ---------- Logging ----------
        if t % 1000 == 0:
            with torch.no_grad():
                pi_train = actor(S_t[:4096])
                Q_mean_log = critics.forward(S_t[:4096], pi_train, keepdim=False).mean().item()

            print(
                f"[{t}/{steps}] "
                f"critic_loss={critic_loss.item():.4f} "
                f"Q_mean@pi={Q_mean_log:.3f} "
                f"(lam_base={lam_base_val.item():.4f}, "
                f"lam_conf_mean={lam_conf_mean_val.item():.4f}, "
                f"w_conf_mean={w_conf_mean_val.item():.3f})"
            )

    # ---------- Save checkpoint ----------
    out = {
        "env_name": env_name,
        "seed": seed,
        "K": K,
        "actor": actor.state_dict(),
        "critics": critics.state_dict(),
        "s_mean": s_mean,
        "s_std": s_std,
        "cfg": dict(
            gamma=gamma,
            tau=tau,
            base_w=base_w,
            beta_uq=beta_uq,
            w_min=w_min,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            policy_delay=policy_delay,
            steps=steps,
            bs=bs,
            target_noise=target_noise,
            noise_clip=noise_clip,
        ),
        "algo": "td3bc_u_conf_weighted",
    }
    out_path = f"td3bc_u_{env_name.replace('-', '_')}_seed{seed}.pt"
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
    p.add_argument("--base_w", type=float, default=2.5)
    
    # Confidence Hyperparameters
    p.add_argument("--beta_uq", type=float, default=5.0)  # Sensitivity. Try 5.0 or 10.0
    p.add_argument("--w_min", type=float, default=0.05)   # Min trust. Try 0.05 or 0.1
    
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--policy_delay", type=int, default=2)
    p.add_argument("--target_noise", type=float, default=0.2)
    p.add_argument("--noise_clip", type=float, default=0.5)
    args = p.parse_args()

    main(
        env_name=args.env,
        seed=args.seed,
        steps=args.steps,
        bs=args.bs,
        K=args.K,
        gamma=args.gamma,
        tau=args.tau,
        base_w=args.base_w,
        beta_uq=args.beta_uq,
        w_min=args.w_min,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        policy_delay=args.policy_delay,
        target_noise=args.target_noise,
        noise_clip=args.noise_clip,
    )