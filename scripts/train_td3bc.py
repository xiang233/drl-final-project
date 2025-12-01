# scripts/train_td3bc.py

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
         alpha_bc=2.5,  # The alpha from the paper (2.5 default)
         actor_lr=3e-4,
         critic_lr=3e-4,
         policy_delay=2,
         target_noise=0.2,
         noise_clip=0.5):
    """
    Plain TD3+BC baseline with Q-normalized BC coefficient (alpha / mean(|Q|)).
    """

    set_seed(seed)
    _, data = load_d4rl(env_name, seed)

    S = data["S"].astype(np.float32)
    A = data["A"].astype(np.float32)
    Rp = data.get("rewards", None)
    Sn = data.get("S_next", None)
    terminals = data.get("terminals", None)
    timeouts = data.get("timeouts", None)

    if Rp is None or Sn is None:
        raise ValueError("Dataset missing rewards or next states; required for TD3-style training.")

    if terminals is None:
        terminals = np.zeros((S.shape[0],), dtype=np.float32)
    if timeouts is None:
        timeouts = np.zeros((S.shape[0],), dtype=np.float32)

    # normalize states
    s_mean, s_std = data["s_mean"], data["s_std"]
    S = (S - s_mean) / (s_std + 1e-6)
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
    S_t = torch.from_numpy(S).to(device)
    A_t = torch.from_numpy(A).to(device)
    R_t = torch.from_numpy(Rp.squeeze().astype(np.float32)).to(device)
    Sn_t = torch.from_numpy(Sn).to(device)
    D_t = torch.from_numpy(done_mask.squeeze().astype(np.float32)).to(device)

    # initialize placeholder for lambda logging
    lam = torch.tensor(alpha_bc).to(device) 
    
    for t in range(1, steps + 1):
        idx = make_batch_indices(N, bs, rng)

        s = S_t[idx]
        a = A_t[idx]
        r = R_t[idx]
        s2 = Sn_t[idx]
        d = D_t[idx]

        # ----- critic update -----
        with torch.no_grad():
            a2 = actor_targ(s2)
            noise = torch.randn_like(a2) * target_noise
            noise = torch.clamp(noise, -noise_clip, noise_clip)
            a2_noisy = torch.clamp(a2 + noise, -1.0, 1.0)
            
            Qt = critics_t.forward(s2, a2_noisy, keepdim=True)
            Qt_min = torch.min(Qt, dim=0).values.squeeze(-1)
            y = r + gamma * (1.0 - d) * Qt_min

        # Get Q-values for the BC normalization term
        Qs = critics.forward(s, a, keepdim=True).squeeze(-1)  # [K,B]
        
        # Critic loss uses the mean Q over the ensemble
        critic_loss = torch.mean((Qs.transpose(0, 1) - y.unsqueeze(-1)) ** 2)

        crt_opt.zero_grad()
        critic_loss.backward()
        crt_opt.step()

        # ----- delayed actor update -----
        if t % policy_delay == 0:
            s_detach = s
            pi_s = actor(s_detach)

            # --- DYNAMIC BC COEFFICIENT CALCULATION ---
            with torch.no_grad():
                Q_bc = critics.forward(s_detach, a, keepdim=False).mean(dim=1)  # [B]
                Q_mean_abs = Q_bc.abs().mean()
                
                # TD3+BC normalization: lambda = alpha / mean(|Q(s,a)|)
                lam = alpha_bc / (Q_mean_abs + 1e-8)
                
            # Q_pi is Q(s, pi(s)), must be computed *outside* no_grad for policy gradient
            Q_pi_all = critics.forward(s_detach, pi_s, keepdim=False)  # [B,K]
            Q_pi_mean = Q_pi_all.mean(dim=1)  # [B]

            bc_term = torch.sum((pi_s - a) ** 2, dim=1) # [B]
            
            # FIX: Loss restructured to minimize [ -lambda * Q + ||pi - a||^2 ]
            actor_loss = -(lam * Q_pi_mean - bc_term).mean()

            act_opt.zero_grad()
            actor_loss.backward()
            act_opt.step()

            # soft target updates
            soft_update_(critics, critics_t, tau)
            soft_update_(actor, actor_targ, tau)

        if t % 1000 == 0:
            with torch.no_grad():
                pi_train = actor(S_t[:4096])
                Q_mean = critics.forward(S_t[:4096], pi_train, keepdim=False).mean().item()
                if t % policy_delay == 0:
                     log_lam = lam.item()
                else:
                     log_lam = float('nan')
            print(
                f"[{t}/{steps}] "
                f"critic_loss={critic_loss.item():.4f} "
                f"Q_mean@pi={Q_mean:.3f} "
                f"(lambda={log_lam:.4f})"
            )

    # save checkpoint
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
            alpha_bc=alpha_bc,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            policy_delay=policy_delay,
            steps=steps,
            bs=bs,
            target_noise=target_noise,
            noise_clip=noise_clip
        ),
        "algo": "td3bc",
    }
    out_path = f"td3bc_{env_name.replace('-', '_')}_seed{seed}.pt"
    torch.save(out, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hopper-medium-replay-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--bs", type=int, default=1024)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha_bc", type=float, default=2.5) # The alpha from the paper
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--policy_delay", type=int, default=2)
    parser.add_argument("--target_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    args = parser.parse_args()

    main(
        env_name=args.env,
        seed=args.seed,
        steps=args.steps,
        bs=args.bs,
        K=args.K,
        gamma=args.gamma,
        tau=args.tau,
        alpha_bc=args.alpha_bc,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        policy_delay=args.policy_delay,
        target_noise=args.target_noise,
        noise_clip=args.noise_clip,
    )