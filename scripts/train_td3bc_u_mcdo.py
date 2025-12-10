# scripts/train_td3bc_u_mcdo.py
#
# TD3+BC-UA with MC-dropout-based pessimistic Q(s,pi(s)).

import argparse
import numpy as np
import torch
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
         base_w=2.5,       # alpha in TD3+BC: lambda = base_w / E|Q|
         alpha_uq=1.0,     # strength of uncertainty penalty on Q
         actor_lr=3e-4,
         critic_lr=3e-4,
         policy_delay=2,
         target_noise=0.2,
         noise_clip=0.5,
         dropout_p=0.1,
         T_uq=5):
    """
    TD3+BC-UA with Pessimistic Q and MC-dropout uncertainty.

    Actor objective:
        L_pi = - E[ lambda * ( Q_mean(s, pi(s)) - alpha_uq * sigma_Q(s, pi(s)) )
                     - ||pi(s) - a||^2 ]

    where:
      - lambda = base_w / E_s[ |Q(s, a_dataset)| ]
      - sigma_Q is std over Monte Carlo dropout + ensemble at (s, pi(s))
    """

    set_seed(seed)
    _, data = load_d4rl(env_name, seed)

    S = data["S"].astype(np.float32)
    A = data["A"].astype(np.float32)
    Rp = data["rewards"].astype(np.float32)
    Sn = data["S_next"].astype(np.float32)
    terminals = data.get("terminals", None)
    timeouts = data.get("timeouts", None)

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

    critics = CriticEnsemble(
        obs_dim, act_dim, K=K, hid=256, dropout_p=dropout_p
    ).to(device)
    critics_t = critics.clone_targets().to(device)
    critics_t.load_state_dict(critics.state_dict())
    critics_t.eval()  # deterministic targets

    act_opt = optim.Adam(actor.parameters(), lr=actor_lr)
    crt_opt = optim.Adam(critics.parameters(), lr=critic_lr)

    rng = np.random.default_rng(seed)
    done_mask = np.clip(terminals + timeouts, 0, 1).astype(np.float32)

    S_t = torch.from_numpy(S).to(device)
    A_t = torch.from_numpy(A).to(device)
    R_t = torch.from_numpy(Rp.squeeze()).to(device)
    Sn_t = torch.from_numpy(Sn).to(device)
    D_t = torch.from_numpy(done_mask.squeeze()).to(device)

    # logging placeholders
    lam_val = torch.tensor(base_w, device=device)
    sigma_Q_mean_val = torch.tensor(0.0, device=device)
    Q_pess_mean_val = torch.tensor(0.0, device=device)

    for t in range(1, steps + 1):
        idx = make_batch_indices(N, bs, rng)
        s = S_t[idx]
        a = A_t[idx]
        r = R_t[idx]
        s2 = Sn_t[idx]
        d = D_t[idx]

        # ---------- Critic update ----------
        with torch.no_grad():
            # target policy with smoothing (TD3)
            a2 = actor_targ(s2)
            noise = torch.randn_like(a2) * target_noise
            noise = torch.clamp(noise, -noise_clip, noise_clip)
            a2_noisy = torch.clamp(a2 + noise, -1.0, 1.0)

            Qt = critics_t.forward(s2, a2_noisy, keepdim=True)  # [K,B,1]
            Qt_min = torch.min(Qt, dim=0).values.squeeze(-1)    # [B]
            y = r + gamma * (1.0 - d) * Qt_min                  # [B]

        Qs = critics.forward(s, a, keepdim=True).squeeze(-1)    # [K,B]
        critic_loss = torch.mean((Qs.transpose(0, 1) - y.unsqueeze(-1)) ** 2)

        crt_opt.zero_grad()
        critic_loss.backward()
        crt_opt.step()

        # ---------- Actor update (delayed) ----------
        if t % policy_delay == 0:
            s_detach = s
            pi_s = actor(s_detach)  # [B, act_dim]

            # MC-dropout over critic ensemble for Q(s, pi(s)) WITH grad
            Q_samples = []
            critics.train()  # enable dropout for MC sampling
            for _ in range(T_uq):
                Q_pi_all = critics.forward(s_detach, pi_s, keepdim=False)  # [B,K]
                Q_samples.append(Q_pi_all.unsqueeze(0))  # [1,B,K]
            critics.train()  # keep train mode for critic regularization

            Q_samples = torch.cat(Q_samples, dim=0)  # [T,B,K]
            Q_pi_mean = Q_samples.mean(dim=(0, 2))   # [B]
            sigma_Q = Q_samples.std(dim=(0, 2))      # [B]

            # TD3+BC lambda normalization (no grad)
            with torch.no_grad():
                critics.eval()
                Q_bc = critics.forward(s_detach, a, keepdim=False).mean(dim=1)  # [B]
                critics.train()
                Q_mean_abs = Q_bc.abs().mean()
                lam = base_w / (Q_mean_abs + 1e-8)

            # Pessimistic Q
            Q_pess = Q_pi_mean - alpha_uq * sigma_Q  # [B]

            # BC term (no UA weight)
            bc_term = torch.sum((pi_s - a) ** 2, dim=1)  # [B]

            # Actor loss: minimize [ -lambda * Q_pess + ||pi - a||^2 ]
            actor_loss = -(lam * Q_pess - bc_term).mean()

            act_opt.zero_grad()
            actor_loss.backward()
            act_opt.step()

            # soft targets
            soft_update_(critics, critics_t, tau)
            soft_update_(actor, actor_targ, tau)

            # update logging scalars
            lam_val = lam.detach()
            sigma_Q_mean_val = sigma_Q.mean().detach()
            Q_pess_mean_val = Q_pess.mean().detach()

        # ---------- Logging ----------
        if t % 1000 == 0:
            with torch.no_grad():
                critics.eval()
                pi_train = actor(S_t[:4096])
                Q_mean_log = critics.forward(S_t[:4096], pi_train, keepdim=False).mean().item()
                critics.train()

            print(
                f"[{t}/{steps}] "
                f"critic_loss={critic_loss.item():.4f} "
                f"Q_mean@pi={Q_mean_log:.3f} "
                f"(lambda={lam_val.item():.4f}, "
                f"sigma_Q={sigma_Q_mean_val.item():.3f}, "
                f"Q_pess={Q_pess_mean_val.item():.3f})"
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
            alpha_uq=alpha_uq,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            policy_delay=policy_delay,
            steps=steps,
            bs=bs,
            target_noise=target_noise,
            noise_clip=noise_clip,
            dropout_p=dropout_p,
            T_uq=T_uq,
        ),
        "algo": "td3bc_u_pessimistic_mcdo",
    }
    out_path = f"td3bc_u_mcdo_{env_name.replace('-', '_')}_seed{seed}.pt"
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
    p.add_argument("--alpha_uq", type=float, default=1.0)
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--policy_delay", type=int, default=2)
    p.add_argument("--target_noise", type=float, default=0.2)
    p.add_argument("--noise_clip", type=float, default=0.5)
    p.add_argument("--dropout_p", type=float, default=0.1)
    p.add_argument("--T_uq", type=int, default=5)
    args = p.parse_args()

    main(env_name=args.env,
         seed=args.seed,
         steps=args.steps,
         bs=args.bs,
         K=args.K,
         gamma=args.gamma,
         tau=args.tau,
         base_w=args.base_w,
         alpha_uq=args.alpha_uq,
         actor_lr=args.actor_lr,
         critic_lr=args.critic_lr,
         policy_delay=args.policy_delay,
         target_noise=args.target_noise,
         noise_clip=args.noise_clip,
         dropout_p=args.dropout_p,
         T_uq=args.T_uq)
