import argparse
import os
import numpy as np
import torch
import gym

from ua.datasets import load_d4rl
from ua.utils import set_seed
from scripts.estimate_return import load_policy


def make_gym_env(env_name: str):
    if env_name.startswith("hopper-"):
        gym_id = "Hopper-v3"
    elif env_name.startswith("walker2d-"):
        gym_id = "Walker2d-v3"
    else:
        gym_id = env_name
    return gym.make(gym_id)


def eval_policy_in_env(
    ckpt_path: str,
    env_name: str,
    episodes: int = 10,
    render: bool = False,
    seed: int | None = None,
):
    # Load D4RL 
    if seed is None:
        seed = 0
    set_seed(seed)
    _, data = load_d4rl(env_name, seed)

    s_mean = data["s_mean"]
    s_std = data["s_std"]
    s_dim = data["S"].shape[1]
    a_dim = data["A"].shape[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load policy 
    pi = load_policy(ckpt_path, s_dim, a_dim, device)
    pi.eval()

    env = make_gym_env(env_name)
    env.seed(seed)

    returns = []

    for ep in range(episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs, _info = obs
        done = False
        ep_ret = 0.0

        while not done:
            s = (obs - s_mean) / (s_std + 1e-6)
            s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(device)

            with torch.no_grad():
                a = pi(s_t).cpu().numpy()[0]

            obs, reward, done, info = env.step(a)
            ep_ret += float(reward)

            if render:
                env.render()

        returns.append(ep_ret)
        print(f"[EP {ep}] return = {ep_ret:.2f}")

    env.close()

    returns = np.array(returns, dtype=np.float32)

    normalized = None
    try:
        import d4rl  # noqa: F401

        env2 = gym.make(env_name)
        normalized = np.array(
            [env2.get_normalized_score(r) * 100.0 for r in returns],
            dtype=np.float32,
        )
        env2.close()
    except Exception:
        pass

    return returns, normalized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, required=True,
                    help="D4RL env name, e.g. hopper-medium-replay-v2")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to checkpoint .pt")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw, norm = eval_policy_in_env(
        ckpt_path=args.ckpt,
        env_name=args.env,
        episodes=args.episodes,
        render=args.render,
        seed=args.seed,
    )

    print("\n=== Raw returns ===")
    print("Returns:", raw)
    print(f"Mean = {raw.mean():.2f}  Std = {raw.std():.2f}")

    if norm is not None:
        print("\n=== D4RL-normalized scores (if d4rl available) ===")
        print("Scores:", norm)
        print(f"Mean = {norm.mean():.2f}  Std = {norm.std():.2f}")
    else:
        print("\n[INFO] d4rl not available or env not registered; "
              "skipping normalized scores.")


if __name__ == "__main__":
    main()
