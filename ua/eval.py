
import numpy as np

def evaluate(env, policy, episodes=10, seed=42):
    if env is None:
        return float("nan"), float("nan")
    returns = []
    for ep in range(episodes):
        obs, done, ret = env.reset(seed=seed+ep), False, 0.0
        while not done:
            a = policy.act(obs, eval_mode=True)
            obs, r, done, _ = env.step(a)
            ret += r
        returns.append(ret)
    raw = np.mean(returns)
    try:
        norm = env.get_normalized_score(raw) * 100.0
    except Exception:
        norm = raw
    return raw, norm
