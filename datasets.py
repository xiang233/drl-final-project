import gym
import d4rl              
import numpy as np

def load_d4rl(name: str, seed: int = 0):
    env = gym.make(name)
    env.seed(seed)
    dataset = env.get_dataset()
    S = dataset["observations"].astype(np.float32)
    A = dataset["actions"].astype(np.float32)
    R = dataset["rewards"].astype(np.float32)
    S2 = dataset["next_observations"].astype(np.float32)
    D = dataset["terminals"].astype(np.float32)
    # z-score stats (state only; actions often not normalized in MuJoCo)
    s_mean, s_std = S.mean(0), S.std(0) + 1e-6
    return env, {"S":S,"A":A,"R":R,"S2":S2,"D":D, "s_mean":s_mean, "s_std":s_std}

def z_norm_states(S, s_mean, s_std):
    return (S - s_mean) / s_std
