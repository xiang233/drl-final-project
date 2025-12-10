# ua/utils.py
import os, random, numpy as np, torch

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def mc_dropout_q_stats(critics, s, a, T: int = 10):
    """
    MC-dropout estimate of Q(s,a): mean and std over both
    dropout masks and ensemble members.

    critics: CriticEnsemble
    s: [B, obs_dim]
    a: [B, act_dim]
    returns: (Q_mean: [B], Q_std: [B])   (no gradients)
    """
    was_training = critics.training
    critics.train()  # ensure dropout active

    samples = []
    with torch.no_grad():
        for _ in range(T):
            # [B, K]
            q = critics.forward(s, a, keepdim=False)
            samples.append(q.unsqueeze(0))  # [1, B, K]

    if not was_training:
        critics.eval()

    Q_samples = torch.cat(samples, dim=0)   # [T, B, K]
    Q_mean = Q_samples.mean(dim=(0, 2))     # [B]
    Q_std  = Q_samples.std(dim=(0, 2))      # [B]
    return Q_mean, Q_std

def mc_dropout_q_mean(critics, s, a, T: int = 10):
    Q_mean, _ = mc_dropout_q_stats(critics, s, a, T)
    return Q_mean
