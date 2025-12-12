
import os, random, numpy as np, torch

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def mc_dropout_q_stats(critics, s, a, T: int = 10):
    was_training = critics.training
    critics.train()  

    samples = []
    with torch.no_grad():
        for _ in range(T):
            q = critics.forward(s, a, keepdim=False)
            samples.append(q.unsqueeze(0))  

    if not was_training:
        critics.eval()

    Q_samples = torch.cat(samples, dim=0) 
    Q_mean = Q_samples.mean(dim=(0, 2))     
    Q_std  = Q_samples.std(dim=(0, 2))    
    return Q_mean, Q_std

def mc_dropout_q_mean(critics, s, a, T: int = 10):
    Q_mean, _ = mc_dropout_q_stats(critics, s, a, T)
    return Q_mean
