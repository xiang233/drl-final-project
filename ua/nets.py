
import copy
import torch
import torch.nn as nn
from typing import List

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

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, act_dim)
        )
    def forward(self, s):
        return torch.tanh(self.net(s))

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hid=256, dropout_p: float = 0.0):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hid), nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hid, 1)
        )
    def forward(self, s, a):
        return self.q(torch.cat([s, a], dim=-1))

class CriticEnsemble(nn.Module):
    def __init__(self, obs_dim, act_dim, K=4, hid=256, dropout_p: float = 0.0):
        super().__init__()
        self.members = nn.ModuleList(
            [Critic(obs_dim, act_dim, hid=hid, dropout_p=dropout_p) for _ in range(K)]
        )

    def forward(self, s, a, keepdim=False):
        outs: List[torch.Tensor] = [m(s, a) for m in self.members]  
        Q = torch.stack(outs, dim=0)  
        if keepdim:
            return Q
        return Q.squeeze(-1).transpose(0, 1) 

    def clone_targets(self):
        tgt = copy.deepcopy(self)
        for p in tgt.parameters():
            p.requires_grad_(False)
        return tgt

def soft_update_(online: nn.Module, target: nn.Module, tau: float):
    with torch.no_grad():
        for p, tp in zip(online.parameters(), target.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)
