import numpy as np, torch, torch.nn as nn, torch.optim as optim

class MLP(nn.Module):
    def __init__(self, din, dout=1, hid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, dout)
        )
    def forward(self, x): return self.net(x)

def fqe_evaluate(policy, data, gamma=0.99, iters=20000, bs=1024, lr=3e-4, device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    S, A, S_next = data["S"], data["A"], data["S_next"]
    R, D = data["rewards"].reshape(-1,1), data["terminals"].reshape(-1,1)
    s_mean, s_std = data["s_mean"], data["s_std"]

    def z(x): return (x - s_mean)/(s_std + 1e-6)
    Sz, Snext_z = z(S), z(S_next)

    din = Sz.shape[1] + A.shape[1]
    q = MLP(din, 1).to(device)
    opt = optim.Adam(q.parameters(), lr=lr)
    mse = nn.MSELoss()

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Sz), torch.from_numpy(A),
        torch.from_numpy(Snext_z), torch.from_numpy(R), torch.from_numpy(D)
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)

    q.train()
    for t, (s, a, sp, r, d) in enumerate(dl, start=1):
        s, a, sp, r, d = s.to(device), a.to(device), sp.to(device), r.to(device), d.to(device)

        with torch.no_grad():
            a_pi_sp = policy(sp)                     
            tgt = r + gamma * (1.0 - d) * q(torch.cat([sp, a_pi_sp], dim=-1))

        pred = q(torch.cat([s, a], dim=-1))
        loss = mse(pred, tgt)
        opt.zero_grad(); loss.backward(); opt.step()

        if t >= iters: break

    q.eval()
    with torch.no_grad():
        s = torch.from_numpy(Sz).to(device)
        v = q(torch.cat([s, policy(s)], dim=-1)).mean().item()
    return v
