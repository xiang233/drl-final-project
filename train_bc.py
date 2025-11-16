# scripts/train_bc.py
import torch, torch.nn as nn, torch.optim as optim
from ua.datasets import load_d4rl, z_norm_states
from ua.utils import set_seed

class MLP(nn.Module):
    def __init__(self, din, dout, hid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, dout)
        )
    def forward(self, x): return self.net(x)

def main(env_name="hopper-medium-replay-v2", seed=0, steps=20000, bs=1024):
    set_seed(seed)
    env, data = load_d4rl(env_name, seed)
    S, A = data["S"], data["A"]
    S = z_norm_states(S, data["s_mean"], data["s_std"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pi = MLP(S.shape[1], A.shape[1]).to(device)
    opt = optim.Adam(pi.parameters(), lr=3e-4)
    mse = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(S), torch.from_numpy(A))
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    pi.train()
    for t, (s, a) in enumerate(dl):
        s, a = s.to(device), a.to(device)
        loss = mse(pi(s), a)
        opt.zero_grad(); loss.backward(); opt.step()
        if (t+1) % 100 == 0:
            print(f"step {t+1}: BC loss {loss.item():.4f}")
        if (t+1) >= steps: break
    torch.save({"model":pi.state_dict()}, f"bc_{env_name.replace('-','_')}.pt")

if __name__ == "__main__":
    main()
