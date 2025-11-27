# scripts/measure_ood.py
import argparse, os, os.path as osp
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

from ua.datasets import load_d4rl, z_norm_states

# --- lightweight policy loaders for our two checkpoints ---
def load_policy(ckpt_path, din, dout, device):
    state = torch.load(ckpt_path, map_location=device)
    # try common keys
    sd = None
    for k in ["model", "actor", "pi", "policy"]:
        if isinstance(state, dict) and k in state:
            sd = state[k]
            break
    if sd is None and isinstance(state, dict):
        # maybe the whole thing is the state_dict
        sd = state

    # simple MLP head (matches train_bc; also fine if td3bc_u saves the actor with same dims)
    import torch.nn as nn
    class MLP(nn.Module):
        def __init__(self, din, dout, hid=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(din, hid), nn.ReLU(),
                nn.Linear(hid, hid), nn.ReLU(),
                nn.Linear(hid, dout)
            )
        def forward(self, x): return self.net(x)

    pi = MLP(din, dout).to(device)
    if sd is not None:
        pi.load_state_dict(sd, strict=False)
    pi.eval()
    return pi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kstate", type=int, default=10, help="k for kNN in state space")
    ap.add_argument("--ckpt", type=str, required=True, help="path to policy .pt file")
    ap.add_argument("--plot", action="store_true", help="show & save histogram png")
    ap.add_argument("--save_npz", action="store_true", help="save distances as .npz")
    ap.add_argument("--out", type=str, default="", help="optional basename override")
    args = ap.parse_args()

    # load dataset (states/actions + z-norm stats)
    _, data = load_d4rl(args.env, args.seed)
    S, A = data["S"], data["A"]
    s_mean, s_std = data["s_mean"], data["s_std"]
    Sn = z_norm_states(S, s_mean, s_std)  # normalized states

    device = "cuda" if torch.cuda.is_available() else "cpu"
    din, dout = S.shape[1], A.shape[1]
    pi = load_policy(args.ckpt, din, dout, device)

    # policy actions on all states
    with torch.no_grad():
        s_t = torch.from_numpy(Sn).to(device)
        a_pi = pi(s_t).cpu().numpy().astype(np.float32)

    # kNN in normalized state space; compute per-sample min action distance to neighbors' behavior actions
    k = max(1, args.kstate)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(Sn)
    dists, idxs = nbrs.kneighbors(Sn, return_distance=True)  # dists unused (state dists)
    # min over neighbors of ||a_pi(s) - a_behavior(s_neighbor)||
    diffs = a_pi[:, None, :] - A[idxs]                       # [N, k, act_dim]
    act_d = np.linalg.norm(diffs, axis=-1)                   # [N, k]
    min_act_d = act_d.min(axis=1).astype(np.float32)         # [N]

    # simple stats in console
    def stat(x):
        return dict(mean=float(x.mean()),
                    p50=float(np.percentile(x, 50)),
                    p90=float(np.percentile(x, 90)),
                    p95=float(np.percentile(x, 95)))
    st = stat(min_act_d)
    print(f"OOD proxy (kNN-in-state min action distance) stats for {osp.basename(args.ckpt)} @ {args.env}:")
    print({k: round(v, 4) for k, v in st.items()})

    # outputs
    base = args.out.strip() or f"{osp.splitext(osp.basename(args.ckpt))[0]}.ood_{args.env}"
    if args.save_npz:
        npz_path = f"{base}.npz"
        np.savez(npz_path, d=min_act_d, env=args.env, ckpt=args.ckpt, kstate=args.kstate)
        print(f"Saved distances: {npz_path}")

    if args.plot:
        plt.figure(figsize=(8,5))
        plt.hist(min_act_d, bins=100, density=True)
        plt.xlabel("kNN-in-state min action distance")
        plt.ylabel("density")
        plt.title(f"OOD proxy: {osp.basename(args.ckpt)} @ {args.env}")
        png_path = f"{base}.png"
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        print(f"Saved plot: {png_path}")
        # also show interactively if you like:
        # plt.show()

if __name__ == "__main__":
    main()
