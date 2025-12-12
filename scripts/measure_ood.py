import argparse
import os.path as osp

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from ua.datasets import load_d4rl, z_norm_states


# Policy architectures & loader 

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, din, dout, hid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, dout),
        )

    def forward(self, x):
        return self.net(x)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hid=256):
        super().__init__()
        self.backbone = MLP(state_dim, action_dim, hid=hid)

    def forward(self, s):
        return torch.tanh(self.backbone(s))


def load_policy(ckpt_path, s_dim, a_dim, device):
    state = torch.load(ckpt_path, map_location=device)
    algo = state.get("algo", None)

    # IQL
    if algo == "iql":
        pi = PolicyNetwork(s_dim, a_dim).to(device)
        actor_sd = state.get("model")
        if actor_sd is None:
            raise KeyError(
                f"Expected 'model' key in IQL checkpoint {ckpt_path}, "
                f"found keys: {list(state.keys())}"
            )
        pi.load_state_dict(actor_sd, strict=True)
        pi.eval()
        return pi

    # IQL-U 
    if algo == "iql_u":
        pi = PolicyNetwork(s_dim, a_dim).to(device)
        actor_sd = state.get("actor") or state.get("model")
        if actor_sd is None:
            raise KeyError(
                f"Expected 'actor' or 'model' in IQL-U checkpoint {ckpt_path}, "
                f"found keys: {list(state.keys())}"
            )
        pi.load_state_dict(actor_sd, strict=True)
        pi.eval()
        return pi

    # TD3BC-U 
    if algo == "td3bc_u" or (
        "actor" in state and "critics" in state and "K" in state and "v" not in state
    ):
        try:
            from ua.nets import Actor
        except ImportError as e:
            raise ImportError(
                "Could not import Actor from ua.nets; needed to load TD3BC-U policy."
            ) from e

        pi = Actor(s_dim, a_dim).to(device)
        actor_sd = state.get("actor")
        if actor_sd is None:
            raise KeyError(
                f"Expected 'actor' key in TD3BC-U checkpoint {ckpt_path}, "
                f"found keys: {list(state.keys())}"
            )
        pi.load_state_dict(actor_sd, strict=True)
        pi.eval()
        return pi

    # BC 
    if "model" in state and "q" not in state and "v" not in state:
        pi = MLP(s_dim, a_dim).to(device)
        pi.load_state_dict(state["model"], strict=True)
        pi.eval()
        return pi


    key = None
    if "model" in state:
        key = "model"
    elif "actor" in state:
        key = "actor"

    if key is None:
        raise ValueError(
            f"Could not determine how to load policy from {ckpt_path}. "
            f"algo={algo}, keys={list(state.keys())}"
        )

    print(f"[WARN] Unknown algo '{algo}' in {ckpt_path}; "
          f"falling back to MLP using state['{key}'].")

    pi = MLP(s_dim, a_dim).to(device)
    pi.load_state_dict(state[key], strict=False)
    pi.eval()
    return pi


# OOD measurement 

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kstate", type=int, default=10,
                    help="k for kNN in state space")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="path to policy .pt file")
    ap.add_argument("--plot", action="store_true",
                    help="show & save histogram png")
    ap.add_argument("--save_npz", action="store_true",
                    help="save distances as .npz")
    ap.add_argument("--out", type=str, default="",
                    help="optional basename override")
    ap.add_argument("--max_samples", type=int, default=200000,
                    help="max number of states to use for kNN (subsample if larger; "
                         "set <=0 to use all)")
    args = ap.parse_args()

    # load dataset
    _, data = load_d4rl(args.env, args.seed)
    S, A = data["S"], data["A"]
    s_mean, s_std = data["s_mean"], data["s_std"]
    Sn = z_norm_states(S, s_mean, s_std) 

    N = Sn.shape[0]
    print(f"[measure_ood] Loaded {args.env} with N={N} transitions.")

    if args.max_samples > 0 and N > args.max_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(N, size=args.max_samples, replace=False)
        Sn_sub = Sn[idx]
        A_sub = A[idx]
        print(f"[measure_ood] Subsampling to {args.max_samples} states for kNN.")
    else:
        Sn_sub = Sn
        A_sub = A
        idx = None 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    s_dim, a_dim = S.shape[1], A.shape[1]
    pi = load_policy(args.ckpt, s_dim, a_dim, device)

    with torch.no_grad():
        s_t = torch.from_numpy(Sn_sub).to(device)
        a_pi = pi(s_t).cpu().numpy().astype(np.float32)

    # kNN 
    k = max(1, args.kstate)
    print(f"[measure_ood] Fitting NearestNeighbors on {Sn_sub.shape[0]} states (k={k})...")
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1).fit(Sn_sub)
    print("[measure_ood] Querying kNN...")
    dists, idxs = nbrs.kneighbors(Sn_sub, return_distance=True)

    diffs = a_pi[:, None, :] - A_sub[idxs]           
    act_d = np.linalg.norm(diffs, axis=-1)           
    min_act_d = act_d.min(axis=1).astype(np.float32) 

    def stat(x):
        return dict(mean=float(x.mean()),
                    p50=float(np.percentile(x, 50)),
                    p90=float(np.percentile(x, 90)),
                    p95=float(np.percentile(x, 95)))

    st = stat(min_act_d)
    print(f"OOD proxy (kNN-in-state min action distance) stats for "
          f"{osp.basename(args.ckpt)} @ {args.env}:")
    print({k: round(v, 4) for k, v in st.items()})

    # outputs
    base = args.out.strip() or f"{osp.splitext(osp.basename(args.ckpt))[0]}.ood_{args.env}"
    if args.save_npz:
        npz_path = f"{base}.npz"
        np.savez(
            npz_path,
            d=min_act_d,
            env=args.env,
            ckpt=args.ckpt,
            kstate=args.kstate,
            idx=idx if idx is not None else np.arange(len(min_act_d)),
        )
        print(f"Saved distances: {npz_path}")

    if args.plot:
        plt.figure(figsize=(8, 5))
        plt.hist(min_act_d, bins=100, density=True)
        plt.xlabel("kNN-in-state min action distance")
        plt.ylabel("density")
        plt.title(f"OOD proxy: {osp.basename(args.ckpt)} @ {args.env}")
        png_path = f"{base}.png"
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()
