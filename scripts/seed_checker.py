python - << 'EOF'
import torch

files = [
    "bc_hopper_medium_replay_v2.pt",
    "bc_walker2d_medium_v2.pt",
    "td3bc_u_hopper_medium_replay_v2.pt",
    "iql_u_hopper_medium_replay_v2.pt",
    "iql_u_walker2d_medium_v2.pt",
]

for f in files:
    try:
        ckpt = torch.load(f, map_location="cpu")
        print(f, "→ seed =", ckpt.get("seed", "UNKNOWN"))
    except:
        print(f, "→ FAILED TO LOAD")
EOF
