
import argparse
import glob
import os
import os.path as osp
import sys
import torch

from scripts.measure_ood import main as measure_ood_main 

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pt_dir", type=str, default="generated_data",
        help="Directory containing .pt checkpoints."
    )
    ap.add_argument(
        "--glob", type=str, default="*.pt",
        help="Glob pattern to select checkpoints (default: '*.pt')"
    )
    ap.add_argument(
        "--max_samples", type=int, default=200000,
        help="Max number of states to use for kNN (passed to measure_ood.py)"
    )
    ap.add_argument(
        "--kstate", type=int, default=10,
        help="k for kNN in state space (passed to measure_ood.py)"
    )
    args = ap.parse_args()

    ckpts = sorted(glob.glob(osp.join(args.pt_dir, args.glob)))
    if not ckpts:
        print(f"[WARN] No checkpoints matched pattern in {args.pt_dir}: {args.glob}")
        return

    print(f"[INFO] Found {len(ckpts)} checkpoints to process for OOD measurement.")

    for path in ckpts:
        stem = osp.splitext(osp.basename(path))[0]
        
        try:
            state = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}. Skipping OOD.")
            continue
        
        env_name = state.get("env_name", None)
        seed = state.get("seed", 0)

        if env_name is None:
             print(f"[WARN] Skipping {path}: no 'env_name' in checkpoint.")
             continue

        out_name_prefix = f"{stem}.ood_{env_name}"
        out_path_full = osp.join(args.pt_dir, out_name_prefix)
        ood_npz_path = f"{out_path_full}.npz"
        
        if osp.exists(ood_npz_path):
            print(f"[SKIP] OOD NPZ already exists: {out_name_prefix}.npz")
            continue

        print(f"[OOD RUN] Measuring {osp.basename(path)} on {env_name}...")
        
        try:
            command = (
                f"python -m scripts.measure_ood "
                f"--env {env_name} --seed {seed} --ckpt {path} "
                f"--save_npz --out {out_path_full} --max_samples {args.max_samples} "
                f"--kstate {args.kstate}"
            )
            os.system(command)

        except Exception as e:
            print(f"[ERROR] OOD measurement failed for {path}: {e}")
            continue

    print("[INFO] Finished OOD measurement step.")


if __name__ == "__main__":
    main()