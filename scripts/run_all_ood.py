# scripts/run_all_ood.py (CORRECTED EXECUTION FOR OOD NAMING)

import argparse
import glob
import os
import os.path as osp
import sys
import torch

# IMPORTANT: Since you fixed PYTHONPATH, we can import successfully
# Note: measure_ood_main is unused here, but the import helps Python resolve paths.
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
    # Add an argument for kNN value
    ap.add_argument(
        "--kstate", type=int, default=10,
        help="k for kNN in state space (passed to measure_ood.py)"
    )
    args = ap.parse_args()

    # Find all checkpoints matching the pattern
    ckpts = sorted(glob.glob(osp.join(args.pt_dir, args.glob)))
    if not ckpts:
        print(f"[WARN] No checkpoints matched pattern in {args.pt_dir}: {args.glob}")
        return

    print(f"[INFO] Found {len(ckpts)} checkpoints to process for OOD measurement.")

    for path in ckpts:
        stem = osp.splitext(osp.basename(path))[0]
        
        # Load the checkpoint just to get env_name and seed
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

        # --- FIX: Construct the output name to match the convention ---
        # The target file name should be: <stem>.ood_<env_name>.npz
        # We pass everything *before* the .npz extension to the --out argument.
        out_name_prefix = f"{stem}.ood_{env_name}"
        out_path_full = osp.join(args.pt_dir, out_name_prefix)
        ood_npz_path = f"{out_path_full}.npz"
        
        if osp.exists(ood_npz_path):
            print(f"[SKIP] OOD NPZ already exists: {out_name_prefix}.npz")
            continue

        # --- Execute measure_ood_main via shell call ---
        print(f"[OOD RUN] Measuring {osp.basename(path)} on {env_name}...")
        
        try:
            # CORRECTED COMMAND: --out now contains the full desired filename prefix.
            command = (
                f"python -m scripts.measure_ood "
                f"--env {env_name} --seed {seed} --ckpt {path} "
                f"--save_npz --out {out_path_full} --max_samples {args.max_samples} "
                f"--kstate {args.kstate}"
            )
            # print(command) # Debugging line to verify command string
            os.system(command)

        except Exception as e:
            print(f"[ERROR] OOD measurement failed for {path}: {e}")
            continue

    print("[INFO] Finished OOD measurement step.")


if __name__ == "__main__":
    main()