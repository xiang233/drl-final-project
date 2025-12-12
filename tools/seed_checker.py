
import glob
import os
import re

import torch

SEED_RE = re.compile(r"_seed(\d+)\.pt$")

def main():
    pt_files = sorted(glob.glob("*.pt"))
    if not pt_files:
        print("No .pt files found in current directory.")
        return

    for path in pt_files:
        m = SEED_RE.search(os.path.basename(path))
        if not m:
            print(f"[SKIP]      {path}: no '_seedN' suffix in filename")
            continue

        seed_from_name = int(m.group(1))

        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"[FAIL LOAD] {path}: could not load checkpoint ({e})")
            continue

        seed_in_ckpt = ckpt.get("seed", None)

        if seed_in_ckpt is None:
            print(
                f"[WARN]      {path}: filename seed={seed_from_name}, "
                f"but checkpoint has no 'seed' key"
            )
        elif seed_in_ckpt == seed_from_name:
            print(f"[OK]        {path}: seed match ({seed_from_name})")
        else:
            print(
                f"[MISMATCH]  {path}: filename seed={seed_from_name}, "
                f"checkpoint seed={seed_in_ckpt}"
            )

if __name__ == "__main__":
    main()
