#!/usr/bin/env python
import os
import glob
import csv
import re

import torch

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
CKPT_DIR = os.path.join(BASE_DIR, "generated_data")
FQE_CSV = os.path.join(CKPT_DIR, "fqe_results.csv")


def infer_method_from_name(stem):
    """Infer algorithm name from checkpoint filename stem."""
    if stem.startswith("bc_"):
        return "bc"
    if stem.startswith("td3bc_u_"):
        return "td3bc_u"
    if stem.startswith("iql_u_"):
        return "iql_u"
    if stem.startswith("iql_"):
        return "iql"
    return "unknown"


def parse_seed_from_name(stem):
    """Parse ..._seedX from filename stem, returns int or None."""
    m = re.search(r"_seed(\d+)", stem)
    if m:
        return int(m.group(1))
    return None


def load_fqe_table(path):
    """Load generated_data/fqe_results.csv into a list of dicts (or empty list)."""
    if not os.path.exists(path):
        print(f"[INFO] No FQE CSV found at {path}; will mark FQE as missing.")
        return []

    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"[INFO] Loaded {len(rows)} rows from {path}")
    return rows


def has_fqe_record(fqe_rows, env_name, method, seed):
    """Check if (env, method, seed) appears in FQE CSV."""
    if not fqe_rows:
        return False
    for r in fqe_rows:
        renv = r.get("env", "")
        rmethod = r.get("method", "")
        rseed = r.get("seed", "")
        try:
            rseed_int = int(rseed)
        except Exception:
            continue
        if renv == env_name and rmethod == method and rseed_int == seed:
            return True
    return False


def check_one_ckpt(pt_path, fqe_rows):
    stem = os.path.splitext(os.path.basename(pt_path))[0]
    ckpt_dir = os.path.dirname(pt_path)

    try:
        state = torch.load(pt_path, map_location="cpu")
    except Exception as e:
        return {
            "file": pt_path,
            "ok": False,
            "error": f"failed to load: {e}",
            "env": "?",
            "method": "?",
            "seed": None,
            "seed_name": parse_seed_from_name(stem),
            "seed_mismatch": None,
            "has_ood": False,
            "has_fqe": False,
        }

    # metadata from checkpoint
    env_name = state.get("env_name", "?")
    algo = state.get("algo", None)
    seed_ckpt = state.get("seed", None)

    # method
    if algo is not None:
        method = algo
    else:
        method = infer_method_from_name(stem)

    # seed from name (if any)
    seed_name = parse_seed_from_name(stem)

    seed_mismatch = None
    if seed_ckpt is not None and seed_name is not None:
        seed_mismatch = (seed_ckpt != seed_name)

    # OOD files: <stem>.ood_<env_name>.npz / .png, in same dir as checkpoint
    ood_base = os.path.join(ckpt_dir, f"{stem}.ood_{env_name}")
    ood_npz = f"{ood_base}.npz"
    ood_png = f"{ood_base}.png"
    has_ood = os.path.exists(ood_npz) or os.path.exists(ood_png)

    # FQE entry in generated_data/fqe_results.csv
    has_fqe = False
    if isinstance(seed_ckpt, int):
        has_fqe = has_fqe_record(fqe_rows, env_name, method, seed_ckpt)

    return {
        "file": pt_path,
        "ok": True,
        "error": None,
        "env": env_name,
        "method": method,
        "seed": seed_ckpt,
        "seed_name": seed_name,
        "seed_mismatch": seed_mismatch,
        "has_ood": has_ood,
        "has_fqe": has_fqe,
    }


def main():
    print(f"[INFO] Checking *.pt in {CKPT_DIR} ...")
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")))
    if not ckpts:
        print("[WARN] No .pt files found in generated_data/.")
        return

    fqe_rows = load_fqe_table(FQE_CSV)

    rows = [check_one_ckpt(p, fqe_rows) for p in ckpts]

    # Pretty print summary
    print("\n=== Checkpoint completeness summary ===")
    header = (
        "file",
        "env",
        "method",
        "seed_ckpt",
        "seed_name",
        "seed_mismatch",
        "OOD_npz/png",
        "FQE_in_csv",
        "status",
    )
    print("{:<40} {:<24} {:<10} {:<9} {:<9} {:<13} {:<11} {:<11} {}".format(*header))

    for r in rows:
        if not r["ok"]:
            status = f"ERROR: {r['error']}"
        else:
            missing = []
            if not r["has_ood"]:
                missing.append("OOD")
            if not r["has_fqe"]:
                missing.append("FQE")
            if r["seed_mismatch"]:
                missing.append("SEED_MISMATCH")
            status = "OK" if not missing else "MISSING:" + ",".join(missing)

        print("{:<40} {:<24} {:<10} {:<9} {:<9} {:<13} {:<11} {:<11} {}".format(
            os.path.basename(r["file"]),
            r["env"],
            r["method"],
            str(r["seed"]),
            str(r["seed_name"]),
            str(r["seed_mismatch"]),
            "✓" if r["has_ood"] else "✗",
            "✓" if r["has_fqe"] else "✗",
            status,
        ))

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
