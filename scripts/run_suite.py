# scripts/run_suite.py
import argparse, os, sys, subprocess, time
from pathlib import Path

def slug(env: str) -> str:
    # hopper-medium-replay-v2 -> hopper_medium_replay_v2
    return env.replace("-", "_")

def exists(p: str) -> bool:
    return Path(p).exists()

def format_ckpt(pattern: str, env: str, seed: int, cwd: str) -> str:
    return pattern.format(env=env, env_slug=slug(env), seed=seed, cwd=cwd)

def run_argv(argv, extra_env=None):
    print("→", " ".join(argv))
    env = os.environ.copy()
    if extra_env: env.update(extra_env)
    return subprocess.run(argv, check=True, env=env)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", required=True,
                    help="e.g. hopper-medium-replay-v2 walker2d-medium-v2")
    ap.add_argument("--seed-lo", type=int, default=0)
    ap.add_argument("--seed-hi", type=int, default=2)
    ap.add_argument("--steps-bc", type=int, default=20000)
    ap.add_argument("--steps-u", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--kstate", type=int, default=10)
    ap.add_argument("--fqe-iters", type=int, default=20000)

    # Filename patterns (can use {env}, {env_slug}, {seed}, {cwd})
    ap.add_argument("--bc-ckpt-pattern",
        default="{cwd}/bc_{env_slug}.pt",
        help="Pattern for BC ckpt; can use {env}, {env_slug}, {seed}, {cwd}.")
    ap.add_argument("--u-ckpt-pattern",
        default="{cwd}/td3bc_u_{env_slug}_seed{seed}.pt",
        help="Pattern for TD3BC-U ckpt; can use {env}, {env_slug}, {seed}, {cwd}.")

    # Training script module names
    ap.add_argument("--train-bc-script", default="scripts.train_bc")
    ap.add_argument("--train-u-script", default="scripts.train_td3bc_u")

    # Results CSV used by summarize_csv
    ap.add_argument("--results-csv", default="results/suite_results.csv")
    args = ap.parse_args()

    cwd = os.getcwd()
    Path("results").mkdir(exist_ok=True)

    for env_name in args.envs:
        for seed in range(args.seed_lo, args.seed_hi + 1):
            bc_ckpt = format_ckpt(args.bc_ckpt_pattern, env_name, seed, cwd)
            u_ckpt  = format_ckpt(args.u_ckpt_pattern,  env_name, seed, cwd)

            # (1) Train or reuse BC
            if args.steps_bc > 0 and not exists(bc_ckpt):
                print(f"• Training BC for {env_name} seed={seed}")
                run_argv([
                    sys.executable, "-m", args.train_bc_script,
                    "--env", env_name,
                    "--seed", str(seed),
                    "--steps", str(args.steps_bc),
                    "--bs", str(args.bs),
                ])
                default_bc = f"{cwd}/bc_{slug(env_name)}.pt"
                if default_bc != bc_ckpt and exists(default_bc):
                    Path(default_bc).rename(bc_ckpt)
            else:
                print(f"✓ Using existing BC ckpt: {bc_ckpt}")

            # (2) Train or reuse TD3BC-U
            if args.steps_u > 0 and not exists(u_ckpt):
                print(f"• Training TD3BC-U for {env_name} seed={seed}")
                run_argv([
                    sys.executable, "-m", args.train_u_script,
                    "--env", env_name,
                    "--seed", str(seed),
                    "--steps", str(args.steps_u),
                    "--K", str(args.K),
                ])
                default_u = f"{cwd}/td3bc_u_{slug(env_name)}.pt"
                if default_u != u_ckpt and exists(default_u):
                    Path(default_u).rename(u_ckpt)
            else:
                print(f"✓ Using existing TD3BC-U ckpt: {u_ckpt}")

            # (3) OOD + FQE for BC
            if exists(bc_ckpt):
                run_argv([
                    sys.executable, "-m", "scripts.measure_ood",
                    "--env", env_name, "--seed", str(seed),
                    "--ckpt", bc_ckpt,
                    "--kstate", str(args.kstate),
                    "--plot",
                    "--save_npz",      # correct flag
                ])
                run_argv([
                    sys.executable, "-m", "scripts.estimate_return",
                    "--env", env_name, "--seed", str(seed),
                    "--ckpt", bc_ckpt,
                    "--iters", str(args.fqe_iters),
                ])
            else:
                print(f"! Missing BC ckpt for {env_name} seed={seed}: {bc_ckpt} (skipping eval)")

            # (4) OOD + FQE for TD3BC-U
            if exists(u_ckpt):
                run_argv([
                    sys.executable, "-m", "scripts.measure_ood",
                    "--env", env_name, "--seed", str(seed),
                    "--ckpt", u_ckpt,
                    "--kstate", str(args.kstate),
                    "--plot",
                    "--save_npz",
                ])
                run_argv([
                    sys.executable, "-m", "scripts.estimate_return",
                    "--env", env_name, "--seed", str(seed),
                    "--ckpt", u_ckpt,
                    "--iters", str(args.fqe_iters),
                ])
            else:
                print(f"! Missing TD3BC-U ckpt for {env_name} seed={seed}: {u_ckpt} (skipping eval)")

    # (5) Summarize to markdown/csv (expects results/suite_results.csv)
    run_argv([
        sys.executable, "-m", "scripts.summarize_csv",
        "--csv", args.results_csv,
    ])

if __name__ == "__main__":
    main()
