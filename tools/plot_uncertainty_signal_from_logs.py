#!/usr/bin/env python
# tools/plot_uncertainty_signal_from_logs.py

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


RE_ENV = re.compile(r"env_name=([A-Za-z0-9\-]+)")
RE_PEARSON_DS = re.compile(r"Pearson r\(sigma_ds, td_error\)\s*=\s*([0-9\.\-eE]+)")
RE_SPEARMAN_DS = re.compile(r"Spearman .*?\(sigma_ds, td_error\)\s*=\s*([0-9\.\-eE]+)")
RE_PEARSON_PI = re.compile(r"Pearson r\(sigma_pi, td_error\)\s*=\s*([0-9\.\-eE]+)")
RE_SPEARMAN_PI = re.compile(r"Spearman .*?\(sigma_pi, td_error\)\s*=\s*([0-9\.\-eE]+)")

RE_BIN_LINE = re.compile(
    r"sigma_(ds|pi)\s*~\s*([0-9\.\-eE]+)\s*->\s*mean TD error\s*=\s*([0-9\.\-eE]+)\s*\(n=\s*(\d+)\)"
)


def parse_log(path: Path):
    txt = path.read_text(errors="ignore")

    # env name
    m = RE_ENV.search(txt)
    env = m.group(1) if m else path.stem

    # correlations (optional, used for titles)
    def grab(rx):
        mm = rx.search(txt)
        return float(mm.group(1)) if mm else None

    corr = {
        "pearson_ds": grab(RE_PEARSON_DS),
        "spearman_ds": grab(RE_SPEARMAN_DS),
        "pearson_pi": grab(RE_PEARSON_PI),
        "spearman_pi": grab(RE_SPEARMAN_PI),
    }

    # binned curves
    bins = {"ds": {"x": [], "y": [], "n": []}, "pi": {"x": [], "y": [], "n": []}}
    for mm in RE_BIN_LINE.finditer(txt):
        which = mm.group(1)  # ds or pi
        x = float(mm.group(2))
        y = float(mm.group(3))
        n = int(mm.group(4))
        bins[which]["x"].append(x)
        bins[which]["y"].append(y)
        bins[which]["n"].append(n)

    if len(bins["ds"]["x"]) == 0 and len(bins["pi"]["x"]) == 0:
        raise ValueError(f"No binned lines parsed from {path}")

    return env, corr, bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="Paths to analyze_uncertainty_signal stdout logs")
    ap.add_argument("--out", type=str, default="results/uq_signal_binned.pdf", help="Output figure path (.pdf/.png)")
    ap.add_argument("--title", type=str, default="Binned |TD error| vs σ_Q", help="Figure title")
    args = ap.parse_args()

    parsed = [parse_log(Path(p)) for p in args.logs]

    # keep stable order if you pass them as hopper walker antmaze
    fig, axes = plt.subplots(1, len(parsed), figsize=(5 * len(parsed), 4), constrained_layout=True)
    if len(parsed) == 1:
        axes = [axes]

    for ax, (env, corr, bins) in zip(axes, parsed):
        # plot ds and pi
        if bins["ds"]["x"]:
            ax.plot(bins["ds"]["x"], bins["ds"]["y"], marker="o", linestyle="-", label=r"dataset actions ($\sigma_{\mathrm{ds}}$)")
        if bins["pi"]["x"]:
            ax.plot(bins["pi"]["x"], bins["pi"]["y"], marker="o", linestyle="--", label=r"policy actions ($\sigma_{\pi}$)")

        # label
        ax.set_xlabel(r"bin center $\sigma_Q$")
        ax.set_ylabel(r"mean $|\delta|$ in bin")

        # title with correlations if available
        if corr["pearson_ds"] is not None and corr["pearson_pi"] is not None:
            ax.set_title(
                f"{env}\n"
                f"r_ds={corr['pearson_ds']:.3f}, ρ_ds={corr['spearman_ds']:.3f} | "
                f"r_pi={corr['pearson_pi']:.3f}, ρ_pi={corr['spearman_pi']:.3f}"
            )
        else:
            ax.set_title(env)

        ax.legend()

    fig.suptitle(args.title)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"[OK] Saved: {out}")


if __name__ == "__main__":
    main()
