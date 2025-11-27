# scripts/summarize_csv.py
import argparse
import pandas as pd
from pathlib import Path

AGG_COLS = ["fqe","ood_mean","ood_p50","ood_p90","ood_p95"]

def agg(df):
    # Drop exact duplicates so repeated runs don't double-count
    df = df.drop_duplicates(subset=["env","seed","method","ckpt"])
    g = (df.groupby(["env","method"], dropna=False)[AGG_COLS]
            .agg(['mean','std','count'])
            .reset_index())

    # Flatten multiindex columns like ('fqe','mean') -> 'fqe_mean'
    g.columns = [
        c if isinstance(c, str) else "_".join([x for x in c if x])
        for c in g.columns.to_list()
    ]
    return g

def fmt_cell(m, s, n):
    # Handle NaN std when n=1
    try:
        s_str = f"{s:.2f}" if pd.notna(s) else "nan"
        return f"{m:.2f} ± {s_str} (n={int(n)})"
    except Exception:
        return "—"

def to_markdown(out_df):
    lines = ["| env | method | FQE | OOD mean | OOD p50 | OOD p90 | OOD p95 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in out_df.iterrows():
        FQE   = fmt_cell(r["fqe_mean"],      r["fqe_std"],      r["fqe_count"])
        M     = fmt_cell(r["ood_mean_mean"], r["ood_mean_std"], r["ood_mean_count"])
        P50   = fmt_cell(r["ood_p50_mean"],  r["ood_p50_std"],  r["ood_p50_count"])
        P90   = fmt_cell(r["ood_p90_mean"],  r["ood_p90_std"],  r["ood_p90_count"])
        P95   = fmt_cell(r["ood_p95_mean"],  r["ood_p95_std"],  r["ood_p95_count"])
        lines.append(f"| {r['env']} | {r['method']} | {FQE} | {M} | {P50} | {P90} | {P95} |")
    return "\n".join(lines)

def main(csv_path):
    df = pd.read_csv(csv_path)
    out = agg(df)

    # Write CSV (flattened aggregates)
    csv_out = Path(csv_path).with_name("suite_results_summary.csv")
    out.to_csv(csv_out, index=False)

    # Write Markdown
    md = to_markdown(out)
    md_out = Path(csv_path).with_name("suite_results_summary.md")
    md_out.write_text(md)

    print("Wrote:")
    print(f"- {csv_out}")
    print(f"- {md_out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/suite_results.csv")
    args = ap.parse_args()
    main(args.csv)
