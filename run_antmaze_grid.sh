#!/usr/bin/env bash

STD_CSV="results/fqe_standard_antmaze_umaze.csv"
GATED_CSV="results/fqe_sigma_gated_bounds_antmaze_umaze.csv"

mkdir -p results

for seed in 0 1 2; do
  ckpt_td3="generated_data/td3bc_antmaze_umaze_v2_seed${seed}.pt"
  ckpt_iql="generated_data/iql_u_conf_weighted_antmaze_umaze_v2_seed${seed}.pt"

  for ckpt in "$ckpt_td3" "$ckpt_iql"; do
    echo "===================================================="
    echo "Env: antmaze-umaze-v2 | Seed: $seed | Ckpt: $ckpt"

    if [ ! -f "$ckpt" ]; then
      echo "[WARN] Missing ckpt: $ckpt, skipping."
      continue
    fi

    # 1) Standard FQE (old implementation)
    python -m scripts.estimate_return_old \
      --env antmaze-umaze-v2 \
      --seed "$seed" \
      --ckpt "$ckpt" \
      --iters 20000 \
      --csv "$STD_CSV"

    # 2) σ-gated FQE over a grid of bounds
    for sigma_low in 0.001 0.003 0.006; do
      for sigma_high in 0.010 0.015 0.020; do
        python -m tools.run_sigma_gated_fqe \
          --ckpt "$ckpt" \
          --sigma_low "$sigma_low" \
          --sigma_high "$sigma_high" \
          --iters 20000 \
          --csv "$GATED_CSV"
      done
    done
  done
done
