#!/usr/bin/env bash
# Fast installation / data sanity check (about 5 minutes on a single GPU).
# Verifies the dataset loads, the full architecture builds, and all three heads
# produce finite losses. The resulting scores are NOT meaningful.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

"$PY" src/train_msf_pscrp.py \
  --name smoke \
  --checkpoint_dir checkpoint/smoke \
  --data_dir data/mthgnn_psc_v8hq_complete_core_strict_temporal_v3_pure_defect \
  --seed 42 --epochs 2 --batch_size 1024 --embed_dim 320 --num_heads 8 \
  --max_neighbors 50 --max_hist_inspections 8 --stage1_epochs 1 \
  --eval_every 1 --patience 2 --view_fusion_init prior --num_workers 4
