#!/usr/bin/env bash
# Full-history configuration (manuscript-aligned splits).
#
# Data : mthgnn_psc_v8hq_complete_core_strict_temporal_v3_pure_defect
#        (train 59,131 / val 1,500 / test 3,181, records from 1999)
# Note : the pre-2020 training records carry a deficiency count but no category
#        codes. They are excluded from the detection and classification losses
#        and from those two metrics via the has_supervision mask, but they still
#        contribute historical/relational context to later records.

set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

"$PY" src/train_msf_pscrp.py \
  --name msf_pscrp_full_seed42 \
  --checkpoint_dir checkpoint/msf_pscrp_full \
  --data_dir data/mthgnn_psc_v8hq_complete_core_strict_temporal_v3_pure_defect \
  --seed 42 \
  --epochs 60 \
  --batch_size 2048 \
  --lr 5e-4 \
  --weight_decay 1e-3 \
  --embed_dim 320 \
  --num_heads 8 \
  --max_neighbors 50 \
  --max_hist_inspections 8 \
  --dropout 0.15 \
  --label_smoothing 0.05 \
  --cb_beta 0.9 \
  --classification_weight 2.5 \
  --sub_class_weight 0.5 \
  --detain_weight 0.5 \
  --stage1_detection_weight 2.0 \
  --stage1_classification_weight 0.5 \
  --stage1_epochs 15 \
  --eval_every 2 \
  --patience 18 \
  --view_fusion_init prior \
  --num_workers 8
