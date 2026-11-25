#!/usr/bin/env bash
# Recommended configuration: annotation-era training subset (release baseline).
#
# Data : mthgnn_psc_v8hq_strict_v3_ann2020  (train 27,286 / val 1,500 / test 3,181)
# Model: 5 meta-path views + Transformer category encoder + FFT branch +
#        cross-attention fusion, 3 prediction heads, two-stage loss schedule.
# Inputs: the record's own covariates, its relational context and earlier
#         inspections of the same ship. All neighbours precede the target record.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

"$PY" src/train_msf_pscrp.py \
  --name msf_pscrp_ann2020_seed42 \
  --checkpoint_dir checkpoint/msf_pscrp_ann2020 \
  --data_dir data/mthgnn_psc_v8hq_strict_v3_ann2020 \
  --seed 42 \
  --epochs 12 \
  --batch_size 1024 \
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
  --stage1_epochs 3 \
  --eval_every 2 \
  --patience 12 \
  --view_fusion_init prior \
  --num_workers 4
