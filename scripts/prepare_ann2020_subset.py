#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the annotation-era training subset used by the recommended configuration.

The PSC corpus used in this work carries category-level deficiency codes only
from 2020 onward. Earlier inspection records report a deficiency count but no
codes, so their category label vector is empty. Training on those records treats
them as defect-free negatives, which injects roughly 17k false negatives into
the detector and makes the defect-view neighbourhoods systematically empty
(68% empty in training vs 7% at test time).

This script keeps only training inspections on or after --cutoff. Validation and
test splits are copied unchanged, so the reported test set is untouched.

Usage
-----
python scripts/prepare_ann2020_subset.py \
    --src  data/mthgnn_psc_v8hq_complete_core_strict_temporal_v3_pure_defect \
    --dst  data/mthgnn_psc_v8hq_strict_v3_ann2020 \
    --cutoff 2020-01-01
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source dataset directory")
    ap.add_argument("--dst", required=True, help="output dataset directory")
    ap.add_argument("--cutoff", default="2020-01-01", help="keep training records on/after this date")
    a = ap.parse_args()

    src_splits = os.path.join(a.src, "splits")
    dst_splits = os.path.join(a.dst, "splits")
    if not os.path.isdir(src_splits):
        sys.exit(f"splits/ not found under {a.src}")

    os.makedirs(dst_splits, exist_ok=True)

    # 1. copy every auxiliary file (vocabularies, graph caches, statistics)
    for name in os.listdir(a.src):
        p = os.path.join(a.src, name)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(a.dst, name))
    for name in os.listdir(src_splits):
        if name == "train_enriched_fixed.json":
            continue
        shutil.copy2(os.path.join(src_splits, name), os.path.join(dst_splits, name))

    # 2. filter the training split by timestamp
    cut = int(np.datetime64(a.cutoff).astype("datetime64[s]").astype(np.int64))
    src_train = os.path.join(src_splits, "train_enriched_fixed.json")
    dst_train = os.path.join(dst_splits, "train_enriched_fixed.json")

    n_in = n_out = n_pos = n_annot = 0
    with open(src_train, encoding="utf-8") as f, open(dst_train, "w", encoding="utf-8") as g:
        for line in f:
            s = json.loads(line)
            n_in += 1
            if s["timestamp"] < cut:
                continue
            g.write(json.dumps(s, ensure_ascii=False) + "\n")
            n_out += 1
            if sum(s.get("labels_main", [])) > 0:
                n_pos += 1
            if s.get("has_supervision", True):
                n_annot += 1

    print(f"[prepare] source train : {n_in}")
    print(f"[prepare] kept train   : {n_out}  ({n_out / max(n_in, 1):.1%})")
    print(f"[prepare] defect-positive: {n_pos} ({n_pos / max(n_out, 1):.3f})")
    print(f"[prepare] annotated      : {n_annot} ({n_annot / max(n_out, 1):.3f})")
    print(f"[prepare] written to {dst_train}")


if __name__ == "__main__":
    main()
