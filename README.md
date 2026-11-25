# MSF-PSCRP

Reference implementation of **Multi-Semantic Fusion for Port State Control Risk
Prediction (MSF-PSCRP)**. The model fuses five ship-relation views and the target
ship's own inspection history to predict three inspection outcomes jointly:
whether a deficiency will be found, which deficiency categories will be recorded,
and whether the ship will be detained.

This repository contains the reference implementation of the model, together with
the training configuration and the evaluation protocol used for the reported
results.

---

## 1. Method

| Module | What it does |
|---|---|
| Meta-path-guided relational semantic encoding | Builds five relation views of a ship (port, flag country, MoU, management company, historical deficiencies), applies per-view temporal attention over up to 50 neighbours, and fuses the views with learnable weights initialised from domain priors (`defect_view` 0.35, `company_view` 0.25, `port_view` 0.15, `country_view` 0.15, `mou_view` 0.10). |
| History-aware semantic enhancement | Encodes the number of past inspections, the cumulative deficiency-category composition via a single-layer Transformer with masked mean pooling, and the recent-window change pattern via an FFT branch (DC / low-frequency gated, high-frequency kept separately). |
| Multi-semantic fusion prediction | Cross-attention fusion of relational and historical semantics, followed by three heads (defect existence, 35-category multi-label deficiencies, detention) trained with a two-stage loss schedule. |

Two-stage schedule: for the first `--stage1_epochs` epochs the detection loss
dominates (Huber-style weighting 2.0 vs 0.5); afterwards the classification
weight is raised to 2.5 and the sub-category auxiliary head to 0.5.

Model size with the default configuration: **91.3M parameters**.

## 2. Input and supervision protocol

Three constraints are applied in the data pipeline and in the training code:

1. **Inspection-time inputs.** A record is described by its own covariates, its
   relational context and the outcomes of earlier inspections of the same ship. Its
   deficiency codes enter the computation only as supervision targets
   (`sub_class_loss` and the metrics); the forward pass always takes
   `hist_defect_subs`, the accumulated codes of earlier inspections.
2. **Temporal ordering.** Every view neighbour and every historical inspection
   satisfies `timestamp(neighbour) < timestamp(target)`; records that share the
   target timestamp are excluded.
3. **Annotation-aware supervision.** Records that carry a deficiency count but no
   category codes (the corpus has none before 2020) are unlabeled for the defect
   existence and category tasks, and are therefore excluded from those losses and
   metrics via the `has_supervision` mask.

Dataset audit of the feature set:

| Check | train | val | test |
|---|---|---|---|
| view-neighbour timestamps inspected | 1,247,045 | 299,767 | 640,620 |
| future / same-time violations | **0 / 0** | **0 / 0** | **0 / 0** |
| history timestamps inspected | 52,821 | 11,056 | 23,363 |
| `>=` target violations | **0** | **0** | **0** |
| samples where `hist_defect_subs` equals current `labels_sub` | **0** | **0** | **0** |

## 3. Data

The feature dataset is **not** included: it is derived from port State control
inspection records that the authors are not free to redistribute. The directory
layout expected by the code is:

```
data/<dataset>/
├── statistics.json                  # class maps, vocab sizes, split sizes
├── entities.txt, relations.txt
├── ship_attr_psc.json, port_attr.json, port_name_map.json
├── company_supplement_map.json
├── graph_v2_cache_keep35.pt
└── splits/
    ├── train_enriched_fixed.json    # one JSON object per line
    ├── val_enriched_fixed.json
    └── test_enriched_fixed.json
```

Each record provides: `timestamp`, `ship_imo`, `port`, `port_country`, `port_mou`,
`flag`, `company`, `ship_type`, `gross_ton`, `labels_main` (35-dim multi-hot),
`labels_sub` (536-dim multi-hot), `is_detain`, `has_supervision`,
`hist_inspection_count`, `hist_defect_cats`, `hist_defect_subs`,
`hist_insp_seq_ts`, `view_neighbors[5]`, `view_neighbor_ts[5]`,
`view_neighbor_delta_years[5]`.

Splits are chronological. The full-history set covers 1999-01 to 2024-12 for
training, 2024-12-31 to 2025-03-30 for validation, and 2025-03-31 to 2025-10-31
for test:

| Dataset | train | val | test | defect-positive (test) | detention (test) |
|---|---|---|---|---|---|
| full history | 59,131 | 1,500 | 3,181 | 0.667 | 0.021 |
| annotation era (2020-01 onward) | 27,286 | 1,500 | 3,181 | 0.667 | 0.021 |

`scripts/prepare_ann2020_subset.py` builds the second one from the first.

## 4. Quick start

```bash
pip install -r requirements.txt

# 1. sanity check: loads the data, builds the model, runs 2 epochs (~5 min)
bash scripts/run_smoke.sh

# 2. recommended configuration (~10 min on one modern GPU)
bash scripts/run_recommended_ann2020.sh

# 3. manuscript-aligned full-history configuration (~3 h)
bash scripts/run_full_history.sh
```

Set `PYTHON=/path/to/python` if the interpreter is not on `PATH`. Results are
written to `checkpoint/<run>/<name>_results.json`, together with the exact
configuration used.

Post-hoc protocol analysis on a finished checkpoint (threshold sweep, per-class
calibration, sub-class to main-class fusion):

```bash
python src/eval_protocol_search.py --ckpt checkpoint/msf_pscrp_ann2020/msf_pscrp_ann2020_seed42_best.ckpt \
    --data_dir data/mthgnn_psc_v8hq_strict_v3_ann2020
```

## 5. Evaluation protocol

Metrics are computed once on the chronological test split, with decision
thresholds selected on the validation split. Three metrics correspond to the three
tasks: defect-existence F1, deficiency-category macro-F1 (reported over categories
with at least five positive test samples, denoted HQ), and detention F1; macro AUC
and Hit@5 summarise the ranking quality of the category head.

Every run writes `<name>_results.json` next to the checkpoint, containing all
metrics together with the exact configuration used.

## 6. Citation

```bibtex
@article{msfpscrp2026,
  title   = {Multi-Semantic Fusion for Port State Control Risk Prediction},
  journal = {Ocean Engineering},
  year    = {2026}
}
```

## 7. License

MIT. See `LICENSE`.
