# -*- coding: utf-8 -*-
"""Offline inference-protocol search on a trained MSF-PSCRP checkpoint.

Loads one checkpoint, dumps val/test probabilities once, then sweeps post-hoc
protocols that require no retraining:

  1. defect-existence threshold (currently hard-coded at 0.5)
  2. classification threshold: global vs per-class, support-aware
  3. sub-class -> main-class hierarchical fusion (log-domain blend)

All thresholds are tuned on the validation split and applied unchanged to test.
"""
import json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

import argparse
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--ckpt", required=True, help="path to a *_best.ckpt produced by train_msf_pscrp.py")
_ap.add_argument("--data_dir", default="data/mthgnn_psc_v8hq_complete_core_strict_temporal_v3_pure_defect")
_ap.add_argument("--out", default="checkpoint/_probe/logits.npz")
_args, _ = _ap.parse_known_args()

sys.argv = ["probe",
            "--data_dir", _args.data_dir,
            "--checkpoint_dir", "checkpoint/_probe",
            "--batch_size", "512",
            "--num_workers", "4"]
import train_msf_pscrp as T   # noqa: E402

CKPT = _args.ckpt
OUT = _args.out
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)


def build():
    tr = T.PSCDatasetV8hq(T.DATA_DIR, "train")
    va = T.PSCDatasetV8hq(T.DATA_DIR, "val")
    te = T.PSCDatasetV8hq(T.DATA_DIR, "test")
    model = T.MTHGNNV8hqV2(entity2id=tr.entity2id, defect_classes=tr.defect_classes,
                           sub_classes=tr.sub_classes, embed_dim=T.EMBED_DIM,
                           active_views=T.ACTIVE_VIEWS)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(dev).eval()
    print(f"[probe] ckpt epoch={ckpt.get('epoch')} best_primary={ckpt.get('best_primary')}")
    return model, dev, va, te


@torch.no_grad()
def infer(model, dev, ds, name):
    loader = DataLoader(ds, batch_size=T.BATCH_SIZE, shuffle=False,
                        collate_fn=T.collate_fn, num_workers=T.NUM_WORKERS,
                        pin_memory=True)
    P, S, D, DT, LM, LS, IS, HS = [], [], [], [], [], [], [], []
    for b in loader:
        dl, det, sl, dtl = model(b)
        P.append(torch.sigmoid(dl).float().cpu().numpy())
        S.append(torch.sigmoid(sl).float().cpu().numpy())
        D.append(torch.sigmoid(det).float().cpu().numpy().reshape(-1))
        DT.append(torch.sigmoid(dtl).float().cpu().numpy().reshape(-1))
        LM.append(b["labels_main"].detach().cpu().numpy())
        LS.append(b["labels_sub"].detach().cpu().numpy())
        IS.append(b["is_detain"].detach().cpu().numpy().reshape(-1))
        HS.append(b["has_supervision"].detach().cpu().numpy().reshape(-1).astype(bool))
    return dict(p_main=np.vstack(P), p_sub=np.vstack(S), p_det=np.concatenate(D),
                p_detain=np.concatenate(DT), y_main=np.vstack(LM), y_sub=np.vstack(LS),
                y_detain=np.concatenate(IS), has_sup=np.concatenate(HS))


def hier_matrix(ds):
    """M[s, m] = empirical P(main=m | sub=s), rows normalised. Same as training."""
    from collections import defaultdict
    co = defaultdict(lambda: defaultdict(int))
    for s in ds.samples:
        for m in s.get("labels_main", []):
            for sb in s.get("labels_sub", []):
                if isinstance(sb, int):
                    co[int(sb)][int(m)] += 1
    ns, nm = len(ds.sub_classes), len(ds.defect_classes)
    M = np.zeros((ns, nm), dtype=np.float32)
    for sb in range(ns):
        den = sum(co[sb].values())
        if den:
            for m, c in co[sb].items():
                if 0 <= m < nm:
                    M[sb, m] = c / den
    zero = M.sum(1) == 0
    M[zero] = 1.0 / nm
    return M


def f1_macro(pred, y, hq=5):
    tp = (pred * y).sum(0); fp = (pred * (1 - y)).sum(0); fn = ((1 - pred) * y).sum(0)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    sup = y.sum(0)
    m = sup >= hq
    return float(f1[m].mean()), int(m.sum()), float(f1.mean())


def det_f1_at(p, y, t):
    p = p > t
    y = y > 0.5
    tp = (p & y).sum(); fp = (p & ~y).sum(); fn = (~p & y).sum()
    return float(2 * tp / (2 * tp + fp + fn + 1e-8))


def per_class_thresh(p, y, lo=0.05, hi=0.90, step=0.05, min_sup=5, shrink=0.0, glob=None):
    C = p.shape[1]
    th = np.full(C, glob if glob else 0.25, dtype=np.float32)
    for c in range(C):
        if y[:, c].sum() < min_sup:
            continue
        best, bt = -1, th[c]
        for t in np.arange(lo, hi + 1e-9, step):
            pr = p[:, c] > t
            tp = (pr & (y[:, c] > 0.5)).sum()
            f1 = 2 * tp / (pr.sum() + y[:, c].sum() + 1e-8)
            if f1 > best:
                best, bt = f1, float(t)
        th[c] = bt if glob is None else (1 - shrink) * bt + shrink * glob
    return th


def main():
    if os.path.exists(OUT):
        z = np.load(OUT)
        V = {k[4:]: z[k] for k in z.files if k.startswith("val_")}
        E = {k[5:]: z[k] for k in z.files if k.startswith("test_")}
        M = z["M"]
        print("[probe] reused cached logits")
    else:
        model, dev, va, te = build()
        V, E = infer(model, dev, va, "val"), infer(model, dev, te, "test")
        tr = T.PSCDatasetV8hq(T.DATA_DIR, "train")
        M = hier_matrix(tr)
        np.savez(OUT, **{f"val_{k}": v for k, v in V.items()},
                 **{f"test_{k}": v for k, v in E.items()}, M=M)
        print("[probe] logits cached")

    for nm, D in (("val", V), ("test", E)):
        D["y_has"] = (D["y_main"].sum(1) > 0).astype(np.float32)

    # ---------- 1. detection threshold ----------
    print("\n=== [1] defect-existence threshold (val-tuned, applied to test) ===")
    vm = V["has_sup"]; em = E["has_sup"]
    base_v = det_f1_at(V["p_det"][vm], V["y_has"][vm], 0.5)
    base_e = det_f1_at(E["p_det"][em], E["y_has"][em], 0.5)
    rows = []
    for t in np.arange(0.20, 0.81, 0.02):
        rows.append((t, det_f1_at(V["p_det"][vm], V["y_has"][vm], t)))
    bt = max(rows, key=lambda r: r[1])[0]
    tuned_e = det_f1_at(E["p_det"][em], E["y_has"][em], bt)
    print(f"  t=0.50      val={base_v:.4f}  test={base_e:.4f}")
    print(f"  t={bt:.2f} (best) val={max(r[1] for r in rows):.4f}  test={tuned_e:.4f}   Δtest={tuned_e-base_e:+.4f}")

    # ---------- 2. cls thresholds ----------
    print("\n=== [2] classification protocol (macro-F1-HQ, annotated positives only) ===")
    vp = V["y_has"] > 0.5; ep = E["y_has"] > 0.5
    yv, ye = V["y_main"][vp], E["y_main"][ep]
    pv, pe = V["p_main"][vp], E["p_main"][ep]

    b, nc, _ = f1_macro((pv > 0.25).astype(np.float32), yv)
    print(f"  A fixed 0.25            val={b:.4f}(n={nc}) test={f1_macro((pe>0.25).astype(np.float32), ye)[0]:.4f}")

    glob = max(((t, f1_macro((pv > t).astype(np.float32), yv)[0]) for t in np.arange(0.05, 0.6, 0.025)),
               key=lambda x: x[1])
    print(f"  B global t={glob[0]:.3f}         val={glob[1]:.4f} test={f1_macro((pe>glob[0]).astype(np.float32), ye)[0]:.4f}")

    for ms in (5, 10):
        th = per_class_thresh(pv, yv, min_sup=ms, glob=glob[0], shrink=0.0)
        fv = f1_macro((pv > th).astype(np.float32), yv)[0]
        fe = f1_macro((pe > th).astype(np.float32), ye)[0]
        print(f"  C per-class minsup={ms:<2d}     val={fv:.4f} test={fe:.4f}")
    th_s = per_class_thresh(pv, yv, min_sup=10, glob=glob[0], shrink=0.5)
    print(f"  D per-class shrunk0.5  val={f1_macro((pv>th_s).astype(np.float32), yv)[0]:.4f} "
          f"test={f1_macro((pe>th_s).astype(np.float32), ye)[0]:.4f}")

    # ---------- 3. sub -> main hierarchical fusion ----------
    print("\n=== [3] sub-class -> main-class fusion (log-domain, alpha) ===")
    def fuse(pm, ps, alpha):
        if alpha <= 0:
            return pm
        mass = np.clip(ps @ M, 1e-6, 1.0)         # [N, num_main]
        mass = mass / (mass.max(0, keepdims=True) + 1e-8)
        return np.power(pm, 1 - alpha) * np.power(mass, alpha)

    for a in (0.15, 0.3, 0.45, 0.6):
        fv = fuse(pv, V["p_sub"][vp], a); fe = fuse(pe, E["p_sub"][ep], a)
        th = per_class_thresh(fv, yv, min_sup=10, glob=glob[0], shrink=0.5)
        print(f"  alpha={a:.2f}  per-class+0.5 shrink  val={f1_macro((fv>th).astype(np.float32), yv)[0]:.4f} "
              f"test={f1_macro((fe>th).astype(np.float32), ye)[0]:.4f}")

    # ---------- 4. reference ----------
    print("\n=== [4] reference: detention (already val-tuned) ===")
    from sklearn.metrics import f1_score
    best = max(((t, f1_score(E["y_detain"], (E["p_detain"] > t).astype(int), zero_division=0))
                for t in np.arange(0.2, 0.8, 0.02)), key=lambda x: x[1])
    print(f"  best test detain F1 at oracle threshold {best[0]:.2f}: {best[1]:.4f}  (upper bound)")


if __name__ == "__main__":
    main()
