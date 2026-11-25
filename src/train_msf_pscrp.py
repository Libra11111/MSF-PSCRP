"""
MSF-PSCRP - Multi-Semantic Fusion for Port State Control Risk Prediction
============================================================
Clean, paper-aligned training script.

Three modules from the manuscript:
  (a) Meta-path-guided relational semantic encoding: five relation views
      (Port / Flag-Country / MoU / Company / Deficiency), per-view attention
      and learnable view fusion.
  (b) History-aware semantic enhancement: inspection count, deficiency-category
      composition (Transformer encoder) and recent-window FFT changes.
  (c) Multi-semantic fusion for multi-task prediction: joint prediction of
      defect existence, deficiency categories and detention.

INPUT AND SUPERVISION RULES:
  - The history branch reads hist_defect_subs, the accumulated deficiency codes of
    earlier inspections of the same ship.
  - labels_sub (codes of the current inspection) serves as an auxiliary supervision
    target (sub_class_loss) and in the metrics.
  - Pair with: mthgnn_psc_v8hq_complete_core_strict_temporal_v3_pure_defect

ANNOTATION MASK:
  - has_supervision marks inspections that carry category-level deficiency
    codes. Earlier records (pre-2020 in this corpus) report a deficiency count
    but no codes; they are unlabeled for defect existence and deficiency
    category, so both the losses and the metrics for these two tasks are
    restricted to annotated inspections. Detention is recorded for every
    inspection and is always supervised.

Usage:
  python train_msf_pscrp.py --data_dir <dataset> --checkpoint_dir <outdir>
"""

from __future__ import annotations
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import label_ranking_average_precision_score

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# =============================================================================
# Config
# =============================================================================

DATA_DIR = "data/mthgnn_psc_v8hq_complete_core_strict_temporal_v3_pure_defect"
CHECKPOINT_DIR = "checkpoint/msf_pscrp"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--embed_dim", type=int, default=320)
parser.add_argument("--time_dim", type=int, default=96)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--dropout", type=float, default=0.15)
parser.add_argument("--label_smoothing", type=float, default=0.05)
parser.add_argument("--focal_gamma", type=int, default=2)
parser.add_argument("--cb_beta", type=float, default=0.9)
parser.add_argument("--detection_weight", type=float, default=1.0)
parser.add_argument("--classification_weight", type=float, default=2.5)
parser.add_argument("--sub_class_weight", type=float, default=0.5)
parser.add_argument("--stage1_detection_weight", type=float, default=2.0,
                   help="Detection loss weight before stage1_epochs")
parser.add_argument("--stage1_classification_weight", type=float, default=0.5,
                   help="Classification loss weight before stage1_epochs")
parser.add_argument("--detain_weight", type=float, default=0.3,
                   help="Weight for detention-risk auxiliary task")
parser.add_argument("--detain_pos_weight_cap", type=float, default=50.0,
                   help="Maximum positive-class weight for detention BCE")
parser.add_argument("--hier_loss_weight", type=float, default=0.0,
                   help="[v3-HIER] Hierarchical consistency loss weight: KL(P_sub || P_main @ M^T) "
                        "where M is the sub→main incidence matrix derived from training co-occurrences.")
parser.add_argument("--hier_soft_temperature", type=float, default=1.0,
                   help="[v3-HIER] Softmax temperature for the hierarchical KL target distribution.")
parser.add_argument("--patience", type=int, default=18)
parser.add_argument("--eval_every", type=int, default=2)
parser.add_argument("--max_neighbors", type=int, default=50)
parser.add_argument("--top_k", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--deterministic", action="store_true", default=False,
                   help="Enable deterministic mode (cudnn deterministic + benchmark off) for bit-exact reproducibility")
parser.add_argument("--name", type=str, default="mthgnn_v8hq_v3")
parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Dataset directory")
parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR, help="Checkpoint/output directory")
parser.add_argument("--num_heads", type=int, default=8)
parser.add_argument("--stage1_epochs", type=int, default=15)
parser.add_argument("--lr_scheduler", type=str, default="cosine",
                   choices=["warm_restart", "cosine", "step"])
parser.add_argument("--tune_cls_threshold", action="store_true", default=True)
parser.add_argument("--active_views", type=str, default="")
parser.add_argument("--use_defect_sub_view", action="store_true", default=False,
                   help="Add DefectSub view (fine-grained 521-class defect neighbors)")
parser.add_argument("--seq_num_layers", type=int, default=3)
parser.add_argument("--fft_low_dim", type=int, default=96)
parser.add_argument("--fft_high_dim", type=int, default=96)
parser.add_argument("--max_hist_inspections", type=int, default=8)
parser.add_argument("--disable_fft", action="store_true")
parser.add_argument("--use_amp", action="store_true", default=True)
parser.add_argument("--ship_type_embed_dim", type=int, default=32)
parser.add_argument("--disable_seq_encoder", action="store_true", help="禁用序列编码器（消融实验）")
parser.add_argument("--disable_cross_attn", action="store_true", help="禁用交叉注意力（消融实验）")
parser.add_argument("--disable_transformer", action="store_true", help="禁用Transformer分支，仅保留FFT（消融实验）")
parser.add_argument("--disable_time_encoding", action="store_true",
                   help="Disable temporal encoding in meta-path view attention")
parser.add_argument("--disable_cb_loss", action="store_true", help="禁用Class-Balanced Loss，改用标准BCE（消融实验）")
parser.add_argument("--disable_port_embed", action="store_true", help="禁用港口节点嵌入（消融实验：去掉港口差异化特征，仅保留port_view聚合）")
parser.add_argument("--view_fusion_init", type=str, default="prior",
                   choices=["prior", "uniform"],
                   help="Initialize meta-path view fusion weights from domain prior or uniform weights")
parser.add_argument("--cls_include_clean", action="store_true", default=False,
                   help="Also supervise annotated defect-free inspections with all-zero "
                        "category targets (calibrates the classification head).")
parser.add_argument("--freeze_view_fusion", action="store_true",
                   help="Fix meta-path view fusion weights after initialization")
parser.add_argument("--eval_only", action="store_true", help="仅评估checkpoint，不训练")
parser.add_argument("--eval_ckpt", type=str, default="", help="评估指定checkpoint路径（与eval_only配合）")
parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers数量")
parser.add_argument("--prefetch_factor", type=int, default=4, help="每worker预取batch数")
parser.add_argument("--select_det_weight", type=float, default=0.4,
                   help="Validation selection weight for defect-existence F1")
parser.add_argument("--select_cls_weight", type=float, default=1.0,
                   help="Validation selection weight for class macro-F1-CB/HQ [ENHANCED: now defaults to 1.0]")
parser.add_argument("--select_auc_weight", type=float, default=0.2,
                   help="Validation selection weight for macro-AUC")
parser.add_argument("--select_hit_weight", type=float, default=0.4,
                   help="Validation selection weight for Hit@K")
parser.add_argument("--select_detain_weight", type=float, default=0.2,
                   help="Validation selection weight for detention F1")
parser.add_argument("--select_sub_hit_weight", type=float, default=0.0,
                   help="Validation selection weight for subclass Top-K hit")
args = parser.parse_args()

DATA_DIR = args.data_dir
CHECKPOINT_DIR = args.checkpoint_dir
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

EMBED_DIM = args.embed_dim
TIME_DIM = args.time_dim
BATCH_SIZE = args.batch_size
DROPOUT = args.dropout
FOCAL_GAMMA = args.focal_gamma
CB_BETA = args.cb_beta
LABEL_SMOOTHING = args.label_smoothing
DETECTION_WEIGHT = args.detection_weight
CLASSIFICATION_WEIGHT = args.classification_weight
SUB_CLASS_WEIGHT = args.sub_class_weight
STAGE1_DETECTION_WEIGHT = args.stage1_detection_weight
STAGE1_CLASSIFICATION_WEIGHT = args.stage1_classification_weight
DETAIN_WEIGHT = args.detain_weight
HIER_LOSS_WEIGHT = args.hier_loss_weight
HIER_SOFT_TEMPERATURE = args.hier_soft_temperature
TOP_K = args.top_k
MAX_NEIGHBORS = args.max_neighbors
STAGE1_EPOCHS = args.stage1_epochs
DETECTION_THRESHOLD = 0.5
CLASSIFICATION_THRESHOLD = 0.25
MAX_HIST = args.max_hist_inspections
FFT_LOW_DIM = args.fft_low_dim
FFT_HIGH_DIM = args.fft_high_dim
DISABLE_FFT = args.disable_fft
USE_AMP = args.use_amp
SHIP_TYPE_EMBED_DIM = args.ship_type_embed_dim
DISABLE_SEQ_ENCODER = args.disable_seq_encoder
DISABLE_CROSS_ATTN = args.disable_cross_attn
DISABLE_TRANSFORMER = args.disable_transformer
DISABLE_TIME_ENCODING = args.disable_time_encoding
DISABLE_CB_LOSS = args.disable_cb_loss
DISABLE_PORT_EMBED = args.disable_port_embed
VIEW_FUSION_INIT = args.view_fusion_init
FREEZE_VIEW_FUSION = args.freeze_view_fusion
CLS_INCLUDE_CLEAN = args.cls_include_clean
NUM_HEADS = args.num_heads
USE_DEFECT_SUB_VIEW = args.use_defect_sub_view
DEFECT_SUB_VIEW_MAX_NEIGHBORS = 20
NUM_WORKERS = args.num_workers

SELECT_DET_WEIGHT = args.select_det_weight
SELECT_CLS_WEIGHT = args.select_cls_weight
SELECT_AUC_WEIGHT = args.select_auc_weight
SELECT_HIT_WEIGHT = args.select_hit_weight
SELECT_DETAIN_WEIGHT = args.select_detain_weight
SELECT_SUB_HIT_WEIGHT = args.select_sub_hit_weight
PREFETCH_FACTOR = args.prefetch_factor

ACTIVE_VIEWS = args.active_views.split(',') if args.active_views else [
    'port_view', 'country_view', 'mou_view', 'defect_view', 'company_view']

if USE_DEFECT_SUB_VIEW and 'defect_sub_view' not in ACTIVE_VIEWS:
    ACTIVE_VIEWS = ACTIVE_VIEWS + ['defect_sub_view']

print(f"[INFO] Active views: {ACTIVE_VIEWS}")
print(f"[INFO] DefectSub View: {USE_DEFECT_SUB_VIEW} ({DEFECT_SUB_VIEW_MAX_NEIGHBORS} neighbors)")
print(f"[INFO] FFT: low={FFT_LOW_DIM}, high={FFT_HIGH_DIM}, disabled={DISABLE_FFT}")
print(f"[INFO] SEQ: layers={args.seq_num_layers}, max_insp={MAX_HIST}")
print(f"[INFO] Ship type embed dim: {SHIP_TYPE_EMBED_DIM}")
print(f"[INFO] AMP: {USE_AMP}, num_heads={NUM_HEADS}")
print(f"[INFO] Time encoding disabled: {DISABLE_TIME_ENCODING}")
print(f"[INFO] View fusion: init={VIEW_FUSION_INIT}, frozen={FREEZE_VIEW_FUSION}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

# =============================================================================
# Cross-Attention Fusion Module (NEW in v2)
# =============================================================================

class CrossAttentionFusion(nn.Module):
    """Cross-attention between entity context and sequence representation."""
    def __init__(self, query_dim: int, key_dim: int, num_heads: int = 4, dropout: float = DROPOUT):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        assert query_dim % num_heads == 0
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(query_dim))

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B = query.size(0)
        Q = self.q_proj(query).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=DROPOUT, training=self.training)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, 1, -1).squeeze(1)
        return self.out_proj(out)


# =============================================================================
# Per-Inspection Sequence Encoder (Time-Domain Branch)
# =============================================================================

class InspectionSequenceEncoder(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int,
                 num_heads: int = 8, num_layers: int = 3,
                 dropout: float = DROPOUT):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.insp_proj = nn.Sequential(
            nn.Linear(num_classes, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Embedding(MAX_HIST + 1, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 3, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, seq_matrix: torch.Tensor) -> torch.Tensor:
        B, L, C = seq_matrix.shape
        insp_emb = self.insp_proj(seq_matrix)

        cls_tokens = self.cls_token.expand(B, 1, -1)
        seq_emb = torch.cat([cls_tokens, insp_emb], dim=1)

        seq_len = seq_emb.size(1)
        positions = torch.arange(seq_len, device=seq_emb.device).unsqueeze(0).expand(B, -1)
        seq_emb = seq_emb + self.pos_embed(positions)

        encoded = self.transformer(seq_emb)
        return encoded[:, 0, :]


# =============================================================================
# FFT Frequency-Domain Branch
# =============================================================================

class FFTDefectAnalyzer(nn.Module):
    def __init__(self, num_classes: int = 35,
                 low_dim: int = FFT_LOW_DIM, high_dim: int = FFT_HIGH_DIM,
                 max_hist: int = None):
        super().__init__()
        self.num_classes = num_classes
        self.low_dim = low_dim
        self.high_dim = high_dim
        self.max_hist = max_hist if max_hist is not None else MAX_HIST

        self.dc_proj = nn.Sequential(
            nn.Linear(num_classes, low_dim),
            nn.LayerNorm(low_dim),
            nn.ReLU(),
        )

        self.low_freq_proj = nn.Sequential(
            nn.Linear(num_classes * 2, low_dim),
            nn.LayerNorm(low_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

        high_feat_dim = num_classes * (self.max_hist - 3)
        self.high_freq_proj = nn.Sequential(
            nn.Linear(high_feat_dim, high_dim),
            nn.LayerNorm(high_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

        self.freq_gate = nn.Sequential(
            nn.Linear(low_dim, low_dim),
            nn.Sigmoid()
        )

    def forward(self, seq_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, C = seq_matrix.shape
        fft_mag = torch.abs(torch.fft.fft(seq_matrix, dim=1))[:, :L, :]

        dc = fft_mag[:, 0, :]
        low_freq = fft_mag[:, 1:min(3, L), :].transpose(1, 2)
        if L > 3:
            high_freq = fft_mag[:, 3:L, :].transpose(1, 2)
        else:
            high_freq = torch.zeros(B, C, 1, device=seq_matrix.device, dtype=seq_matrix.dtype)

        dc_emb = self.dc_proj(dc)
        low_freq_emb = self.low_freq_proj(low_freq.reshape(B, -1))
        high_freq_emb = self.high_freq_proj(high_freq.reshape(B, -1))

        gate = self.freq_gate(low_freq_emb)
        low_freq_emb = gate * low_freq_emb + (1 - gate) * dc_emb

        return low_freq_emb, high_freq_emb


# =============================================================================
# Dual-Branch Sequence Encoder
# =============================================================================

class DualBranchSeqEncoder(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int,
                 low_dim: int = FFT_LOW_DIM, high_dim: int = FFT_HIGH_DIM):
        super().__init__()
        self.seq_encoder = InspectionSequenceEncoder(num_classes, embed_dim,
                                            num_heads=NUM_HEADS,
                                            num_layers=args.seq_num_layers)
        self.fft_analyzer = FFTDefectAnalyzer(num_classes, low_dim, high_dim, max_hist=MAX_HIST)

        fusion_dim_full = embed_dim + low_dim + high_dim + embed_dim
        fusion_dim_short = embed_dim * 2  # seq_emb + cumulative_hist
        self.fusion_proj_full = nn.Sequential(
            nn.Linear(fusion_dim_full, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
        self.fusion_proj_short = nn.Sequential(
            nn.Linear(fusion_dim_short, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

    def forward(self, seq_matrix: torch.Tensor,
                cumulative_hist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if DISABLE_TRANSFORMER:
            seq_emb_out = torch.zeros(cumulative_hist.shape[0], self.seq_encoder.embed_dim,
                                     device=cumulative_hist.device, dtype=cumulative_hist.dtype)
            seq_emb = seq_emb_out
        else:
            seq_emb = self.seq_encoder(seq_matrix)
            seq_emb_out = seq_emb

        if DISABLE_FFT and DISABLE_TRANSFORMER:
            fused = torch.cat([seq_emb_out, cumulative_hist], dim=-1)
            return self.fusion_proj_short(fused), seq_emb_out
        elif DISABLE_FFT:
            fused = torch.cat([seq_emb, cumulative_hist], dim=-1)
            return self.fusion_proj_short(fused), seq_emb_out
        elif DISABLE_TRANSFORMER:
            low_freq, high_freq = self.fft_analyzer(seq_matrix)
            fused = torch.cat([seq_emb_out, low_freq, high_freq, cumulative_hist], dim=-1)
            return self.fusion_proj_full(fused), seq_emb_out
        else:
            low_freq, high_freq = self.fft_analyzer(seq_matrix)
            fused = torch.cat([seq_emb, low_freq, high_freq, cumulative_hist], dim=-1)
            return self.fusion_proj_full(fused), seq_emb_out


# =============================================================================
# V7 Components
# =============================================================================

class EnhancedTimeEncoder(nn.Module):
    def __init__(self, time_dim: int = TIME_DIM, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.w = nn.Parameter(torch.randn(time_dim) * 0.1)
        self.b = nn.Parameter(torch.randn(time_dim) * 0.1)
        self.scale = math.sqrt(1.0 / time_dim)
        self.time_delta_proj = nn.Sequential(
            nn.Linear(1, time_dim // 2), nn.ReLU(),
            nn.Linear(time_dim // 2, time_dim))
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, embed_dim), nn.LayerNorm(embed_dim),
            nn.ReLU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, delta_t: torch.Tensor,
                time_deltas: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_enc = self.scale * torch.cos(self.w * delta_t.unsqueeze(-1) + self.b)
        if time_deltas is not None and time_deltas.numel() > 0:
            base_enc = base_enc + self.time_delta_proj(time_deltas.mean(dim=-1, keepdim=True))
        return self.time_proj(base_enc)

    def encode_neighbor_delta(self, delta_years: torch.Tensor) -> torch.Tensor:
        """Encode per-neighbor time gaps; shape [..., N] -> [..., N, embed_dim]."""
        base_enc = self.scale * torch.cos(self.w * delta_years.unsqueeze(-1) + self.b)
        return self.time_proj(base_enc)


class ViewTemporalAttention(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, time_dim: int = TIME_DIM,
                 num_heads: int = 8, dropout: float = DROPOUT):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.time_proj = nn.Linear(time_dim, embed_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Dropout(dropout), nn.LayerNorm(embed_dim))

    def forward(self, query: torch.Tensor, keys: torch.Tensor,
                time_encoding: torch.Tensor,
                neighbor_time_encoding: Optional[torch.Tensor] = None,
                key_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = query.size(0)
        te = time_encoding if time_encoding.size(-1) == self.embed_dim else self.time_proj(time_encoding)
        if neighbor_time_encoding is not None:
            # New strict-temporal path: target inspection time conditions the query,
            # and each neighbor receives its own delta-time encoding.
            Q = self.W_Q(query + te).unsqueeze(1)
            nte = neighbor_time_encoding
            if nte.size(-1) != self.embed_dim:
                nte = self.time_proj(nte)
            K = self.W_K(keys + nte)
        else:
            # Backward-compatible path for old datasets without per-neighbor deltas.
            Q = self.W_Q(query).unsqueeze(1)
            K = self.W_K(keys)
            K = K + te.unsqueeze(1)
        V = self.W_V(keys)
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        valid_any = None
        if key_mask is not None:
            mask = key_mask.to(device=scores.device, dtype=torch.bool).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, -1e4)
            valid_any = key_mask.to(device=scores.device, dtype=torch.float32).any(dim=-1).unsqueeze(-1)
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=DROPOUT, training=self.training)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, 1, self.embed_dim).squeeze(1)
        if valid_any is not None:
            context = context * valid_any
        return self.out_proj(context)


class HistDefectTokenEncoder(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int, dropout: float = 0.25, nhead: int = 4):
        super().__init__()
        self.token_bias = nn.Parameter(torch.zeros(num_classes, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 2,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.out_ln = nn.LayerNorm(embed_dim)

    def forward(self, hist_multihot: torch.Tensor) -> torch.Tensor:
        B, C = hist_multihot.shape
        tok = self.token_bias.unsqueeze(0).expand(B, C, -1) * hist_multihot.unsqueeze(-1)
        pad = hist_multihot <= 0.5
        active = hist_multihot.sum(dim=1) > 0.5
        pooled = torch.zeros(B, self.token_bias.shape[1], device=hist_multihot.device, dtype=tok.dtype)
        if active.any():
            sub_tok = tok[active]
            sub_pad = pad[active]
            sub_out = self.encoder(sub_tok, src_key_padding_mask=sub_pad)
            m = (~sub_pad).float().unsqueeze(-1)
            pooled[active] = (sub_out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.out_ln(pooled)


# =============================================================================
# Loss Functions
# =============================================================================

class DetectionLoss(nn.Module):
    def __init__(self, pos_weight: float = 14.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor(self.pos_weight, device=logits.device))


class ClassBalancedLoss(nn.Module):
    """Class-Balanced Loss with label smoothing (used in V8HQ v2)."""
    def __init__(self, beta: float = 0.9, label_smoothing: float = 0.05,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        if self.class_weights is not None:
            bce = bce * self.class_weights.unsqueeze(0)
        return bce.mean()


class SubClassLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets)


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification (MSCOCO SOTA).
    Adaptively down-weights easy negatives based on their probability.
    γ_neg: negative discount (higher = more aggressive discounting, 4 recommended)
    γ_pos: positive boost (lower = more focus on hard positives, 0 recommended)
    clip: probability floor for negatives (prevents overdiscounting)
    
    Reference: Li et al., "Bridging the Gap between Classification and Localization
               for Weakly Supervised Object Localization" (ASL, CVPR 2020)
    """
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0,
                 clip: float = 0.05, eps: float = 1e-8,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Asymmetric probabilities
        p_pos = probs.clamp(min=self.eps)
        p_neg = (1 - probs).clamp(min=self.eps)

        # Positive loss (standard BCE)
        pos_loss = targets * torch.log(p_pos)
        # Negative loss with asymmetric discounting
        neg_loss = (1 - targets) * torch.log(p_neg)
        loss = -(pos_loss + neg_loss)

        # Asymmetric modulation
        # Adaptive hard negative mining: down-weight easy negatives
        asym_neg = ((1 - targets) * (1 - probs.pow(self.gamma_neg))).clamp(min=self.clip)
        # Adaptive positive focusing: down-weight easy positives  
        asym_pos = (targets * probs.pow(self.gamma_pos))
        loss = loss * (asym_pos + asym_neg)

        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)

        return loss.mean()


# =============================================================================
# Dataset and Collate
# =============================================================================

class PSCDatasetV8hq(Dataset):
    def __init__(self, data_dir: str, split: str):
        self.data_dir = data_dir
        self.split = split
        self.samples = []
        self.entity2id = {}
        self.defect_classes = {}
        self.sub_classes = {}
        self.defect_text_to_idx = {}
        self.defect_sub_neighbors = {}  # {local_idx: [entity_ids]}
        self._load_data()
        if USE_DEFECT_SUB_VIEW:
            self._load_defect_sub_neighbors()

    def _load_data(self):
        with open(os.path.join(self.data_dir, "entities.txt"), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    self.entity2id[parts[0]] = int(parts[1])

        with open(os.path.join(self.data_dir, "statistics.json"), encoding="utf-8") as f:
            stats = json.load(f)
            self.defect_classes = stats.get("main_class_to_idx", stats.get("defect_classes", {}))
            self.sub_classes = stats.get("sub_class_to_idx", stats.get("sub_classes", {}))
            self.defect_text_to_idx = {n: i for n, i in self.defect_classes.items()}

        split_file = os.path.join(self.data_dir, "splits", f"{self.split}_enriched_fixed.json")
        if not os.path.exists(split_file):
            split_file = os.path.join(self.data_dir, "splits", f"{self.split}.json")
        with open(split_file, encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def _load_defect_sub_neighbors(self):
        fpath = os.path.join(self.data_dir, "splits", f"defect_sub_neighbors_{self.split}.json")
        if os.path.exists(fpath):
            with open(fpath, encoding="utf-8") as f:
                self.defect_sub_neighbors = json.load(f)
            print(f"  [DefectSub View] Loaded {len(self.defect_sub_neighbors)} neighbor entries from {fpath}")
        else:
            print(f"  [DefectSub View] WARNING: {fpath} not found. DefectSub view will be empty.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx].copy()
        sample['defect_classes'] = self.defect_classes
        sample['sub_classes'] = self.sub_classes
        sample['defect_text_to_idx'] = self.defect_text_to_idx
        if USE_DEFECT_SUB_VIEW and str(idx) in self.defect_sub_neighbors:
            sample['neighbors_defect_sub'] = self.defect_sub_neighbors[str(idx)]
        else:
            sample['neighbors_defect_sub'] = []
        return sample


def collate_fn(batch):
    B = len(batch)
    num_classes_main = len(batch[0].get('defect_classes', {}))
    num_classes_sub = len(batch[0].get('sub_classes', {}))
    defect_to_idx = batch[0].get('defect_text_to_idx', {})
    sub_to_idx = batch[0].get('sub_classes', {})

    # ---- Per-inspection sequence matrix ----
    seq_matrix = torch.zeros(B, MAX_HIST, num_classes_main, dtype=torch.float32)
    for i, b in enumerate(batch):
        insp_seq = b.get('hist_insp_seq', [])
        for j, insp_defects in enumerate(insp_seq):
            if j >= MAX_HIST:
                break
            for defect_text in insp_defects:
                if defect_text in defect_to_idx:
                    idx = defect_to_idx[defect_text]
                    if idx < num_classes_main:
                        seq_matrix[i, j, idx] = 1.0

    # ---- Cumulative history ----
    hist_defect_cats_list = []
    for b in batch:
        cats = b.get('hist_defect_cats', [])
        tensor = torch.zeros(num_classes_main, dtype=torch.float32)
        for cat_text in cats:
            if cat_text in defect_to_idx:
                tensor[defect_to_idx[cat_text]] = 1.0
        hist_defect_cats_list.append(tensor)
    hist_defect_cats = torch.stack(hist_defect_cats_list)

    # ---- Causal historical sub-defects ----
    # Do not use labels_sub here: labels_sub is the current inspection target.
    # Clean causal datasets can provide hist_defect_subs as text labels.
    hist_defect_subs_list = []
    for b in batch:
        subs = b.get('hist_defect_subs', [])
        tensor = torch.zeros(num_classes_sub, dtype=torch.float32)
        for sub_text in subs:
            if sub_text in sub_to_idx:
                tensor[int(sub_to_idx[sub_text])] = 1.0
        hist_defect_subs_list.append(tensor)
    hist_defect_subs = torch.stack(hist_defect_subs_list)

    # ---- Labels ----
    labels_main_list = []
    for b in batch:
        raw = b.get('labels_main', [])
        tensor = torch.zeros(num_classes_main, dtype=torch.float32)
        if len(raw) == num_classes_main and all(v in (0, 1) for v in raw):
            tensor = torch.tensor(raw, dtype=torch.float32)
        else:
            for idx in raw:
                if 0 <= int(idx) < num_classes_main:
                    tensor[int(idx)] = 1.0
        labels_main_list.append(tensor)
    labels_main = torch.stack(labels_main_list)

    labels_sub_list = []
    for b in batch:
        raw = b.get('labels_sub', [])
        tensor = torch.zeros(num_classes_sub, dtype=torch.float32)
        if len(raw) == num_classes_sub and all(v in (0, 1) for v in raw):
            tensor = torch.tensor(raw, dtype=torch.float32)
        else:
            for idx in raw:
                if 0 <= int(idx) < num_classes_sub:
                    tensor[int(idx)] = 1.0
        labels_sub_list.append(tensor)
    labels_sub = torch.stack(labels_sub_list)

    # ---- Neighbors ----
    view_nbr_lists = {v: [] for v in ACTIVE_VIEWS}
    view_delta_lists = {v: [] for v in ACTIVE_VIEWS}
    has_edge_time_deltas = any(isinstance(b.get('view_neighbor_delta_years'), dict) for b in batch)
    for b in batch:
        vn = b.get('view_neighbors', {})
        vd = b.get('view_neighbor_delta_years', {})
        if vn:
            for view_name in ACTIVE_VIEWS:
                nbrs = vn.get(view_name, [])
                view_nbr_lists[view_name].append(nbrs if isinstance(nbrs, list) else [])
                deltas = vd.get(view_name, []) if isinstance(vd, dict) else []
                view_delta_lists[view_name].append(deltas if isinstance(deltas, list) else [])
        else:
            for view_name in ACTIVE_VIEWS:
                view_nbr_lists[view_name].append([])
                view_delta_lists[view_name].append([])

    neighbors_dict = {}
    neighbor_delta_dict = {}
    neighbor_mask_dict = {}
    view_key_map = {
        'port_view': 'neighbors_port', 'country_view': 'neighbors_country',
        'mou_view': 'neighbors_mou', 'defect_view': 'neighbors_defect',
        'company_view': 'neighbors_company',
    }
    for view_name in ACTIVE_VIEWS:
        if view_name == 'defect_sub_view':
            continue
        view_key = view_key_map.get(view_name, view_name)
        nbr_lists = view_nbr_lists.get(view_name, [])
        delta_lists = view_delta_lists.get(view_name, [])
        padded = []
        padded_deltas = []
        padded_masks = []
        for nbrs, deltas in zip(nbr_lists, delta_lists):
            if isinstance(nbrs, list):
                ids = [int(x) for x in nbrs[:MAX_NEIGHBORS]]
                valid_len = len(ids)
                ids += [0] * (MAX_NEIGHBORS - len(ids))
            else:
                ids = [0] * MAX_NEIGHBORS
                valid_len = 0
            if isinstance(deltas, list):
                ds = [float(x) for x in deltas[:MAX_NEIGHBORS]]
                ds += [0.0] * (MAX_NEIGHBORS - len(ds))
            else:
                ds = [0.0] * MAX_NEIGHBORS
            mask = [1.0] * valid_len + [0.0] * (MAX_NEIGHBORS - valid_len)
            padded.append(ids[:MAX_NEIGHBORS])
            padded_deltas.append(ds[:MAX_NEIGHBORS])
            padded_masks.append(mask[:MAX_NEIGHBORS])
        neighbors_dict[view_key] = padded
        if has_edge_time_deltas:
            neighbor_delta_dict[view_key] = torch.tensor(padded_deltas, dtype=torch.float32)
            neighbor_mask_dict[view_key] = torch.tensor(padded_masks, dtype=torch.bool)

    # defect_sub_view: separate max_neighbors
    if USE_DEFECT_SUB_VIEW and 'defect_sub_view' in ACTIVE_VIEWS:
        sub_nbr_lists = []
        for b in batch:
            nbrs = b.get('neighbors_defect_sub', [])
            if isinstance(nbrs, list):
                ids = [int(x) for x in nbrs[:DEFECT_SUB_VIEW_MAX_NEIGHBORS]]
                ids += [0] * (DEFECT_SUB_VIEW_MAX_NEIGHBORS - len(ids))
            else:
                ids = [0] * DEFECT_SUB_VIEW_MAX_NEIGHBORS
            sub_nbr_lists.append(ids[:DEFECT_SUB_VIEW_MAX_NEIGHBORS])
        neighbors_dict['neighbors_defect_sub'] = sub_nbr_lists

    # ---- Detain / supervision ----
    is_detain_list = []
    for b in batch:
        v = b.get('is_detain', 0)
        if isinstance(v, list):
            v = v[0] if v else 0
        is_detain_list.append(float(v))
    is_detain = torch.tensor(is_detain_list, dtype=torch.float32).unsqueeze(-1)

    has_sup_list = []
    for b in batch:
        v = b.get('has_supervision', 0)
        if isinstance(v, list):
            v = v[0] if v else 0
        has_sup_list.append(bool(v))
    has_sup = torch.tensor(has_sup_list, dtype=torch.bool)

    num_hist_list = []
    for b in batch:
        v = b.get('hist_inspection_count')
        if v is None:
            v = b.get('num_hist_inspections', 0)
        num_hist_list.append(int(v) if v else 0)

    # ---- Ship type: integer index ----
    ship_type_idx = torch.zeros(B, dtype=torch.long)
    ship_type_map = {}
    all_types = [b.get('ship_type', '') for b in batch]
    for t in all_types:
        if t and t not in ship_type_map:
            ship_type_map[t] = len(ship_type_map) + 1
    for i, t in enumerate(all_types):
        ship_type_idx[i] = ship_type_map.get(t, 0)

    # ---- Ship gross ton ----
    gross_ton_list = []
    for b in batch:
        gt = float(b.get('gross_ton', 0) or 0)
        gross_ton_list.append(gt)
    mean_gt = sum(gross_ton_list) / len(gross_ton_list) if gross_ton_list else 1.0
    gross_ton_norm = torch.tensor([gt / (mean_gt + 1e-6) for gt in gross_ton_list], dtype=torch.float32).unsqueeze(-1)

    # ---- Port attributes ----
    port_lat_list = [float(b.get('port_lat', 0) or 0) for b in batch]
    port_lon_list = [float(b.get('port_lon', 0) or 0) for b in batch]
    ml = sum(port_lat_list) / len(port_lat_list) if port_lat_list else 1.0
    mo = sum(port_lon_list) / len(port_lon_list) if port_lon_list else 1.0
    sl = max(port_lat_list) - min(port_lat_list) if len(set(port_lat_list)) > 1 else 1.0
    so = max(port_lon_list) - min(port_lon_list) if len(set(port_lon_list)) > 1 else 1.0
    port_lat_norm = torch.tensor([(v - ml) / (sl + 1e-8) for v in port_lat_list], dtype=torch.float32).unsqueeze(-1)
    port_lon_norm = torch.tensor([(v - mo) / (so + 1e-8) for v in port_lon_list], dtype=torch.float32).unsqueeze(-1)

    # ---- Port country / MOU indices ----
    port_country_idx = torch.zeros(B, dtype=torch.long)
    port_country_map = {}
    all_pcs = [b.get('port_country', '') for b in batch]
    for pc in all_pcs:
        if pc and pc not in port_country_map:
            port_country_map[pc] = len(port_country_map) + 1
    for i, pc in enumerate(all_pcs):
        port_country_idx[i] = port_country_map.get(pc, 0)

    port_mou_idx = torch.zeros(B, dtype=torch.long)
    port_mou_map = {}
    all_pms = [b.get('port_mou', '') for b in batch]
    for pm in all_pms:
        if pm and pm not in port_mou_map:
            port_mou_map[pm] = len(port_mou_map) + 1
    for i, pm in enumerate(all_pms):
        port_mou_idx[i] = port_mou_map.get(pm, 0)

    batch_out = {
        'ship_imos': [b.get('ship_imo', '') for b in batch],
        'ports': [b.get('port', '') for b in batch],
        'mous': [b.get('mou', '') for b in batch],
        'flags': [b.get('flag', '') for b in batch],
        'companies': [b.get('company', '') for b in batch],
        'ship_types': [b.get('ship_type', '') for b in batch],
        'timestamp_rel': torch.tensor([b.get('timestamp_rel', 0.0) for b in batch], dtype=torch.float32),
        'hist_defect_cats': hist_defect_cats,
        'hist_defect_subs': hist_defect_subs,
        'labels_main': labels_main,
        'labels_sub': labels_sub,
        'is_detain': is_detain,
        'has_supervision': has_sup,
        'num_hist_inspections': num_hist_list,
        'neighbors': neighbors_dict,
        'hist_insp_seq_matrix': seq_matrix,
        'defect_classes': num_classes_main,
        'sub_classes': num_classes_sub,
        'ship_type_idx': ship_type_idx,
        'gross_ton_norm': gross_ton_norm,
        'port_lat_norm': port_lat_norm,
        'port_lon_norm': port_lon_norm,
        'port_country_idx': port_country_idx,
        'port_mou_idx': port_mou_idx,
    }
    if has_edge_time_deltas:
        batch_out['neighbor_time_deltas'] = neighbor_delta_dict
        batch_out['neighbor_masks'] = neighbor_mask_dict
    return batch_out


# =============================================================================
# V8HQ v2 MTHGNN Model
# =============================================================================

class MTHGNNV8hqV2(nn.Module):
    def __init__(self, entity2id: dict, defect_classes: dict, sub_classes: dict,
                 embed_dim: int = EMBED_DIM, time_dim: int = TIME_DIM,
                 active_views: List[str] = None,
                 num_ship_types: int = 100, num_port_countries: int = 200,
                 num_port_mous: int = 15, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.entity2id = entity2id
        self.defect_classes = defect_classes
        self.num_defect_classes = len(defect_classes)
        self.sub_classes = sub_classes
        self.num_sub_classes = len(sub_classes)
        self.active_views = active_views or list(ACTIVE_VIEWS)

        num_total = max(self.entity2id.values()) + 2

        self.ship_embed = nn.Embedding(num_total, embed_dim, padding_idx=0)
        self.port_embed = nn.Embedding(num_total, embed_dim, padding_idx=0)
        self.country_embed = nn.Embedding(num_total, embed_dim, padding_idx=0)
        self.mou_embed = nn.Embedding(num_total, embed_dim, padding_idx=0)
        self.company_embed = nn.Embedding(num_total, embed_dim, padding_idx=0)

        # ---- Ship attribute embeddings (fixed dims) ----
        self.ship_type_embed = nn.Embedding(num_ship_types, SHIP_TYPE_EMBED_DIM)
        # Ship attr: ship_type_embed(32) + gross_ton(1) = 33

        # ---- Port attribute embeddings (fixed dims) ----
        self.port_country_embed = nn.Embedding(num_port_countries, 32)
        self.port_mou_embed = nn.Embedding(num_port_mous, 16)
        # Port attr: country_embed(32) + mou_embed(16) + lat(1) + lon(1) = 50

        self.time_encoder = EnhancedTimeEncoder(time_dim, embed_dim)

        self.seq_encoder = DualBranchSeqEncoder(
            num_classes=self.num_defect_classes,
            embed_dim=embed_dim,
            low_dim=FFT_LOW_DIM,
            high_dim=FFT_HIGH_DIM,
        )

        self.view_attentions = nn.ModuleDict({
            name: ViewTemporalAttention(embed_dim, time_dim, num_heads=NUM_HEADS)
            for name in self.active_views
        })

        prior_weights = {
            'port_view': 0.15, 'country_view': 0.15, 'mou_view': 0.10,
            'defect_view': 0.35, 'company_view': 0.25
        }
        if VIEW_FUSION_INIT == "uniform":
            prior = torch.ones(len(self.active_views), dtype=torch.float32)
            prior = prior / prior.sum()
        else:
            prior = torch.tensor([prior_weights.get(v, 0.2) for v in self.active_views], dtype=torch.float32)
            prior = prior / prior.sum()
        self.view_fusion_scores = nn.Parameter(torch.log(prior.clamp_min(1e-6)))
        if FREEZE_VIEW_FUSION:
            self.view_fusion_scores.requires_grad_(False)

        self.hist_main_proj = nn.Sequential(
            nn.Linear(self.num_defect_classes, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(embed_dim, embed_dim))
        self.hist_sub_proj = nn.Sequential(
            nn.Linear(self.num_sub_classes, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(embed_dim // 2, embed_dim // 2))
        self.hist_token_encoder = HistDefectTokenEncoder(
            self.num_defect_classes, embed_dim, dropout=DROPOUT)

        self.history_proj = nn.Sequential(
            nn.Linear(1, embed_dim // 4), nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim // 4))

        # Cross-attention: entity context attends to sequence features (NEW in v2)
        self.cross_attn_seq = CrossAttentionFusion(
            query_dim=embed_dim, key_dim=embed_dim, num_heads=NUM_HEADS)
        self.cross_attn_view = CrossAttentionFusion(
            query_dim=embed_dim, key_dim=embed_dim, num_heads=NUM_HEADS)

        # Record projection input dim:
        # ship(256) + port(256) + country(256) + mou(256) + company(256) = 5*256 = 1280
        # + history(64) + hist_sub(128)                              = 192
        # + seq_fused(256) + fused(256)                             = 512
        # + seq_cross(256) + view_cross(256)                       = 512
        # + extra: ship_attrs(33) + port_attrs(50)                 = 83
        # Total = 2579
        rec_input_dim = (embed_dim * 5 + embed_dim // 4 + embed_dim // 2
                         + embed_dim * 2 + embed_dim * 2 + 33 + 50)
        # = 1024 + 64 + 128 + 512 + 512 + 83 = 2323
        self.record_proj = nn.Sequential(
            nn.Linear(rec_input_dim, embed_dim * 3),
            nn.LayerNorm(embed_dim * 3), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(DROPOUT / 2))

        # ---- Task Heads (stronger in v2) ----
        # Detection head: with residual
        self.detection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.LayerNorm(embed_dim // 2),
            nn.ReLU(), nn.Dropout(DROPOUT / 2),
            nn.Linear(embed_dim // 2, 1))

        # Defect classification head: 3-layer MLP (stronger in v2)
        self.defect_head = nn.Sequential(
            nn.Linear(embed_dim + 1, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(embed_dim, embed_dim // 2), nn.LayerNorm(embed_dim // 2),
            nn.GELU(), nn.Dropout(DROPOUT / 2),
            nn.Linear(embed_dim // 2, self.num_defect_classes))

        # Sub-class head: stronger in v2
        self.sub_class_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.LayerNorm(embed_dim // 2),
            nn.ReLU(), nn.Dropout(DROPOUT / 2),
            nn.Linear(embed_dim // 2, self.num_sub_classes))

        # Detain head: stronger in v2
        self.detain_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.LayerNorm(embed_dim // 2),
            nn.ReLU(), nn.Dropout(DROPOUT / 2),
            nn.Linear(embed_dim // 2, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def _get_entity_id(self, entity_str: str, prefix: str) -> int:
        key = f"{prefix}:{entity_str}" if entity_str else ''
        return self.entity2id.get(key, 0) if key else 0

    def _get_view_embeddings(self, batch: dict, ship_emb: torch.Tensor,
                            time_enc: torch.Tensor) -> List[torch.Tensor]:
        view_embs = []
        view_key_map = {
            'port_view': 'neighbors_port', 'country_view': 'neighbors_country',
            'mou_view': 'neighbors_mou', 'defect_view': 'neighbors_defect',
            'company_view': 'neighbors_company',
            'defect_sub_view': 'neighbors_defect_sub',
        }
        max_n_map = {
            'defect_sub_view': DEFECT_SUB_VIEW_MAX_NEIGHBORS,
        }
        for view_name in self.active_views:
            view_key = view_key_map.get(view_name, view_name)
            nbr_lists = batch['neighbors'].get(view_key, [])
            max_n = max_n_map.get(view_name, MAX_NEIGHBORS)
            nbr_ids = []
            for nbrs in nbr_lists:
                ids = [int(x) if x else 0 for x in nbrs]
                ids += [0] * (max_n - len(ids))
                nbr_ids.append(ids[:max_n])
            # All views use Ship entity embeddings
            nbr_tensor = torch.tensor(nbr_ids, dtype=torch.long, device=ship_emb.device)
            nbr_emb = self.ship_embed(nbr_tensor)
            delta_tensor = None
            mask_tensor = None
            if 'neighbor_time_deltas' in batch:
                delta_tensor = batch['neighbor_time_deltas'].get(view_key)
                if isinstance(delta_tensor, torch.Tensor):
                    delta_tensor = delta_tensor.to(ship_emb.device)
                    if delta_tensor.size(1) != max_n:
                        delta_tensor = delta_tensor[:, :max_n]
                    neighbor_time_enc = self.time_encoder.encode_neighbor_delta(delta_tensor)
                else:
                    neighbor_time_enc = None
            else:
                neighbor_time_enc = None
            if 'neighbor_masks' in batch:
                mask_tensor = batch['neighbor_masks'].get(view_key)
                if isinstance(mask_tensor, torch.Tensor):
                    mask_tensor = mask_tensor.to(ship_emb.device)
                    if mask_tensor.size(1) != max_n:
                        mask_tensor = mask_tensor[:, :max_n]
                else:
                    mask_tensor = None
            v_emb = self.view_attentions[view_name](
                ship_emb,
                nbr_emb,
                time_encoding=time_enc,
                neighbor_time_encoding=neighbor_time_enc,
                key_mask=mask_tensor,
            )
            view_embs.append(v_emb)
        return view_embs

    def forward(self, batch: dict):
        device = next(self.parameters()).device
        B = len(batch['ship_imos'])

        def to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return x

        # Move batch tensors to device
        for k in ['hist_defect_cats', 'hist_defect_subs', 'labels_sub', 'hist_insp_seq_matrix',
                   'timestamp_rel', 'ship_type_idx', 'gross_ton_norm',
                   'port_lat_norm', 'port_lon_norm', 'port_country_idx', 'port_mou_idx']:
            if k in batch:
                batch[k] = to_device(batch[k])

        ship_ids = torch.tensor(
            [self._get_entity_id(imo, 'Ship') for imo in batch['ship_imos']],
            dtype=torch.long, device=device)
        port_ids = torch.tensor(
            [self._get_entity_id(p, 'Port') for p in batch['ports']],
            dtype=torch.long, device=device)
        mou_ids = torch.tensor(
            [self._get_entity_id(m, 'MoU') for m in batch['mous']],
            dtype=torch.long, device=device)
        country_ids = torch.tensor(
            [self._get_entity_id(f, 'Country') for f in batch['flags']],
            dtype=torch.long, device=device)
        company_ids = torch.tensor(
            [self._get_entity_id(c, 'Company') for c in batch['companies']],
            dtype=torch.long, device=device)

        ship_emb = self.ship_embed(ship_ids)
        port_emb = self.port_embed(port_ids)
        mou_emb = self.mou_embed(mou_ids)
        country_emb = self.country_embed(country_ids)
        company_emb = self.company_embed(company_ids)

        num_hist = torch.tensor(batch['num_hist_inspections'],
                              dtype=torch.float32, device=device)
        num_hist_norm = (num_hist / (num_hist.max() + 1e-6)).unsqueeze(-1)
        history_emb = self.history_proj(num_hist_norm)

        hist_main = batch.get('hist_defect_cats', torch.zeros(B, self.num_defect_classes, device=device))
        # History branch: accumulated codes of earlier inspections.
        # labels_sub (codes of the current inspection) is used only in the
        # auxiliary sub_class_loss.
        hist_sub = batch.get('hist_defect_subs', torch.zeros(B, self.num_sub_classes, device=device))
        hist_main_emb = self.hist_main_proj(hist_main)
        hist_main_emb = hist_main_emb + self.hist_token_encoder(hist_main)
        hist_sub_emb = self.hist_sub_proj(hist_sub)

        seq_matrix = batch.get('hist_insp_seq_matrix')
        if seq_matrix is None or seq_matrix.numel() == 0:
            seq_matrix = torch.zeros(B, 1, self.num_defect_classes, device=device)
        if DISABLE_SEQ_ENCODER:
            seq_fused = torch.zeros(B, self.embed_dim, device=device)
            seq_raw_emb = torch.zeros(B, self.embed_dim, device=device)
        else:
            seq_fused, seq_raw_emb = self.seq_encoder(seq_matrix, hist_main_emb)

        ts = batch['timestamp_rel'].to(device)
        if DISABLE_TIME_ENCODING:
            time_enc = torch.zeros(B, self.embed_dim, device=device)
        else:
            time_enc = self.time_encoder(ts)

        view_embs = self._get_view_embeddings(batch, ship_emb, time_enc)
        fused = sum(v * w for v, w in zip(
            view_embs, F.softmax(self.view_fusion_scores, dim=-1).unbind()))

        # ---- Cross-attention: entity context attends to sequence & view features (v2) ----
        # Entity context: ship + port
        entity_context = ship_emb + port_emb
        if DISABLE_CROSS_ATTN:
            seq_cross = torch.zeros(B, self.embed_dim, device=device)
            view_cross = torch.zeros(B, self.embed_dim, device=device)
        else:
            seq_cross = self.cross_attn_seq(entity_context, seq_raw_emb)
            view_cross = self.cross_attn_view(entity_context, fused)

        # ---- Ship attributes ----
        ship_type_emb = self.ship_type_embed(batch['ship_type_idx'].to(device))
        gross_ton = batch['gross_ton_norm'].to(device)

        # ---- Port attributes ----
        if DISABLE_PORT_EMBED:
            port_country_emb = torch.zeros(B, SHIP_TYPE_EMBED_DIM, device=device, dtype=torch.float32)
            port_mou_emb = torch.zeros(B, 16, device=device, dtype=torch.float32)
            port_lat = torch.zeros(B, 1, device=device, dtype=torch.float32)
            port_lon = torch.zeros(B, 1, device=device, dtype=torch.float32)
        else:
            port_country_emb = self.port_country_embed(batch['port_country_idx'].to(device))
            port_mou_emb = self.port_mou_embed(batch['port_mou_idx'].to(device))
            port_lat = batch['port_lat_norm'].to(device)
            port_lon = batch['port_lon_norm'].to(device)

        extra_emb = torch.cat([ship_type_emb, gross_ton, port_country_emb,
                               port_mou_emb, port_lat, port_lon], dim=-1)

        # Final record representation
        # ship(256) + port(256) + country(256) + mou(256)
        # + history(64) + hist_sub(128)
        # + seq_fused(256) + fused(256)
        # + seq_cross(256) + view_cross(256)
        # + extra(83)
        # = 2323
        record_repr = torch.cat([
            ship_emb, port_emb, country_emb, mou_emb, company_emb,
            history_emb, hist_sub_emb,
            seq_fused, fused,
            seq_cross, view_cross,
            extra_emb
        ], dim=-1)
        record_emb = self.record_proj(record_repr)

        detection_logits = self.detection_head(record_emb).squeeze(-1)
        detection_probs = torch.sigmoid(detection_logits)

        classification_input = torch.cat(
            [record_emb, detection_probs.unsqueeze(-1)], dim=-1)
        defect_logits = self.defect_head(classification_input)
        sub_class_logits = self.sub_class_head(record_emb)
        detain_logits = self.detain_head(record_emb).squeeze(-1)

        return defect_logits, detection_logits, sub_class_logits, detain_logits


# =============================================================================
# Metrics
# =============================================================================

def compute_binary_metrics(labels, preds, threshold=0.5, prefix="binary"):
    labels = np.asarray(labels).astype(bool)
    preds = np.asarray(preds).astype(float)
    pred_labels = preds > threshold

    tp = (labels & pred_labels).sum()
    fp = ((~labels) & pred_labels).sum()
    fn = (labels & (~pred_labels)).sum()
    tn = ((~labels) & (~pred_labels)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / max(len(labels), 1)

    try:
        auc = float(roc_auc_score(labels.astype(np.float32), preds))
    except ValueError:
        auc = 0.0
    try:
        ap = float(average_precision_score(labels.astype(np.float32), preds))
    except ValueError:
        ap = 0.0

    return {
        f'{prefix}_f1': float(f1),
        f'{prefix}_precision': float(precision),
        f'{prefix}_recall': float(recall),
        f'{prefix}_accuracy': float(accuracy),
        f'{prefix}_auc': auc,
        f'{prefix}_ap': ap,
        f'{prefix}_threshold': float(threshold),
        f'{prefix}_support': int(labels.sum()),
    }


def search_best_binary_threshold(labels, preds, lo=0.05, hi=0.80, step=0.02,
                                  min_pos_support=10):
    """Search threshold maximizing F1.

    For low-support positive classes (e.g., detention with only 20 val positives),
    we fallback to 0.5 because threshold search becomes unstable. The
    `min_pos_support` argument prevents overfitting to noisy small-sample F1.
    """
    labels = np.asarray(labels).astype(bool)
    preds = np.asarray(preds).astype(float)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return 0.5
    # If too few positive samples, threshold search is unreliable
    if n_pos < min_pos_support:
        return 0.5
    best_f1, best_thresh = -1.0, 0.5
    for thresh in np.arange(lo, hi + step * 0.5, step):
        metrics = compute_binary_metrics(labels, preds, thresh, prefix="tmp")
        f1 = metrics["tmp_f1"]
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(thresh)
    return best_thresh


def compute_metrics(labels_main, preds_defect, preds_detection, threshold_defect=0.25,
                    labels_detain=None, preds_detain=None, threshold_detain=0.5,
                    per_class_thresholds: Optional[np.ndarray] = None):
    """
    Comprehensive evaluation for multi-label PSC defect prediction.
    Supports per-class threshold optimization for improved F1.
    All metrics are computed on the FULL test set (all samples),
    NOT filtered by has_defect (unlike cls_macro_f1 which uses pos_mask).
    """
    gt_has_defect = labels_main.any(axis=1)
    pred_has_defect = (preds_detection > DETECTION_THRESHOLD)

    # --- Detection metrics ---
    dt_tp = (gt_has_defect & pred_has_defect).sum()
    dt_fp = ((~gt_has_defect) & pred_has_defect).sum()
    dt_fn = (gt_has_defect & (~pred_has_defect)).sum()
    det_prec = dt_tp / (dt_tp + dt_fp + 1e-8)
    det_rec = dt_tp / (dt_tp + dt_fn + 1e-8)
    det_f1 = 2 * det_prec * det_rec / (det_prec + det_rec + 1e-8)

    # --- Classification metrics on ALL samples ---
    num_classes = preds_defect.shape[1]
    num_samples = preds_defect.shape[0]

    # Macro-AUC: per-class AUC, then average
    # Only compute AUC for classes that have both positive and negative samples in labels
    aucs = []
    aps = []
    for c in range(num_classes):
        col_labels = labels_main[:, c]
        col_preds = preds_defect[:, c]
        if col_labels.sum() > 0 and col_labels.sum() < len(col_labels):
            try:
                aucs.append(roc_auc_score(col_labels, col_preds))
                aps.append(average_precision_score(col_labels, col_preds))
            except ValueError:
                pass
    macro_auc = float(np.mean(aucs)) if aucs else 0.0
    macro_ap = float(np.mean(aps)) if aps else 0.0

    # Micro-AUC: flatten all (sample, label) pairs
    try:
        micro_auc = float(roc_auc_score(labels_main.ravel(), preds_defect.ravel()))
    except ValueError:
        micro_auc = 0.0
    try:
        micro_ap = float(average_precision_score(labels_main.ravel(), preds_defect.ravel()))
    except ValueError:
        micro_ap = 0.0

    # LRAP: Label Ranking Average Precision
    # Measures how well the ranking of labels matches ground truth
    try:
        lrap = float(label_ranking_average_precision_score(labels_main, preds_defect))
    except ValueError:
        lrap = 0.0

    # --- Classification metrics on HAS_DEFECT samples (original behavior) ---
    pos_mask = gt_has_defect
    if pos_mask.sum() == 0:
        cm = {
            'cls_macro_f1': 0.0, 'cls_micro_f1': 0.0,
            'cls_precision': 0.0, 'cls_recall': 0.0,
            'n_pos_eval': 0,
            'cls_macro_f1_hq': 0.0,
        }
    else:
        pos_preds = preds_defect[pos_mask]
        pos_labels = labels_main[pos_mask]
        # Per-class threshold: apply different thresholds to each of 35 classes
        if per_class_thresholds is not None:
            # per_class_thresholds shape: (num_classes,)
            binary_preds = (pos_preds > per_class_thresholds[np.newaxis, :]).astype(np.float32)
        else:
            binary_preds = (pos_preds > threshold_defect).astype(np.float32)
        tp = (pos_labels * binary_preds).sum(axis=0)
        fp = ((1 - pos_labels) * binary_preds).sum(axis=0)
        fn = (pos_labels * (1 - binary_preds)).sum(axis=0)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1_per_class = 2 * prec * rec / (prec + rec + 1e-8)

        # Macro-F1 (all active classes, zero-shot classes give NaN)
        cls_macro_f1 = float(np.nanmean(f1_per_class))

        # Macro-F1-HQ: only classes with at least HQ_MIN_SUPPORT positive samples
        # in pos_labels. HQ_MIN_SUPPORT=5 ensures F1 is statistically meaningful
        # (a class with only 1 positive sample produces unreliable F1).
        HQ_MIN_SUPPORT = 5
        hq_mask = pos_labels.sum(axis=0) >= HQ_MIN_SUPPORT
        if hq_mask.sum() > 0:
            cls_macro_f1_hq = float(np.nanmean(f1_per_class[hq_mask]))
        else:
            cls_macro_f1_hq = 0.0

        # HQ mask coverage for transparency
        n_hq_classes = int(hq_mask.sum())

        cm = {
            'cls_macro_f1': cls_macro_f1,
            'cls_macro_f1_hq': cls_macro_f1_hq,
            'cls_micro_f1': float(tp.sum() / (tp.sum() + 0.5 * fp.sum() + 0.5 * fn.sum())),
            'cls_precision': float(tp.sum() / (tp.sum() + fp.sum() + 1e-8)),
            'cls_recall': float(tp.sum() / (tp.sum() + fn.sum() + 1e-8)),
            'n_pos_eval': int(pos_mask.sum()),
            'n_hq_classes': n_hq_classes,
            'hq_min_support': HQ_MIN_SUPPORT,
        }

    # --- Top-K Hit (on has_defect samples) ---
    n_pos = cm['n_pos_eval']
    if n_pos > 0:
        pos_preds = preds_defect[pos_mask]
        pos_labels = labels_main[pos_mask]
        top_k = min(TOP_K, pos_labels.shape[1])
        top_k_indices = np.argsort(-pos_preds, axis=1)[:, :top_k]
        hits = sum(1 for i in range(n_pos)
                   if set(np.where(pos_labels[i] > 0.5)[0]) & set(top_k_indices[i]))
        top_k_hit = hits / n_pos
    else:
        top_k_hit = 0.0

    # --- Primary score ---
    primary = 0.3 * det_f1 + 0.4 * cm['cls_macro_f1'] + 0.3 * top_k_hit

    metrics = {
        # Detection
        'det_f1': float(det_f1),
        'det_precision': float(det_prec),
        'det_recall': float(det_rec),
        # Classification (has_defect samples)
        'cls_macro_f1': cm['cls_macro_f1'],
        'cls_macro_f1_hq': cm['cls_macro_f1_hq'],
        'cls_macro_f1_cb': cm['cls_macro_f1_hq'],
        'cls_micro_f1': cm['cls_micro_f1'],
        'cls_precision': cm['cls_precision'],
        'cls_recall': cm['cls_recall'],
        'top_k_hit': float(top_k_hit),
        'n_pos_eval': cm['n_pos_eval'],
        'primary': float(primary),
        # New ranking/AUC metrics (full dataset)
        'macro_auc': macro_auc,
        'micro_auc': micro_auc,
        'macro_ap': macro_ap,
        'micro_ap': micro_ap,
        'lrap': lrap,
    }
    if labels_detain is not None and preds_detain is not None:
        detain_metrics = compute_binary_metrics(
            labels_detain, preds_detain, threshold_detain, prefix="detain")
        metrics.update(detain_metrics)
        metrics['mt_primary'] = float(
            0.25 * metrics['det_f1']
            + 0.30 * metrics['cls_macro_f1_hq']
            + 0.25 * metrics['top_k_hit']
            + 0.20 * metrics['detain_f1'])
    return metrics


def search_best_per_class_thresholds(labels_main, preds_defect, lo=0.05, hi=0.85, step=0.05, min_pos_support=3):
    """Per-class threshold optimization: find best threshold for each class independently.
    This is critical for multi-label classification where different classes have
    very different optimal thresholds.
    Returns: numpy array of shape (num_classes,) with best threshold per class.
    """
    num_classes = preds_defect.shape[1]
    best_thresholds = np.full(num_classes, 0.25)  # default 0.25
    gt_has_defect = labels_main.any(axis=1)
    pos_mask = gt_has_defect
    pos_preds = preds_defect[pos_mask]
    pos_labels = labels_main[pos_mask]

    if len(pos_labels) == 0:
        return best_thresholds

    for c in range(num_classes):
        c_labels = pos_labels[:, c]
        c_preds = pos_preds[:, c]
        n_pos = int(c_labels.sum())
        # If too few positives, use default
        if n_pos < min_pos_support:
            best_thresholds[c] = 0.25
            continue
        # Search threshold for this class
        best_f1, best_t = -1.0, 0.25
        for thresh in np.arange(lo, hi + step * 0.5, step):
            binary_preds = (c_preds > thresh).astype(np.float32)
            tp = (c_labels * binary_preds).sum()
            fp = ((1 - c_labels) * binary_preds).sum()
            fn = (c_labels * (1 - binary_preds)).sum()
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best_f1:
                best_f1, best_t = f1, float(thresh)
        best_thresholds[c] = best_t
    return best_thresholds


def search_best_threshold(labels_main, preds_defect):
    best_f1, best_thresh = 0, 0.25
    gt = labels_main.any(axis=1)
    pos_mask = gt
    if pos_mask.sum() == 0:
        return best_thresh
    pos_preds = preds_defect[pos_mask]
    pos_labels = labels_main[pos_mask]
    for thresh in np.arange(0.10, 0.45, 0.02):
        bp = (pos_preds > thresh).astype(np.float32)
        tp = (pos_labels * bp).sum(axis=0)
        fp = ((1 - pos_labels) * bp).sum(axis=0)
        fn = (pos_labels * (1 - bp)).sum(axis=0)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        mf1 = float(np.nanmean(f1))
        if mf1 > best_f1:
            best_f1, best_thresh = mf1, thresh
    return best_thresh


def estimate_binary_pos_weight(samples, key: str, cap: float) -> float:
    pos, total = 0, 0
    for sample in samples:
        value = sample.get(key, 0)
        if isinstance(value, list):
            value = value[0] if value else 0
        pos += 1 if float(value or 0) > 0 else 0
        total += 1
    neg = max(total - pos, 1)
    if pos <= 0:
        return 1.0
    return float(min(neg / max(pos, 1), cap))


# =============================================================================
# Training
# =============================================================================

def compute_loss(batch, defect_logits, detection_logits, sub_class_logits,
                detain_logits, criterion_detection, criterion_defect, criterion_sub,
                criterion_detain, epoch, hier_matrix=None, hier_loss_weight=0.0,
                hier_temperature=1.0):
    device = defect_logits.device
    labels_main = batch['labels_main'].to(device)
    labels_sub = batch['labels_sub'].to(device)
    is_detain = batch['is_detain'].to(device).squeeze(-1)
    has_sup = batch['has_supervision'].to(device)

    labels_has_defect = (labels_main.sum(dim=-1) > 0).float()

    has_sup_1d = has_sup.squeeze(-1) if has_sup.dim() > 1 else has_sup

    # Defect-existence supervision is defined for annotated inspections.
    # Records created before category-level coding was introduced hold
    # deficiency_count > 0 but no category label (labels_main is all-zero); they
    # are excluded from the detection loss and from the detection metrics, while
    # still contributing relational and historical context through the
    # pre-computed views.
    if has_sup_1d.sum() > 0:
        detection_loss = criterion_detection(
            detection_logits[has_sup_1d], labels_has_defect[has_sup_1d])
    else:
        detection_loss = torch.tensor(0.0, device=device)

    # Category supervision: by default the classification loss is applied to
    # defect-positive inspections only (the operating regime of the metric).
    # --cls_include_clean additionally supervises annotated defect-free
    # inspections (all-zero targets), which calibrates the "absent" side.
    if CLS_INCLUDE_CLEAN:
        mask = has_sup_1d
    else:
        mask = labels_has_defect.bool() & has_sup_1d
    if mask.sum() > 0:
        classification_loss = criterion_defect(defect_logits[mask], labels_main[mask])
    else:
        classification_loss = torch.tensor(0.0, device=device)

    sub_class_loss = criterion_sub(sub_class_logits, labels_sub)
    detain_loss = criterion_detain(detain_logits, is_detain)

    # ---- Hierarchical consistency loss (v3-HIER) ----
    # Builds a soft target: P_sub_target = sigmoid(defect_logits) @ M^T
    # where M[s, m] = co-occurrence frequency of sub s under main m (normalized per row).
    # The intuition: predicted sub-class distribution should be consistent with predicted
    # main-class distribution via the hierarchy. labels_sub is NOT used as input feature here.
    hier_loss = torch.tensor(0.0, device=device)
    if hier_matrix is not None and hier_loss_weight > 0.0:
        main_probs = torch.sigmoid(defect_logits).float()    # [B, num_main]  cast to fp32 for matmul
        M = hier_matrix.to(device).float()                   # [num_sub, num_main]
        sub_target = main_probs @ M.t()                      # [B, num_sub]
        sub_target = sub_target.clamp(min=1e-6, max=1.0 - 1e-6)
        sub_pred = torch.sigmoid(sub_class_logits).float().clamp(min=1e-6, max=1.0 - 1e-6)
        hier_loss = -(sub_target * torch.log(sub_pred) +
                      (1 - sub_target) * torch.log(1 - sub_pred)).mean()

    if epoch < STAGE1_EPOCHS:
        det_w, cls_w, sub_w = STAGE1_DETECTION_WEIGHT, STAGE1_CLASSIFICATION_WEIGHT, 0.1
    else:
        det_w, cls_w, sub_w = DETECTION_WEIGHT, CLASSIFICATION_WEIGHT, SUB_CLASS_WEIGHT

    total = (det_w * detection_loss + cls_w * classification_loss
             + sub_w * sub_class_loss + DETAIN_WEIGHT * detain_loss
             + hier_loss_weight * hier_loss)
    return (total, detection_loss.detach(), classification_loss.detach(),
            sub_class_loss.detach(), detain_loss.detach(),
            hier_loss.detach() if isinstance(hier_loss, torch.Tensor) else hier_loss)


def train_epoch(model, loader, optimizer, criterion_detection, criterion_defect,
               criterion_sub, criterion_detain, epoch, scheduler=None, use_amp=False, scaler=None,
               hier_matrix=None, hier_loss_weight=0.0, hier_temperature=1.0):
    model.train()
    total_loss, total_det, total_cls, total_sub, total_detain, total_hier, n_batches = 0, 0, 0, 0, 0, 0, 0

    for batch in tqdm(loader, ncols=100, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                defect_logits, detection_logits, sub_logits, detain_logits = model(batch)
            loss, det_l, cls_l, sub_l, detain_l, hier_l = compute_loss(
                batch, defect_logits, detection_logits, sub_logits, detain_logits,
                criterion_detection, criterion_defect, criterion_sub, criterion_detain, epoch,
                hier_matrix=hier_matrix, hier_loss_weight=hier_loss_weight,
                hier_temperature=hier_temperature)

            if not math.isfinite(loss.item()):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            defect_logits, detection_logits, sub_logits, detain_logits = model(batch)
            loss, det_l, cls_l, sub_l, detain_l, hier_l = compute_loss(
                batch, defect_logits, detection_logits, sub_logits, detain_logits,
                criterion_detection, criterion_defect, criterion_sub, criterion_detain, epoch,
                hier_matrix=hier_matrix, hier_loss_weight=hier_loss_weight,
                hier_temperature=hier_temperature)
            if torch.isnan(loss) or not math.isfinite(loss.item()):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_det += det_l.item()
        total_cls += cls_l.item()
        total_sub += sub_l.item()
        total_detain += detain_l.item()
        total_hier += float(hier_l) if isinstance(hier_l, torch.Tensor) else hier_l
        n_batches += 1

    if scheduler is not None:
        scheduler.step()

    return {'loss': total_loss / max(n_batches, 1),
            'det_loss': total_det / max(n_batches, 1),
            'cls_loss': total_cls / max(n_batches, 1),
            'sub_loss': total_sub / max(n_batches, 1),
            'detain_loss': total_detain / max(n_batches, 1),
            'hier_loss': total_hier / max(n_batches, 1)}


@torch.no_grad()
def evaluate(model, loader, tune_threshold=False,
             cls_threshold: Optional[float] = None,
             detain_threshold: Optional[float] = None,
             per_class_thresholds: Optional[np.ndarray] = None):
    model.eval()
    all_d, all_l, all_det, all_detain, all_detain_l = [], [], [], [], []
    all_sub, all_sub_l = [], []
    all_sup = []
    for batch in tqdm(loader, ncols=100, desc="Evaluating", leave=False):
        defect_logits, detection_logits, sub_logits, detain_logits = model(batch)
        all_d.append(torch.sigmoid(defect_logits).cpu().numpy())
        all_sub.append(torch.sigmoid(sub_logits).cpu().numpy())
        all_sub_l.append(batch['labels_sub'].cpu().numpy())
        det = torch.sigmoid(detection_logits).cpu().numpy()
        if det.ndim == 2:
            det = det.squeeze(-1)
        all_det.append(det)
        detain = torch.sigmoid(detain_logits).cpu().numpy()
        if detain.ndim == 2:
            detain = detain.squeeze(-1)
        all_detain.append(detain)
        all_detain_l.append(batch['is_detain'].cpu().numpy().reshape(-1))
        all_l.append(batch['labels_main'].cpu().numpy())
        all_sup.append(batch['has_supervision'].cpu().numpy().reshape(-1).astype(bool))

    preds_defect = np.vstack(all_d)
    preds_sub = np.vstack(all_sub)
    preds_detection = np.concatenate(all_det)
    preds_detain = np.concatenate(all_detain)
    labels_main = np.vstack(all_l)
    labels_sub = np.vstack(all_sub_l)
    labels_detain = np.concatenate(all_detain_l)
    has_sup = np.concatenate(all_sup).astype(bool)

    # Defect-existence and category metrics are computed only on annotated
    # inspections (has_supervision=True). Non-annotated records are unlabeled
    # for these two tasks, not negative. Detention is recorded for every
    # inspection, so its metrics keep the full sample.
    n_total_eval = int(len(has_sup))
    n_annot_eval = int(has_sup.sum())
    lm_ann = labels_main[has_sup]
    pd_ann = preds_defect[has_sup]
    pdet_ann = preds_detection[has_sup]

    best_thresh = CLASSIFICATION_THRESHOLD if cls_threshold is None else cls_threshold
    best_detain_thresh = 0.5 if detain_threshold is None else detain_threshold
    best_per_class_thresh = per_class_thresholds
    if tune_threshold:
        if cls_threshold is None:
            best_thresh = search_best_threshold(lm_ann, pd_ann)
            # Enhanced: also search per-class thresholds
            best_per_class_thresh = search_best_per_class_thresholds(lm_ann, pd_ann)
        if detain_threshold is None:
            best_detain_thresh = search_best_binary_threshold(
                labels_detain, preds_detain, min_pos_support=50)
    metrics = compute_metrics(
        lm_ann, pd_ann, pdet_ann, best_thresh,
        labels_detain=labels_detain, preds_detain=preds_detain,
        threshold_detain=best_detain_thresh,
        per_class_thresholds=best_per_class_thresh)
    metrics['n_total_eval'] = n_total_eval
    metrics['n_annotated_eval'] = n_annot_eval
    sub_pos_mask = labels_sub.sum(axis=1) > 0
    n_sub_pos = int(sub_pos_mask.sum())
    if n_sub_pos > 0:
        sub_top_k = min(TOP_K, labels_sub.shape[1])
        sub_top_idx = np.argsort(-preds_sub[sub_pos_mask], axis=1)[:, :sub_top_k]
        sub_true = labels_sub[sub_pos_mask]
        sub_hits = sum(
            1 for i in range(n_sub_pos)
            if set(np.where(sub_true[i] > 0.5)[0]) & set(sub_top_idx[i])
        )
        metrics['sub_top_k_hit'] = float(sub_hits / n_sub_pos)
        metrics['sub_prob_max_mean'] = float(np.max(preds_sub[sub_pos_mask], axis=1).mean())
    else:
        metrics['sub_top_k_hit'] = 0.0
        metrics['sub_prob_max_mean'] = 0.0
    metrics['sub_n_pos_eval'] = n_sub_pos
    return metrics, best_thresh, best_detain_thresh, best_per_class_thresh


def main():
    print(f"\n{'='*60}")
    print(f"  MTHGNN V8HQ v3 — Classification-Focused")
    print(f"  embed={EMBED_DIM}, batch={BATCH_SIZE}, lr={args.lr}")
    print(f"  CB Loss: beta={CB_BETA}, label_smooth={LABEL_SMOOTHING}, cls_w={CLASSIFICATION_WEIGHT}")
    print(f"  FFT: low={FFT_LOW_DIM}, high={FFT_HIGH_DIM}, disabled={DISABLE_FFT}")
    print(f"  SEQ: layers={args.seq_num_layers}, max_insp={MAX_HIST}")
    print(f"  Heads: {NUM_HEADS}, Cross-attn: {'disabled' if DISABLE_CROSS_ATTN else 'enabled'}")
    print(f"  AMP BF16: {USE_AMP}")
    print(f"{'='*60}\n")

    train_dataset = PSCDatasetV8hq(DATA_DIR, "train")
    val_dataset = PSCDatasetV8hq(DATA_DIR, "val")
    test_dataset = PSCDatasetV8hq(DATA_DIR, "test")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                           collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=pin_memory)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    seq_count = sum(1 for s in train_dataset.samples
                   if s.get('hist_insp_seq') and len(s.get('hist_insp_seq', [])) > 0)
    print(f"[V8HQ v2] Samples with per-inspection sequences: {seq_count}/{len(train_dataset)}")

    ship_types_all = set()
    port_countries_all = set()
    port_mous_all = set()
    for s in train_dataset.samples:
        t = s.get('ship_type', '')
        if t: ship_types_all.add(t)
        pc = s.get('port_country', '')
        if pc: port_countries_all.add(pc)
        pm = s.get('port_mou', '')
        if pm: port_mous_all.add(pm)

    num_ship_types = max(len(ship_types_all) + 2, 100)
    num_port_countries = max(len(port_countries_all) + 2, 200)
    num_port_mous = max(len(port_mous_all) + 2, 15)
    print(f"  Vocab: ship_types={num_ship_types}, port_countries={num_port_countries}, port_mous={num_port_mous}")

    num_classes = len(train_dataset.defect_classes)
    pos_counts = np.zeros(num_classes)
    for sample in train_dataset.samples:
        raw = sample.get('labels_main', [])
        if len(raw) == num_classes and all(v in (0, 1) for v in raw):
            for i, v in enumerate(raw):
                if v == 1:
                    pos_counts[i] += 1
        else:
            for idx in raw:
                if 0 <= int(idx) < num_classes:
                    pos_counts[int(idx)] += 1

    effective_num = 1.0 - np.power(CB_BETA, pos_counts)
    class_weights = (1.0 - CB_BETA) / (effective_num + 1e-8)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = np.clip(class_weights, 0.3, 3.0)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # ---- eval_only: 先从checkpoint读取配置来初始化正确的模型 ----
    if args.eval_only:
        ckpt_path = args.eval_ckpt or os.path.join(CHECKPOINT_DIR, f"{args.name}_best.ckpt")
        print(f"\n[EVAL ONLY] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_config = ckpt.get("config", {})
        # 用checkpoint中的配置覆盖ACTIVE_VIEWS
        if ckpt_config.get("active_views"):
            av = ckpt_config["active_views"]
            active_views_list = av.split(",") if isinstance(av, str) else av
        elif args.active_views:
            active_views_list = args.active_views.split(",")
        else:
            active_views_list = ["port_view","country_view","mou_view","defect_view","company_view"]
        global ACTIVE_VIEWS
        ACTIVE_VIEWS = active_views_list
        # 用正确的ACTIVE_VIEWS重建模型
        model = MTHGNNV8hqV2(
            train_dataset.entity2id, train_dataset.defect_classes,
            train_dataset.sub_classes, EMBED_DIM, TIME_DIM, ACTIVE_VIEWS,
            num_ship_types=num_ship_types,
            num_port_countries=num_port_countries,
            num_port_mous=num_port_mous,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        class_weights_tensor = class_weights_tensor.to(device)
        print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        model.load_state_dict(ckpt["model"], strict=False)
        best_epoch = ckpt.get("epoch", 0)
        best_primary = ckpt.get("best_primary", 0)
        best_thresh = ckpt.get("threshold", CLASSIFICATION_THRESHOLD)
        best_detain_thresh = ckpt.get("detain_threshold", 0.5)
        best_per_class_thresh = ckpt.get("per_class_thresholds")
        print(f"[EVAL] Epoch={best_epoch}, best_primary={best_primary:.4f}")
        tm_test, _, _, _ = evaluate(
            model, test_loader, cls_threshold=best_thresh,
            detain_threshold=best_detain_thresh,
            per_class_thresholds=best_per_class_thresh)
        print(f"\n{'='*60}")
        print(f"  Test Results (eval_only, best epoch={best_epoch}):")
        print(f"  Det F1:          {tm_test['det_f1']:.4f}")
        print(f"  Class Macro F1:  {tm_test['cls_macro_f1']:.4f}")
        print(f"  Class Macro F1 HQ/CB: {tm_test['cls_macro_f1_hq']:.4f}")
        print(f"  Detention F1:    {tm_test.get('detain_f1', 0):.4f}")
        print(f"  Macro AUC:       {tm_test.get('macro_auc', 0):.4f}")
        print(f"  Hit@5:          {tm_test['top_k_hit']:.4f}")
        print(f"{'='*60}")
        results = {
            "name": args.name,             "best_epoch": best_epoch,
            "best_val_det_f1": best_primary, "best_threshold": best_thresh,
            "best_detain_threshold": best_detain_thresh,
            "per_class_thresholds_enabled": best_per_class_thresh is not None,
            "test_metrics": tm_test, "config": vars(args)
        }
        out_path = os.path.join(CHECKPOINT_DIR, f"{args.name}_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {out_path}")
        return results

    model = MTHGNNV8hqV2(
        entity2id=train_dataset.entity2id,
        defect_classes=train_dataset.defect_classes,
        sub_classes=train_dataset.sub_classes,
        embed_dim=EMBED_DIM,
        active_views=ACTIVE_VIEWS,
    )
    pin_memory = torch.cuda.is_available()
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=pin_memory)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    seq_count = sum(1 for s in train_dataset.samples
                   if s.get('hist_insp_seq') and len(s.get('hist_insp_seq', [])) > 0)
    print(f"[V8HQ v2] Samples with per-inspection sequences: {seq_count}/{len(train_dataset)}")

    # Count unique values for embedding tables
    ship_types_all = set()
    port_countries_all = set()
    port_mous_all = set()
    for s in train_dataset.samples:
        t = s.get('ship_type', '')
        if t: ship_types_all.add(t)
        pc = s.get('port_country', '')
        if pc: port_countries_all.add(pc)
        pm = s.get('port_mou', '')
        if pm: port_mous_all.add(pm)

    num_ship_types = max(len(ship_types_all) + 2, 100)
    num_port_countries = max(len(port_countries_all) + 2, 200)
    num_port_mous = max(len(port_mous_all) + 2, 15)
    print(f"  Vocab: ship_types={num_ship_types}, port_countries={num_port_countries}, port_mous={num_port_mous}")

    num_classes = len(train_dataset.defect_classes)
    pos_counts = np.zeros(num_classes)
    for sample in train_dataset.samples:
        raw = sample.get('labels_main', [])
        if len(raw) == num_classes and all(v in (0, 1) for v in raw):
            for i, v in enumerate(raw):
                if v == 1:
                    pos_counts[i] += 1
        else:
            for idx in raw:
                if 0 <= int(idx) < num_classes:
                    pos_counts[int(idx)] += 1

    effective_num = 1.0 - np.power(CB_BETA, pos_counts)
    class_weights = (1.0 - CB_BETA) / (effective_num + 1e-8)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = np.clip(class_weights, 0.3, 3.0)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    model = MTHGNNV8hqV2(
        train_dataset.entity2id, train_dataset.defect_classes,
        train_dataset.sub_classes, EMBED_DIM, TIME_DIM, ACTIVE_VIEWS,
        num_ship_types=num_ship_types,
        num_port_countries=num_port_countries,
        num_port_mous=num_port_mous,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    class_weights_tensor = class_weights_tensor.to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU (CUDA unavailable)")

    # ---- eval_only 模式：跳过训练，直接评估 ----
    if args.eval_only:
        ckpt_path = args.eval_ckpt or os.path.join(CHECKPOINT_DIR, f"{args.name}_best.ckpt")
        print(f"\n[EVAL ONLY] Loading checkpoint: {ckpt_path}")
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 collate_fn=collate_fn, num_workers=NUM_WORKERS,
                                 pin_memory=torch.cuda.is_available())
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        best_epoch = ckpt.get("epoch", 0)
        best_primary = ckpt.get("best_primary", 0)
        best_thresh = ckpt.get("threshold", CLASSIFICATION_THRESHOLD)
        best_detain_thresh = ckpt.get("detain_threshold", 0.5)
        best_per_class_thresh = ckpt.get("per_class_thresholds")
        print(f"[EVAL] Epoch={best_epoch}, best_primary={best_primary:.4f}")
        tm_test, _, _, _ = evaluate(
            model, test_loader, cls_threshold=best_thresh,
            detain_threshold=best_detain_thresh,
            per_class_thresholds=best_per_class_thresh)
        print(f"\n{'='*60}")
        print(f"  Test Results (eval_only, best epoch={best_epoch}):")
        print(f"  Det F1:          {tm_test['det_f1']:.4f}")
        print(f"  Class Macro F1:  {tm_test['cls_macro_f1']:.4f}")
        print(f"  Class Macro F1 HQ/CB: {tm_test['cls_macro_f1_hq']:.4f}")
        print(f"  Detention F1:    {tm_test.get('detain_f1', 0):.4f}")
        print(f"  Macro AUC:       {tm_test.get('macro_auc', 0):.4f}")
        print(f"  Hit@5:          {tm_test['top_k_hit']:.4f}")
        print(f"{'='*60}")
        results = {
            "name": args.name,             "best_epoch": best_epoch,
            "best_val_det_f1": best_primary, "best_threshold": best_thresh,
            "best_detain_threshold": best_detain_thresh,
            "per_class_thresholds_enabled": best_per_class_thresh is not None,
            "test_metrics": tm_test, "config": vars(args)
        }
        out_path = os.path.join(CHECKPOINT_DIR, f"{args.name}_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {out_path}")
        return results

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == "warm_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    criterion_detection = DetectionLoss(pos_weight=14.0)
    # 支持三种缺陷分类损失:
    # 1. Class-Balanced Loss (默认): CB + label smoothing
    # 2. Asymmetric Focal Loss (--disable_cb_loss + focal_gamma>0): 自适应hard negative mining
    # 3. 标准 BCE (--disable_cb_loss + focal_gamma=0)
    if DISABLE_CB_LOSS:
        if FOCAL_GAMMA > 0:
            print(f"[FOCAL] Using Asymmetric Loss (gamma_neg={FOCAL_GAMMA}, gamma_pos=1)")
            criterion_defect = AsymmetricLoss(
                gamma_neg=FOCAL_GAMMA, gamma_pos=1.0,
                class_weights=class_weights_tensor)
        else:
            print("[ABLATION] Using standard BCE instead of Class-Balanced Loss")
            criterion_defect = nn.BCEWithLogitsLoss(
                pos_weight=class_weights_tensor.to(device) if class_weights_tensor is not None else None,
                reduction='mean')
    else:
        # Class-Balanced Loss with label smoothing (proven stable for PSC)
        print(f"[CB-LOSS] Using Class-Balanced Loss (beta={CB_BETA}, smooth={args.label_smoothing})")
        criterion_defect = ClassBalancedLoss(
            beta=CB_BETA, label_smoothing=args.label_smoothing,
            class_weights=class_weights_tensor)
    criterion_sub = SubClassLoss()
    detain_pos_weight = estimate_binary_pos_weight(
        train_dataset.samples, "is_detain", args.detain_pos_weight_cap)
    criterion_detain = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(detain_pos_weight, dtype=torch.float32, device=device))
    print(f"Detain task: weight={DETAIN_WEIGHT}, pos_weight={detain_pos_weight:.2f}")

    use_amp = USE_AMP and torch.cuda.is_available() and torch.cuda.device_count() > 0
    scaler = torch.amp.GradScaler("cuda", init_scale=2**12) if use_amp else None
    if use_amp:
        print(f"AMP BF16: ENABLED (init_scale={scaler.get_scale():.0f})")
    # Reproducibility: deterministic mode for bit-exact re-runs
    if getattr(args, 'deterministic', False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Reproducibility: deterministic mode ON (slower but reproducible)")
    else:
        torch.backends.cudnn.benchmark = True

    best_primary, best_epoch, patience_counter = 0, 0, 0
    best_thresh = CLASSIFICATION_THRESHOLD
    best_detain_thresh = 0.5
    best_per_class_thresh = None

    # ---- Build sub→main hierarchy matrix from training data (v3-HIER) ----
    hier_matrix = None
    if HIER_LOSS_WEIGHT > 0.0:
        num_defect_main = len(train_dataset.defect_classes)
        num_defect_sub = len(train_dataset.sub_classes)
        from collections import defaultdict
        co_counts = defaultdict(lambda: defaultdict(int))   # co_counts[main][sub] = count
        sub_marginal = defaultdict(int)
        main_marginal = defaultdict(int)
        print("[HIER] Building sub→main hierarchy matrix from training data...")
        with open(os.path.join(DATA_DIR, "splits", "train_enriched_fixed.json"), encoding="utf-8") as f:
            n_train_hier = 0
            for line in f:
                if n_train_hier > 200000:   # cap for efficiency
                    break
                s = json.loads(line)
                lm = s.get("labels_main", [])
                ls = s.get("labels_sub", [])
                for m in lm:
                    if 0 <= int(m) < num_defect_main:
                        main_marginal[int(m)] += 1
                for s_idx in ls:
                    if 0 <= int(s_idx) < num_defect_sub:
                        sub_marginal[int(s_idx)] += 1
                for m in lm:
                    if 0 <= int(m) < num_defect_main:
                        for s_idx in ls:
                            if 0 <= int(s_idx) < num_defect_sub:
                                co_counts[int(m)][int(s_idx)] += 1
                n_train_hier += 1
        M = torch.zeros(num_defect_sub, num_defect_main, dtype=torch.float32)
        # Use empirical P(sub | main): for each sub, distribute mass over mains it appears with
        for s_idx in range(num_defect_sub):
            denom = 0
            for m_idx in range(num_defect_main):
                denom += co_counts[m_idx].get(s_idx, 0)
            if denom > 0:
                for m_idx in range(num_defect_main):
                    c = co_counts[m_idx].get(s_idx, 0)
                    if c > 0:
                        # Smoothed estimate: P(sub | main) = count(main, sub) / count(sub)
                        M[s_idx, m_idx] = c / denom
        # Sub-classes never seen get uniform over mains as fallback
        zero_rows = (M.sum(dim=1) == 0)
        if zero_rows.any():
            M[zero_rows] = 1.0 / num_defect_main
        hier_matrix = M
        nz = (M > 0).float().mean().item()
        print(f"[HIER] Built hierarchy matrix: shape={tuple(M.shape)}, "
              f"non-zero density={nz:.4f}, hier_loss_weight={HIER_LOSS_WEIGHT}")

    for epoch in range(args.epochs):
        tm = train_epoch(model, train_loader, optimizer,
                        criterion_detection, criterion_defect, criterion_sub, criterion_detain,
                        epoch, scheduler, use_amp=use_amp, scaler=scaler,
                        hier_matrix=hier_matrix, hier_loss_weight=HIER_LOSS_WEIGHT,
                        hier_temperature=HIER_SOFT_TEMPERATURE)
        print(f"Epoch {epoch+1}: loss={tm['loss']:.4f}, "
              f"det={tm['det_loss']:.4f}, cls={tm['cls_loss']:.4f}, "
              f"sub={tm['sub_loss']:.4f}, hier={tm.get('hier_loss', 0.0):.4f}, "
              f"detain={tm['detain_loss']:.4f}")

        if (epoch + 1) % args.eval_every == 0:
            vm, val_thresh, val_detain_thresh, val_per_class_thresh = evaluate(
                model, val_loader, tune_threshold=args.tune_cls_threshold)
            selection_score = (
                SELECT_DET_WEIGHT * vm['det_f1']
                + SELECT_CLS_WEIGHT * vm['cls_macro_f1_hq']
                + SELECT_AUC_WEIGHT * vm.get('macro_auc', 0.0)
                + SELECT_HIT_WEIGHT * vm['top_k_hit']
                + SELECT_DETAIN_WEIGHT * vm.get('detain_f1', 0.0)
                + SELECT_SUB_HIT_WEIGHT * vm.get('sub_top_k_hit', 0.0)
            )
            print(f"  Val: det_f1={vm['det_f1']:.4f}, "
                  f"cls_macro={vm['cls_macro_f1']:.4f}, "
                  f"cls_macro_hq={vm['cls_macro_f1_hq']:.4f}, "
                  f"macro_auc={vm.get('macro_auc', 0):.4f}, "
                  f"top5={vm['top_k_hit']:.4f}, "
                  f"sub_top5={vm.get('sub_top_k_hit', 0):.4f}, "
                  f"sub_pmax={vm.get('sub_prob_max_mean', 0):.4f}, "
                  f"detain_f1={vm.get('detain_f1', 0):.4f}, "
                  f"select={selection_score:.4f}")

            if selection_score > best_primary:
                best_primary = selection_score
                best_epoch = epoch + 1
                patience_counter = 0
                best_thresh = val_thresh
                best_detain_thresh = val_detain_thresh
                best_per_class_thresh = val_per_class_thresh
                torch.save({
                    'model': model.state_dict(),
                    'epoch': epoch + 1,
                    'val_metrics': vm,
                    'threshold': best_thresh,
                    'detain_threshold': best_detain_thresh,
                    'per_class_thresholds': best_per_class_thresh,
                    'best_primary': best_primary,
                    'config': vars(args)
                }, os.path.join(CHECKPOINT_DIR, f"{args.name}_best.ckpt"))
                print(f"  [*] Best saved! selection={best_primary:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"[!] Early stopping at epoch {epoch+1}")
                    break

    best_ckpt_path = os.path.join(CHECKPOINT_DIR, f"{args.name}_best.ckpt")
    if os.path.exists(best_ckpt_path):
        print(f"\nLoading best model (epoch {best_epoch})...")
        ckpt = torch.load(best_ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model'])
        best_epoch = ckpt.get('epoch', best_epoch)
        # Use per-class thresholds from best validation checkpoint
        test_per_class_thresh = ckpt.get('per_class_thresholds')
    else:
        # Happens when epochs < eval_every (e.g. a smoke run): no validation was
        # performed, so score the weights that are currently in memory.
        print(f"\n[!] No validation checkpoint found (epochs < eval_every); "
              f"evaluating the final weights instead.")
        test_per_class_thresh = best_per_class_thresh

    tm_test, _, _, _ = evaluate(
        model, test_loader, cls_threshold=best_thresh,
        detain_threshold=best_detain_thresh,
        per_class_thresholds=test_per_class_thresh)

    print(f"\n{'='*60}")
    print(f"  V8HQ v2 Enhanced Test Results (best by composite score on val):")
    print(f"  Det F1:          {tm_test['det_f1']:.4f}")
    print(f"  Class Macro F1:  {tm_test['cls_macro_f1']:.4f}")
    print(f"  Class Macro F1 HQ/CB: {tm_test['cls_macro_f1_hq']:.4f}")
    print(f"  Detention F1:    {tm_test.get('detain_f1', 0):.4f}")
    print(f"  Macro AUC:       {tm_test.get('macro_auc', 0):.4f}")
    print(f"  Hit@5:          {tm_test['top_k_hit']:.4f}")
    print(f"  Best val epoch:  {best_epoch}")
    if test_per_class_thresh is not None:
        print(f"  Per-class thresholds: enabled ({len(test_per_class_thresh)} classes)")
    print(f"{'='*60}")

    results = {
        'name': args.name,         'best_epoch': best_epoch,
        'best_val_det_f1': best_primary, 'best_threshold': best_thresh,
        'best_detain_threshold': best_detain_thresh,
        'per_class_thresholds_enabled': test_per_class_thresh is not None,
        'test_metrics': tm_test, 'config': vars(args)
    }
    with open(os.path.join(CHECKPOINT_DIR, f"{args.name}_results.json"), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


if __name__ == "__main__":
    main()
