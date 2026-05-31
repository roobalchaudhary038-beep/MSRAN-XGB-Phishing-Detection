#!/usr/bin/env python3
"""
=============================================================================
Advanced Phishing Detection Framework
Multi-Scale Residual Attention Network (MSRAN) + XGBoost Ensemble

Proposed to outperform the paper:
  Decision Tree: 96.90%  | DNN: 96.08%  | SVM: 94.40%

Usage:
  python phishing_detection_advanced.py [--dataset path/to/data.csv]

Dataset sources:
  Kaggle : https://www.kaggle.com/datasets/akashkr/phishing-website-dataset
  UCI    : https://archive.ics.uci.edu/dataset/327/phishing+websites
  (save as phishing_dataset.csv or Training Dataset.arff)

Install:
  pip install scikit-learn xgboost lightgbm shap lime seaborn pandas scipy torch
=============================================================================
"""

import argparse, os, sys, warnings, json, gc
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# ── sklearn ──────────────────────────────────────────────────────────────────
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, learning_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance as sk_perm

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from lime import lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

# ── PyTorch ───────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] PyTorch {torch.__version__} | device: {DEVICE}")
    torch.manual_seed(42)
except ImportError:
    HAS_TORCH = False
    print("[WARN] torch not found – DNN models skipped.")

SEED = 42
np.random.seed(SEED)

OUT = Path("phishing_results")
(OUT / "plots").mkdir(parents=True, exist_ok=True)
(OUT / "models").mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
PALETTE = [
    "#E74C3C", "#3498DB", "#2ECC71", "#9B59B6",
    "#F39C12", "#1ABC9C", "#E67E22", "#34495E", "#E91E63", "#607D8B",
    "#795548", "#FF5722",
]

# =============================================================================
# 1.  DATA
# =============================================================================

_FEATURE_NAMES = [
    "UsingIP", "LongURL", "ShortURL", "Symbol@", "Redirecting//",
    "PrefixSuffix-", "SubDomains", "HTTPS", "DomainRegLen", "Favicon",
    "NonStdPort", "HTTPSDomainURL", "RequestURL", "AnchorURL",
    "LinksInScriptTags", "ServerFormHandler", "InfoEmail", "AbnormalURL",
    "WebsiteForwarding", "StatusBarCust", "DisableRightClick",
    "UsingPopupWindow", "IframeRedirection", "AgeofDomain",
    "DNSRecording", "WebsiteTraffic", "PageRank", "GoogleIndex",
    "LinksPointingToPage", "StatsReport",
]


def _arff_to_df(path):
    data, in_data, cols = [], False, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            low = line.lower()
            if low.startswith("@attribute"):
                cols.append(line.split()[1].strip("'\""))
            elif low.startswith("@data"):
                in_data = True
            elif in_data:
                try:
                    data.append([float(v) for v in line.split(",")])
                except ValueError:
                    pass
    return pd.DataFrame(data, columns=cols) if data else None


def load_dataset(filepath=None):
    candidates = ([filepath] if filepath else []) + [
        "phishing_dataset.csv", "dataset.csv",
        "Training Dataset.arff", "PhishingData.arff", "phishing.arff",
    ]
    df = None
    for c in candidates:
        if c and os.path.exists(c):
            print(f"[INFO] Loading: {c}")
            df = _arff_to_df(c) if c.endswith(".arff") else pd.read_csv(c)
            break

    if df is None:
        print("[INFO] Dataset not found → using synthetic data for demo.")
        return _synthetic()

    print(f"[INFO] Raw shape: {df.shape}")
    target = next(
        (c for c in ["Result","result","class","Class","label","target"]
         if c in df.columns),
        df.columns[-1]
    )
    X = df.drop(columns=[target])
    y = df[target].copy()
    X = X.loc[:, ~X.columns.str.lower().isin(["index","unnamed: 0"])]
    X = X.loc[:, ~X.columns.str.startswith("Unnamed")]
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    uv = set(y.unique())
    if uv == {-1, 1}:   y = (y == 1).astype(int)
    elif uv == {1, 2}:  y = (y - 1).astype(int)
    else:               y = (y > 0).astype(int)
    print(f"[INFO] Legitimate={int((y==1).sum())}  Phishing={int((y==0).sum())}")
    return X.values.astype(np.float32), y.values.astype(int), list(X.columns)


def _synthetic():
    """Structured synthetic dataset that mimics real phishing patterns."""
    rng = np.random.default_rng(SEED)
    n_l, n_p, nf = 6157, 4898, 30
    # Legitimate: mostly +1; Phishing: mixed but biased toward -1
    Xl = rng.choice([-1,1], (n_l, nf), p=[0.15, 0.85]).astype(np.float32)
    Xp = rng.choice([-1,1], (n_p, nf), p=[0.72, 0.28]).astype(np.float32)
    # Make key features more discriminative
    for feat_idx in [7, 8, 25, 26, 23, 24]:   # HTTPS, DomainRegLen, Traffic…
        Xl[:, feat_idx] = rng.choice([-1,1], n_l, p=[0.08, 0.92])
        Xp[:, feat_idx] = rng.choice([-1,1], n_p, p=[0.88, 0.12])
    X = np.vstack([Xl, Xp])
    y = np.hstack([np.ones(n_l, int), np.zeros(n_p, int)])
    idx = rng.permutation(len(y))
    return X[idx], y[idx], _FEATURE_NAMES


# =============================================================================
# 2.  PYTORCH MSRAN
# =============================================================================

class ResBlock(nn.Module):
    def __init__(self, in_d, out_d, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(in_d, out_d) if in_d != out_d else nn.Identity()
        self.net  = nn.Sequential(
            nn.Linear(in_d, out_d),
            nn.BatchNorm1d(out_d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_d, out_d),
            nn.BatchNorm1d(out_d),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.proj(x) + self.net(x))


class MSRAN(nn.Module):
    """
    Multi-Scale Residual Attention Network for tabular phishing detection.

    Innovations over the paper's DNN (200→100→1):
      ✓ Residual connections → no vanishing gradients
      ✓ Feature-wise gated attention → selective feature focus
      ✓ Multi-scale hierarchy (256→128→64) → richer representations
      ✓ GELU activations → smoother optimisation landscape
      ✓ LayerNorm after attention → stable training
      ✓ AdamW + cosine-annealing → better generalisation
    """

    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        # stem
        self.stem = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # scale-1 256
        self.s1a = ResBlock(256, 256, dropout)
        self.s1b = ResBlock(256, 256, dropout)
        # scale-2 128
        self.s2a = ResBlock(256, 128, dropout)
        self.s2b = ResBlock(128, 128, dropout)
        # feature-wise attention gate on 128-dim
        self.att_gate = nn.Sequential(
            nn.Linear(128, 128),
            nn.Sigmoid(),
        )
        self.att_norm = nn.LayerNorm(128)
        # scale-3 64
        self.s3 = ResBlock(128, 64, dropout)
        # skip from scale-1 → 64
        self.skip = nn.Sequential(nn.Linear(256, 64), nn.GELU())
        self.merge_norm = nn.BatchNorm1d(64)
        # head
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x   = self.stem(x)
        s1  = self.s1b(self.s1a(x))
        s2  = self.s2b(self.s2a(s1))
        g   = self.att_gate(s2)
        s2  = self.att_norm(s2 + s2 * g)
        s3  = self.s3(s2)
        out = self.merge_norm(s3 + self.skip(s1))
        return torch.sigmoid(self.head(out)).squeeze(1)


class MSRANWrapper:
    """sklearn-style wrapper so MSRAN can be used with permutation_importance etc."""

    def __init__(self, input_dim, dropout=0.3, lr=1e-3, wd=1e-4,
                 epochs=200, batch=256, patience=25):
        self.input_dim = input_dim
        self.dropout   = dropout
        self.lr        = lr
        self.wd        = wd
        self.epochs    = epochs
        self.batch     = batch
        self.patience  = patience
        self.model_    = None
        self.history_  = {"train_loss": [], "val_loss": [],
                          "train_acc":  [], "val_acc":  [],
                          "train_auc":  [], "val_auc":  []}

    def _make_loader(self, X, y, shuffle=True):
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        return DataLoader(TensorDataset(Xt, yt),
                          batch_size=self.batch, shuffle=shuffle)

    def _auc(self, probs, labels):
        try:
            return roc_auc_score(labels, probs)
        except Exception:
            return 0.5

    def fit(self, X_tr, y_tr, X_val=None, y_val=None):
        self.model_ = MSRAN(self.input_dim, self.dropout).to(DEVICE)
        opt = optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.wd)

        # cosine annealing
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)

        # class weights for imbalance
        n0, n1 = (y_tr == 0).sum(), (y_tr == 1).sum()
        w      = torch.tensor([len(y_tr) / (2 * n0),
                               len(y_tr) / (2 * n1)],
                              dtype=torch.float32, device=DEVICE)

        tr_loader = self._make_loader(X_tr, y_tr, shuffle=True)
        do_val    = X_val is not None

        best_val_auc, no_imp, best_state = 0.0, 0, None

        for ep in range(1, self.epochs + 1):
            # ── train ──────────────────────────────────────────────────────
            self.model_.train()
            t_loss, t_correct, t_tot = 0., 0, 0
            for Xb, yb in tr_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                pred = self.model_(Xb)
                wt   = w[yb.long()]
                loss = nn.functional.binary_cross_entropy(pred, yb, weight=wt)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                opt.step()
                t_loss    += loss.item() * len(yb)
                t_correct += ((pred > 0.5) == yb.bool()).sum().item()
                t_tot     += len(yb)
            sched.step()

            t_loss /= t_tot
            t_acc   = t_correct / t_tot

            # ── validate ────────────────────────────────────────────────────
            if do_val:
                self.model_.eval()
                with torch.no_grad():
                    Xv = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
                    yv = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
                    vp = self.model_(Xv)
                    wv = w[yv.long()]
                    v_loss = nn.functional.binary_cross_entropy(vp, yv, weight=wv).item()
                    v_acc  = ((vp > 0.5) == yv.bool()).float().mean().item()
                    v_auc  = self._auc(vp.cpu().numpy(), y_val)

                self.history_["val_loss"].append(v_loss)
                self.history_["val_acc"].append(v_acc)
                self.history_["val_auc"].append(v_auc)

                if v_auc > best_val_auc:
                    best_val_auc, no_imp = v_auc, 0
                    best_state = {k: v.clone() for k, v in
                                  self.model_.state_dict().items()}
                else:
                    no_imp += 1
                    if no_imp >= self.patience:
                        print(f"  Early stop at epoch {ep}  best_val_auc={best_val_auc:.4f}")
                        break

            self.history_["train_loss"].append(t_loss)
            self.history_["train_acc"].append(t_acc)

            if ep % 20 == 0:
                msg = f"  ep{ep:4d}  loss={t_loss:.4f}  acc={t_acc:.4f}"
                if do_val:
                    msg += f"  val_auc={v_auc:.4f}"
                print(msg)

        if best_state:
            self.model_.load_state_dict(best_state)
        torch.save(self.model_.state_dict(), OUT / "models" / "msran_best.pt")
        return self

    def predict_proba_raw(self, X):
        self.model_.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            p  = self.model_(Xt).cpu().numpy()
        return p

    def predict_proba(self, X):
        p = self.predict_proba_raw(X)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba_raw(X) > 0.5).astype(int)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# =============================================================================
# 3.  UTILITIES
# =============================================================================

def get_metrics(name, y_true, y_pred, y_prob=None):
    return dict(
        Model=name,
        Accuracy=accuracy_score(y_true, y_pred) * 100,
        Precision=precision_score(y_true, y_pred, zero_division=0),
        Recall=recall_score(y_true, y_pred, zero_division=0),
        F1=f1_score(y_true, y_pred, zero_division=0),
        MCC=matthews_corrcoef(y_true, y_pred),
        Kappa=cohen_kappa_score(y_true, y_pred),
        AUC=roc_auc_score(y_true, y_prob) if y_prob is not None else None,
        ErrorRate=(1 - accuracy_score(y_true, y_pred)) * 100,
    )


def save_plot(name):
    plt.tight_layout()
    plt.savefig(OUT / "plots" / name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {name}")


# =============================================================================
# 4.  PLOTS
# =============================================================================

def plot_class_dist(y):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Dataset Class Distribution", fontsize=14, fontweight="bold")
    counts = [(y == 1).sum(), (y == 0).sum()]
    labs   = ["Legitimate", "Phishing"]
    clrs   = ["#2ECC71", "#E74C3C"]
    axes[0].bar(labs, counts, color=clrs, edgecolor="white", width=0.5)
    for i, c in enumerate(counts):
        axes[0].text(i, c + 30, str(int(c)), ha="center", fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Sample Counts")
    axes[1].pie(counts, labels=labs, autopct="%1.1f%%", colors=clrs,
                startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Class Proportion")
    save_plot("01_class_distribution.png")


def plot_correlation(X, y, feat):
    df = pd.DataFrame(X, columns=feat)
    df["Target"] = y
    corr = df.corr()
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.suptitle("Feature Correlation Analysis", fontsize=14, fontweight="bold")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, ax=axes[0],
                linewidths=0.3, vmin=-1, vmax=1,
                xticklabels=corr.columns, yticklabels=corr.columns,
                cbar_kws={"label": "Correlation"})
    axes[0].set_title("Full Feature Correlation Matrix")
    axes[0].tick_params(labelsize=7)
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")
    tc   = corr["Target"].drop("Target").sort_values()
    clrs = ["#3498DB" if v >= 0 else "#E74C3C" for v in tc.values]
    axes[1].barh(tc.index, tc.values, color=clrs, edgecolor="white", height=0.7)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_xlabel("Correlation with Target  (Legit=1 / Phishing=0)")
    axes[1].set_title("Feature–Target Point-Biserial Correlation")
    axes[1].grid(axis="x", alpha=0.3)
    save_plot("02_correlation_heatmap.png")


def plot_training_curves(history, model_name="MSRAN"):
    """Plot training curves from MSRANWrapper.history_"""
    h = history
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training History – {model_name}", fontsize=14, fontweight="bold")

    def _ep(key):
        return range(1, len(h[key]) + 1)

    if h["train_loss"]:
        axes[0].plot(_ep("train_loss"), h["train_loss"], color="#E74C3C", lw=2, label="Train")
        if h["val_loss"]:
            axes[0].plot(_ep("val_loss"), h["val_loss"], color="#3498DB", lw=2, ls="--", label="Val")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Binary Cross-Entropy")

    if h["train_acc"]:
        axes[1].plot(_ep("train_acc"), [v*100 for v in h["train_acc"]], "#E74C3C", lw=2, label="Train")
        if h["val_acc"]:
            axes[1].plot(_ep("val_acc"), [v*100 for v in h["val_acc"]], "#3498DB", lw=2, ls="--", label="Val")
        axes[1].set_title("Accuracy (%)"); axes[1].legend(); axes[1].grid(alpha=0.3)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")

    if h["val_auc"]:
        axes[2].plot(_ep("val_auc"), [v*100 for v in h["val_auc"]], "#2ECC71", lw=2, label="Val AUC")
        axes[2].set_title("Validation AUC-ROC (%)"); axes[2].legend(); axes[2].grid(alpha=0.3)
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("AUC (%)")
        axes[2].axhline(96.90, color="red", ls="--", lw=1, alpha=0.5, label="Paper DT acc")

    save_plot("03_training_curves.png")


def plot_roc(roc_data):
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_title("ROC Curve Comparison – All Models", fontsize=14, fontweight="bold")
    for i, (name, yt, yp) in enumerate(roc_data):
        if yp is None:
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        a  = auc(fpr, tpr)
        lw = 3 if ("MSRAN" in name or "Ensemble" in name) else 1.5
        ax.plot(fpr, tpr, lw=lw, color=PALETTE[i % len(PALETTE)],
                label=f"{name}  (AUC={a:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random (AUC=0.50)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    save_plot("04_roc_curves.png")


def plot_cms(cms_data):
    n    = len(cms_data)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.8 * rows))
    fig.suptitle("Confusion Matrices – All Models", fontsize=14, fontweight="bold", y=1.01)
    axes = np.array(axes).flatten()
    for i, (name, cm) in enumerate(cms_data):
        ax   = axes[i]
        cmn  = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cmn, ax=ax, cmap="Blues", annot=False, cbar=False,
                    linewidths=0.5, vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                clr = "white" if cmn[r, c] > 0.55 else "black"
                ax.text(c + 0.5, r + 0.5, f"{cm[r,c]}\n({cmn[r,c]:.1%})",
                        ha="center", va="center", fontsize=9, fontweight="bold", color=clr)
        acc = (cm[0, 0] + cm[1, 1]) / cm.sum() * 100
        ax.set_title(f"{name}\n{acc:.2f}%", fontweight="bold", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_xticklabels(["Phishing", "Legit"], fontsize=8)
        ax.set_yticklabels(["Phishing", "Legit"], fontsize=8, rotation=0)
    for i in range(len(cms_data), len(axes)):
        axes[i].set_visible(False)
    save_plot("05_confusion_matrices.png")


def plot_metric_bars(df):
    metric_cols = ["Accuracy", "Precision", "Recall", "F1"]
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold")
    for idx, m in enumerate(metric_cols):
        ax = axes[idx // 2][idx % 2]
        ds  = df.sort_values(m, ascending=True)
        vals = ds[m].values if m == "Accuracy" else ds[m].values * 100
        clrs = [
            "#E74C3C" if "Ensemble" in n else
            "#FF8A65" if "MSRAN" in n else
            "#3498DB" if "DNN" in n else "#95A5A6"
            for n in ds["Model"]
        ]
        bars = ax.barh(ds["Model"], vals, color=clrs, edgecolor="white",
                       height=0.65, linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(v + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}%", va="center", fontsize=8, fontweight="bold")
        ax.axvline(96.90, color="red", ls="--", lw=1.2, alpha=0.55,
                   label="Paper DT (96.90%)")
        ax.set_title(f"Comparison: {m}", fontweight="bold")
        ax.set_xlabel(f"{m} (%)")
        ax.set_xlim([max(0, float(vals.min()) - 5), 101])
        ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
    save_plot("06_metric_bars.png")


def plot_statistical(df):
    fig = plt.figure(figsize=(22, 8))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])

    # heatmap
    ax1  = fig.add_subplot(gs[0])
    cols = [c for c in ["Accuracy","Precision","Recall","F1","AUC","MCC","Kappa"]
            if c in df.columns]
    data = df.set_index("Model")[cols].astype(float).copy()
    for c in cols:
        if c != "Accuracy":
            data[c] = data[c] * 100
    sns.heatmap(data, annot=True, fmt=".2f", cmap="RdYlGn",
                ax=ax1, linewidths=0.4, vmin=50, vmax=100,
                cbar_kws={"label": "Score (%)"})
    ax1.set_title("Performance Heatmap – All Models", fontsize=13, fontweight="bold")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha="right")

    # radar
    ax2    = fig.add_subplot(gs[1], polar=True)
    cats   = ["Acc\n(%)", "Prec\n(%)", "Recall\n(%)", "F1\n(%)", "AUC\n(%)"]
    N      = len(cats)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]
    top5   = df.nlargest(5, "Accuracy")
    for i, (_, row) in enumerate(top5.iterrows()):
        vals = [
            float(row["Accuracy"]),
            float(row["Precision"]) * 100,
            float(row["Recall"])    * 100,
            float(row["F1"])        * 100,
            (float(row["AUC"]) * 100 if row["AUC"] is not None else 95.0),
        ] + [float(row["Accuracy"])]
        ax2.plot(angles, vals, lw=2, color=PALETTE[i], label=row["Model"])
        ax2.fill(angles, vals, alpha=0.08, color=PALETTE[i])
    ax2.set_xticks(angles[:-1]); ax2.set_xticklabels(cats, fontsize=9)
    ax2.set_ylim([85, 101])
    ax2.set_title("Radar – Top 5 Models", fontsize=12, fontweight="bold", pad=20)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.5, 1.15), fontsize=8)
    ax2.grid(True)

    fig.suptitle("Statistical Comparison", fontsize=14, fontweight="bold")
    save_plot("07_statistical_comparison.png")


def plot_learning_curves_plot(lc_models, X, y):
    n = len(lc_models)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n + 1, figsize=(6 * (n + 1), 6))
    if n + 1 == 1:
        axes = [axes]
    fig.suptitle("Learning Curves & CV Distribution", fontsize=13, fontweight="bold")
    kf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv_scores = {}
    for i, (mname, model) in enumerate(lc_models.items()):
        ax = axes[i]
        tsz, tsc, vsc = learning_curve(
            model, X, y, cv=kf, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 8), scoring="accuracy"
        )
        tm, ts = tsc.mean(1) * 100, tsc.std(1) * 100
        vm, vs = vsc.mean(1) * 100, vsc.std(1) * 100
        ax.plot(tsz, tm, "o-", color="#E74C3C", lw=2, label="Train")
        ax.fill_between(tsz, tm - ts, tm + ts, alpha=0.15, color="#E74C3C")
        ax.plot(tsz, vm, "s-", color="#3498DB", lw=2, label="Val")
        ax.fill_between(tsz, vm - vs, vm + vs, alpha=0.15, color="#3498DB")
        ax.set_title(f"Learning Curve: {mname}", fontweight="bold")
        ax.set_xlabel("Training Samples"); ax.set_ylabel("Accuracy (%)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylim([80, 102])
        cv_scores[mname] = cross_val_score(model, X, y, cv=kf, scoring="accuracy") * 100
    ax = axes[n]
    bp = ax.boxplot(list(cv_scores.values()), labels=list(cv_scores.keys()),
                    patch_artist=True, notch=True, showmeans=True)
    for j, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[j]); patch.set_alpha(0.75)
    ax.set_ylabel("Accuracy (%)"); ax.set_title("5-Fold CV Distribution", fontweight="bold")
    ax.axhline(96.90, color="red", ls="--", lw=1.2, label="Paper DT (96.90%)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    save_plot("08_learning_curves.png")


def plot_feature_importance(fi_list, feat):
    n = len(fi_list)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 12))
    if n == 1:
        axes = [axes]
    fig.suptitle("Feature Importance Analysis", fontsize=13, fontweight="bold")
    for ax, (mname, imp) in zip(axes, fi_list):
        k = min(20, len(imp))
        idx  = np.argsort(imp)[-k:]
        vals = imp[idx]
        norm_vals = vals / (vals.max() + 1e-12)
        cmap  = plt.cm.RdYlGn
        clrs  = [cmap(v) for v in norm_vals]
        bars  = ax.barh(range(k), vals, color=clrs, edgecolor="white")
        ax.set_yticks(range(k))
        ax.set_yticklabels([feat[i] for i in idx], fontsize=9)
        ax.set_xlabel("Importance Score")
        ax.set_title(f"{mname}\nTop-{k} Feature Importance", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(v + 5e-4, bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}", va="center", fontsize=7)
    save_plot("09_feature_importance.png")


def plot_shap_tree(model, X_test, feat, name):
    if not HAS_SHAP:
        return None
    print(f"  SHAP ({name})…")
    samp = X_test[: min(600, len(X_test))]
    exp  = shap.TreeExplainer(model)
    sv   = exp.shap_values(samp)
    if isinstance(sv, list):
        sv = sv[1]                     # class-1 SHAP values for binary classifiers

    mean_abs = np.abs(sv).mean(0)
    k   = min(20, len(mean_abs))
    idx = np.argsort(mean_abs)[-k:]

    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    fig.suptitle(f"SHAP Explainability – {name}", fontsize=14, fontweight="bold")

    # ── bar ──
    ax = axes[0]
    vals    = mean_abs[idx]
    norm_v  = vals / (vals.max() + 1e-12)
    ax.barh(range(k), vals, color=[plt.cm.plasma(v) for v in norm_v], edgecolor="white")
    ax.set_yticks(range(k))
    ax.set_yticklabels([feat[i] for i in idx], fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|"); ax.set_title("Global Feature Importance", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # ── beeswarm ──
    ax   = axes[1]
    top  = np.argsort(mean_abs)[-10:]
    Xt   = samp[:, top]
    Xn   = (Xt - Xt.min(0)) / (Xt.max(0) - Xt.min(0) + 1e-8)
    svt  = sv[:, top]
    for j in range(len(top)):
        jit = np.random.normal(0, 0.08, len(svt))
        sc  = ax.scatter(svt[:, j], np.full(len(svt), j) + jit,
                         c=Xn[:, j], cmap="RdBu", alpha=0.4, s=10, vmin=0, vmax=1)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([feat[i] for i in top], fontsize=9)
    ax.axvline(0, color="black", lw=1); ax.set_xlabel("SHAP Value")
    ax.set_title("SHAP Beeswarm Plot", fontweight="bold")
    plt.colorbar(sc, ax=ax, label="Feature value\n(Low↔High)", shrink=0.6)
    ax.grid(alpha=0.3)

    # ── waterfall (instance 0) ──
    ax   = axes[2]
    sv0  = sv[0]
    top12= np.argsort(np.abs(sv0))[-12:]
    clrs = ["#E74C3C" if v > 0 else "#3498DB" for v in sv0[top12]]
    ax.barh(range(12), sv0[top12], color=clrs, edgecolor="white")
    ax.set_yticks(range(12))
    ax.set_yticklabels([feat[i] for i in top12], fontsize=9)
    ax.axvline(0, color="black", lw=1); ax.set_xlabel("SHAP Value")
    ax.set_title("Waterfall – Instance #0", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#E74C3C", label="→ Phishing"),
                       Patch(facecolor="#3498DB", label="→ Legitimate")],
              fontsize=9, loc="lower right")

    save_plot(f"10_shap_{name.replace(' ','_')}.png")
    return mean_abs


def plot_shap_nn(wrapper, X_test, feat):
    """SHAP for PyTorch MSRAN via KernelExplainer (sampling-based)."""
    if not HAS_SHAP:
        return
    print("  SHAP (MSRAN – KernelExplainer, may take ~1 min)…")
    bg   = shap.sample(X_test, 80)
    samp = X_test[:120]

    def predict_fn(x):
        return wrapper.predict_proba(x.astype(np.float32))[:, 1]

    exp = shap.KernelExplainer(predict_fn, bg)
    sv  = exp.shap_values(samp, nsamples=100, l1_reg="num_features(10)")
    if isinstance(sv, list):
        sv = sv[0]

    mean_abs = np.abs(sv).mean(0)
    k   = min(20, len(mean_abs))
    idx = np.argsort(mean_abs)[-k:]
    vals = mean_abs[idx]
    norm = vals / (vals.max() + 1e-12)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.barh(range(k), vals, color=[plt.cm.plasma(v) for v in norm], edgecolor="white")
    ax.set_yticks(range(k))
    ax.set_yticklabels([feat[i] for i in idx], fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("SHAP Feature Importance – MSRAN", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    save_plot("10_shap_MSRAN.png")


def plot_lime_analysis(predict_fn, X_test, feat, model_name):
    if not HAS_LIME:
        print("  [SKIP] lime not installed."); return
    print(f"  LIME ({model_name})…")
    exp_obj = lime_tabular.LimeTabularExplainer(
        X_test, feature_names=feat,
        class_names=["Phishing", "Legitimate"],
        mode="classification", random_state=SEED,
    )
    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    fig.suptitle(f"LIME Local Explanations – {model_name}", fontsize=13, fontweight="bold")
    for k, ax in enumerate(axes):
        e     = exp_obj.explain_instance(X_test[k], predict_fn, num_features=10)
        items = e.as_list(label=1)
        feats, vals = zip(*items)
        clrs  = ["#E74C3C" if v > 0 else "#3498DB" for v in vals]
        ax.barh(range(len(feats)), vals, color=clrs, edgecolor="white")
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=8)
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("Feature Weight")
        ax.set_title(f"Instance #{k}", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    save_plot(f"11_lime_{model_name.replace(' ','_')}.png")


def plot_permutation(model, X_test, y_test, feat, name):
    print(f"  Permutation importance ({name})…")
    r   = sk_perm(model, X_test, y_test, n_repeats=10, random_state=SEED, scoring="accuracy")
    imp = r.importances_mean
    std = r.importances_std
    k   = min(20, len(imp))
    idx = np.argsort(imp)[-k:]
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.barh(range(k), imp[idx], xerr=std[idx], capsize=4,
            color=[plt.cm.viridis(v) for v in np.linspace(0.2, 0.9, k)],
            edgecolor="white")
    ax.set_yticks(range(k))
    ax.set_yticklabels([feat[i] for i in idx], fontsize=9)
    ax.set_xlabel("Accuracy Drop (Permutation Importance)")
    ax.set_title(f"Permutation Feature Importance – {name}", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    save_plot(f"12_permutation_{name.replace(' ','_')}.png")


def plot_acc_error(df):
    fig, ax = plt.subplots(figsize=(16, 7))
    x   = np.arange(len(df))
    w   = 0.35
    acc = df["Accuracy"].values
    err = df["ErrorRate"].values
    b1  = ax.bar(x - w/2, acc, w, label="Accuracy (%)", color="#2ECC71", edgecolor="white", alpha=0.85)
    b2  = ax.bar(x + w/2, err, w, label="Error Rate (%)", color="#E74C3C", edgecolor="white", alpha=0.85)
    for b in b1: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2,
                          f"{b.get_height():.2f}", ha="center", fontsize=7, fontweight="bold")
    for b in b2: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
                          f"{b.get_height():.2f}", ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("(%)")
    ax.set_title("Accuracy vs Error Rate – All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    ax.axhline(96.90, color="navy", ls="--", lw=1.2, alpha=0.5, label="Paper DT 96.90%")
    save_plot("14_accuracy_vs_error.png")


def plot_summary_table(df):
    fig, ax = plt.subplots(figsize=(20, max(4, len(df) * 0.65 + 1.5)))
    ax.axis("off")
    col_labels = list(df.columns)

    def fmt(v, col):
        if not isinstance(v, float) and not isinstance(v, np.floating):
            return str(v)
        if col in ("Accuracy", "ErrorRate"):
            return f"{v:.2f}%"
        return f"{v:.4f}" if v is not None else "N/A"

    cell_text = [[fmt(row[c], c) for c in col_labels] for _, row in df.iterrows()]
    row_clrs  = []
    for name in df["Model"]:
        if "Ensemble" in name: row_clrs.append(["#FDECEA"] * len(col_labels))
        elif "MSRAN" in name:  row_clrs.append(["#FDF3E3"] * len(col_labels))
        else:                   row_clrs.append(["#FFFFFF"] * len(col_labels))

    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   rowColours=[r[0] for r in row_clrs],
                   cellColours=row_clrs,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.15, 1.6)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Complete Results Summary", fontsize=14, fontweight="bold", pad=20)
    save_plot("13_results_table.png")


# =============================================================================
# 5.  MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None)
    p.add_argument("--epochs",  type=int, default=200)
    p.add_argument("--batch",   type=int, default=256)
    p.add_argument("--no-shap", action="store_true", help="Skip SHAP (faster)")
    args = p.parse_args()

    print("\n" + "="*70)
    print("  Advanced Phishing Detection – MSRAN + XGBoost Ensemble")
    print("="*70)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1] Loading data…")
    X, y, feat = load_dataset(args.dataset)
    n_feat = X.shape[1]

    plot_class_dist(y)
    plot_correlation(X, y, feat)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.33, random_state=SEED, stratify=y)
    X_trn, X_val, y_trn, y_val = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=SEED, stratify=y_tr)
    print(f"  Train={len(X_trn)}  Val={len(X_val)}  Test={len(X_te)}")

    sc      = StandardScaler()
    X_tr_s  = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)
    X_trn_s = sc.fit_transform(X_trn)
    X_val_s = sc.transform(X_val)

    # ── 2. Baseline ML ────────────────────────────────────────────────────────
    print("\n[2] Training baseline ML models…")
    baselines = {
        "Decision Tree":      DecisionTreeClassifier(criterion="gini", random_state=SEED),
        "SVM":                SVC(kernel="rbf", C=10, probability=True, random_state=SEED),
        "Logistic Regression":LogisticRegression(max_iter=2000, random_state=SEED),
        "KNN":                KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "LDA":                LinearDiscriminantAnalysis(),
        "Naive Bayes":        GaussianNB(),
        "Random Forest":      RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    }
    if HAS_XGB:
        baselines["XGBoost"] = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=SEED,
            use_label_encoder=False, n_jobs=-1)
    if HAS_LGB:
        baselines["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbose=-1, n_jobs=-1)

    trained = {}
    for name, m in baselines.items():
        print(f"  {name}…", end=" ", flush=True)
        m.fit(X_tr_s, y_tr)
        trained[name] = m
        print(f"{m.score(X_te_s, y_te)*100:.2f}%")

    # ── 3. MSRAN (PyTorch) ────────────────────────────────────────────────────
    msran_wrapper = None
    if HAS_TORCH:
        print("\n[3] Training MSRAN (PyTorch)…")
        msran_wrapper = MSRANWrapper(
            input_dim=n_feat, dropout=0.3, lr=1e-3, wd=1e-4,
            epochs=args.epochs, batch=args.batch, patience=25
        )
        msran_wrapper.fit(X_trn_s, y_trn, X_val_s, y_val)
        plot_training_curves(msran_wrapper.history_, "MSRAN")
        msran_acc = msran_wrapper.score(X_te_s, y_te) * 100
        print(f"  MSRAN test accuracy: {msran_acc:.2f}%")

    # ── 4. Ensemble ───────────────────────────────────────────────────────────
    print("\n[4] Building ensemble…")
    base_ens = trained.get("XGBoost", trained.get("LightGBM", trained["Random Forest"]))
    xgb_te_p = base_ens.predict_proba(X_te_s)[:, 1]

    if msran_wrapper is not None:
        # search best blend weight
        xgb_tr_p   = base_ens.predict_proba(X_tr_s)[:, 1]
        msran_tr_p  = msran_wrapper.predict_proba_raw(X_tr_s)
        best_acc, best_w = 0.0, 0.5
        for w in np.arange(0.05, 1.0, 0.05):
            ep  = w * msran_tr_p + (1 - w) * xgb_tr_p
            acc = accuracy_score(y_tr, (ep > 0.5).astype(int))
            if acc > best_acc:
                best_acc, best_w = acc, w
        print(f"  Optimal MSRAN weight: {best_w:.2f}")
        msran_te_p  = msran_wrapper.predict_proba_raw(X_te_s)
        ens_proba   = best_w * msran_te_p + (1 - best_w) * xgb_te_p
        ens_name    = "MSRAN+XGB Ensemble"
    else:
        ens_proba   = xgb_te_p
        ens_name    = base_ens.__class__.__name__ + " (Best)"
        best_w      = 0.0
    ens_pred = (ens_proba > 0.5).astype(int)

    # ── 5. Collect results ────────────────────────────────────────────────────
    print("\n[5] Computing metrics…")
    all_res, roc_data, cm_data = [], [], []

    for name, m in trained.items():
        yp  = m.predict(X_te_s)
        ypr = m.predict_proba(X_te_s)[:, 1] if hasattr(m, "predict_proba") else None
        all_res.append(get_metrics(name, y_te, yp, ypr))
        roc_data.append((name, y_te, ypr))
        cm_data.append((name, confusion_matrix(y_te, yp)))

    if msran_wrapper is not None:
        mp  = msran_wrapper.predict_proba_raw(X_te_s)
        mpd = (mp > 0.5).astype(int)
        all_res.append(get_metrics("MSRAN", y_te, mpd, mp))
        roc_data.append(("MSRAN", y_te, mp))
        cm_data.append(("MSRAN", confusion_matrix(y_te, mpd)))

    all_res.append(get_metrics(ens_name, y_te, ens_pred, ens_proba))
    roc_data.append((ens_name, y_te, ens_proba))
    cm_data.append((ens_name, confusion_matrix(y_te, ens_pred)))

    df_res = pd.DataFrame(all_res).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    df_res.to_csv(OUT / "results_table.csv", index=False)

    print("\n── Results ──────────────────────────────────────────")
    print(df_res[["Model","Accuracy","Precision","Recall","F1","AUC"]].to_string(index=False))

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    print("\n[6] Generating plots…")
    plot_roc(roc_data)
    plot_cms(cm_data)
    plot_metric_bars(df_res)
    plot_statistical(df_res)
    plot_acc_error(df_res)
    plot_summary_table(df_res)

    lc_m = {k: v for k, v in trained.items()
            if k in ["Decision Tree","Random Forest","XGBoost","LightGBM","Logistic Regression"]}
    plot_learning_curves_plot(dict(list(lc_m.items())[:3]), X_tr_s, y_tr)

    fi = [(n, m.feature_importances_) for n, m in trained.items()
          if hasattr(m, "feature_importances_")][:3]
    plot_feature_importance(fi, feat)

    # ── 7. SHAP ───────────────────────────────────────────────────────────────
    if not args.no_shap:
        print("\n[7] SHAP analysis…")
        if HAS_XGB and "XGBoost" in trained:
            plot_shap_tree(trained["XGBoost"], X_te_s, feat, "XGBoost")
        if "Random Forest" in trained:
            plot_shap_tree(trained["Random Forest"], X_te_s, feat, "Random_Forest")
        if msran_wrapper is not None:
            plot_shap_nn(msran_wrapper, X_te_s, feat)
    else:
        print("\n[7] SHAP skipped (--no-shap).")

    # ── 8. LIME ───────────────────────────────────────────────────────────────
    print("\n[8] LIME analysis…")
    if HAS_LIME:
        if msran_wrapper is not None:
            plot_lime_analysis(msran_wrapper.predict_proba, X_te_s, feat, "MSRAN")
        elif "XGBoost" in trained:
            plot_lime_analysis(trained["XGBoost"].predict_proba, X_te_s, feat, "XGBoost")

    # ── 9. Permutation importance ─────────────────────────────────────────────
    print("\n[9] Permutation importance…")
    if "Random Forest" in trained:
        plot_permutation(trained["Random Forest"], X_te_s, y_te, feat, "Random_Forest")

    # ── 10. Statistical significance ──────────────────────────────────────────
    print("\n[10] Statistical significance (Wilcoxon signed-rank)…")
    kf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    dt_cv   = cross_val_score(trained["Decision Tree"], X_tr_s, y_tr, cv=kf, scoring="accuracy")
    xgb_cv  = cross_val_score(base_ens,                 X_tr_s, y_tr, cv=kf, scoring="accuracy")
    rf_cv   = cross_val_score(trained["Random Forest"], X_tr_s, y_tr, cv=kf, scoring="accuracy")

    ref_cv = xgb_cv  # our best ML model
    _, p_dt  = stats.wilcoxon(ref_cv, dt_cv)  if len(set(ref_cv - dt_cv)) > 1 else (0, 1)
    _, p_rf  = stats.wilcoxon(ref_cv, rf_cv)  if len(set(ref_cv - rf_cv)) > 1 else (0, 1)
    sig      = lambda p: "✓ significant" if p < 0.05 else "○ not significant"
    print(f"  XGBoost vs DT : p={p_dt:.4f}  {sig(p_dt)}")
    print(f"  XGBoost vs RF : p={p_rf:.4f}  {sig(p_rf)}")

    # ── Final summary ─────────────────────────────────────────────────────────
    best = df_res.iloc[0]
    print("\n" + "="*70)
    print("  FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"\n  Best model  : {best['Model']}")
    print(f"  Accuracy    : {best['Accuracy']:.4f}%")
    print(f"  Precision   : {best['Precision']:.4f}")
    print(f"  Recall      : {best['Recall']:.4f}")
    print(f"  F1-Score    : {best['F1']:.4f}")
    if best["AUC"] is not None:
        print(f"  AUC-ROC     : {best['AUC']:.4f}")
    print(f"  MCC         : {best['MCC']:.4f}")
    print(f"  Cohen Kappa : {best['Kappa']:.4f}")
    print(f"\n  Paper DT (96.90%) → Improvement: +{best['Accuracy']-96.90:.2f}%")
    print(f"\n  Output: {OUT.resolve()}/")
    plots = sorted((OUT/"plots").glob("*.png"))
    print(f"  Plots generated: {len(plots)}")
    for pf in plots:
        print(f"    {pf.name}")
    print("="*70)


if __name__ == "__main__":
    main()
