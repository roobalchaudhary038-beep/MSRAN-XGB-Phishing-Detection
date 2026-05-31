# MSRAN-XGB Phishing Detection

> **Multi-Scale Residual Attention Network (MSRAN) + XGBoost Ensemble for Phishing Website Detection**

A deep learning framework that outperforms state-of-the-art baselines from the reference paper:

| Model | Accuracy |
|-------|----------|
| Existing: Decision Tree (paper) | 96.90% |
| Existing: DNN (paper) | 96.08% |
| Existing: SVM (paper) | 94.40% |
| **MSRAN + XGBoost Ensemble (ours)** | **> 97%** |

---

 Overview

This repository implements two phishing detection pipelines:

- **`phishing_detection_advanced.py`** — Core framework: MSRAN architecture + XGBoost ensemble with SHAP, LIME, and permutation importance
- **`phishing_full_analysis.py`** — Extended analysis: ROC with confidence intervals, calibration curves, PR curves, comparison against 5 prior baselines

### Key Innovations over the Paper's DNN

- ✅ Residual connections — prevents vanishing gradients
- ✅ Feature-wise gated attention — selective feature focus
- ✅ Multi-scale hierarchy (256 → 128 → 64) — richer representations
- ✅ GELU activations — smoother optimisation landscape
- ✅ LayerNorm after attention — stable training
- ✅ AdamW + cosine-annealing with warm restarts — better generalisation
- ✅ Class-weighted BCE loss — handles class imbalance

---

Installation

```bash
pip install scikit-learn xgboost lightgbm shap lime seaborn pandas scipy torch
```

---

 Dataset

| Source | Format | Auto-detected filename |
|--------|--------|----------------------|
| Kaggle – Phishing Website Dataset | CSV | `phishing_dataset.csv` |
| UCI – Phishing Websites | ARFF | `Training Dataset.arff` |
| Synthetic (built-in) | — | Auto-used if no file found |

---

Usage

```bash
python phishing_detection_advanced.py --dataset phishing_dataset.csv
python phishing_detection_advanced.py --no-shap
python phishing_full_analysis.py --dataset phishing_dataset.csv
```

---

Author

Roobal Chaudhary
GitHub: https://github.com/roobalchaudhary038-beep


MIT License
