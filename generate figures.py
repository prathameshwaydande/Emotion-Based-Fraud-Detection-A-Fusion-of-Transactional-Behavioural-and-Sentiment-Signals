# generate_figures.py
# Creates dissertation figures 3–8 from your saved fusion model + data
# Figures saved under: data/project/figures/

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from catboost import CatBoostClassifier
import joblib

# Optional: use seaborn ONLY for heatmap; comment out if you prefer pure matplotlib
try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

# -----------------------------
# Config / paths
# -----------------------------
BASE_DIR = Path("data/project")          # where your model + csvs were saved
MODEL_PATH = BASE_DIR / "fusion_fraud_model.pkl"
META_PATH  = BASE_DIR / "fusion_fraud_meta.json"
TRANS_CSV  = BASE_DIR / "transactions.csv"
KS_CSV     = BASE_DIR / "keystrokes.csv"
TXT_CSV    = BASE_DIR / "text.csv"
FIG_DIR    = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Utilities
# -----------------------------
def safe_load_meta(meta_path):
    thr = 0.5
    method = "default"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        thr = float(meta.get("best_threshold", thr))
        method = meta.get("threshold_method", method)
    return thr, method

def assemble_dataset(trans_csv, ks_csv, txt_csv, target_col="is_fraud"):
    """Re-assemble features exactly like the training step."""
    # Import the function from your module to stay consistent
    from fraud_fusion_starter import assemble_dataset
    X, y, num_cols, cat_cols = assemble_dataset(str(trans_csv), str(ks_csv), str(txt_csv), target_col=target_col)
    return X, y, num_cols, cat_cols

def plot_and_save(fig, outpath, tight=True):
    if tight:
        plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# 1) Load model + meta + data
# -----------------------------
print("📂 Loading model:", MODEL_PATH)
model = joblib.load(MODEL_PATH)

best_thr, thr_method = safe_load_meta(META_PATH)
print(f"✅ Using decision threshold = {best_thr:.3f} (method={thr_method})")

# Make sure data exists; if not, generate via your run_demo function
if not (TRANS_CSV.exists() and KS_CSV.exists() and TXT_CSV.exists()):
    print("⚠️ Data files not found. Generating via run_demo_in_spyder(...).")
    from fraud_fusion_starter import run_demo_in_spyder
    run_demo_in_spyder(kaggle_csv=None, base_dir=str(BASE_DIR))

print("📦 Assembling dataset…")
X_full, y_full, num_cols, cat_cols = assemble_dataset(TRANS_CSV, KS_CSV, TXT_CSV)

# Keep a reproducible split (same as your evaluate script: 80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.20, random_state=42, stratify=y_full
)

# -----------------------------
# 2) Evaluate Fusion model (Figures 3–6)
# -----------------------------
print("\n🎯 Evaluating Fusion model…")
# Predict proba on the pipeline
y_proba = model.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= best_thr).astype(int)

auc = roc_auc_score(y_test, y_proba)
print(f"Fusion AUC: {auc:.4f}")
print(classification_report(y_test, y_pred, digits=4))

# Figure 3: ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig = plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, lw=2, label=f"Fusion (AUC={auc:.3f})")
plt.plot([0,1],[0,1], linestyle="--", lw=1, color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Fusion Model")
plt.legend()
plot_and_save(fig, FIG_DIR / "figure3_roc_fusion.png")

# Figure 4: Precision–Recall
prec, rec, _ = precision_recall_curve(y_test, y_proba)
fig = plt.figure(figsize=(6,5))
plt.plot(rec, prec, lw=2, label="Fusion")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve — Fusion Model")
plt.legend()
plot_and_save(fig, FIG_DIR / "figure4_pr_fusion.png")

# Figure 5: Confusion Matrix at best threshold
cm = confusion_matrix(y_test, y_pred)
fig = plt.figure(figsize=(5,4))
if HAS_SEABORN:
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
else:
    plt.imshow(cm, interpolation="nearest")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.title(f"Confusion Matrix — Fusion (thr={best_thr:.3f})")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plot_and_save(fig, FIG_DIR / "figure5_confusion_fusion.png")

# Figure 6: Feature Importance (Fusion)
# Extract feature names after preprocessing
print("🔍 Computing feature importances…")
pre = model.named_steps["preprocessor"]
clf = model.named_steps["classifier"]

feature_names = pre.get_feature_names_out()
# CatBoost feature importance aligned to transformed features
importances = clf.get_feature_importance(type="FeatureImportance")

fi = pd.DataFrame({"feature": feature_names, "importance": importances})
fi = fi.sort_values("importance", ascending=False)
topN = 20
fig = plt.figure(figsize=(8,7))
plt.barh(fi.head(topN)["feature"][::-1], fi.head(topN)["importance"][::-1])
plt.title("Top Feature Importances — Fusion Model")
plt.xlabel("Importance")
plt.ylabel("Features")
plot_and_save(fig, FIG_DIR / "figure6_feature_importance_fusion.png")

# -----------------------------
# 3) Baseline (transaction-only) vs Fusion (Figure 8)
# -----------------------------
print("\n🧪 Training Baseline (transaction-only) for comparison…")
txn_num = ["transaction_amount", "hour_of_day", "transaction_frequency", "average_amount"]
txn_cat = ["merchant_category", "device_type", "location_cluster"]

# Keep only transaction columns
X_train_txn = X_train[txn_num + txn_cat].copy()
X_test_txn  = X_test[txn_num + txn_cat].copy()

pre_txn = ColumnTransformer([
    ("num", StandardScaler(), txn_num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), txn_cat)
])

# Lightweight baseline CatBoost (fast-ish)
class_weights = [1, (len(y_train) - y_train.sum()) / y_train.sum()]
baseline_clf = CatBoostClassifier(
    verbose=0, random_state=42, eval_metric="AUC",
    iterations=200, depth=4, learning_rate=0.05,
    class_weights=class_weights
)

baseline_pipe = ImbPipeline([
    ("pre", pre_txn),
    ("clf", baseline_clf)
])
baseline_pipe.fit(X_train_txn, y_train)

y_proba_base = baseline_pipe.predict_proba(X_test_txn)[:,1]
# Keep default thr=0.5 for baseline to reflect typical usage
y_pred_base = (y_proba_base >= 0.5).astype(int)

auc_base = roc_auc_score(y_test, y_proba_base)

# Compute metrics
def prec_rec_f1(y_true, y_hat):
    from sklearn.metrics import precision_score, recall_score, f1_score
    return (
        precision_score(y_true, y_hat, zero_division=0),
        recall_score(y_true, y_hat, zero_division=0),
        f1_score(y_true, y_hat, zero_division=0),
    )
p_base, r_base, f1_base = prec_rec_f1(y_test, y_pred_base)
p_fuse, r_fuse, f1_fuse = prec_rec_f1(y_test, y_pred)

# Figure 8: Baseline vs Fusion
labels = ["AUC", "Precision", "Recall", "F1-Score"]
baseline_scores = [auc_base, p_base, r_base, f1_base]
fusion_scores   = [auc, p_fuse, r_fuse, f1_fuse]

x = np.arange(len(labels))
width = 0.35
fig = plt.figure(figsize=(7,5))
ax = plt.gca()
ax.bar(x - width/2, baseline_scores, width, label="Baseline (txn-only)")
ax.bar(x + width/2, fusion_scores,   width, label="Fusion")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_title("Baseline vs Fusion Performance (test set)")
ax.legend()
plot_and_save(fig, FIG_DIR / "figure8_baseline_vs_fusion.png")

print(f"Baseline AUC={auc_base:.4f}, P={p_base:.3f}, R={r_base:.3f}, F1={f1_base:.3f}")
print(f"Fusion   AUC={auc:.4f},   P={p_fuse:.3f}, R={r_fuse:.3f}, F1={f1_fuse:.3f}")

# -----------------------------
# 4) Optional: Figure 7 (CV heatmap) if cv_results.csv exists
# -----------------------------
CV_CSV = BASE_DIR / "cv_results.csv"   # If you saved this during training
if CV_CSV.exists():
    print("\n📊 Found cv_results.csv — building Figure 7 heatmap…")
    cv = pd.read_csv(CV_CSV)
    # Expect columns like: param_classifier__depth, param_classifier__learning_rate, mean_test_score
    if HAS_SEABORN and {"param_classifier__depth", "param_classifier__learning_rate", "mean_test_score"}.issubset(cv.columns):
        pivot = cv.pivot_table(
            index="param_classifier__depth",
            columns="param_classifier__learning_rate",
            values="mean_test_score",
            aggfunc="mean"
        )
        fig = plt.figure(figsize=(6,5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues")
        plt.title("Cross-Validation Hyperparameter Grid (AUC)")
        plt.xlabel("learning_rate"); plt.ylabel("depth")
        plot_and_save(fig, FIG_DIR / "figure7_cv_heatmap.png")
        print("✅ Saved Figure 7.")
    else:
        print("⚠️ cv_results.csv format not recognised; skipping Figure 7.")
else:
    print("\nℹ️ No cv_results.csv found — skipping Figure 7. (Optional)")

print("\n✅ All figures saved in:", FIG_DIR.resolve())
