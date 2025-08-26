# fraud_fusion_starter.py
# Author: Prathamesh Waydande
# Dissertation: Fusion-Based Financial Fraud Detection using Behavioural, Emotional & Transactional Signals

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC
from catboost import CatBoostClassifier

# ----------------------------
# 1. Synthetic Data Generators
# ----------------------------

def prepare_transactions(kaggle_csv: str | None, out_dir: str) -> str:
    """Prepare transactions.csv from Kaggle dataset (creditcard.csv) or synthetic."""
    out_path = Path(out_dir) / "transactions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if kaggle_csv:
        df = pd.read_csv(kaggle_csv)
        # adapt Kaggle creditcard.csv (Time, V1..V28, Amount, Class)
        df = df.rename(columns={"Class": "is_fraud", "Amount": "transaction_amount"})
        df["session_id"] = np.arange(len(df))
        df["user_id"] = np.random.randint(1000, 2000, len(df))
        df["merchant_category"] = np.random.choice(["electronics", "fashion", "groceries"], len(df))
        df["device_type"] = np.random.choice(["mobile", "desktop"], len(df))
        df["location_cluster"] = np.random.choice(["UK", "India", "USA"], len(df))
        df["hour_of_day"] = (df["Time"] // 3600) % 24
        df["transaction_frequency"] = df.groupby("user_id")["transaction_amount"].transform("count")
        df["average_amount"] = df.groupby("user_id")["transaction_amount"].transform("mean")
    else:
        # generate synthetic if no Kaggle dataset
        n = 5000
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "session_id": np.arange(n),
            "user_id": rng.integers(1000, 2000, n),
            "transaction_amount": rng.gamma(2, 50, n),
            "merchant_category": rng.choice(["electronics", "fashion", "groceries"], n),
            "device_type": rng.choice(["mobile", "desktop"], n),
            "location_cluster": rng.choice(["UK", "India", "USA"], n),
            "hour_of_day": rng.integers(0, 24, n),
        })
        df["transaction_frequency"] = df.groupby("user_id")["transaction_amount"].transform("count")
        df["average_amount"] = df.groupby("user_id")["transaction_amount"].transform("mean")
        df["is_fraud"] = rng.choice([0, 1], size=n, p=[0.97, 0.03])

    df.to_csv(out_path, index=False)
    return str(out_path)


def generate_synthetic_keystrokes(transactions_csv: str, out_dir: str, chars_per_form: int = 40) -> str:
    """Generate raw keystroke features linked to transactions."""
    rng = np.random.default_rng(42)
    out_path = Path(out_dir) / "keystrokes.csv"
    tdf = pd.read_csv(transactions_csv)
    rows = []

    for _, r in tdf.iterrows():
        sid = r["session_id"]
        uid = r["user_id"]
        is_f = int(r["is_fraud"]) if "is_fraud" in r else 0
        mean_dwell = 140 if is_f else 110
        dwell_jitter = 30
        mean_flight = 90
        flight_jitter = 25

        t = rng.integers(1_000_000, 9_999_999)
        for _ in range(chars_per_form):
            key = int(rng.integers(30, 126))
            dwell = max(25, rng.normal(mean_dwell, dwell_jitter))
            flight = max(20, rng.normal(mean_flight, flight_jitter))
            if rng.random() < (0.10 if is_f else 0.03):
                t += int(1500 + rng.random() * 1500)
            rows.append({"session_id": sid, "user_id": uid, "key_code": key, "event_type": "down", "t": int(t), "is_backspace": 0})
            rows.append({"session_id": sid, "user_id": uid, "key_code": key, "event_type": "up",   "t": int(t + dwell), "is_backspace": 0})
            t += int(dwell + flight)
            if rng.random() < (0.20 if is_f else 0.05):
                rows.append({"session_id": sid, "user_id": uid, "key_code": 8, "event_type": "down", "t": int(t-10), "is_backspace": 1})
                rows.append({"session_id": sid, "user_id": uid, "key_code": 8, "event_type": "up",   "t": int(t),    "is_backspace": 1})

    pd.DataFrame(rows).to_csv(out_path, index=False)
    return str(out_path)


def generate_synthetic_text(transactions_csv: str, out_dir: str) -> str:
    """Generate sentiment-based text features linked to transactions."""
    rng = np.random.default_rng(42)
    out_path = Path(out_dir) / "text.csv"
    tdf = pd.read_csv(transactions_csv)

    sentiments = []
    for _, r in tdf.iterrows():
        is_f = int(r["is_fraud"]) if "is_fraud" in r else 0
        polarity = rng.normal(0.1 if is_f else 0.4, 0.2)
        subjectivity = rng.uniform(0.3, 0.9)
        sentiments.append({"session_id": r["session_id"], "user_id": r["user_id"], "sentiment_polarity": polarity, "sentiment_subjectivity": subjectivity})

    pd.DataFrame(sentiments).to_csv(out_path, index=False)
    return str(out_path)

# ----------------------------
# 2. Feature Assembly
# ----------------------------

def assemble_dataset(trans_csv: str, ks_csv: str, txt_csv: str, target_col: str = "is_fraud"):
    trans = pd.read_csv(trans_csv)
    ks = pd.read_csv(ks_csv)
    txt = pd.read_csv(txt_csv)

    # Aggregate keystrokes
    ks_agg = ks.groupby("session_id").agg(
        mean_hold_time=("t", "mean"),
        std_hold_time=("t", "std"),
        typing_speed=("t", "count"),
        error_rate=("is_backspace", "mean"),
    ).reset_index()

    full = trans.merge(ks_agg, on="session_id").merge(txt, on=["session_id", "user_id"])
    y = full[target_col]

    num_cols = ["transaction_amount", "hour_of_day", "transaction_frequency", "average_amount",
                "mean_hold_time", "std_hold_time", "typing_speed", "error_rate",
                "sentiment_polarity", "sentiment_subjectivity"]
    cat_cols = ["merchant_category", "device_type", "location_cluster"]

    return full[num_cols + cat_cols], y, num_cols, cat_cols

# ----------------------------
# 3. Training
# ----------------------------

def train_model(X, y, num_cols, cat_cols):
    cat_indices = [X.columns.get_loc(c) for c in cat_cols]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    model = CatBoostClassifier(verbose=0, random_state=42, eval_metric="AUC",
                               class_weights=[1, (len(y) - sum(y)) / sum(y)])

    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTENC(categorical_features=cat_indices, random_state=42)),
        ("classifier", model)
    ])

    param_dist = {
        "classifier__depth": [4, 6, 8],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__l2_leaf_reg": [1, 3, 5],
        "classifier__iterations": [200, 500]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=5,
        scoring="roc_auc", cv=cv, n_jobs=-1, random_state=42, verbose=2
    )
    search.fit(X, y)

    print("\n✅ Best params:", search.best_params_)
    return search.best_estimator_

# ----------------------------
# 4. Orchestration
# ----------------------------

def run_demo_in_spyder(kaggle_csv: str | None = None, base_dir: str = "./data/project"):
    print("[1/4] Preparing transactions...")
    trans_csv = prepare_transactions(kaggle_csv, base_dir)
    print("transactions.csv →", trans_csv)

    print("[2/4] Generating keystrokes & text...")
    ks_csv = generate_synthetic_keystrokes(trans_csv, base_dir)
    txt_csv = generate_synthetic_text(trans_csv, base_dir)
    print("keystrokes.csv →", ks_csv)
    print("text.csv →", txt_csv)

    print("[3/4] Assembling dataset...")
    X, y, num_cols, cat_cols = assemble_dataset(trans_csv, ks_csv, txt_csv)

    print("[4/4] Training model...")
    model = train_model(X, y, num_cols, cat_cols)

    out_path = Path(base_dir) / "fusion_fraud_model.pkl"
    joblib.dump(model, out_path)
    print(f"\n✅ Model saved at {out_path}")
    import json
    from sklearn.metrics import precision_recall_curve, average_precision_score

    def pick_best_threshold(y_true, y_prob, method="f1", cost_fn=5.0, cost_fp=1.0):
        prec, rec, thr = precision_recall_curve(y_true, y_prob)
        if method == "f1":
            f1 = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9)
            best_idx = int(np.nanargmax(f1))
            return float(thr[best_idx]), {
                "precision": float(prec[best_idx]),
                "recall": float(rec[best_idx]),
                "f1": float(f1[best_idx])
            }
        elif method == "cost":
            best_thr, best_cost = 0.5, float("inf")
            for t in np.linspace(0.01, 0.99, 99):
                pred = (y_prob >= t).astype(int)
                fp = int(((pred==1) & (y_true==0)).sum())
                fn = int(((pred==0) & (y_true==1)).sum())
                cost = fn*cost_fn + fp*cost_fp
                if cost < best_cost:
                    best_thr, best_cost = t, cost
            return best_thr, {"cost": best_cost}
        else:
            return 0.5, {}

    # Re-run train/test split inside this function for threshold tuning
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    best_thr, meta = pick_best_threshold(y_test.values, y_proba, method="f1")

    meta_out = {
        "threshold": best_thr,
        "method": "f1",
        "metrics": meta,
        "average_precision": average_precision_score(y_test, y_proba)
    }
    with open(Path(base_dir) / "fusion_fraud_meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"\n✅ Saved best threshold = {best_thr:.3f} (method=f1)")
    print(f"Meta saved to {Path(base_dir) / 'fusion_fraud_meta.json'}")
    return model

if __name__ == "__main__":
    run_demo_in_spyder(kaggle_csv=None)
