# streamlit_app.py
# Premium Streamlit UI for Fusion-Based Fraud Detection
# Works with your existing fraud_fusion_starter.py

import os
from pathlib import Path
import json
import time
import io

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# sentiment (optional but recommended)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_OK = True
except Exception:
    _VADER_OK = False

# --- your module ---
from fraud_fusion_starter import (
    prepare_transactions,
    generate_synthetic_keystrokes,
    generate_synthetic_text,
    assemble_dataset,
    train_model,
    run_demo_in_spyder,
)

# ========== CONFIG ==========
st.set_page_config(
    page_title="Fusion Fraud Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path("data/project")
MODEL_PATH = BASE / "fusion_fraud_model.pkl"
META_PATH  = BASE / "fusion_fraud_meta.json"
TRANS_CSV  = BASE / "transactions.csv"
KS_CSV     = BASE / "keystrokes.csv"
TXT_CSV    = BASE / "text.csv"

# ========== STYLES ==========
st.markdown("""
<style>
:root {
  --bg: #0b132b;
  --card: #1c2541;
  --accent: #5bc0be;
  --accent2: #3a506b;
  --text: #e8f1f2;
}
html, body, [class*="css"]  {
  color: var(--text);
  background-color: var(--bg);
}
section.main>div { max-width: 1400px; }
.big-header {
  background: linear-gradient(135deg, var(--accent2), var(--accent));
  color: white; padding: 22px 26px; border-radius: 18px; margin-bottom: 14px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}
.kpi {
  background: var(--card);
  border-radius: 16px; padding: 18px 20px; height: 120px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  border: 1px solid #2c365e;
}
.kpi h3 { font-size: 14px; margin: 0; color: #b5c6d6; letter-spacing: .5px; }
.kpi .val { font-size: 28px; font-weight: 800; margin-top: 6px; color: #ffffff; }
.card {
  background: var(--card); border: 1px solid #2c365e;
  border-radius: 16px; padding: 18px 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
button[kind="primary"] { background: var(--accent) !important; }
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] { background: var(--card); padding: 10px 16px; border-radius: 12px; color: #cfe7ea; }
.stTabs [aria-selected="true"] { background: var(--accent2); color: white; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="big-header">
  <h2 style="margin:0; font-weight:800;">Fusion Fraud Intelligence</h2>
  <div style="opacity:.9">Behavioural • Emotional • Transactional — enterprise-grade detection</div>
</div>
""", unsafe_allow_html=True)

# ========== HELPERS ==========

@st.cache_resource(show_spinner=False)
def load_model_and_meta():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return model, meta

@st.cache_data(show_spinner=False)
def load_dataset():
    if not (TRANS_CSV.exists() and KS_CSV.exists() and TXT_CSV.exists()):
        return None
    X, y, num_cols, cat_cols = assemble_dataset(str(TRANS_CSV), str(KS_CSV), str(TXT_CSV))
    return X, y, num_cols, cat_cols

def score_dataframe(pipe, df, thr):
    proba = pipe.predict_proba(df)[:, 1]
    pred = (proba >= thr).astype(int)
    return proba, pred

def ensure_project_ready(kaggle_csv_path=None):
    BASE.mkdir(parents=True, exist_ok=True)
    if kaggle_csv_path and Path(kaggle_csv_path).exists():
        trans = prepare_transactions(kaggle_csv_path, str(BASE))
    else:
        trans = prepare_transactions(None, str(BASE))
    ks = generate_synthetic_keystrokes(trans, str(BASE))
    txt = generate_synthetic_text(trans, str(BASE))
    # Training
    X, y, num_cols, cat_cols = assemble_dataset(trans, ks, txt)
    model = train_model(X, y, num_cols, cat_cols)
    joblib.dump(model, MODEL_PATH)
    # Pick threshold (re-using simple f1 method)
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_curve, average_precision_score
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    prec, rec, thr = precision_recall_curve(yte, p)
    f1 = (2*prec[:-1]*rec[:-1])/(prec[:-1]+rec[:-1]+1e-9)
    best_idx = int(np.nanargmax(f1))
    best_thr = float(thr[best_idx])
    meta = {
        "threshold": best_thr,
        "method": "f1",
        "metrics": {"precision": float(prec[best_idx]), "recall": float(rec[best_idx]), "f1": float(f1[best_idx])},
        "average_precision": float(average_precision_score(yte, p))
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    return model, meta

def get_vader_sentiment(text: str):
    if not _VADER_OK or not text:
        return 0.0, 0.5
    s = SentimentIntensityAnalyzer().polarity_scores(text)
    # Map to your two fields
    polarity = float(s["compound"])
    subjectivity = float(1 - s["neu"])
    return polarity, subjectivity

def feature_importance_dataframe(pipe, feature_df: pd.DataFrame):
    # Map CatBoost importances back to transformed feature names
    try:
        clf = pipe.named_steps.get("classifier", None) or pipe.named_steps.get("clf", None)
        pre = pipe.named_steps.get("preprocessor", None) or pipe.named_steps.get("pre", None)
        if clf is None or pre is None: return None
        # get transformed feature names
        ohe_names = []
        if hasattr(pre, "get_feature_names_out"):
            ohe_names = list(pre.get_feature_names_out())
        else:
            # ColumnTransformer stores transformers_ after fit
            ohe_names = [f"f{i}" for i in range(clf.feature_importances_.shape[0])]
        imp = clf.get_feature_importance() if hasattr(clf, "get_feature_importance") else clf.feature_importances_
        return pd.DataFrame({"feature": ohe_names, "importance": imp}).sort_values("importance", ascending=False).head(25)
    except Exception:
        return None

# ========== SIDEBAR ==========
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("", ["🏠 Dashboard", "⚡ Single Session", "📦 Batch Scoring", "🛠 Setup / Train"])

st.sidebar.markdown("---")
st.sidebar.caption("Model files")
st.sidebar.write(f"**Model**: {'✅' if MODEL_PATH.exists() else '❌'}  \n**Meta**: {'✅' if META_PATH.exists() else '❌'}")
st.sidebar.write(f"**Transactions**: {'✅' if TRANS_CSV.exists() else '❌'}")
st.sidebar.write(f"**Keystrokes**: {'✅' if KS_CSV.exists() else '❌'}")
st.sidebar.write(f"**Text**: {'✅' if TXT_CSV.exists() else '❌'}")

# ========== LOAD ==========
model, meta = load_model_and_meta()
data = load_dataset()

# ========== PAGES ==========
if page == "🏠 Dashboard":
    st.markdown("#### Executive Overview")
    col1, col2, col3, col4 = st.columns(4)
    if model is not None and meta is not None and data is not None:
        X, y, *_ = data
        thr = float(meta.get("threshold", 0.5))
        proba, pred = score_dataframe(model, X, thr)
        auc_val = float(((__import__("sklearn.metrics").metrics).roc_auc_score)(y, proba))
        tp = int(((pred==1)&(y==1)).sum())
        fp = int(((pred==1)&(y==0)).sum())
        fn = int(((pred==0)&(y==1)).sum())
        tn = int(((pred==0)&(y==0)).sum())
        with col1:
            st.markdown('<div class="kpi"><h3>AUC</h3><div class="val">{:.4f}</div></div>'.format(auc_val), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="kpi"><h3>Threshold</h3><div class="val">{:.3f}</div></div>'.format(thr), unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="kpi"><h3>Detected Frauds</h3><div class="val">{}</div></div>'.format(tp), unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="kpi"><h3>False Positives</h3><div class="val">{}</div></div>'.format(fp), unsafe_allow_html=True)

        c1, c2 = st.columns((2,1))
        with c1:
            st.markdown("##### Score Distribution")
            fig = px.histogram(pd.DataFrame({"score": proba}), x="score", nbins=40, template="plotly_dark")
            fig.add_vline(x=thr, line_dash="dash", line_color="#5bc0be")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("##### Confusion Matrix")
            cm = np.array([[tn, fp],[fn, tp]])
            z = cm
            fig2 = go.Figure(data=go.Heatmap(
                z=z, x=["Pred:Legit","Pred:Fraud"], y=["True:Legit","True:Fraud"],
                text=z, texttemplate="%{text}", colorscale="blues"))
            fig2.update_layout(template="plotly_dark", height=320)
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Top Features (model view)"):
            imp_df = feature_importance_dataframe(model, X)
            if imp_df is not None:
                st.dataframe(imp_df, use_container_width=True)
            else:
                st.info("Feature importances not available for this pipeline.")

    else:
        st.info("Model or data not found. Go to **Setup / Train** to build the project.")

elif page == "⚡ Single Session":
    st.markdown("#### Score a Single Session")
    st.caption("Fill the form below. Optionally type a note to derive sentiment & keystroke speed.")

    # ---- left: inputs ----
    c1, c2 = st.columns((1,1))
    with c1:
        st.markdown("###### Transaction")
        amt = st.slider("Transaction amount", 1.0, 5000.0, 129.0, 1.0)
        hour = st.slider("Hour of day", 0, 23, 14)
        merchant = st.selectbox("Merchant category", ["electronics","fashion","groceries","travel","crypto","gift"], index=0)
        device = st.selectbox("Device type", ["mobile","desktop","tablet"], index=0)
        loc = st.selectbox("Location cluster", ["UK","India","USA","LC00","LC01","LC02"], index=0)
        # basic frequency/profiles (demo)
        tx_freq = st.number_input("Transaction frequency (user)", min_value=1, value=3)
        avg_amt = st.number_input("User average amount", min_value=1.0, value=180.0, step=1.0)

    with c2:
        st.markdown("###### Behaviour & Text")
        st.caption("Click **Start**, type your note, then **Stop** to estimate typing speed.")
        if "typing_start" not in st.session_state:
            st.session_state.typing_start = None

        text = st.text_area("Type a short note (reason/note)", height=120, placeholder="e.g., 'Please cancel immediately, unusual activity detected'")

        cols_t = st.columns(3)
        with cols_t[0]:
            if st.button("▶️ Start"):
                st.session_state.typing_start = time.time()
        with cols_t[1]:
            if st.button("⏸ Stop"):
                if st.session_state.typing_start:
                    st.session_state.typing_duration = time.time() - st.session_state.typing_start
                else:
                    st.session_state.typing_duration = None
        with cols_t[2]:
            st.write("")

        duration = st.session_state.get("typing_duration", None)
        if duration and text:
            typing_speed = round(len(text) / max(duration, 0.1), 2)  # chars/sec
        else:
            typing_speed = 3.0  # default

        # simple knobs to approximate other keystroke features
        mean_hold = st.slider("Mean hold time (ms)", 60, 250, 110)
        std_hold  = st.slider("Std hold time (ms)", 5, 120, 30)
        error_rate = st.slider("Error rate (backspace ratio)", 0.0, 0.5, 0.05, 0.01)

        # sentiment
        if _VADER_OK:
            pol, subj = get_vader_sentiment(text or "")
        else:
            pol = 0.0; subj = 0.5

    st.markdown("---")

    if st.button("💡 Score Session", use_container_width=True):
        # Build single-row input aligned to your training features
        # Your training used: num_cols = ["transaction_amount","hour_of_day","transaction_frequency","average_amount",
        # "mean_hold_time","std_hold_time","typing_speed","error_rate","sentiment_polarity","sentiment_subjectivity"]
        # cat_cols = ["merchant_category","device_type","location_cluster"]

        row = pd.DataFrame([{
            "transaction_amount": amt,
            "hour_of_day": hour,
            "transaction_frequency": tx_freq,
            "average_amount": avg_amt,
            "mean_hold_time": mean_hold,
            "std_hold_time": std_hold,
            "typing_speed": typing_speed,
            "error_rate": error_rate,
            "sentiment_polarity": pol,
            "sentiment_subjectivity": subj,
            "merchant_category": merchant,
            "device_type": device,
            "location_cluster": loc
        }])

        if model is None or meta is None:
            st.error("Model not found. Please train it on the Setup page first.")
        else:
            thr = float(meta.get("threshold", 0.5))
            p = model.predict_proba(row)[:, 1][0]
            is_fraud = (p >= thr)
            # Pretty gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(p*100),
                number={'suffix': "%"},
                title={'text': "Fraud Probability"},
                gauge={
                    'axis': {'range': [0,100]},
                    'bar': {'color': '#5bc0be'},
                    'steps': [
                        {'range':[0, 30], 'color':'#1c2541'},
                        {'range':[30, 70], 'color':'#3a506b'},
                        {'range':[70, 100], 'color':'#5bc0be'}
                    ]}))
            fig.update_layout(template="plotly_dark", height=260)
            st.plotly_chart(fig, use_container_width=True)

            verdict = "🚨 Fraud" if is_fraud else "✅ Legit"
            st.success(f"**Decision:** {verdict}  \n**Score:** {p:.4f} (threshold {thr:.3f})")

            with st.expander("View feature payload"):
                st.json(row.to_dict(orient="records")[0])

elif page == "📦 Batch Scoring":
    st.markdown("#### Batch Scoring")
    st.caption("Upload a CSV of **already-assembled features** or your **transactions.csv** (will try to merge with existing keystrokes & text).")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    mode = st.radio("Input type", ["Assembled feature table", "Raw transactions.csv"], horizontal=True)

    if st.button("Run batch scoring", use_container_width=True):
        if model is None or meta is None:
            st.error("Model not found. Train it on Setup / Train first.")
        elif uploaded is None:
            st.error("Please upload a CSV.")
        else:
            thr = float(meta.get("threshold", 0.5))
            raw = pd.read_csv(uploaded)

            try:
                if mode == "Raw transactions.csv":
                    # Try to use existing local keystrokes/text for merge:
                    if KS_CSV.exists() and TXT_CSV.exists():
                        # assemble expects file paths, but we have a DataFrame → save temp
                        tmp_trans = BASE / "tmp_upload_transactions.csv"
                        raw.to_csv(tmp_trans, index=False)
                        X, y, num_cols, cat_cols = assemble_dataset(str(tmp_trans), str(KS_CSV), str(TXT_CSV))
                        df_to_score = X.copy()
                        tmp_trans.unlink(missing_ok=True)
                    else:
                        st.warning("Local keystrokes/text not found; expecting assembled features.")
                        df_to_score = raw.copy()
                else:
                    df_to_score = raw.copy()

                proba, pred = score_dataframe(model, df_to_score, thr)
                out = df_to_score.copy()
                out["fraud_proba"] = proba
                out["fraud_pred"] = pred

                st.success(f"Scored {len(out)} rows.")
                st.dataframe(out.head(30), use_container_width=True)
                buf = io.BytesIO()
                out.to_csv(buf, index=False)
                st.download_button("⬇️ Download scored CSV", data=buf.getvalue(), file_name="scored.csv", mime="text/csv")
            except Exception as e:
                st.exception(e)

elif page == "🛠 Setup / Train":
    st.markdown("#### Project Setup / Training")
    st.caption("If you haven’t trained yet, do it here. You can use Kaggle `creditcard.csv` placed in this folder, or go fully synthetic.")

    kaggle_present = Path("creditcard.csv").exists()
    st.write(f"**Kaggle creditcard.csv present:** {'✅' if kaggle_present else '❌'}")

    use_kaggle = st.checkbox("Use Kaggle creditcard.csv if available", value=kaggle_present)
    if st.button("🚀 Build dataset & train model", use_container_width=True):
        with st.spinner("Preparing data and training..."):
            try:
                model2, meta2 = ensure_project_ready("creditcard.csv" if use_kaggle and kaggle_present else None)
                st.success("Training complete.")
                st.json(meta2)
                st.info("Reload the page to refresh the Dashboard.")
            except Exception as e:
                st.exception(e)

    st.markdown("---")
    st.caption("Utilities")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Recompute dataset (synthetic)"):
            try:
                prepare_transactions(None, str(BASE))
                generate_synthetic_keystrokes(str(TRANS_CSV), str(BASE))
                generate_synthetic_text(str(TRANS_CSV), str(BASE))
                st.success("Synthetic dataset regenerated.")
            except Exception as e:
                st.exception(e)
    with c2:
        if st.button("Clear cached data"):
            load_dataset.clear()
            load_model_and_meta.clear()
            st.success("Cleared.")
    with c3:
        if st.button("Delete model files"):
            try:
                if MODEL_PATH.exists(): MODEL_PATH.unlink()
                if META_PATH.exists(): META_PATH.unlink()
                st.success("Deleted model/meta.")
            except Exception as e:
                st.exception(e)
