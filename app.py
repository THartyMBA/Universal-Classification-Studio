# app.py
"""
Universal Classification Studio  🔮
---------------------------------
Upload **any** CSV (mixed numeric + categorical columns), pick the *target*
column, choose a model, and click **Train**.

The app will

1. auto-encode categoricals, scale / impute numerics,
2. split your data into train / test,
3. fit a scikit-learn classifier,
4. show accuracy, ROC-AUC, a ROC curve (for binary targets),
5. append the predicted class-probability to your data,
6. let you download the scored file.

Perfect for portfolio demos such as lead-scoring or credit-risk modeling.
"""

import io
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ───────────────────────────────────────────────────────
# Caching helpers
# ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

@st.cache_resource(show_spinner=False)
def build_model(model_name: str):
    """Return an (un-fitted) sklearn estimator given a friendly name."""
    if model_name == "Gradient Boosting":
        return GradientBoostingClassifier()
    if model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, n_jobs=-1)
    raise ValueError("Unknown model")

def make_pipeline(df: pd.DataFrame, target_col: str, model_name: str):
    numeric_cols = df.drop(columns=[target_col]).select_dtypes(include=["number"]).columns
    categorical_cols = df.drop(columns=[target_col]).select_dtypes(exclude=["number"]).columns

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
    model = build_model(model_name)
    full_pipe = Pipeline(steps=[("pre", pre), ("clf", model)])
    return full_pipe

def train_and_score(df: pd.DataFrame, target_col: str, model_name: str, test_size: float):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    pipe = make_pipeline(df, target_col, model_name)
    pipe.fit(X_train, y_train)

    # Probability of positive class (for binary) or class 1 (for multiclass)
    probas = pipe.predict_proba(X_test)
    y_pred = np.argmax(probas, axis=1) if probas.shape[1] > 2 else (probas[:, 1] > 0.5)

    acc = accuracy_score(y_test, y_pred)

    # ROC-AUC (handles multi-class with one-vs-rest when >2 classes)
    try:
        auc = roc_auc_score(y_test, probas, multi_class="ovr" if probas.shape[1] > 2 else "raise")
    except ValueError:
        auc = np.nan  # occurs if only one label present in y_test

    metrics = {"accuracy": acc, "roc_auc": auc, "classes": pipe.classes_}

    # Build ROC curve only for binary
    roc_data = None
    if probas.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_test, probas[:, 1])
        roc_data = (fpr, tpr)

    # Predict on entire dataset to return scored DataFrame
    scored_df = df.copy()
    scored_df["pred_proba"] = (
        pipe.predict_proba(df.drop(columns=[target_col]))[:, 1]
        if probas.shape[1] == 2
        else pipe.predict_proba(df.drop(columns=[target_col])).max(axis=1)  # highest class prob.
    )

    return pipe, metrics, roc_data, scored_df

def plot_roc(fpr, tpr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name="ROC", mode="lines"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        showlegend=False,
        width=600,
        height=450,
    )
    return fig

# ───────────────────────────────────────────────────────
# Streamlit UI
# ───────────────────────────────────────────────────────
st.set_page_config(page_title="Universal Classification Studio", layout="wide")
st.title("🧠 Universal Classification Studio")
st.info(
    "🔔 **Demo Notice**  \n"
    "This application is a streamlined proof-of-concept, **not** an "
    "enterprise-grade product.  \n\n"
    "Need production-level performance, security or custom features? "
    "[Get in touch](mailto:drtom@paced.drtomharty.com) and let’s build a tailored solution.",
    icon="💡",
)

uploaded = st.file_uploader("📂 Upload a CSV file", type=["csv"])
if not uploaded:
    st.info("Awaiting CSV upload…")
    st.stop()

df = load_csv(uploaded)
st.subheader("Preview")
st.dataframe(df.head())

target_col = st.selectbox("🎯 Choose the target column (Y)", df.columns)
model_name = st.selectbox("🤖 Choose an algorithm", ["Gradient Boosting", "Random Forest", "Logistic Regression"])
test_size = st.slider("Test-size (validation split %)", 0.1, 0.5, 0.2, step=0.05)

if st.button("🚀 Train model"):
    with st.spinner("Crunching numbers…"):
        model, metrics, roc_data, scored_df = train_and_score(df, target_col, model_name, test_size)

    # Metrics
    st.success("Model trained!")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    col2.metric("ROC-AUC", "N/A" if np.isnan(metrics['roc_auc']) else f"{metrics['roc_auc']:.3f}")

    # ROC plot
    if roc_data:
        st.plotly_chart(plot_roc(*roc_data), use_container_width=False)

    # Scored sample
    st.subheader("Scored data (top 10 rows)")
    st.dataframe(scored_df.head(10))

    # Downloads
    csv_bytes = scored_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download scored CSV", csv_bytes, "scored_data.csv", "text/csv")

    # Let advanced users grab the fitted model
    pkl_bytes = pickle.dumps(model)
    st.download_button("💾 Download trained model (.pkl)", pkl_bytes, "model.pkl", "application/octet-stream")

