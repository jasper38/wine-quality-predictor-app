
import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Paths
# -------------------------------
PIPELINE_PATH = "wine_quality_pipeline.joblib"
META_PATH = "metadata.joblib"

# -------------------------------
# Load pipeline and metadata
# -------------------------------
@st.cache_resource
def load_pipeline():
    pipeline = joblib.load(PIPELINE_PATH)
    meta = joblib.load(META_PATH)
    return pipeline, meta

pipeline, meta = load_pipeline()
feature_cols = meta['feature_cols']

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Boutique Wine Predictor 🍷", layout="centered")
st.markdown("<h1 style='text-align: center; color: #8B0000;'>🍷 Boutique Wine Quality Predictor</h1>", unsafe_allow_html=True)
st.write("<p style='text-align: center;'>Enter wine features to predict premium quality.</p>", unsafe_allow_html=True)

# -------------------------------
# Input Section
# -------------------------------
input_method = st.radio("Input Method:", ["Single Sample", "Batch CSV Upload"], index=0, horizontal=True)

def single_sample_input():
    st.subheader("Wine Sample Features")
    data = {}
    col1, col2 = st.columns(2)

    # Left column
    with col1:
        for i, col in enumerate(feature_cols):
            if i % 2 == 0:
                data[col] = st.number_input(col, value=0.0, format="%.6f")
    # Right column
    with col2:
        for i, col in enumerate(feature_cols):
            if i % 2 == 1:
                data[col] = st.number_input(col, value=0.0, format="%.6f")

    return pd.DataFrame([data])

def batch_csv_input():
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        # Compute engineered features if missing
        if 'acidity_ratio' not in df.columns:
            df['acidity_ratio'] = df['fixed_acidity'] / (df['volatile_acidity'] + 1e-6)
        if 'sulfur_ratio' not in df.columns:
            df['sulfur_ratio'] = df['free_sulfur_dioxide'] / (df['total_sulfur_dioxide'] + 1e-6)
        # Add missing columns
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        # Reorder
        input_df = df[feature_cols]
        st.subheader("Uploaded CSV Preview")
        st.dataframe(df.head())
        return input_df
    return None

input_df = single_sample_input() if input_method == "Single Sample" else batch_csv_input()

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Quality"):
    if input_df is not None and not input_df.empty:
        Xt = input_df[feature_cols]
        pred_prob = pipeline.predict_proba(Xt)[:, 1]
        pred_class = pipeline.predict(Xt)

        st.markdown("### Prediction Results:")
        for i, pred in enumerate(pred_class):
            label = "GOOD WINE" if pred == 1 else "NOT GOOD"
            confidence = pred_prob[i] * 100
            st.write(f"**Prediction:** {label}")
            st.write(f"**Confidence:** {confidence:.2f} percent")
            st.write("---")