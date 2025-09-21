
import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Paths
# -------------------------------
PIPELINE_PATH = "wine_quality_pipeline.joblib"
META_PATH = "model_metadata.joblib"

# -------------------------------
# Load pipeline and metadata
# -------------------------------
@st.cache_resource
def load_pipeline_and_meta():
    pipeline_dict = joblib.load(PIPELINE_PATH)
    pipeline = pipeline_dict['calibrator']  # calibrated classifier
    preprocessor = pipeline_dict['preprocessor']
    metadata = joblib.load(META_PATH)       # metadata separately
    feature_cols = metadata['feature_cols']
    return pipeline, preprocessor, feature_cols

pipeline, preprocessor, feature_cols = load_pipeline_and_meta()

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

    col1, col2, col3 = st.columns(3)

    # Left column
    with col1:
        data['fixed_acidity'] = st.number_input("Fixed Acidity", value=7.0, format="%.2f")
        data['volatile_acidity'] = st.number_input("Volatile Acidity", value=0.300, format="%.3f")
        data['citric_acid'] = st.number_input("Citric Acid", value=0.3, format="%.2f")
        data['residual_sugar'] = st.number_input("Residual Sugar", value=2.5, format="%.2f")

    # Middle column
    with col2:
        data['chlorides'] = st.number_input("Chlorides", value=0.080, format="%.3f")
        data['free_sulfur_dioxide'] = st.number_input("Free Sulfur Dioxide", value=15.0, format="%.2f")
        data['total_sulfur_dioxide'] = st.number_input("Total Sulfur Dioxide", value=46.0, format="%.2f")
        data['density'] = st.number_input("Density", value=0.99600, format="%.5f")

    # Right column
    with col3:
        data['ph'] = st.number_input("pH", value=3.3, format="%.2f")
        data['sulphates'] = st.number_input("Sulphates", value=0.65, format="%.2f")
        data['alcohol'] = st.number_input("Alcohol", value=10.0, format="%.2f")
        data['acidity_ratio'] = st.number_input("Acidity Ratio", value=7.0/0.3, format="%.3f")
        data['sulfur_ratio'] = st.number_input("Sulfur Ratio", value=15.0/46.0, format="%.3f")

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

        # Ensure all required feature columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        # Reorder to match training
        input_df = df[feature_cols]

        st.subheader("Uploaded CSV Preview")
        st.dataframe(df.head())  # show original CSV preview only
        return input_df
    return None

# Choose input method
input_df = single_sample_input() if input_method == "Single Sample" else batch_csv_input()

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Quality"):
    if input_df is not None and not input_df.empty:
        # Transform using preprocessor
        Xt = preprocessor.transform(input_df)
        preds = pipeline.predict(Xt)
        probs = pipeline.predict_proba(Xt)

        st.markdown("### Prediction Results:")
        for i, pred in enumerate(preds):
            label = "Good 🍷" if pred == 1 else "Not Good ❌"
            confidence = probs[i][pred] * 100
            st.write(f"**Prediction:** {label}")
            st.write(f"**Confidence:** {confidence:.2f} percent")
            if len(preds) > 1:
                st.write("---")  # separator for batch inputs
