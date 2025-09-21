import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load model and metadata
# -------------------------------
@st.cache_resource
def load_model_and_metadata():
    model = joblib.load("wine_quality_pipeline.joblib")
    metadata = joblib.load("model_metadata.joblib")
    return model, metadata

model, metadata = load_model_and_metadata()
feature_names = metadata.get("feature_names")
label_mapping = metadata.get("label_mapping", {0: "Not Good ❌", 1: "Good 🍷"})

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Boutique Wine Predictor 🍷", layout="centered")

st.markdown("<h1 style='text-align: center; color: #8B0000;'>🍷 Boutique Wine Quality Predictor</h1>", unsafe_allow_html=True)
st.write("<p style='text-align: center;'>Enter wine features to predict premium quality.</p>", unsafe_allow_html=True)

# -------------------------------
# Input Section (Single / Batch)
# -------------------------------
input_method = st.radio("Input Method:", ["Single Sample", "Batch CSV Upload"], index=0, horizontal=True)

def single_sample_input():
    st.subheader("Wine Sample Features")
    col1, col2 = st.columns(2)

    data = {}
    # Left column
    with col1:
        data['fixed_acidity'] = st.number_input("Fixed Acidity", value=7.0)
        data['volatile_acidity'] = st.number_input("Volatile Acidity", value=0.3)
        data['citric_acid'] = st.number_input("Citric Acid", value=0.3)
        data['residual_sugar'] = st.number_input("Residual Sugar", value=2.5)
        data['chlorides'] = st.number_input("Chlorides", value=0.08)
        data['free_sulfur_dioxide'] = st.number_input("Free Sulfur Dioxide", value=15.0)

    # Right column
    with col2:
        data['total_sulfur_dioxide'] = st.number_input("Total Sulfur Dioxide", value=46.0)
        data['density'] = st.number_input("Density", value=0.996)
        data['pH'] = st.number_input("pH", value=3.3)
        data['sulphates'] = st.number_input("Sulphates", value=0.65)
        data['alcohol'] = st.number_input("Alcohol", value=10.0)
        # New engineered features
        data['acidity_ratio'] = st.number_input("Acidity Ratio", value=7.0/0.3)
        data['sulfur_ratio'] = st.number_input("Sulfur Ratio", value=15.0/46.0)

    return pd.DataFrame([data])

def batch_csv_input():
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Normalize headers
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        st.subheader("Uploaded CSV Preview")
        st.dataframe(df.head())
        return df
    return None

# Get input
if input_method == "Single Sample":
    input_df = single_sample_input()
else:
    input_df = batch_csv_input()

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Quality"):
    if input_df is not None and not input_df.empty:
        # Fill missing columns with 0
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0.0

        input_df = input_df[feature_names]

        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)

        results = []
        for i, pred in enumerate(predictions):
            label = label_mapping[pred]
            confidence = probabilities[i][pred] * 100
            results.append({"Prediction": label, "Confidence (%)": round(confidence, 2)})

        results_df = pd.DataFrame(results)

        # Display results directly below inputs
        st.markdown("### Prediction Results")
        def highlight_quality(row):
            color = "#c6f5c6" if "Good" in row["Prediction"] else "#f5c6c6"
            return [f"background-color: {color}"]*len(row)

        st.dataframe(pd.concat([input_df.reset_index(drop=True), results_df], axis=1).style.apply(highlight_quality, axis=1))

        # Optional: confidence progress bars
        st.subheader("Confidence Visual")
        for i, conf in enumerate(probabilities[:,1]*100):
            st.write(f"Sample {i+1}:")
            st.progress(int(conf))

# -------------------------------
# Feature Importance
# -------------------------------
if hasattr(model, "feature_importances_"):
    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=model.feature_importances_, y=feature_names, palette="mako", ax=ax)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)