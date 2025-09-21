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
# App title 
# -------------------------------
st.set_page_config(page_title="Boutique Wine Predictor 🍷", layout="wide")
st.markdown("<h1 style='text-align: center; color: #8B0000;'>🍷 Boutique Wine Quality Predictor</h1>", unsafe_allow_html=True)
st.write("""
Enter the chemical properties of a wine sample, or upload a CSV file with multiple samples,
to predict whether it meets premium quality standards.
""")

# -------------------------------
# Sidebar: Input Method
# -------------------------------
input_method = st.sidebar.radio("Input Method:", ["Single Sample", "Batch CSV Upload"])

# -------------------------------
# Input Sections
# -------------------------------
def single_sample_input():
    st.sidebar.subheader("Acidity")
    fixed_acidity = st.sidebar.number_input("Fixed Acidity", value=7.0)
    volatile_acidity = st.sidebar.number_input("Volatile Acidity", value=0.3)
    citric_acid = st.sidebar.number_input("Citric Acid", value=0.3)
    
    st.sidebar.subheader("Sugar & Sulfur")
    residual_sugar = st.sidebar.number_input("Residual Sugar", value=2.5)
    chlorides = st.sidebar.number_input("Chlorides", value=0.08)
    free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", value=15.0)
    total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", value=46.0)
    
    st.sidebar.subheader("Other Properties")
    density = st.sidebar.number_input("Density", value=0.996)
    pH = st.sidebar.number_input("pH", value=3.3)
    sulphates = st.sidebar.number_input("Sulphates", value=0.65)
    alcohol = st.sidebar.number_input("Alcohol", value=10.0)

    data = {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }
    return pd.DataFrame([data])

def batch_csv_input():
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
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
        input_df = input_df[feature_names]  # ensure correct column order
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)

        results = []
        for i, pred in enumerate(predictions):
            label = label_mapping[pred]
            confidence = probabilities[i][pred] * 100
            results.append({
                "Prediction": label,
                "Confidence (%)": round(confidence, 2)
            })

        results_df = pd.DataFrame(results)
        st.success("✅ Predictions Complete")
        
        # Highlight Good/Not Good rows
        def highlight_quality(row):
            color = "#c6f5c6" if "Good" in row["Prediction"] else "#f5c6c6"
            return [f"background-color: {color}"]*len(row)
        
        st.dataframe(pd.concat([input_df.reset_index(drop=True), results_df], axis=1).style.apply(highlight_quality, axis=1))
        
        # Display confidence as progress bars
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