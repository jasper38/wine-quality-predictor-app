# 🍷 Boutique Wine Quality Predictor

## Overview
This Streamlit web application allows users to predict whether a red wine sample meets **premium quality standards**. The model uses chemical attributes of wine and classifies each sample as **Good Quality** or **Not Good Quality**, along with a confidence score to support decision-making.  

The app supports:  
- **Single sample input** via sidebar  
- **Batch prediction** via CSV upload  
- **Feature importance visualization**  

---

## Live Demo
[Your Streamlit Cloud URL here]  
*(Replace with your deployed app link)*

---

## Files in this Repository

| File | Description |
|------|-------------|
| `wine_quality_app_polished.py` | Main Streamlit application |
| `wine_quality_pipeline.joblib` | Trained RandomForest pipeline |
| `model_metadata.joblib` | Metadata containing feature names and label mapping |
| `README.md` | This file with instructions |

---

## Requirements

Install the required packages using pip:

```bash
pip install streamlit pandas joblib matplotlib seaborn
```

---

## How to Run Locally

1. Clone this repository:  
```bash
git clone https://github.com/yourusername/wine-quality-app.git
cd wine-quality-app
```

2. Ensure `wine_quality_pipeline.joblib` and `model_metadata.joblib` are in the same directory as the app.  

3. Run the Streamlit app:  
```bash
streamlit run wine_quality_app_polished.py
```

4. Open the provided localhost URL in your browser to access the app.  

---

## Usage Instructions

### Single Sample Prediction
1. Select **Single Sample** in the sidebar.  
2. Input the chemical properties of the wine.  
3. Click **Predict Quality**.  
4. View prediction result, confidence, and optionally see feature importance.  

### Batch Prediction
1. Select **Batch CSV Upload** in the sidebar.  
2. Upload a CSV file containing one or more wine samples (columns must match the trained feature names).  
3. Click **Predict Quality**.  
4. Results will show for all samples, including prediction and confidence.  

---

## Notes
- The model defines **Good Quality** as `quality >= 7` and **Not Good Quality** otherwise.  
- Confidence scores are displayed as percentages.  
- Feature importance helps understand which chemical attributes influence the model the most.  

---

## Deployment on Streamlit Cloud
1. Push this repository to GitHub.  
2. Go to [Streamlit Cloud](https://share.streamlit.io/) and connect your GitHub repo.  
3. Set the main file as `wine_quality_app_polished.py`.  
4. Deploy and share the public URL for submission.

