# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load Models
rf_model = joblib.load("ckd_rf_model.pkl")
ab_model = joblib.load("ckd_ab_model.pkl")

# Page Config
st.set_page_config(page_title="CKD Prediction App", layout="wide")

st.markdown("<h1 style='text-align:center; color:#2C3E50;'> Chronic Kidney Disease Prediction</h1>", unsafe_allow_html=True)
st.write("---")

st.write("Upload a CSV file **(without classification column)** to compare predictions from Random Forest and AdaBoost models.")

# File Upload
uploaded_file = st.file_uploader(" Upload Patient Data (CSV)", type=["csv"])

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    st.subheader(" Uploaded Data Preview")
    st.dataframe(df_input.head())

    # Encode categorical features
    df_processed = df_input.copy()
    cat_cols = df_processed.select_dtypes(include=["object"]).columns
    le = LabelEncoder()
    for col in cat_cols:
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))

    # Make Predictions
    rf_preds = rf_model.predict(df_processed)
    ab_preds = ab_model.predict(df_processed)

    rf_labels = ["CKD" if p == 1 else "Not CKD" for p in rf_preds]
    ab_labels = ["CKD" if p == 1 else "Not CKD" for p in ab_preds]

    # Create a combined results DataFrame
    df_results = df_input.copy()
    df_results["Random Forest Prediction"] = rf_labels
    df_results["AdaBoost Prediction"] = ab_labels

    # Display Confidence (accuracy)
    st.write(" Model Confidence Levels")
    col1, col2 = st.columns(2)

    # Confidence = model's training/test accuracy (loaded from training script)
    # Here, we estimate using predict_proba mean confidence
    rf_conf = np.mean(np.max(rf_model.predict_proba(df_processed), axis=1))
    ab_conf = np.mean(np.max(ab_model.predict_proba(df_processed), axis=1))

    with col1:
        st.metric(label="Random Forest Confidence", value=f"{rf_conf*100:.2f}%")
    with col2:
        st.metric(label="AdaBoost Confidence", value=f"{ab_conf*100:.2f}%")

    # Side-by-side Result Display
    st.write(" Patient-wise Predictions (Side by Side)")

    def highlight_predictions(row):
        rf_color = "#ff9999" if row["Random Forest Prediction"] == "CKD" else "#b3ffb3"
        ab_color = "#ff9999" if row["AdaBoost Prediction"] == "CKD" else "#b3ffb3"
        # Highlight both prediction columns individually
        styles = []
        for col in row.index:
            if col == "Random Forest Prediction":
                styles.append(f"background-color: {rf_color}; color: black;")
            elif col == "AdaBoost Prediction":
                styles.append(f"background-color: {ab_color}; color: black;")
            else:
                styles.append("")
        return styles

    st.dataframe(df_results.style.apply(highlight_predictions, axis=1))

    # Download Option
    csv = df_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=" Download Combined Predictions as CSV",
        data=csv,
        file_name="ckd_predictions_comparison.csv",
        mime="text/csv",
    )


