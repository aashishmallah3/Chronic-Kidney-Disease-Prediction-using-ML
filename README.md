# Chronic Kidney Disease Prediction using ML

A machine learning web application that predicts Chronic Kidney Disease (CKD) from patient clinical data, comparing predictions from two classification models — **Random Forest** and **AdaBoost** — side by side through an interactive **Streamlit** interface.

---

## 🩺 Overview

Chronic Kidney Disease is a progressive condition that, if detected early, can be managed effectively. This project uses supervised machine learning to classify patients as **CKD** or **Not CKD** based on 24 clinical attributes (blood pressure, blood glucose, hemoglobin, serum creatinine, etc.).

The app allows a user to upload a CSV of patient records (without the diagnosis label) and instantly see predictions and confidence scores from both trained models.

---

## ✨ Features

- Trains and compares two classifiers: **Random Forest** and **AdaBoost**
- Interactive Streamlit UI for uploading patient data and viewing results
- Side-by-side prediction comparison with color-coded highlighting (CKD / Not CKD)
- Model confidence scores displayed per prediction batch
- Downloadable CSV of combined prediction results

---

## 📁 Project Structure

```
├── model.py                 # Trains RF & AdaBoost models, saves .pkl files and test set
├── app.py                   # Streamlit web app for uploading data & viewing predictions
├── ckd.csv                  # Full source dataset (400 patient records)
├── test_ckd.csv             # Held-out sample (25 records, no label) for testing the app
├── ckd_rf_model.pkl         # Trained Random Forest model
├── ckd_ab_model.pkl         # Trained AdaBoost model
└── README.md
```

---

## 📊 Dataset

The dataset (`ckd.csv`) contains 400 patient records with 24 clinical features plus a `classification` target column (`ckd` / `notckd`), including:

- **Vitals & physical:** age, blood pressure (bp)
- **Urine tests:** specific gravity (sg), albumin (al), sugar (su), red blood cells (rbc), pus cells (pc), pus cell clumps (pcc), bacteria (ba)
- **Blood tests:** blood glucose random (bgr), blood urea (bu), serum creatinine (sc), sodium (sod), potassium (pot), hemoglobin (hemo), packed cell volume (pcv), white blood cell count (wc), red blood cell count (rc)
- **Medical history:** hypertension (htn), diabetes mellitus (dm), coronary artery disease (cad), appetite, pedal edema (pe), anemia (ane)

---

## ⚙️ How It Works

### 1. Model Training (`model.py`)
- Loads `ckd.csv` and drops rows with missing values
- Cleans and encodes the target (`ckd` → 1, `notckd` → 0)
- Splits off 25 random rows as a held-out test set, saved (label-free) as `test_ckd.csv`
- Label-encodes categorical features
- Trains a `RandomForestClassifier` and an `AdaBoostClassifier` (100 estimators each)
- Evaluates accuracy on the test split
- Saves both trained models as `.pkl` files using `joblib`

### 2. Prediction App (`app.py`)
- Loads the two saved models
- Accepts a CSV upload (same feature columns, no `classification` column)
- Encodes categorical columns and generates predictions from both models
- Displays per-patient predictions side by side with color highlighting
- Shows average prediction confidence for each model
- Lets the user download the combined results as a CSV

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
git clone <your-repo-url>
cd chronic-kidney-disease-prediction
pip install -r requirements.txt
```

**requirements.txt** (suggested)
```
streamlit
pandas
numpy
scikit-learn
joblib
```

### 1. Train the models
```bash
python model.py
```
This generates `ckd_rf_model.pkl`, `ckd_ab_model.pkl`, and `test_ckd.csv`.

### 2. Run the web app
```bash
streamlit run app.py
```
Open the local URL shown in your terminal, then upload `test_ckd.csv` (or any similarly formatted CSV) to get predictions.

---

## 🖥️ Usage

1. Launch the app with `streamlit run app.py`
2. Upload a patient data CSV **without** the `classification` column
3. Review the uploaded data preview
4. View model confidence levels for Random Forest and AdaBoost
5. Inspect side-by-side, color-coded predictions per patient
6. Download the combined results as a CSV

---

## ⚠️ Notes & Limitations

- Categorical encoding in `app.py` is fit fresh on each upload rather than reusing the encoder from training — for consistent results, uploaded data should use the same categorical value formats as the training set.
- Rows with missing values are dropped during training, which reduces the effective dataset size.
- This tool is built for **educational/demonstration purposes only** and is **not a substitute for professional medical diagnosis**.

---

## 🧰 Tech Stack

- **Python**
- **scikit-learn** — model training (Random Forest, AdaBoost)
- **pandas / numpy** — data handling
- **Streamlit** — web interface
- **joblib** — model serialization

---

## 📄 License

This project is intended for educational purposes.
