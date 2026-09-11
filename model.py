# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load and prepare dataset
df = pd.read_csv("ckd.csv")
df = df.dropna()

# Fix target encoding
df["classification"] = df["classification"].str.strip().str.lower()
df["classification"] = df["classification"].map({"ckd": 1, "notckd": 0})

# Separate 25 rows for testing
test_data = df.sample(25, random_state=42)
train_data = df.drop(test_data.index)

# Drop 'classification' from test CSV (no ground truth given to app)
test_data_no_label = test_data.drop("classification", axis=1)
test_data_no_label.to_csv("test_ckd.csv", index=False)

# Split features and labels
X_train = train_data.drop("classification", axis=1)
y_train = train_data["classification"]

X_test = test_data.drop("classification", axis=1)
y_test = test_data["classification"]

# Encode categorical columns
cat_cols = X_train.select_dtypes(include=["object"]).columns
le = LabelEncoder()
for col in cat_cols:
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# Train AdaBoost Model
ab_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ab_model.fit(X_train, y_train)
ab_pred = ab_model.predict(X_test)
ab_acc = accuracy_score(y_test, ab_pred)

# Save both models and accuracy
joblib.dump(rf_model, "ckd_rf_model.pkl")
joblib.dump(ab_model, "ckd_ab_model.pkl")

print("✅ Models trained and saved:")
print(f"   - Random Forest Accuracy: {rf_acc*100:.2f}%")
print(f"   - AdaBoost Accuracy: {ab_acc*100:.2f}%")
print("✅ Test dataset saved as test_ckd.csv (without classification column)")
