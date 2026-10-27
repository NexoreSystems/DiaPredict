import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score

DATA_DIR = r"C:\Users\Admin\OneDrive\Рабочий стол\Dowload_datasets"
INPUT_FILE = os.path.join(DATA_DIR, "diabetes_master_longitudinal.csv")

print("1/4 Loading dataset...")
df = pd.read_csv(INPUT_FILE)

# --- TIER 1: Diabetes Risk Model ---
print("\n2/4 Training Tier 1 Model (Diabetes Risk Prediction)...")

t1_features = ['age', 'glucose_fasting', 'HbA1c', 'bmi', 'systolic_bp', 'triglycerides']
for col in t1_features:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

df_tier1 = df.groupby('patient_id').last().reset_index()

X_t1 = df_tier1[t1_features]
y_t1 = df_tier1['dx_type2_diabetes']

X_train_t1, X_test_t1, y_train_t1, y_test_t1 = train_test_split(
    X_t1, y_t1, test_size=0.2, random_state=42, stratify=y_t1
)

clf_tier1 = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
clf_tier1.fit(X_train_t1, y_train_t1)

t1_preds = clf_tier1.predict_proba(X_test_t1)[:, 1]
t1_auc = roc_auc_score(y_test_t1, t1_preds)
print(f"--> Tier 1 ROC-AUC Score: {t1_auc:.4f}")

# --- TIER 2: HbA1c Forecast Model ---
print("\n3/4 Training Tier 2 Model (HbA1c Forecast)...")

df['next_HbA1c'] = df.groupby('patient_id')['HbA1c'].shift(-1)
df_tier2 = df.dropna(subset=['HbA1c', 'next_HbA1c', 'glucose_fasting'])

t2_features = ['HbA1c', 'glucose_fasting', 'bmi', 'age']
X_t2 = df_tier2[t2_features]
y_t2 = df_tier2['next_HbA1c']

X_train_t2, X_test_t2, y_train_t2, y_test_t2 = train_test_split(
    X_t2, y_t2, test_size=0.2, random_state=42
)

reg_tier2 = Ridge(alpha=1.0)
reg_tier2.fit(X_train_t2, y_train_t2)

t2_preds = reg_tier2.predict(X_test_t2)
t2_rmse = np.sqrt(mean_squared_error(y_test_t2, t2_preds))
t2_r2 = r2_score(y_test_t2, t2_preds)

print(f"--> Tier 2 RMSE (Prediction Error): {t2_rmse:.4f}")
print(f"--> Tier 2 R^2 Score: {t2_r2:.4f}")

# --- SAVE MODELS ---
print("\n4/4 Saving models to files...")
model_t1_path = os.path.join(DATA_DIR, "tier1_diabetes_risk.joblib")
model_t2_path = os.path.join(DATA_DIR, "tier2_hba1c_forecast.joblib")

joblib.dump(clf_tier1, model_t1_path)
joblib.dump(reg_tier2, model_t2_path)

print("\n==========================================")
print("SUCCESS: MODELS TRAINED AND SAVED!")
print("==========================================")