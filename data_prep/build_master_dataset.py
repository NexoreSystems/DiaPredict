import os
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\Admin\OneDrive\Рабочий стол\Dowload_datasets"

print("1/5 Loading source files...")
df_patients = pd.read_csv(os.path.join(DATA_DIR, "patients.csv"))
df_labs = pd.read_csv(os.path.join(DATA_DIR, "lab_results.csv"))

print(f"Loaded: {len(df_patients)} patients, {len(df_labs)} lab results.")

# 2. Filter patients with >= 180 days history
print("\n2/5 Filtering longitudinal data (>= 180 days span)...")
df_labs['test_date'] = pd.to_datetime(df_labs['test_date'])

patient_history = df_labs.groupby('patient_id')['test_date'].agg(['min', 'max', 'count'])
patient_history['days_span'] = (patient_history['max'] - patient_history['min']).dt.days

valid_patients = patient_history[(patient_history['days_span'] >= 180) & (patient_history['count'] >= 3)].index
print(f"Found {len(valid_patients)} patients with >= 180 days history.")

df_labs_filtered = df_labs[df_labs['patient_id'].isin(valid_patients)].copy()

# 3. Pivot laboratory tests
print("\n3/5 Pivoting lab tests into feature columns...")
df_labs_pivot = df_labs_filtered.pivot_table(
    index=['patient_id', 'test_date'],
    columns='test_name',
    values='value',
    aggfunc='mean'
).reset_index()

df_labs_pivot.columns.name = None

# 4. Merge with demographics
print("\n4/5 Merging with patient demographics...")
master_df = df_labs_pivot.merge(df_patients, on='patient_id', how='left')
master_df = master_df.sort_values(by=['patient_id', 'test_date']).reset_index(drop=True)

# 5. Forward fill missing lab values per patient (handling NaNs in time-series)
print("\n5/5 Propagating missing lab values (Forward Fill)...")
lab_cols = ['HbA1c', 'glucose_fasting', 'BMI', 'systolic_bp', 'diastolic_bp', 'eGFR', 'creatinine', 'HDL', 'LDL']
lab_cols_existing = [c for c in lab_cols if c in master_df.columns]

# Forward fill within each patient's timeline
master_df[lab_cols_existing] = master_df.groupby('patient_id')[lab_cols_existing].ffill().bfill()

# Save final dataset
output_file = os.path.join(DATA_DIR, "diabetes_master_longitudinal.csv")
master_df.to_csv(output_file, index=False)

print("\n==========================================")
print(f"SUCCESSFULLY CREATED: {output_file}")
print(f"Total Rows: {len(master_df)}")
print(f"Unique Patients: {master_df['patient_id'].nunique()}")
print("\nFirst 3 rows preview:")
print(master_df[['patient_id', 'test_date', 'HbA1c', 'glucose_fasting', 'age', 'dx_type2_diabetes']].head(3))
print("==========================================")