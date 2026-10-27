import os
import pandas as pd

# Точний шлях до вашої папки через OneDrive
DATA_DIR = r"C:\Users\Admin\OneDrive\Рабочий стол\Dowload_datasets"

print(f"Зчитування даних з: {DATA_DIR}\n")

try:
    patients = pd.read_csv(os.path.join(DATA_DIR, "patients.csv"), nrows=2)
    lab_results = pd.read_csv(os.path.join(DATA_DIR, "lab_results.csv"), nrows=2)
    diagnoses = pd.read_csv(os.path.join(DATA_DIR, "diagnoses.csv"), nrows=2)

    print("=== PATIENTS COLUMNS ===")
    print(patients.columns.tolist())

    print("\n=== LAB RESULTS COLUMNS ===")
    print(lab_results.columns.tolist())

    print("\n=== DIAGNOSES COLUMNS ===")
    print(diagnoses.columns.tolist())

except Exception as e:
    print(f"Помилка: {e}")