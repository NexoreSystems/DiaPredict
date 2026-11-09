"""Build encounter-level records from observed source data without filling gaps."""
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data_raw"
OUTPUT_FILE = BASE_DIR / "diabetes_master_longitudinal.csv"
CHUNK_SIZE = 500_000


def build_master_dataset() -> pd.DataFrame:
    """Return one row per observed patient encounter, preserving missing labs as NaN.

    Blood pressure and BMI come from the patient profile and are therefore
    patient-level covariates, not longitudinal measurements. No forward/backward
    fill is performed on any lab result.
    """
    patients = pd.read_csv(RAW_DIR / "patients.csv")
    patients["patient_id"] = patients["patient_id"].astype(str)

    encounter_parts = []
    lab_parts = []
    usecols = ["patient_id", "test_date", "test_name", "value"]
    for chunk in pd.read_csv(RAW_DIR / "lab_results.csv", usecols=usecols, chunksize=CHUNK_SIZE):
        chunk["patient_id"] = chunk["patient_id"].astype(str)
        chunk["test_date"] = pd.to_datetime(chunk["test_date"], errors="coerce")
        chunk = chunk.dropna(subset=["patient_id", "test_date"])
        encounter_parts.append(chunk[["patient_id", "test_date"]].drop_duplicates())
        lab_parts.append(chunk[chunk["test_name"].isin({"HbA1c", "glucose_fasting", "triglycerides"})])

    encounters = pd.concat(encounter_parts, ignore_index=True).drop_duplicates()
    labs = pd.concat(lab_parts, ignore_index=True)
    observed_labs = labs.pivot_table(
        index=["patient_id", "test_date"],
        columns="test_name",
        values="value",
        aggfunc="mean",
    ).reset_index()
    observed_labs.columns.name = None

    # Keep every actual lab encounter in chronological history, including visits
    # where one or more of the three model lab measurements were not taken.
    master = encounters.merge(observed_labs, on=["patient_id", "test_date"], how="left")
    master = master.merge(patients, on="patient_id", how="left", validate="many_to_one")
    master = master.sort_values(["patient_id", "test_date"], kind="stable").reset_index(drop=True)
    return master


def main() -> None:
    master = build_master_dataset()
    master.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved observed encounter data: {OUTPUT_FILE}")
    print(f"Rows: {len(master):,}; patients: {master['patient_id'].nunique():,}")
    for field in ("HbA1c", "glucose_fasting", "triglycerides"):
        coverage = master[field].notna().mean() * 100 if field in master else 0
        print(f"{field} observed on {coverage:.1f}% of encounter rows")
    print("No missing laboratory values were filled.")


if __name__ == "__main__":
    main()
