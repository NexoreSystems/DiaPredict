"""Train Tier 1 to forecast a first recorded E11 diagnosis within 365 days.

The outcome is a future coded diagnosis, not biological disease onset. The script
uses raw dated diagnoses and laboratory records, excludes patients with a prior
E11 record at the index date, and requires a full year of observed follow-up for
negative examples. Patient-level static diagnosis flags are never used as labels.

Run from the project directory:
    python train_tier1_forecast.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parent
RAW = BASE / "data_raw"
HORIZON_DAYS = 365
SEED = 42
FEATURES = ["glucose_fasting", "HbA1c"]


def say(message: str) -> None:
    print(message, flush=True)


def read_event_dates() -> tuple[pd.Series, pd.Series]:
    """Return each patient's first E11 date and last dated diagnosis encounter."""
    first_e11: dict[str, pd.Timestamp] = {}
    last_diagnosis: dict[str, pd.Timestamp] = {}
    for chunk in pd.read_csv(RAW / "diagnoses.csv", chunksize=100_000, dtype=str):
        dates = pd.to_datetime(chunk["visit_date"], errors="coerce")
        for patient_id, dt in pd.DataFrame({"patient_id": chunk["patient_id"], "date": dates}).groupby("patient_id")["date"].max().items():
            if pd.notna(dt) and (patient_id not in last_diagnosis or dt > last_diagnosis[patient_id]):
                last_diagnosis[patient_id] = dt

        primary = chunk["primary_icd10"].fillna("").str.strip().str.upper().str.startswith("E11")
        secondary = chunk["secondary_icd10s"].fillna("").str.upper().str.split("|").map(
            lambda codes: any(code.strip().startswith("E11") for code in codes)
        )
        mask = primary | secondary
        if mask.any():
            event_rows = pd.DataFrame({"patient_id": chunk.loc[mask, "patient_id"],
                                       "date": dates.loc[mask]}).dropna(subset=["date"])
            for patient_id, dt in event_rows.groupby("patient_id")["date"].min().items():
                if patient_id not in first_e11 or dt < first_e11[patient_id]:
                    first_e11[patient_id] = dt
    return pd.Series(first_e11, name="first_e11"), pd.Series(last_diagnosis, name="last_diagnosis")


def read_laboratory_history() -> tuple[pd.DataFrame, pd.Series]:
    """Build past-only lab features at dates and return last observed lab date."""
    needed = {"glucose_fasting", "HbA1c"}
    kept = []
    last_lab: dict[str, pd.Timestamp] = {}
    for chunk in pd.read_csv(RAW / "lab_results.csv", usecols=["patient_id", "test_date", "test_name", "value"],
                             chunksize=250_000):
        chunk["test_date"] = pd.to_datetime(chunk["test_date"], errors="coerce")
        for patient_id, dt in chunk.groupby("patient_id")["test_date"].max().items():
            if pd.notna(dt) and (patient_id not in last_lab or dt > last_lab[patient_id]):
                last_lab[patient_id] = dt
        chunk = chunk[chunk["test_name"].isin(needed)].copy()
        if not chunk.empty:
            chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
            kept.append(chunk.dropna(subset=["patient_id", "test_date", "value"]))

    labs = pd.concat(kept, ignore_index=True)
    wide = (labs.pivot_table(index=["patient_id", "test_date"], columns="test_name",
                             values="value", aggfunc="mean")
            .reset_index())
    wide.columns.name = None
    for col in needed:
        if col not in wide:
            wide[col] = np.nan
    wide = wide.sort_values(["patient_id", "test_date"], kind="stable")
    # Only dates with an actually measured fasting glucose can be prediction landmarks.
    glucose_observed = wide["glucose_fasting"].notna()
    # Forward fill only: never copy a measurement from a later date into the past.
    wide[list(needed)] = wide.groupby("patient_id", sort=False)[list(needed)].ffill()
    index = wide.loc[glucose_observed].copy()
    return index, pd.Series(last_lab, name="last_lab")


def make_cohort() -> pd.DataFrame:
    say("Reading dated diagnosis events...")
    first_e11, last_diagnosis = read_event_dates()
    say("Reading lab history and selecting eligible glucose visits...")
    index, last_lab = read_laboratory_history()
    index["patient_id"] = index["patient_id"].astype(str)
    index = index.join(first_e11, on="patient_id").join(last_diagnosis, on="patient_id").join(last_lab, on="patient_id")
    index["followup_end"] = index[["last_diagnosis", "last_lab"]].max(axis=1)
    index["horizon_end"] = index["test_date"] + pd.to_timedelta(HORIZON_DAYS, unit="D")

    prior = index["first_e11"].notna() & (index["first_e11"] <= index["test_date"])
    incident = index["first_e11"].notna() & (index["first_e11"] > index["test_date"]) & (index["first_e11"] <= index["horizon_end"])
    observed_control = ((index["first_e11"].isna() | (index["first_e11"] > index["horizon_end"])) &
                        (index["followup_end"] >= index["horizon_end"]))
    cohort = index.loc[~prior & (incident | observed_control)].copy()
    cohort["target"] = incident.loc[cohort.index].astype(int)
    # Forecasts use only observed glucose and HbA1c values available by index date.
    # Do not impute missing patient measurements.
    # Keep one reproducible eligible visit per patient to avoid overweighting
    # people with frequent testing and to match one current-visit prediction in app.py.
    cohort = cohort.dropna(subset=["test_date", "target", *FEATURES])
    cohort = (cohort.sample(frac=1.0, random_state=SEED)
              .drop_duplicates("patient_id", keep="first")
              .rename(columns={"test_date": "index_date"}))
    say(f"Eligible patients: {len(cohort):,}; future E11 within 365 days: {int(cohort['target'].sum()):,}; "
        f"observed controls: {int((cohort['target'] == 0).sum()):,}; prior E11 excluded: {int(prior.sum()):,}")
    if cohort["target"].nunique() != 2:
        raise RuntimeError("Eligible cohort does not contain both outcome classes.")
    return cohort.sort_values("index_date", kind="stable").reset_index(drop=True)


def choose_threshold(y: np.ndarray, p: np.ndarray, target_recall: float = 0.80) -> tuple[float, float, float]:
    thresholds = np.unique(p)
    rows = []
    for threshold in thresholds:
        pred = p >= threshold
        rec = recall_score(y, pred, zero_division=0)
        prec = precision_score(y, pred, zero_division=0)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append((float(threshold), float(rec), float(f1), float(prec)))
    valid = [row for row in rows if row[1] >= target_recall]
    screen = max(valid, key=lambda row: row[0])[0] if valid else min(rows, key=lambda row: row[0])[0]
    best_f1 = max(rows, key=lambda row: row[2])[0]
    # A separate high-risk band requires at least 50% precision on validation.
    # If the cohort cannot support that, leave the high band empty (threshold=1).
    high_candidates = [row for row in rows if row[3] >= 0.50]
    high = min(high_candidates, key=lambda row: row[0])[0] if high_candidates else 1.0
    return screen, best_f1, max(float(screen), float(high))


def main() -> None:
    cohort = make_cohort()
    # Patient appears once, so date-ordered partitions also keep all records per patient together.
    train_end = cohort["index_date"].quantile(0.70)
    val_end = cohort["index_date"].quantile(0.85)
    train = cohort[cohort["index_date"] <= train_end]
    valid = cohort[(cohort["index_date"] > train_end) & (cohort["index_date"] <= val_end)]
    test = cohort[cohort["index_date"] > val_end]
    if min(train["target"].sum(), valid["target"].sum(), test["target"].sum()) < 20:
        raise RuntimeError("A chronological split has too few positive cases for a credible evaluation.")
    say(f"Chronological patient split: train={len(train):,}, validation={len(valid):,}, test={len(test):,}; "
        f"date cutoffs={train_end.date()} / {val_end.date()}")

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=SEED))
    model.fit(train[FEATURES], train["target"])
    p_val_raw = np.clip(model.predict_proba(valid[FEATURES])[:, 1], 1e-6, 1 - 1e-6)
    # Separate Platt calibration on validation data; the final test remains untouched.
    calibrator = LogisticRegression(max_iter=1000, random_state=SEED)
    calibrator.fit(np.log(p_val_raw / (1 - p_val_raw)).reshape(-1, 1), valid["target"])
    p_val = calibrator.predict_proba(np.log(p_val_raw / (1 - p_val_raw)).reshape(-1, 1))[:, 1]
    screen_threshold, f1_threshold, high_threshold = choose_threshold(valid["target"].to_numpy(), p_val)

    p_test_raw = np.clip(model.predict_proba(test[FEATURES])[:, 1], 1e-6, 1 - 1e-6)
    p_test = calibrator.predict_proba(np.log(p_test_raw / (1 - p_test_raw)).reshape(-1, 1))[:, 1]
    y_test = test["target"].to_numpy()
    pred = p_test >= screen_threshold
    pred_high = p_test >= high_threshold
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    metrics = {
        "test_period_start": str(test["index_date"].min().date()),
        "test_period_end": str(test["index_date"].max().date()),
        "n_test": int(len(test)), "n_positive_test": int(y_test.sum()),
        "prevalence_test": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, p_test)),
        "pr_auc_average_precision": float(average_precision_score(y_test, p_test)),
        "brier_score": float(brier_score_loss(y_test, p_test)),
        "screening_threshold": float(screen_threshold), "high_threshold": float(high_threshold),
        "recall_at_screening_threshold": float(recall_score(y_test, pred, zero_division=0)),
        "precision_at_screening_threshold": float(precision_score(y_test, pred, zero_division=0)),
        "recall_at_high_threshold": float(recall_score(y_test, pred_high, zero_division=0)),
        "precision_at_high_threshold": float(precision_score(y_test, pred_high, zero_division=0)),
        "f1_threshold_validation_only": float(f1_threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    bundle = {
        "model": model, "calibrator": calibrator, "features": FEATURES,
        "name": "Logistic regression, first recorded E11 within 365 days",
        "horizon_days": HORIZON_DAYS,
        "target_definition": "First recorded ICD-10 E11 diagnosis within 365 days after an eligible visit with a fasting-glucose result, among patients without a prior recorded E11 diagnosis at that visit.",
        "thresholds": {"screening": float(screen_threshold), "high": float(high_threshold),
                       "validation_target_recall": 0.80},
        "metrics": metrics,
        "data_note": "Predictors are observed fasting glucose and HbA1c available by the index visit. No missing patient laboratory values are imputed. Patients without both observed predictors are not scored. Outcome is a recorded diagnosis code. This is not external clinical validation or a diagnosis of biological disease onset.",
    }
    experiment_dir = BASE / "results_v2" / "experiments"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    model_path = experiment_dir / "tier1_incident_1y.joblib"
    joblib.dump(bundle, model_path)
    report_path = experiment_dir / "tier1_incident_1y_metrics.json"
    report_path.write_text(json.dumps({"cohort_n": int(len(cohort)),
                                      "cohort_events": int(cohort["target"].sum()),
                                      "train_n": int(len(train)), "train_events": int(train["target"].sum()),
                                      "validation_n": int(len(valid)), "validation_events": int(valid["target"].sum()),
                                      "features": FEATURES,
                                      "data_note": bundle["data_note"],
                                      "test": metrics, "target_definition": bundle["target_definition"]},
                                     indent=2, ensure_ascii=False), encoding="utf-8")
    say(f"Saved model: {model_path.name}")
    say(f"Saved evaluation: {report_path.relative_to(BASE)}")
    say(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
