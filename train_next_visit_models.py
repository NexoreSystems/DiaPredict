"""Train next-visit forecasts on complete, observed patient records only.

Tier 1 predicts whether a first recorded type 2 diabetes diagnosis appears by the estimated
next visit date. Tier 2 predicts HbA1c at that estimated date. Every landmark requires an observed
glucose, HbA1c, triglyceride, BMI, systolic blood pressure, and age value.
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, mean_absolute_error,
                             mean_squared_error, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
RAW = BASE / "data_raw"
SEED = 42
HORIZON_FEATURE = "days_to_next_visit"
REQUIRED = ["age", "glucose_fasting", "HbA1c", "bmi", "systolic_bp", "triglycerides"]
FEATURES = REQUIRED + [HORIZON_FEATURE]
MIN_PRIOR_INTERVALS = 2


def say(message: str) -> None:
    print(message, flush=True)


def read_visit_dates() -> dict[str, list[pd.Timestamp]]:
    """Union dated diagnosis encounters and lab dates into a visit timeline."""
    cache = BASE / ".diapredict_build" / "observed_visit_dates.pkl"
    sources = [RAW / "diagnoses.csv", RAW / "lab_results.csv"]
    stamp = tuple(path.stat().st_mtime_ns for path in sources)
    if cache.exists():
        try:
            with cache.open("rb") as f:
                cached_stamp, cached_dates = pickle.load(f)
            if cached_stamp == stamp:
                return cached_dates
        except Exception:
            pass
    pieces = []
    for path, id_col, date_col in (
        (RAW / "diagnoses.csv", "patient_id", "visit_date"),
        (RAW / "lab_results.csv", "patient_id", "test_date"),
    ):
        for chunk in pd.read_csv(path, usecols=[id_col, date_col], chunksize=400_000, dtype={id_col: str}):
            chunk[date_col] = pd.to_datetime(chunk[date_col], errors="coerce")
            chunk = chunk.dropna(subset=[id_col, date_col])
            chunk = chunk.rename(columns={id_col: "patient_id", date_col: "visit_date"})
            pieces.append(chunk.drop_duplicates())
    dates = pd.concat(pieces, ignore_index=True).drop_duplicates(["patient_id", "visit_date"])
    result = {str(pid): list(pd.to_datetime(group["visit_date"]).sort_values().drop_duplicates())
              for pid, group in dates.groupby("patient_id", sort=False)}
    cache.parent.mkdir(exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump((stamp, result), f, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def read_complete_visits() -> pd.DataFrame:
    """Pivot required labs at their actual dates and merge measured profile fields."""
    kept = []
    needed_labs = {"glucose_fasting", "HbA1c", "triglycerides"}
    for chunk in pd.read_csv(RAW / "lab_results.csv",
                             usecols=["patient_id", "test_date", "test_name", "value"],
                             chunksize=300_000, dtype={"patient_id": str}):
        chunk = chunk[chunk["test_name"].isin(needed_labs)].copy()
        if chunk.empty:
            continue
        chunk["test_date"] = pd.to_datetime(chunk["test_date"], errors="coerce")
        chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
        kept.append(chunk.dropna(subset=["patient_id", "test_date", "value"]))
    labs = pd.concat(kept, ignore_index=True)
    wide = labs.pivot_table(index=["patient_id", "test_date"], columns="test_name",
                            values="value", aggfunc="mean").reset_index()
    wide.columns.name = None

    patients = pd.read_csv(RAW / "patients.csv", dtype={"patient_id": str})
    patients = patients.rename(columns={"bmi": "bmi", "dx_type2_diabetes": "known_t2d"})
    profile_fields = ["patient_id", "age", "bmi", "systolic_bp", "known_t2d"]
    wide = wide.merge(patients[profile_fields], on="patient_id", how="left", validate="many_to_one")
    for column in REQUIRED:
        if column not in wide:
            wide[column] = np.nan
        wide[column] = pd.to_numeric(wide[column], errors="coerce")
    complete = wide.dropna(subset=REQUIRED).copy()
    # Treat impossible/non-finite source values as missing rather than silently using them.
    complete = complete[np.isfinite(complete[REQUIRED]).all(axis=1)]
    return complete.sort_values(["patient_id", "test_date"], kind="stable").reset_index(drop=True)


def read_diagnosis_outcomes() -> tuple[dict[str, pd.Timestamp], dict[str, pd.Timestamp]]:
    first_e11: dict[str, pd.Timestamp] = {}
    last_seen: dict[str, pd.Timestamp] = {}
    for chunk in pd.read_csv(RAW / "diagnoses.csv", chunksize=200_000, dtype=str):
        dates = pd.to_datetime(chunk["visit_date"], errors="coerce")
        rows = pd.DataFrame({"patient_id": chunk["patient_id"], "date": dates}).dropna()
        if not rows.empty:
            for pid, dt in rows.groupby("patient_id")["date"].max().items():
                last_seen[pid] = max(dt, last_seen.get(pid, dt))
        primary = chunk["primary_icd10"].fillna("").str.strip().str.upper().str.startswith("E11")
        secondary = chunk["secondary_icd10s"].fillna("").str.upper().str.split("|").map(
            lambda codes: any(code.strip().startswith("E11") for code in codes))
        event_rows = pd.DataFrame({"patient_id": chunk.loc[primary | secondary, "patient_id"],
                                   "date": dates.loc[primary | secondary]}).dropna()
        for pid, dt in event_rows.groupby("patient_id")["date"].min().items():
            first_e11[pid] = min(dt, first_e11.get(pid, dt))
    return first_e11, last_seen


def add_next_visit_horizon(frame: pd.DataFrame, visit_dates: dict[str, list[pd.Timestamp]]) -> pd.DataFrame:
    rows = []
    for row in frame.itertuples(index=False):
        dates = visit_dates.get(str(row.patient_id), [])
        prior = [d for d in dates if d <= row.test_date]
        gaps = np.diff(np.array(prior, dtype="datetime64[ns]")).astype("timedelta64[D]").astype(int)
        gaps = gaps[gaps > 0]
        if len(gaps) < MIN_PRIOR_INTERVALS:
            continue
        days = max(1, int(round(float(np.median(gaps)))))
        item = row._asdict()
        item[HORIZON_FEATURE] = days
        item["expected_next_date"] = row.test_date + pd.Timedelta(days=days)
        rows.append(item)
    return pd.DataFrame(rows)


def choose_threshold(y: np.ndarray, p: np.ndarray, target_recall: float = 0.80) -> float:
    thresholds = np.unique(p)
    candidates = [float(t) for t in thresholds
                  if recall_score(y, p >= t, zero_division=0) >= target_recall]
    return max(candidates) if candidates else float(min(thresholds))


def bootstrap_ci(y_true, prediction, statistic, *, require_both_classes=False,
                 repeats: int = 2000) -> dict:
    """Patient-level percentile 95% CI on the untouched temporal test set."""
    y_true = np.asarray(y_true)
    prediction = np.asarray(prediction)
    rng = np.random.default_rng(SEED)
    estimates = []
    for _ in range(repeats):
        index = rng.integers(0, len(y_true), len(y_true))
        sample_y = y_true[index]
        if require_both_classes and np.unique(sample_y).size < 2:
            continue
        value = float(statistic(sample_y, prediction[index]))
        if np.isfinite(value):
            estimates.append(value)
    if not estimates:
        return {"lower_95": None, "upper_95": None, "valid_replicates": 0}
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"lower_95": float(low), "upper_95": float(high),
            "valid_replicates": len(estimates)}


def add_classification_intervals(metrics: dict, y: np.ndarray, p: np.ndarray,
                                 threshold: float) -> None:
    predicted = (p >= threshold).astype(int)
    measures = {
        "roc_auc": (roc_auc_score, True),
        "pr_auc_average_precision": (average_precision_score, True),
        "brier_score": (brier_score_loss, False),
        "recall": (lambda actual, guess: recall_score(actual, guess, zero_division=0), False),
        "precision": (lambda actual, guess: precision_score(actual, guess, zero_division=0), False),
    }
    metrics["confidence_intervals_95"] = {
        name: bootstrap_ci(y, p if name in ("roc_auc", "pr_auc_average_precision", "brier_score") else predicted,
                           fn, require_both_classes=both)
        for name, (fn, both) in measures.items()
    }


def format_ci(metrics: dict, key: str, digits: int = 3) -> str:
    interval = metrics.get("confidence_intervals_95", {}).get(key, {})
    if interval.get("lower_95") is None:
        return "не розраховано"
    return f"{interval['lower_95']:.{digits}f}–{interval['upper_95']:.{digits}f}"


def write_human_report(report: dict) -> None:
    tier1, tier2 = report["tier1"]["test"], report["tier2"]["test"]
    app_cohort = report["application_eligible_cohort"]
    lines = [
        "# DiaPredict — звіт оцінювання моделей",
        "",
        "Оцінки наведені на часовій тестовій вибірці, яка не використовувалася для навчання чи вибору порогу. "
        "95% інтервали отримані бутстрепом на рівні пацієнтів.",
        "",
        "## Когорта прогнозу в застосунку",
        "",
        f"- Повні фактично виміряні візити: {report['complete_visits']:,}.",
        f"- Пацієнти з щонайменше трьома повними візитами: {app_cohort['total']:,}.",
        f"- Скринінг без ЦД2: {app_cohort['screening_no_t2d']:,}.",
        f"- Оцінка ускладнень при ЦД2: {app_cohort['confirmed_t2d']:,}.",
        "- Для прогнозу береться останній повний набір полів; неповні результати не заповнюються попередніми значеннями.",
        "",
        "## Tier 1 — прогноз появи ЦД2 до очікуваного наступного візиту",
        "",
        f"- Тестова вибірка: {tier1['n_test']} пацієнтів; позитивних результатів: {tier1['n_positive_test']}.",
        f"- ROC-AUC: {tier1['roc_auc']:.3f} (95% ДІ {format_ci(tier1, 'roc_auc')}).",
        f"- PR-AUC: {tier1['pr_auc_average_precision']:.3f} (95% ДІ {format_ci(tier1, 'pr_auc_average_precision')}).",
        f"- Brier score: {tier1['brier_score']:.3f} (95% ДІ {format_ci(tier1, 'brier_score')}).",
        f"- Поріг сповіщення: {tier1['screening_threshold']:.3f}; чутливість {tier1['recall']:.3f} "
        f"(95% ДІ {format_ci(tier1, 'recall')}); точність позитивних сповіщень {tier1['precision']:.3f} "
        f"(95% ДІ {format_ci(tier1, 'precision')}).",
        "- Ціль — перша поява запису діагнозу ЦД2 після дати оцінки й до очікуваного наступного візиту; "
        "це не встановлює дату біологічного початку хвороби.",
        "",
        "## Tier 2 — HbA1c на розрахункову дату",
        "",
        f"- Тестова вибірка: {tier2['n_test']} пацієнтів.",
        f"- MAE: {tier2['mae']:.3f} п.п. (95% ДІ {format_ci(tier2, 'mae')}); "
        f"RMSE: {tier2['rmse']:.3f} (95% ДІ {format_ci(tier2, 'rmse')}).",
        f"- Простий прогноз без зміни: MAE {tier2['baseline_no_change_mae']:.3f} "
        f"(95% ДІ {format_ci(tier2, 'baseline_no_change_mae')}).",
        f"- Різниця MAE модель мінус базовий прогноз: {tier2['mae_minus_baseline_mae']:.3f} "
        f"(95% ДІ {format_ci(tier2, 'mae_minus_baseline_mae')}); від’ємне значення означає меншу помилку моделі.",
        "",
        "## Обмеження",
        "",
        "Це внутрішня ретроспективна часова перевірка, а не зовнішня перевірка на незалежному наборі даних. "
        "Результати не підтверджують клінічну придатність моделі.",
        "",
    ]
    weak_path = BASE / "results_v2" / "experiments" / "tier1_incident_1y_metrics.json"
    if weak_path.exists():
        weak = json.loads(weak_path.read_text(encoding="utf-8"))["test"]
        lines.extend([
            "## Окремий експеримент Tier 1 на 365 днів",
            "",
            f"ROC-AUC {weak['roc_auc']:.3f} на {weak['n_test']} пацієнтах. Ця версія не є активною моделлю застосунку "
            "і не повинна об’єднуватися з результатами Tier 1 до наступного візиту.",
            "",
        ])
    (BASE / "results_v2" / "evaluation_summary_uk.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    say("Loading dated encounter and laboratory data...")
    visits = read_visit_dates()
    labs = read_complete_visits()
    first_e11, last_diagnosis = read_diagnosis_outcomes()
    say(f"Complete actual lab/profile visits: {len(labs):,}; patients: {labs.patient_id.nunique():,}")
    complete_visit_dates = {
        str(pid): list(pd.to_datetime(group["test_date"]).drop_duplicates().sort_values())
        for pid, group in labs.groupby("patient_id", sort=False)
    }
    landmarks = add_next_visit_horizon(labs, complete_visit_dates)
    landmarks["patient_id"] = landmarks["patient_id"].astype(str)
    landmarks["first_e11"] = landmarks["patient_id"].map(first_e11)
    landmarks["last_diagnosis"] = landmarks["patient_id"].map(last_diagnosis)
    # Diagnosis dates and lab dates both establish observation through the prediction date.
    last_lab = pd.Series({pid: max(ds) for pid, ds in visits.items()}, name="last_lab")
    landmarks["last_seen"] = landmarks["patient_id"].map(last_diagnosis)
    lab_seen = landmarks["patient_id"].map(last_lab)
    landmarks["last_seen"] = pd.concat([landmarks["last_seen"], lab_seen], axis=1).max(axis=1)

    prior_e11 = (landmarks["first_e11"].notna() &
                 (landmarks["first_e11"] <= landmarks["test_date"]))
    incident = (landmarks["first_e11"].notna() &
                (landmarks["first_e11"] > landmarks["test_date"]) &
                (landmarks["first_e11"] <= landmarks["expected_next_date"]))
    observed_control = ((landmarks["first_e11"].isna() |
                         (landmarks["first_e11"] > landmarks["expected_next_date"])) &
                        (landmarks["last_seen"] >= landmarks["expected_next_date"]))
    say(f"Landmarks with at least {MIN_PRIOR_INTERVALS} prior intervals: {len(landmarks):,}; "
        f"prior E11: {int(prior_e11.sum()):,}; incident by expected next visit: {int(incident.sum()):,}; "
        f"observed controls: {int(observed_control.sum()):,}; median forecast interval: "
        f"{landmarks[HORIZON_FEATURE].median():.0f} days")
    t1 = landmarks.loc[~prior_e11 & (incident | observed_control)].copy()
    t1["target"] = incident.loc[t1.index].astype(int)
    # Keep the latest eligible complete landmark per patient; outcome is observed at its personal horizon.
    t1 = (t1.sort_values("test_date", kind="stable").drop_duplicates("patient_id", keep="last")
          .sort_values("test_date", kind="stable").reset_index(drop=True))
    if t1["target"].nunique() != 2:
        raise RuntimeError("Tier 1 complete-case cohort has fewer than two outcome classes.")

    split_70 = t1["test_date"].quantile(0.70)
    split_85 = t1["test_date"].quantile(0.85)
    train = t1[t1.test_date <= split_70]
    valid = t1[(t1.test_date > split_70) & (t1.test_date <= split_85)]
    test = t1[t1.test_date > split_85]
    if min(int(train.target.sum()), int(valid.target.sum()), int(test.target.sum())) < 20:
        raise RuntimeError("Tier 1 chronological split has too few positive outcomes.")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=SEED))
    clf.fit(train[FEATURES], train.target)
    val_raw = np.clip(clf.predict_proba(valid[FEATURES])[:, 1], 1e-6, 1 - 1e-6)
    calibrator = LogisticRegression(max_iter=1000, random_state=SEED)
    calibrator.fit(np.log(val_raw / (1 - val_raw)).reshape(-1, 1), valid.target)
    val_p = calibrator.predict_proba(np.log(val_raw / (1 - val_raw)).reshape(-1, 1))[:, 1]
    threshold = choose_threshold(valid.target.to_numpy(), val_p)
    test_raw = np.clip(clf.predict_proba(test[FEATURES])[:, 1], 1e-6, 1 - 1e-6)
    test_p = calibrator.predict_proba(np.log(test_raw / (1 - test_raw)).reshape(-1, 1))[:, 1]
    y = test.target.to_numpy()
    pred = test_p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    t1_metrics = {
        "test_period_start": str(test.test_date.min().date()),
        "test_period_end": str(test.test_date.max().date()),
        "n_test": int(len(test)), "n_positive_test": int(y.sum()), "prevalence_test": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, test_p)),
        "pr_auc_average_precision": float(average_precision_score(y, test_p)),
        "brier_score": float(brier_score_loss(y, test_p)),
        "screening_threshold": float(threshold),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    add_classification_intervals(t1_metrics, y, test_p, threshold)
    t1_bundle = {
        "model": clf, "calibrator": calibrator, "features": FEATURES,
        "name": "Type 2 diabetes diagnosis recorded by estimated next visit",
        "target_definition": "First recorded type 2 diabetes diagnosis after the index date and on or before the next-visit date estimated from prior visit intervals; this is a recorded diagnosis, not biological disease onset.",
        "thresholds": {"screening": float(threshold)},
        "metrics": t1_metrics,
    }
    joblib.dump(t1_bundle, BASE / "tier1_next_visit.joblib")

    # Reconstruct HbA1c on the estimated next-visit date by interpolation between
    # bracketing measured HbA1c results. This aligns the target with Tier 1's horizon.
    confirmed = landmarks.copy()
    hba1c_events = (labs[["patient_id", "test_date", "HbA1c"]]
                   .dropna(subset=["HbA1c"]).sort_values(["patient_id", "test_date"], kind="stable"))
    next_target = {}
    for pid, grp in hba1c_events.groupby("patient_id", sort=False):
        dates = grp["test_date"].to_numpy()
        vals = grp["HbA1c"].to_numpy(dtype=float)
        next_target[str(pid)] = (dates, vals)
    future_vals, future_dates = [], []
    for row in confirmed.itertuples(index=False):
        pair = next_target.get(str(row.patient_id))
        if pair is None:
            future_vals.append(np.nan); future_dates.append(pd.NaT); continue
        dates, vals = pair
        target_date = np.datetime64(row.expected_next_date)
        upper = np.searchsorted(dates, target_date, side="left")
        if upper >= len(dates):
            future_vals.append(np.nan); future_dates.append(pd.NaT)
        elif dates[upper] == target_date:
            future_vals.append(float(vals[upper])); future_dates.append(pd.Timestamp(dates[upper]))
        elif upper == 0:
            future_vals.append(np.nan); future_dates.append(pd.NaT)
        else:
            lower = upper - 1
            span = float((dates[upper] - dates[lower]) / np.timedelta64(1, "D"))
            offset = float((target_date - dates[lower]) / np.timedelta64(1, "D"))
            value = vals[lower] + (vals[upper] - vals[lower]) * (offset / span)
            future_vals.append(float(value)); future_dates.append(pd.Timestamp(target_date))
    confirmed["next_hba1c"] = future_vals
    confirmed["next_hba1c_date"] = future_dates
    t2 = confirmed.dropna(subset=["next_hba1c"]).copy()
    # One latest usable pair per patient avoids patient leakage across chronological splits.
    t2 = (t2.sort_values("test_date", kind="stable").drop_duplicates("patient_id", keep="last")
          .sort_values("test_date", kind="stable").reset_index(drop=True))
    if len(t2) < 300:
        raise RuntimeError(f"Tier 2 complete-case cohort too small: {len(t2)}")
    s70, s85 = t2.test_date.quantile(0.70), t2.test_date.quantile(0.85)
    tr2 = t2[t2.test_date <= s70]
    va2 = t2[(t2.test_date > s70) & (t2.test_date <= s85)]
    te2 = t2[t2.test_date > s85]
    reg = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.06,
                                        max_iter=250, l2_regularization=1.0, random_state=SEED)
    reg.fit(tr2[FEATURES], tr2.next_hba1c)
    p2 = reg.predict(te2[FEATURES])
    baseline = te2["HbA1c"].to_numpy(dtype=float)
    t2_metrics = {
        "test_period_start": str(te2.test_date.min().date()),
        "test_period_end": str(te2.test_date.max().date()),
        "n_test": int(len(te2)),
        "mae": float(mean_absolute_error(te2.next_hba1c, p2)),
        "rmse": float(np.sqrt(mean_squared_error(te2.next_hba1c, p2))),
        "baseline_no_change_mae": float(mean_absolute_error(te2.next_hba1c, baseline)),
        "baseline_no_change_rmse": float(np.sqrt(mean_squared_error(te2.next_hba1c, baseline))),
        "mean_days_to_target_hba1c": float((te2.next_hba1c_date - te2.test_date).dt.days.mean()),
        "mean_days_to_estimated_visit": float(te2[HORIZON_FEATURE].mean()),
    }
    y2 = te2.next_hba1c.to_numpy(dtype=float)
    model_errors = np.abs(y2 - p2)
    baseline_errors = np.abs(y2 - baseline)
    t2_metrics["mae_minus_baseline_mae"] = float(model_errors.mean() - baseline_errors.mean())
    t2_metrics["confidence_intervals_95"] = {
        "mae": bootstrap_ci(y2, p2, lambda actual, estimate: mean_absolute_error(actual, estimate)),
        "rmse": bootstrap_ci(y2, p2, lambda actual, estimate: np.sqrt(mean_squared_error(actual, estimate))),
        "baseline_no_change_mae": bootstrap_ci(
            y2, baseline, lambda actual, estimate: mean_absolute_error(actual, estimate)),
        "mae_minus_baseline_mae": bootstrap_ci(
            y2, model_errors - baseline_errors, lambda _actual, paired_error: paired_error.mean()),
    }
    t2_bundle = {
        "model": reg, "features": FEATURES, "mode": "level",
        "name": "HbA1c at estimated next visit",
        "target_definition": "HbA1c interpolated at the estimated next-visit date from the nearest measured values before and after that date.",
        "metrics": t2_metrics,
    }
    joblib.dump(t2_bundle, BASE / "tier2_next_visit.joblib")

    latest_complete_landmarks = (landmarks.sort_values(["patient_id", "test_date"], kind="stable")
                                 .groupby("patient_id", sort=False).tail(1))
    diagnosis_flag = (pd.to_numeric(latest_complete_landmarks["known_t2d"], errors="coerce")
                      if "known_t2d" in latest_complete_landmarks else
                      pd.Series(np.nan, index=latest_complete_landmarks.index))
    app_eligible = {
        "total": int(len(latest_complete_landmarks)),
        "screening_no_t2d": int((diagnosis_flag == 0).sum()),
        "confirmed_t2d": int((diagnosis_flag == 1).sum()),
        "unknown_diagnosis": int(diagnosis_flag.isna().sum()),
        "rule": "Latest complete panel with at least three complete visits; incomplete later encounters are not used as model inputs.",
    }
    report = {
        "required_fields": REQUIRED,
        "complete_visits": int(len(labs)),
        "complete_patients": int(labs.patient_id.nunique()),
        "minimum_prior_intervals": MIN_PRIOR_INTERVALS,
        "application_eligible_cohort": app_eligible,
        "tier1": {"n": int(len(t1)), "events": int(t1.target.sum()), "features": FEATURES,
                  "test": t1_metrics, "target_definition": t1_bundle["target_definition"]},
        "tier2": {"n": int(len(t2)), "patients": int(t2.patient_id.nunique()),
                  "features": FEATURES, "test": t2_metrics,
                  "target_definition": t2_bundle["target_definition"]},
        "visit_date_method": "Median of at least two intervals between complete laboratory/profile visits, added to the latest complete visit date.",
        "data_policy": "No forecast input imputation. Each scored row must contain all six required observed fields.",
    }
    out = BASE / "results_v2" / "next_visit_metrics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_human_report(report)
    say(f"Tier 1 complete cohort: {len(t1):,}; events by estimated next visit: {int(t1.target.sum()):,}")
    say(f"Tier 1 temporal test: AUC {t1_metrics['roc_auc']:.3f}, recall {t1_metrics['recall']:.3f}, precision {t1_metrics['precision']:.3f}")
    say(f"Tier 2 complete cohort: {len(t2):,}; test MAE {t2_metrics['mae']:.3f} vs no-change {t2_metrics['baseline_no_change_mae']:.3f}")
    say(f"Application eligible patients: {app_eligible['total']:,}; screening={app_eligible['screening_no_t2d']:,}; complications={app_eligible['confirmed_t2d']:,}")
    say(f"Saved models and report: {out.relative_to(BASE)}")


if __name__ == "__main__":
    main()
