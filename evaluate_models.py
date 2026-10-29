"""
evaluate_models.py — повна оцінка моделей DiaPredict для розділу 4 магістерської.

Запуск з папки проєкту (де лежить app.py і файл .env з MYSQL_PASSWORD):
    python evaluate_models.py                 # дані беруться з MySQL (diabetes_db.patient_records)
    python evaluate_models.py --csv           # або з diabetes_master_longitudinal.csv
    python evaluate_models.py --csv "D:\\шлях\\до\\папки"

Результат: текст у консолі + папка results/ (metrics.txt, metrics.json, графіки PNG).
Розбиття й параметри моделей ті самі, що в train_models.py (random_state=42),
тому Accuracy / ROC-AUC / RMSE мають збігатися з постером.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ARGS = sys.argv[1:]
USE_CSV = "--csv" in ARGS
PATH_ARGS = [a for a in ARGS if not a.startswith("--")]
DATA_DIR = PATH_ARGS[0] if PATH_ARGS else r"C:\Users\Admin\OneDrive\Рабочий стол\Dowload_datasets"
INPUT_FILE = os.path.join(DATA_DIR, "diabetes_master_longitudinal.csv")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)

LINES, RESULTS = [], {}


def say(text=""):
    print(text)
    LINES.append(str(text))


CANON = {c.lower(): c for c in ("patient_id", "test_date", "HbA1c", "glucose_fasting", "bmi",
                                "systolic_bp", "triglycerides", "age", "sex", "dx_type2_diabetes")}


def load_data():
    """MySQL (за замовчуванням) або CSV. Рядки впорядковуються за пацієнтом і датою —
    це потрібно для коректного 'наступного візиту' та 'останнього візиту'."""
    if USE_CSV:
        say(f"Джерело: CSV {INPUT_FILE}")
        data = pd.read_csv(INPUT_FILE)
    else:
        from sqlalchemy import create_engine
        from sqlalchemy.engine import URL
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        url = URL.create("mysql+pymysql", username=os.getenv("MYSQL_USER", "root"),
                         password=os.getenv("MYSQL_PASSWORD", ""), host=os.getenv("MYSQL_HOST", "localhost"),
                         port=int(os.getenv("MYSQL_PORT", "3306")), database=os.getenv("MYSQL_DB", "diabetes_db"))
        say("Джерело: MySQL, база diabetes_db, таблиця patient_records")
        data = pd.read_sql("SELECT * FROM patient_records", create_engine(url))
    data = data.rename(columns={c: CANON[c.lower()] for c in data.columns if c.lower() in CANON})
    data["test_date"] = pd.to_datetime(data["test_date"], errors="coerce")
    return data.sort_values(["patient_id", "test_date"], kind="stable").reset_index(drop=True)


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


# ───────────────────────── 0. Дані ─────────────────────────
say("=== 0. ДАНІ ===")
df = load_data()
T1 = ['age', 'glucose_fasting', 'HbA1c', 'bmi', 'systolic_bp', 'triglycerides']
T2 = ['HbA1c', 'glucose_fasting', 'bmi', 'age']

say(f"Записів (візитів): {len(df)}")
say(f"Пацієнтів: {df['patient_id'].nunique()}")
visits = df.groupby('patient_id').size()
say(f"Візитів на пацієнта: медіана {visits.median():.0f}, мін {visits.min()}, макс {visits.max()}")
if 'test_date' in df.columns:
    d = pd.to_datetime(df['test_date'], errors='coerce')
    say(f"Період даних: {d.min().date()} — {d.max().date()}")
per_patient = df.groupby('patient_id')['dx_type2_diabetes'].max()
say(f"Пацієнтів із діагнозом ЦД2: {int(per_patient.sum())} ({100 * per_patient.mean():.1f}%)")
say("Частка пропусків у ознаках (до заповнення медіаною):")
for c in sorted(set(T1 + T2)):
    if c in df.columns:
        say(f"  {c}: {100 * df[c].isna().mean():.1f}%")
RESULTS['data'] = {'rows': int(len(df)), 'patients': int(df['patient_id'].nunique()),
                   'diabetic_patients': int(per_patient.sum())}

# ───────────────────────── 1. Tier 1 ─────────────────────────
say("\n=== 1. TIER 1: ризик діабету (класифікація) ===")
d1 = df.copy()
for c in T1:
    d1[c] = d1[c].fillna(d1[c].median())
last = d1.groupby('patient_id').last().reset_index()
X, y = last[T1], last['dx_type2_diabetes']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
say(f"Навчальна вибірка: {len(Xtr)} пацієнтів; тестова: {len(Xte)}; частка хворих у тесті {100 * yte.mean():.1f}%")
say(f"Базова точність «завжди ні»: {100 * (1 - yte.mean()):.1f}%")

rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42).fit(Xtr, ytr)
proba = rf.predict_proba(Xte)[:, 1]


def clf_metrics(thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    return {'threshold': thr, 'accuracy': accuracy_score(yte, pred),
            'precision': precision_score(yte, pred, zero_division=0),
            'recall': recall_score(yte, pred), 'f1': f1_score(yte, pred),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


auc = roc_auc_score(yte, proba)
say(f"ROC-AUC: {auc:.4f}")
m = clf_metrics(0.5)
say(f"Поріг 0.5: Accuracy {m['accuracy']:.3f}, Precision {m['precision']:.3f}, "
    f"Recall {m['recall']:.3f}, F1 {m['f1']:.3f}")
say(f"Матриця помилок: TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")

say("\nПідбір порога класифікації (для висновків про покращення Recall):")
say(f"{'поріг':>6} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>6}")
sweep = []
for thr in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
    r = clf_metrics(thr)
    sweep.append(r)
    say(f"{thr:>6} {r['accuracy']:>9.3f} {r['precision']:>10.3f} {r['recall']:>8.3f} {r['f1']:>6.3f}")
best = max(sweep, key=lambda r: r['f1'])
say(f"Найкращий F1 серед перевірених порогів: {best['threshold']} (F1 {best['f1']:.3f}, Recall {best['recall']:.3f})")

say("\nВажливість ознак (RandomForest):")
imp = dict(zip(T1, rf.feature_importances_))
for k, v in sorted(imp.items(), key=lambda kv: -kv[1]):
    say(f"  {k}: {v:.3f}")

say("\nПорівняння моделей (ROC-AUC на тому ж тесті):")
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
hgb = HistGradientBoostingClassifier(random_state=42).fit(Xtr, ytr)
auc_lr = roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])
auc_hgb = roc_auc_score(yte, hgb.predict_proba(Xte)[:, 1])
say(f"  Логістична регресія: {auc_lr:.4f}")
say(f"  RandomForest (у системі): {auc:.4f}")
say(f"  HistGradientBoosting: {auc_hgb:.4f}")

# Перевірка: чи не «підказують» відповідь діагностичні лабораторні критерії
no_lab = [c for c in T1 if c not in ('glucose_fasting', 'HbA1c')]
rf2 = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42).fit(Xtr[no_lab], ytr)
auc_nolab = roc_auc_score(yte, rf2.predict_proba(Xte[no_lab])[:, 1])
say(f"\nAUC без глюкози та HbA1c (лише {', '.join(no_lab)}): {auc_nolab:.4f}")

RESULTS['tier1'] = {'auc': auc, 'at_0.5': m, 'threshold_sweep': sweep, 'importance': imp,
                    'auc_logreg': auc_lr, 'auc_hgb': auc_hgb, 'auc_without_glucose_hba1c': auc_nolab,
                    'baseline_always_no_accuracy': 1 - float(yte.mean())}

# ───────────────────────── 2. Tier 2 ─────────────────────────
say("\n=== 2. TIER 2: прогноз HbA1c на наступний візит (регресія) ===")
d2 = df.copy()
d2['next_HbA1c'] = d2.groupby('patient_id')['HbA1c'].shift(-1)
d2 = d2.dropna(subset=['HbA1c', 'next_HbA1c', 'glucose_fasting'])
say(f"Пар «візит → наступний візит»: {len(d2)}")


def reg_report(name, ytrue, pred):
    r = {'rmse': rmse(ytrue, pred), 'mae': float(mean_absolute_error(ytrue, pred)), 'r2': float(r2_score(ytrue, pred))}
    say(f"  {name:<34} RMSE {r['rmse']:.4f}  MAE {r['mae']:.4f}  R2 {r['r2']:.4f}")
    return r


def run_split(title, Xa, ya, ga, split):
    say(f"\n{title}")
    tr, te = split
    Xtr_, Xte_, ytr_, yte_ = Xa.iloc[tr], Xa.iloc[te], ya.iloc[tr], ya.iloc[te]
    out = {}
    out['persistence'] = reg_report("Базова: «без змін»", yte_, Xte_['HbA1c'])
    out['mean'] = reg_report("Базова: середнє train", yte_, np.full(len(yte_), ytr_.mean()))
    ridge = Ridge(alpha=1.0).fit(Xtr_, ytr_)
    out['ridge'] = reg_report("Ridge (у системі)", yte_, ridge.predict(Xte_))
    hgb_r = HistGradientBoostingRegressor(random_state=42).fit(Xtr_, ytr_)
    out['hgb'] = reg_report("HistGradientBoosting", yte_, hgb_r.predict(Xte_))
    return out, ridge


X2, y2, g2 = d2[T2].reset_index(drop=True), d2['next_HbA1c'].reset_index(drop=True), d2['patient_id'].reset_index(drop=True)

idx = np.arange(len(X2))
tr_r, te_r = train_test_split(idx, test_size=0.2, random_state=42)
out_row, ridge_row = run_split("а) Випадковий поділ за рядками (як у train_models.py — збігається з постером):",
                               X2, y2, g2, (tr_r, te_r))

tr_g, te_g = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X2, y2, groups=g2))
out_grp, ridge_grp = run_split("б) Поділ за пацієнтами (візити одного пацієнта не потрапляють і в train, і в test):",
                               X2, y2, g2, (tr_g, te_g))

say("\nКоефіцієнти Ridge (поділ за рядками): " +
    ", ".join(f"{k}={v:.3f}" for k, v in zip(T2, ridge_row.coef_)) + f", intercept={ridge_row.intercept_:.3f}")
RESULTS['tier2'] = {'pairs': int(len(d2)), 'random_split': out_row, 'group_split': out_grp,
                    'ridge_coef': dict(zip(T2, map(float, ridge_row.coef_))),
                    'ridge_intercept': float(ridge_row.intercept_)}

# ───────────────────────── 3. Графіки ─────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(yte, proba)
    plt.figure(figsize=(5, 4.5))
    plt.plot(fpr, tpr, label=f"RandomForest (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Випадковий класифікатор")
    plt.xlabel("Частка хибнопозитивних")
    plt.ylabel("Частка виявлених хворих (Recall)")
    plt.title("ROC-крива, Tier 1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tier1_roc.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(5.5, 3.5))
    order = sorted(imp.items(), key=lambda kv: kv[1])
    plt.barh([k for k, _ in order], [v for _, v in order], color="#2b4585")
    plt.xlabel("Важливість ознаки")
    plt.title("Важливість ознак, Tier 1")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tier1_importance.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(5.5, 4))
    plt.plot([r['threshold'] for r in sweep], [r['precision'] for r in sweep], marker="o", label="Precision")
    plt.plot([r['threshold'] for r in sweep], [r['recall'] for r in sweep], marker="o", label="Recall")
    plt.plot([r['threshold'] for r in sweep], [r['f1'] for r in sweep], marker="o", label="F1")
    plt.xlabel("Поріг класифікації")
    plt.title("Вплив порога на якість, Tier 1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tier1_threshold.png"), dpi=200)
    plt.close()

    sample = np.random.RandomState(42).choice(te_r, size=min(3000, len(te_r)), replace=False)
    plt.figure(figsize=(5, 5))
    plt.scatter(X2['HbA1c'].iloc[sample], y2.iloc[sample], s=4, alpha=0.4, label="Факт")
    xs = np.linspace(X2['HbA1c'].min(), X2['HbA1c'].max(), 50)
    plt.plot(xs, ridge_row.intercept_ + ridge_row.coef_[0] * xs, color="red", label="Ridge (за HbA1c)")
    plt.plot(xs, xs, "--", color="gray", label="Без змін (y = x)")
    plt.xlabel("HbA1c зараз, %")
    plt.ylabel("HbA1c на наступному візиті, %")
    plt.title("Tier 2: прогноз HbA1c")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tier2_forecast.png"), dpi=200)
    plt.close()
    say(f"\nГрафіки збережено в {OUT_DIR}")
except ImportError:
    say("\nmatplotlib не встановлено — графіки пропущено (pip install matplotlib)")

with open(os.path.join(OUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LINES))
with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=2, default=float)
say(f"Метрики збережено: {os.path.join(OUT_DIR, 'metrics.txt')}")