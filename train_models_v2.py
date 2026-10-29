"""
train_models_v2.py — друга ітерація моделей DiaPredict (для розділів 3–4 магістерської).

Що нового порівняно з train_models.py (v1):
  Tier 1  • прапорець «тригліцериди не вимірювали» (45% пропусків);
          • порівняння 4 моделей (логістична регресія, RF v1, збалансований RF, HistGradientBoosting)
            5-кратною крос-валідацією;
          • абляція ознак (що буде без ТГ, без глюкози та HbA1c);
          • вибір порога класифікації за цільовим Recall для скринінгу (на крос-валідації, не на тесті).
  Tier 2  • історія візитів: попередній HbA1c, динаміка, середнє за історією, номер візиту, інтервал між візитами;
          • прогноз ЗМІНИ HbA1c (а не абсолютного значення) — природніша постановка задачі;
          • порівняння з базовим прогнозом «без змін» за RMSE і MAE;
          • поділ ЗА ПАЦІЄНТАМИ, 5-кратна крос-валідація за пацієнтами + незалежний тест.

Нові моделі зберігаються в tier1_v2.joblib та tier2_v2.joblib. Старі файли НЕ перезаписуються,
тож сервер (app.py) продовжує працювати без змін.

Запуск із папки проєкту (де app.py та .env із MYSQL_PASSWORD):
    python train_models_v2.py            # дані з MySQL
    python train_models_v2.py --csv      # або з diabetes_master_longitudinal.csv
Результати: папка results_v2/ (metrics_v2.txt, metrics_v2.json, графіки).

Обмеження даних, які варто згадати в роботі: пропуски в лабораторних показниках були заповнені
ffill/bfill у build_master_dataset.py; bfill для перших візитів використовує значення з наступних візитів.
"""
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_recall_curve, precision_score,
                             r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
RECALL_TARGET = 0.80          # цільовий Recall для скринінгового порога
ARGS = sys.argv[1:]
USE_CSV = "--csv" in ARGS
PATH_ARGS = [a for a in ARGS if not a.startswith("--")]
DATA_DIR = PATH_ARGS[0] if PATH_ARGS else r"C:\Users\Admin\OneDrive\Рабочий стол\Dowload_datasets"
BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "results_v2")
os.makedirs(OUT_DIR, exist_ok=True)

LINES, RESULTS = [], {}


def say(text=""):
    print(text, flush=True)
    LINES.append(str(text))


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


CANON = {c.lower(): c for c in ("patient_id", "test_date", "HbA1c", "glucose_fasting", "bmi",
                                "systolic_bp", "triglycerides", "age", "sex", "dx_type2_diabetes")}


def load_data():
    if USE_CSV:
        say("Джерело: CSV")
        data = pd.read_csv(os.path.join(DATA_DIR, "diabetes_master_longitudinal.csv"))
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
        say("Джерело: MySQL, diabetes_db.patient_records")
        data = pd.read_sql("SELECT * FROM patient_records", create_engine(url))
    data = data.rename(columns={c: CANON[c.lower()] for c in data.columns if c.lower() in CANON})
    data["test_date"] = pd.to_datetime(data["test_date"], errors="coerce")
    for c in ("age", "HbA1c", "glucose_fasting", "bmi", "systolic_bp", "triglycerides"):
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data["patient_id"] = data["patient_id"].astype(str)
    return data.sort_values(["patient_id", "test_date"], kind="stable").reset_index(drop=True)


say("=== ДАНІ ===")
df = load_data()
say(f"Записів: {len(df)}; пацієнтів: {df['patient_id'].nunique()}")

# ════════════════════════════ TIER 1 ════════════════════════════
say("\n=== TIER 1 (v2): оцінка ризику діабету ===")
T1_BASE = ['age', 'glucose_fasting', 'HbA1c', 'bmi', 'systolic_bp', 'triglycerides']
T1_V2 = T1_BASE + ['tg_missing']

last = df.groupby('patient_id', sort=True).tail(1).reset_index(drop=True)      # останній візит пацієнта
last['tg_missing'] = last['triglycerides'].isna().astype(int)
y1 = last['dx_type2_diabetes'].astype(int).to_numpy()
say(f"Пацієнтів: {len(last)}; хворих: {y1.sum()} ({100 * y1.mean():.1f}%); "
    f"без вимірювання ТГ на останньому візиті: {100 * last['tg_missing'].mean():.1f}%")

idx_all = np.arange(len(last))
idx_tr, idx_te = train_test_split(idx_all, test_size=0.2, random_state=SEED, stratify=y1)
Xtr_all, Xte_all = last.iloc[idx_tr], last.iloc[idx_te]
ytr, yte = y1[idx_tr], y1[idx_te]
say(f"Навчальна вибірка: {len(idx_tr)}, тестова: {len(idx_te)} (частка хворих у тесті {100 * yte.mean():.1f}%)")


def imp():
    return SimpleImputer(strategy='median')


def rf_v1():
    return make_pipeline(imp(), RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED, n_jobs=-1))


def rf_bal():
    return make_pipeline(imp(), RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=20,
                                                       class_weight='balanced_subsample', random_state=SEED, n_jobs=-1))


def logreg():
    return make_pipeline(imp(), StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))


def hgb():
    return HistGradientBoostingClassifier(random_state=SEED)


CANDIDATES = [  # (назва, ознаки, фабрика)
    ("RF v1 (глибина 6, 6 ознак) — поточна модель", T1_BASE, rf_v1),
    ("Логістична регресія +tg_missing", T1_V2, logreg),
    ("RF v1 +tg_missing", T1_V2, rf_v1),
    ("RF збалансований +tg_missing", T1_V2, rf_bal),
    ("HistGradientBoosting +tg_missing", T1_V2, hgb),
]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

say("\nКрос-валідація (5 фолдів, лише навчальна частина):")
say(f"{'модель':<46}{'ROC-AUC':>9}{'PR-AUC':>9}")
cv_res, oof_store = {}, {}
for name, feats, make in CANDIDATES:
    oof = cross_val_predict(make(), Xtr_all[feats], ytr, cv=skf, method='predict_proba')[:, 1]
    cv_res[name] = {'auc': float(roc_auc_score(ytr, oof)), 'pr_auc': float(average_precision_score(ytr, oof))}
    oof_store[name] = oof
    say(f"{name:<46}{cv_res[name]['auc']:>9.4f}{cv_res[name]['pr_auc']:>9.4f}")

best_name = max((c[0] for c in CANDIDATES[1:]), key=lambda n: cv_res[n]['auc'])
best_feats, best_make = next((f, m) for n, f, m in CANDIDATES if n == best_name)
say(f"\nОбрана модель Tier 1 v2: {best_name}")

say("\nАбляція ознак (крос-валідаційний ROC-AUC обраної моделі):")
ABL = {
    "усі ознаки +tg_missing": T1_V2,
    "без прапорця tg_missing": T1_BASE,
    "без тригліцеридів (і без прапорця)": [c for c in T1_BASE if c != 'triglycerides'],
    "без глюкози та HbA1c": [c for c in T1_V2 if c not in ('glucose_fasting', 'HbA1c')],
    "лише глюкоза та HbA1c": ['glucose_fasting', 'HbA1c'],
}
abl_res = {}
for title, feats in ABL.items():
    oof = cross_val_predict(best_make(), Xtr_all[feats], ytr, cv=skf, method='predict_proba')[:, 1]
    abl_res[title] = float(roc_auc_score(ytr, oof))
    say(f"  {title:<40} AUC {abl_res[title]:.4f}")

# Підбір порогів за OOF-прогнозами (на навчальній частині — тест не використовується)
oof_best = oof_store[best_name]
prec, rec, thr = precision_recall_curve(ytr, oof_best)
f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
thr_f1 = float(thr[int(np.argmax(f1s))])
ok = thr[rec[:-1] >= RECALL_TARGET]
thr_screen = float(ok.max()) if len(ok) else float(thr.min())
say(f"\nПороги, підібрані на крос-валідації: максимум F1 → {thr_f1:.3f}; скринінг (Recall ≥ {RECALL_TARGET:.0%}) → {thr_screen:.3f}")

final1 = best_make().fit(Xtr_all[best_feats], ytr)
p_te = final1.predict_proba(Xte_all[best_feats])[:, 1]
v1 = rf_v1().fit(Xtr_all[T1_BASE], ytr)
p_v1 = v1.predict_proba(Xte_all[T1_BASE])[:, 1]


def clf_row(proba, thr_):
    pred = (proba >= thr_).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    return {'threshold': float(thr_), 'accuracy': float(accuracy_score(yte, pred)),
            'precision': float(precision_score(yte, pred, zero_division=0)),
            'recall': float(recall_score(yte, pred)), 'f1': float(f1_score(yte, pred)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


say("\nТЕСТ (незалежна частина, 20% пацієнтів):")
say(f"Базова точність «завжди ні»: {100 * (1 - yte.mean()):.1f}%")
say(f"{'модель / поріг':<38}{'AUC':>7}{'Acc':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}")
t1_rows = {}
for label, proba, t in (("v1: RF, поріг 0.5", p_v1, 0.5),
                        ("v2: поріг 0.5", p_te, 0.5),
                        (f"v2: поріг F1 ({thr_f1:.2f})", p_te, thr_f1),
                        (f"v2: скринінг ({thr_screen:.2f})", p_te, thr_screen)):
    r = clf_row(proba, t)
    r['auc'] = float(roc_auc_score(yte, proba))
    t1_rows[label] = r
    say(f"{label:<38}{r['auc']:>7.3f}{r['accuracy']:>7.3f}{r['precision']:>7.3f}{r['recall']:>7.3f}{r['f1']:>7.3f}")
scr = t1_rows[f"v2: скринінг ({thr_screen:.2f})"]
say(f"Скринінговий режим: TP={scr['tp']} FN={scr['fn']} FP={scr['fp']} TN={scr['tn']}")
say(f"PR-AUC v1: {average_precision_score(yte, p_v1):.3f}; v2: {average_precision_score(yte, p_te):.3f}; "
    f"Brier v2: {brier_score_loss(yte, p_te):.4f}")

pi = permutation_importance(final1, Xte_all[best_feats].iloc[:5000], yte[:5000], scoring='roc_auc',
                            n_repeats=3, random_state=SEED)
imp_t1 = dict(sorted(zip(best_feats, map(float, pi.importances_mean)), key=lambda kv: -kv[1]))
say("\nПерестановочна важливість ознак (падіння AUC на тесті):")
for k, v in imp_t1.items():
    say(f"  {k}: {v:.4f}")

joblib.dump({'model': final1, 'features': best_feats, 'name': best_name,
             'thresholds': {'f1': thr_f1, 'screening': thr_screen, 'recall_target': RECALL_TARGET},
             'test_metrics': t1_rows},
            os.path.join(BASE, "tier1_v2.joblib"))
RESULTS['tier1'] = {'cv': cv_res, 'best': best_name, 'ablation_auc': abl_res, 'thresholds': {'f1': thr_f1, 'screening': thr_screen},
                    'test': t1_rows, 'importance': imp_t1}

# ════════════════════════════ TIER 2 ════════════════════════════
say("\n=== TIER 2 (v2): прогноз HbA1c на наступний візит ===")
d = df.copy()
g = d.groupby('patient_id', sort=False)
d['next_HbA1c'] = g['HbA1c'].shift(-1)
d['prev_HbA1c'] = g['HbA1c'].shift(1)
d['prev2_HbA1c'] = g['HbA1c'].shift(2)
d['visit_index'] = g.cumcount()
d['has_prev'] = d['prev_HbA1c'].notna().astype(int)
d['hba1c_delta1'] = (d['HbA1c'] - d['prev_HbA1c']).fillna(0.0)
d['hba1c_delta2'] = (d['prev_HbA1c'] - d['prev2_HbA1c']).fillna(0.0)
d['hba1c_hist_mean'] = d.groupby('patient_id', sort=False)['HbA1c'].cumsum() / (d['visit_index'] + 1)
d['glucose_delta1'] = (d['glucose_fasting'] - d.groupby('patient_id', sort=False)['glucose_fasting'].shift(1)).fillna(0.0)
d['days_since_prev'] = d.groupby('patient_id', sort=False)['test_date'].diff().dt.days.fillna(0.0)

T2_BASE = ['HbA1c', 'glucose_fasting', 'bmi', 'age']
T2_HIST = T2_BASE + ['systolic_bp', 'has_prev', 'hba1c_delta1', 'hba1c_delta2', 'hba1c_hist_mean',
                     'visit_index', 'glucose_delta1', 'days_since_prev']

d2 = d.dropna(subset=['HbA1c', 'next_HbA1c', 'glucose_fasting']).reset_index(drop=True)
y2 = d2['next_HbA1c'].to_numpy()
groups = d2['patient_id'].to_numpy()
stable = np.abs(y2 - d2['HbA1c'].to_numpy()) <= 0.3
say(f"Пар «візит → наступний візит»: {len(d2)}; пацієнтів: {d2['patient_id'].nunique()}; "
    f"частка «стабільних» переходів (|зміна| ≤ 0.3): {100 * stable.mean():.1f}%")

MODELS2 = {  # назва: (ознаки, режим цілі, оцінювач)
    "Ridge v1 (4 ознаки) — поточна модель": (T2_BASE, 'level', lambda: Ridge(alpha=1.0)),
    "Ridge +історія": (T2_HIST, 'delta', lambda: make_pipeline(imp(), StandardScaler(), Ridge(alpha=1.0))),
    "HistGB +історія (MSE)": (T2_HIST, 'delta', lambda: HistGradientBoostingRegressor(
        learning_rate=0.08, max_iter=300, random_state=SEED)),
    "HistGB +історія (MAE)": (T2_HIST, 'delta', lambda: HistGradientBoostingRegressor(
        loss='absolute_error', learning_rate=0.08, max_iter=300, random_state=SEED)),
}


def fit_predict(spec, tr, te):
    feats, mode, make = spec
    est = make()
    if mode == 'delta':
        est.fit(d2.iloc[tr][feats], y2[tr] - d2.iloc[tr]['HbA1c'].to_numpy())
        return est.predict(d2.iloc[te][feats]) + d2.iloc[te]['HbA1c'].to_numpy(), est
    est.fit(d2.iloc[tr][feats], y2[tr])
    return est.predict(d2.iloc[te][feats]), est


def scores(yt, p):
    return {'rmse': rmse(yt, p), 'mae': float(mean_absolute_error(yt, p)), 'r2': float(r2_score(yt, p))}


gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr2, te2 = next(gss.split(d2, y2, groups=groups))
say(f"Навчальна частина: {len(set(groups[tr2]))} пацієнтів, тест: {len(set(groups[te2]))} пацієнтів (поділ за пацієнтами)")

say("\nКрос-валідація за пацієнтами (5 фолдів, навчальна частина), середнє ± σ:")
cv2 = {k: {'rmse': [], 'mae': []} for k in ["Базова: «без змін»", *MODELS2]}
for f_tr, f_te in GroupKFold(n_splits=5).split(tr2, groups=groups[tr2]):
    a, b = tr2[f_tr], tr2[f_te]
    s = scores(y2[b], d2.iloc[b]['HbA1c'].to_numpy())
    cv2["Базова: «без змін»"]['rmse'].append(s['rmse'])
    cv2["Базова: «без змін»"]['mae'].append(s['mae'])
    for name, spec in MODELS2.items():
        p, _ = fit_predict(spec, a, b)
        s = scores(y2[b], p)
        cv2[name]['rmse'].append(s['rmse'])
        cv2[name]['mae'].append(s['mae'])
say(f"{'модель':<40}{'RMSE':>16}{'MAE':>16}")
cv2_sum = {}
for k, v in cv2.items():
    cv2_sum[k] = {m: [float(np.mean(v[m])), float(np.std(v[m]))] for m in v}
    say(f"{k:<40}{cv2_sum[k]['rmse'][0]:>10.4f} ± {cv2_sum[k]['rmse'][1]:.3f}{cv2_sum[k]['mae'][0]:>10.4f} ± {cv2_sum[k]['mae'][1]:.3f}")

best2 = min(MODELS2, key=lambda k: cv2_sum[k]['rmse'][0] + cv2_sum[k]['mae'][0])
say(f"\nОбрана модель Tier 2 v2 (мінімум RMSE+MAE на крос-валідації): {best2}")

say("\nТЕСТ (незалежні 20% пацієнтів):")
say(f"{'модель':<40}{'RMSE':>8}{'MAE':>8}{'R2':>8}")
base_s = scores(y2[te2], d2.iloc[te2]['HbA1c'].to_numpy())
say(f"{'Базова: «без змін»':<40}{base_s['rmse']:>8.4f}{base_s['mae']:>8.4f}{base_s['r2']:>8.4f}")
t2_test = {"Базова: «без змін»": base_s}
preds_te, fitted = {}, {}
for name, spec in MODELS2.items():
    p, est = fit_predict(spec, tr2, te2)
    preds_te[name], fitted[name] = p, est
    t2_test[name] = scores(y2[te2], p)
    r = t2_test[name]
    say(f"{name:<40}{r['rmse']:>8.4f}{r['mae']:>8.4f}{r['r2']:>8.4f}")
rb, mb = t2_test[best2], base_s
say(f"\nОбрана модель проти «без змін»: RMSE {100 * (rb['rmse'] / mb['rmse'] - 1):+.1f}%, MAE {100 * (rb['mae'] / mb['mae'] - 1):+.1f}%")

st = stable[te2]
say("Похибка MAE окремо для стабільних та змінних переходів (тест):")
for label, mask in (("стабільні (|зміна| ≤ 0.3)", st), ("змінні (|зміна| > 0.3)", ~st)):
    if mask.sum():
        mb_ = mean_absolute_error(y2[te2][mask], d2.iloc[te2]['HbA1c'].to_numpy()[mask])
        mm_ = mean_absolute_error(y2[te2][mask], preds_te[best2][mask])
        say(f"  {label:<28} «без змін» {mb_:.3f} | {best2}: {mm_:.3f}  (n={int(mask.sum())})")

joblib.dump({'model': fitted[best2], 'features': MODELS2[best2][0], 'mode': MODELS2[best2][1], 'name': best2,
             'note': "якщо mode='delta': HbA1c_next = HbA1c + model.predict(X)", 'test_metrics': t2_test},
            os.path.join(BASE, "tier2_v2.joblib"))
RESULTS['tier2'] = {'cv': cv2_sum, 'best': best2, 'test': t2_test, 'stable_share': float(stable.mean())}

# ════════════════════════════ ГРАФІКИ ════════════════════════════
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5.2, 4.6))
    for label, pr, ls in (("v1: RF", p_v1, "--"), (f"v2: {best_name.split(' +')[0]}", p_te, "-")):
        fpr, tpr, _ = roc_curve(yte, pr)
        plt.plot(fpr, tpr, ls, label=f"{label} (AUC {roc_auc_score(yte, pr):.2f})")
    plt.plot([0, 1], [0, 1], ":", color="gray")
    plt.xlabel("Частка хибнопозитивних")
    plt.ylabel("Recall")
    plt.title("ROC-криві Tier 1 (тест)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tier1_roc_v1_v2.png"), dpi=200)
    plt.close()

    ths = np.linspace(0.05, 0.8, 40)
    plt.figure(figsize=(5.5, 4))
    plt.plot(ths, [precision_score(yte, p_te >= t, zero_division=0) for t in ths], label="Precision")
    plt.plot(ths, [recall_score(yte, p_te >= t) for t in ths], label="Recall")
    plt.plot(ths, [f1_score(yte, p_te >= t) for t in ths], label="F1")
    plt.axvline(thr_screen, color="gray", ls=":", label=f"скринінг ({thr_screen:.2f})")
    plt.xlabel("Поріг класифікації")
    plt.title("Вплив порога, Tier 1 v2 (тест)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tier1_threshold_v2.png"), dpi=200)
    plt.close()

    names = list(t2_test)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - 0.2, [t2_test[n]['rmse'] for n in names], 0.4, label="RMSE")
    ax.bar(x + 0.2, [t2_test[n]['mae'] for n in names], 0.4, label="MAE")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" — поточна модель", "").replace(" +історія", "\n+історія") for n in names], fontsize=7)
    ax.set_ylabel("Похибка HbA1c, %")
    ax.set_title("Tier 2: порівняння моделей (тест)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tier2_models_v2.png"), dpi=200)
    plt.close()
    say(f"\nГрафіки збережено в {OUT_DIR}")
except ImportError:
    say("\nmatplotlib не встановлено — графіки пропущено")

with open(os.path.join(OUT_DIR, "metrics_v2.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LINES))
with open(os.path.join(OUT_DIR, "metrics_v2.json"), "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=2, default=float)
say(f"Збережено: tier1_v2.joblib, tier2_v2.joblib, {os.path.join(OUT_DIR, 'metrics_v2.txt')}")