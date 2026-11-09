"""Endocrinology Decision Support System API (FastAPI + MySQL + ML-моделі)."""
import logging
import mimetypes
import os
import time
from functools import lru_cache
from datetime import date
from pathlib import Path
from threading import Lock, Thread
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL

try:  # .env необов'язковий: pip install python-dotenv
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dss")

# Фікс для Windows/MIME-type
mimetypes.add_type("application/javascript", ".js", True)

# ───────────────────────── Конфігурація ─────────────────────────
BASE_DIR = Path(__file__).resolve().parent
TABLE = os.getenv("MYSQL_TABLE", "patient_records_observed")
GLUCOSE_MMOL_TO_MGDL = 18.015
TRIGLYCERIDES_MMOL_TO_MGDL = 88.57

MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
if MYSQL_PASSWORD is None:
    raise RuntimeError(
        "Не задано MYSQL_PASSWORD. Створіть файл .env поруч з app.py "
        "(див. .env.example) або встановіть змінну оточення."
    )

DATABASE_URL = URL.create(
    "mysql+pymysql",
    username=os.getenv("MYSQL_USER", "root"),
    password=MYSQL_PASSWORD,  # спецсимволи в паролі екрануються автоматично
    host=os.getenv("MYSQL_HOST", "localhost"),
    port=int(os.getenv("MYSQL_PORT", "3306")),
    database=os.getenv("MYSQL_DB", "diabetes_db"),
)
# pool_pre_ping прибирає "MySQL server has gone away" після простою
engine = create_engine(DATABASE_URL, pool_recycle=3600, pool_pre_ping=True,
                       pool_size=5, max_overflow=10)

app = FastAPI(title="Endocrinology Decision Support System API")
# JSON зі списком пацієнтів великий (МБ) — стискаємо, щоб він швидше доходив до браузера
app.add_middleware(GZipMiddleware, minimum_size=1000)

class NoCacheStaticFiles(StaticFiles):
    """Браузер перевіряє файли при кожному запиті, тож зміни в JS підхоплюються одразу."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


js_dir = BASE_DIR / "js"
if js_dir.exists():
    app.mount("/js", NoCacheStaticFiles(directory=js_dir), name="js")

t1_model = joblib.load(BASE_DIR / "tier1_diabetes_risk.joblib")      # v1 (резервна)
t2_model = joblib.load(BASE_DIR / "tier2_hba1c_forecast.joblib")     # v1 (резервна)


def _load_bundle(name: str):
    """Моделі другої версії (train_models_v2.py). Якщо файлу немає — працюють моделі v1."""
    path = BASE_DIR / name
    return joblib.load(path) if path.exists() else None


t1_v2 = _load_bundle("tier1_v2.joblib")
t2_v2 = _load_bundle("tier2_v2.joblib")
t1_forecast = _load_bundle("tier1_next_visit.joblib")
t2_next_visit = _load_bundle("tier2_next_visit.joblib")
logger.info("Tier 1: %s | Tier 2: %s",
            f"next visit ({t1_forecast['name']})" if t1_forecast else
            (f"v2 ({t1_v2['name']})" if t1_v2 else "v1"),
            f"next visit ({t2_next_visit['name']})" if t2_next_visit else
            (f"v2 ({t2_v2['name']})" if t2_v2 else "v1"))

T1_DEFAULTS = {"age": 50, "glucose_fasting": 100, "HbA1c": 6.0,
               "bmi": 25, "systolic_bp": 120, "triglycerides": np.nan}
T2_DEFAULTS = {"HbA1c": 6.0, "glucose_fasting": 100, "bmi": 25, "age": 50}
T1_BASE = list(T1_DEFAULTS)

# Tier 1: нижня межа — поріг скринінгового сповіщення. Високий рівень
# не визначається лише за моделлю, бо його поріг не підтвердився на часовому тесті.
RISK_LOW = (t1_forecast["thresholds"]["screening"] if t1_forecast else
            (t1_v2["thresholds"]["screening"] if t1_v2 else 0.3))
REQUIRED_FORECAST_FIELDS = ("age", "glucose_fasting", "HbA1c", "bmi", "systolic_bp", "triglycerides")
MIN_PRIOR_INTERVALS = 2


class LabRecordInput(BaseModel):
    patient_mode: str = "existing"
    patient_id: str = Field(min_length=1, max_length=32)
    age: int = Field(ge=0, le=120)
    sex: str
    dx_type2_diabetes: int = Field(ge=0, le=1)
    test_date: date
    hba1c: float = Field(ge=3, le=20)
    glucose_fasting_mmol: float = Field(ge=1, le=60)
    bmi: float = Field(ge=10, le=80)
    systolic_bp: float = Field(ge=60, le=260)
    diastolic_bp: float = Field(default=80.0, ge=30, le=160)
    triglycerides: float = Field(ge=0.1, le=50)
    # Спосіб життя вводить лікар/лаборант; порожнє значення = "не змінювати"
    smoking: Optional[str] = None
    alcohol: Optional[str] = None
    family_history: Optional[str] = None


class LifestyleInput(BaseModel):
    smoking: Optional[str] = None
    alcohol: Optional[str] = None
    family_history: Optional[str] = None


_id_lock = Lock()  # щоб два одночасні запити не отримали однаковий новий ID


# ───────────────────────── Допоміжні функції ─────────────────────────
def _num(value, default):
    """float зі значенням за замовчуванням для NULL/NaN."""
    return default if value is None or pd.isna(value) else float(value)


@lru_cache(maxsize=1)
def _table_columns() -> dict:
    """{'hba1c': 'HbA1c', ...} — реальні колонки таблиці (регістронезалежно)."""
    return {c["name"].lower(): c["name"] for c in inspect(engine).get_columns(TABLE)}


# Назви колонок у БД можуть відрізнятися регістром/написанням — приводимо їх до тих, що очікує код.
_COLUMN_ALIASES = {
    "hba1c": "HbA1c", "hba1c_percent": "HbA1c", "a1c": "HbA1c",
    "triglycerides": "triglycerides", "triglyceride": "triglycerides", "tg": "triglycerides",
    "triglycerides_mgdl": "triglycerides", "triglycerides_mg_dl": "triglycerides",
    "glucose_fasting": "glucose_fasting", "systolic_bp": "systolic_bp", "bmi": "bmi",
    "age": "age", "sex": "sex", "dx_type2_diabetes": "dx_type2_diabetes",
    "test_date": "test_date", "patient_id": "patient_id",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        canon = _COLUMN_ALIASES.get(str(col).lower())
        if canon and canon != col and canon not in df.columns:
            rename[col] = canon
    return df.rename(columns=rename) if rename else df


def _last_valid(df: pd.DataFrame, col: str):
    """Останнє непорожнє значення колонки (показник міг не вимірюватися на останньому візиті)."""
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return None if s.empty else float(s.iloc[-1])


def _estimate_next_visit(dates):
    """Estimate the next visit by the median of at least two observed prior intervals."""
    observed = pd.Series(pd.to_datetime(list(dates), errors="coerce")).dropna().drop_duplicates().sort_values()
    if len(observed) < MIN_PRIOR_INTERVALS + 1:
        return None, None
    gaps = observed.diff().dt.days.dropna()
    gaps = gaps[gaps > 0]
    if len(gaps) < MIN_PRIOR_INTERVALS:
        return None, None
    days = max(1, int(round(float(gaps.median()))))
    return observed.iloc[-1] + pd.Timedelta(days=days), days


def _complete_latest_visits(df: pd.DataFrame) -> pd.DataFrame:
    """Return each patient's latest complete panel after at least three such visits."""
    if df.empty or "patient_id" not in df or "test_date" not in df:
        return pd.DataFrame()
    d = _normalize_columns(df.copy())
    d["patient_id"] = d["patient_id"].astype(str)
    d["test_date"] = pd.to_datetime(d["test_date"], errors="coerce")
    for field in REQUIRED_FORECAST_FIELDS:
        d[field] = pd.to_numeric(d[field], errors="coerce") if field in d else np.nan
        d[field] = d[field].where(np.isfinite(d[field]))
    d = d.dropna(subset=["patient_id", "test_date"])
    # Merge rows only within one patient/date encounter; never carry values from older visits.
    visits = (d.sort_values(["patient_id", "test_date"], kind="stable")
              .groupby(["patient_id", "test_date"], as_index=False, sort=False).last())
    fields = list(REQUIRED_FORECAST_FIELDS)
    complete_mask = visits[fields].notna().all(axis=1)
    # Match model training: use the latest fully observed panel. Later dates with
    # only partial tests do not become a model input or copy older measurements.
    complete = visits.loc[complete_mask].copy()
    complete = complete.sort_values(["patient_id", "test_date"], kind="stable")
    groups = complete.groupby("patient_id", sort=False)
    complete["interval_days"] = groups["test_date"].diff().dt.days
    complete["previous_hba1c"] = groups["HbA1c"].shift(1)
    complete["previous_glucose"] = groups["glucose_fasting"].shift(1)
    complete["previous_systolic_bp"] = groups["systolic_bp"].shift(1)
    stats = complete.groupby("patient_id", sort=False).agg(
        visits=("test_date", "size"),
        interval_count=("interval_days", "count"),
        median_interval=("interval_days", "median"),
    )
    stats = stats[stats["interval_count"] >= MIN_PRIOR_INTERVALS]
    if stats.empty:
        return pd.DataFrame()
    current = complete.groupby("patient_id", sort=False).tail(1).set_index("patient_id")
    current = current.join(stats, how="inner", rsuffix="_stats")
    current["days_to_next_visit"] = current["median_interval"].round().clip(lower=1).astype(int)
    next_dates = current["test_date"] + pd.to_timedelta(current["days_to_next_visit"], unit="D")
    current["next_visit_date"] = next_dates.dt.date.astype(str)
    return current.reset_index()


def _trig_to_mmol(raw):
    """Database contract: triglycerides are stored in mg/dL."""
    if raw is None:
        return None
    return raw / TRIGLYCERIDES_MMOL_TO_MGDL


def _features(latest: dict, defaults: dict) -> pd.DataFrame:
    return pd.DataFrame([{k: _num(latest.get(k), d) for k, d in defaults.items()}])


def _t1_proba(frame: pd.DataFrame) -> np.ndarray:
    """Tier 1 score: probability of a first recorded type 2 diabetes diagnosis by next visit."""
    if t1_forecast is None:
        return np.full(len(frame), np.nan, dtype=float)
    if t1_forecast is not None:
        features = t1_forecast["features"]
        x = pd.DataFrame({k: (pd.to_numeric(frame[k], errors="coerce") if k in frame.columns
                              else pd.Series(np.nan, index=frame.index)) for k in features if k != "tg_missing"})
        available = x[features].notna().all(axis=1)
        result = np.full(len(x), np.nan, dtype=float)
        if not available.any():
            return result
        raw = np.clip(t1_forecast["model"].predict_proba(x.loc[available, features])[:, 1], 1e-6, 1 - 1e-6)
        calibrator = t1_forecast.get("calibrator")
        if calibrator is not None:
            logits = np.log(raw / (1 - raw)).reshape(-1, 1)
            raw = calibrator.predict_proba(logits)[:, 1]
        result[available.to_numpy()] = raw
        return result
    if t1_v2 is None:
        x = pd.DataFrame({k: (pd.to_numeric(frame[k], errors="coerce") if k in frame.columns
                              else pd.Series(np.nan, index=frame.index)).fillna(d).astype(float)
                          for k, d in T1_DEFAULTS.items()})
        return t1_model.predict_proba(x)[:, 1]
    x = pd.DataFrame({k: (pd.to_numeric(frame[k], errors="coerce").astype(float) if k in frame.columns
                          else pd.Series(np.nan, index=frame.index)) for k in T1_BASE})
    x["tg_missing"] = x["triglycerides"].isna().astype(int)  # як у train_models_v2.py; пропуски заповнює сам pipeline
    return t1_v2["model"].predict_proba(x[t1_v2["features"]])[:, 1]


def _t2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ознаки Tier 2 v2 для кожного візиту (історія пацієнта) — ті самі, що в train_models_v2.py.
    Рядки кожного пацієнта мають іти в хронологічному порядку."""
    d = pd.DataFrame(index=df.index)
    d["patient_id"] = df["patient_id"].astype(str)
    for col, default in (("HbA1c", 6.0), ("glucose_fasting", 100.0), ("bmi", 25.0), ("age", 50.0)):
        d[col] = pd.to_numeric(df[col], errors="coerce").fillna(default) if col in df.columns else default
    d["systolic_bp"] = pd.to_numeric(df["systolic_bp"], errors="coerce") if "systolic_bp" in df.columns else np.nan
    dates = pd.to_datetime(df["test_date"], errors="coerce")
    g = d.groupby("patient_id", sort=False)
    prev, prev2 = g["HbA1c"].shift(1), g["HbA1c"].shift(2)
    d["visit_index"] = g.cumcount()
    d["has_prev"] = prev.notna().astype(int)
    d["hba1c_delta1"] = (d["HbA1c"] - prev).fillna(0.0)
    d["hba1c_delta2"] = (prev - prev2).fillna(0.0)
    d["hba1c_hist_mean"] = g["HbA1c"].cumsum() / (d["visit_index"] + 1)
    d["glucose_delta1"] = (d["glucose_fasting"] - g["glucose_fasting"].shift(1)).fillna(0.0)
    d["days_since_prev"] = dates.groupby(d["patient_id"], sort=False).diff().dt.days.fillna(0.0)
    return d


def _t2_predict(feat: pd.DataFrame) -> np.ndarray:
    """Прогноз HbA1c на наступний візит за ознаками з _t2_features."""
    if t2_next_visit is not None:
        return t2_next_visit["model"].predict(feat[t2_next_visit["features"]])
    if t2_v2 is None:
        return t2_model.predict(feat[list(T2_DEFAULTS)])
    pred = t2_v2["model"].predict(feat[t2_v2["features"]])
    return pred + feat["HbA1c"].to_numpy() if t2_v2.get("mode") == "delta" else pred


LIFESTYLE_TABLE = "patient_lifestyle"
NOT_SET = "Не вказано"
LIFESTYLE_CHOICES = {
    "smoking": ("Не курить", "Колишній курець", "Активний курець"),
    "alcohol": ("Ні", "Помірне вживання", "Часте вживання"),
    "family_history": ("Не обтяжена", "Присутня (1-ша лінія)"),
}


@lru_cache(maxsize=1)
def _ensure_lifestyle_table() -> bool:
    """Окрема таблиця зі способом життя (основну таблицю аналізів не змінюємо)."""
    with engine.begin() as conn:
        conn.execute(text(
            f"CREATE TABLE IF NOT EXISTS {LIFESTYLE_TABLE} ("
            "patient_id VARCHAR(32) NOT NULL PRIMARY KEY, "
            "smoking VARCHAR(32) NULL, alcohol VARCHAR(32) NULL, family_history VARCHAR(48) NULL, "
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
            ") CHARACTER SET utf8mb4"))
    return True


def _clean_lifestyle(smoking, alcohol, family_history) -> dict:
    """Лише непорожні та допустимі значення."""
    out = {}
    for key, value in {"smoking": smoking, "alcohol": alcohol, "family_history": family_history}.items():
        if not value:
            continue
        if value not in LIFESTYLE_CHOICES[key]:
            raise HTTPException(422, f"{key}: недопустиме значення '{value}'")
        out[key] = value
    return out


def _save_lifestyle(conn, pid: str, fields: dict):
    if not fields:
        return
    cols = ["patient_id", *fields]  # ключі — з фіксованого whitelist
    conn.execute(text(
        f"INSERT INTO {LIFESTYLE_TABLE} ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)}) "
        f"ON DUPLICATE KEY UPDATE {', '.join(f'{k} = VALUES({k})' for k in fields)}"),
        {"patient_id": pid, **fields})


def _get_lifestyle(pid: str) -> dict:
    """Реальні дані, введені користувачем; якщо їх немає — 'Не вказано' (нічого не вигадуємо)."""
    out = {k: NOT_SET for k in LIFESTYLE_CHOICES}
    try:
        _ensure_lifestyle_table()
        with engine.connect() as conn:
            row = conn.execute(text(
                f"SELECT smoking, alcohol, family_history FROM {LIFESTYLE_TABLE} WHERE patient_id = :p"),
                {"p": pid}).mappings().first()
        if row:
            out.update({k: row[k] for k in LIFESTYLE_CHOICES if row[k]})
    except Exception:
        logger.exception("lifestyle read")
    return out


def _next_id_from(ids) -> str:
    digits = ids.dropna().astype(str).str.replace(r"\D", "", regex=True)
    nums = pd.to_numeric(digits[digits != ""], errors="coerce").dropna()
    return f"P{(int(nums.max()) + 1 if not nums.empty else 1):07d}"


def _next_patient_id(conn) -> str:
    return _next_id_from(pd.read_sql(text(f"SELECT DISTINCT patient_id FROM {TABLE}"), con=conn)["patient_id"])


def _build_clinical_summary(hba1c, prev_hba1c, glucose_mmol, predicted_hba1c,
                            bmi, bp, smoking, diabetic=False):
    conclusions, actions = [], []
    diff = hba1c - prev_hba1c

    # Для пацієнта з ЦД2 орієнтир — цільові значення (HbA1c < 7%, глюкоза натще ≤ 7.2),
    # а не діагностичні пороги, тому "предіабет" для нього не виводиться.
    off_target = (hba1c >= 7.0 or glucose_mmol > 7.2) if diabetic else (hba1c >= 6.5 or glucose_mmol >= 7.0)
    if off_target:
        if diff > 0.3:
            conclusions.append(f"<b>Глікемія:</b> Декомпенсація (HbA1c {hba1c:.1f}%, +{diff:.1f}% від минулого візиту).")
            actions.append("Негайна інтенсифікація медикаментозного лікування.")
        elif diff < -0.3:
            conclusions.append(f"<b>Глікемія:</b> Перевищення цільового HbA1c ({hba1c:.1f}%), але наявна позитивна динаміка ({diff:.1f}%).")
            actions.append("Продовжувати поточну сахарознижувальну терапію з моніторингом через 3 міс.")
        else:
            conclusions.append(f"<b>Глікемія:</b> Стійка гіперглікемія (HbA1c {hba1c:.1f}%, глюкоза {glucose_mmol:.2f} ммоль/л).")
            actions.append("Корекція дозування препаратів (Метформін / іСГЛT-2).")
    elif diabetic:
        conclusions.append(f"<b>Глікемія:</b> Компенсація ЦД 2 типу (HbA1c {hba1c:.1f}%, глюкоза {glucose_mmol:.2f} ммоль/л — у цільовому діапазоні).")
        actions.append("Продовжувати поточну терапію, контроль HbA1c кожні 3–6 міс.")
    elif hba1c >= 5.7 or glucose_mmol >= 5.6:
        conclusions.append(f"<b>Глікемія:</b> Предиабетичний стан / Порушення толерантності (HbA1c {hba1c:.1f}%).")
        actions.append("Призначити низьковуглеводну дієту (Стіл №9) та 150 хв/тиж аеробних навантажень.")
    else:
        conclusions.append(f"<b>Глікемія:</b> Нормоглікемія (HbA1c {hba1c:.1f}%, Глюкоза {glucose_mmol:.2f} ммоль/л).")
        actions.append("Плановий профілактичний огляд за рекомендованим графіком.")

    if predicted_hba1c > hba1c + 0.2:
        conclusions.append(f"<b>ML Прогноз:</b> Траєкторія підвищення HbA1c до {predicted_hba1c:.1f}% у наступному періоді.")
        actions.append("Посилити контроль модифікації способу життя для запобігання прогресуванню.")

    if bp >= 140 and bmi >= 30:
        conclusions.append(f"<b>Коморбідність:</b> Поєднання ожиріння (ІМТ {bmi:.1f}) та гіпертензії ({bp:.0f} мм рт.ст.).")
        actions.append("Консультація кардіолога, підбір антигіпертензивної терапії (іАПФ/БРА).")
    elif bp >= 130:
        conclusions.append(f"<b>Судинний ризик:</b> Погранична артеріальна гіпертензія ({bp:.0f} мм рт.ст.).")
        actions.append("Добовий моніторинг АТ, обмеження солі в раціоні.")

    if smoking == "Активний курець":
        conclusions.append("<b>Спосіб життя:</b> Тютюнопаління як високий фактор мікроангіопатії.")
        actions.append("Залучення до програми відмови від тютюнопаління.")

    return conclusions, actions


def _risk_assessment(risk_prob: float, hba1c: float, glucose_mmol: float):
    """Keep the model alert band separate from diagnostic-range laboratory findings."""
    auc = (t1_forecast or {}).get("metrics", {}).get("roc_auc")
    if risk_prob is None or not np.isfinite(risk_prob):
        return "Недостатньо даних", "Прогноз не розраховано: потрібні наявні вимірювання глюкози натще та HbA1c."
    above_threshold = risk_prob >= RISK_LOW
    level = "Понад поріг сповіщення" if above_threshold else "Нижче порога сповіщення"
    if auc is not None and auc < 0.60:
        level += " · низька точність на тесті"
        message = (f"Прогноз до очікуваного наступного візиту: {risk_prob * 100:.1f}%. "
                   f"Часовий тест ROC-AUC {auc:.2f}; модель має обмежену розрізнювальну здатність.")
        if hba1c >= 6.5 or glucose_mmol >= 7.0:
            message += (" Окремо: лабораторний результат у діагностичному діапазоні; "
                        "потрібні клінічна оцінка й підтвердження.")
        return level, message
    if hba1c >= 6.5 or glucose_mmol >= 7.0:
        recommendation = (
            "Окремо: лабораторний результат у діагностичному діапазоні. "
            "Потрібна клінічна оцінка й підтвердження; прогноз Tier 1 діагноз не встановлює."
        )
    elif above_threshold:
        recommendation = (
            "Оцінка перевищила внутрішній поріг скринінгового сповіщення. "
            "Розгляньте лабораторне обстеження та клінічну оцінку."
        )
    else:
        recommendation = (
            "Оцінка нижче внутрішнього порога сповіщення. Це не виключає діабет "
            "і не замінює лабораторне обстеження за показаннями."
        )
    return level, recommendation


def _trend_forecast(values):
    """Лінійна екстраполяція на наступний візит за останніми значеннями (не ML-модель)."""
    pts = [float(v) for v in values if v is not None and not pd.isna(v)]
    if len(pts) < 2:
        return None
    slope, intercept = np.polyfit(np.arange(len(pts)), pts, 1)
    return max(float(slope * len(pts) + intercept), 0.0)


def _complication_scores(hba1c, prev_hba1c, glucose_mmol, bp, bmi, trig_mmol, age, predicted_hba1c):
    """Бальна оцінка (0-100) мікро- та макросудинного ризику за клінічними порогами.

    Це прозорий розрахунок за правилами, а не навчена ML-модель: у даних немає міток ускладнень.
    """
    micro = macro = 0
    factors = []  # (бали, текст)

    def add(kind, points, text=None):
        nonlocal micro, macro
        if kind == "micro":
            micro += points
        else:
            macro += points
        if text and points > 0:
            factors.append((points, text))

    # Глікемія
    hba1c_pts = 50 if hba1c >= 10 else 40 if hba1c >= 9 else 30 if hba1c >= 8 else 15 if hba1c >= 7 else 0
    add("micro", hba1c_pts, f"HbA1c {hba1c:.1f}% (ціль до 7%)")
    add("macro", 10 if hba1c >= 9 else 5 if hba1c >= 8 else 0)
    gl_pts = 15 if glucose_mmol >= 11 else 10 if glucose_mmol >= 9 else 5 if glucose_mmol >= 7 else 0
    add("micro", gl_pts, f"глюкоза натще {glucose_mmol:.1f} ммоль/л")

    # Динаміка
    if hba1c - prev_hba1c > 0.3:
        add("micro", 10, f"зростання HbA1c (+{hba1c - prev_hba1c:.1f}%)")
    if predicted_hba1c is not None and predicted_hba1c > hba1c + 0.2:
        add("micro", 10, f"прогноз підвищення HbA1c до {predicted_hba1c:.1f}%")

    # Артеріальний тиск
    add("macro", 40 if bp >= 160 else 30 if bp >= 140 else 15 if bp >= 130 else 0,
        f"систолічний АТ {bp:.0f} мм рт.ст.")
    add("micro", 10 if bp >= 140 else 5 if bp >= 130 else 0)

    # Вага, ліпіди, вік
    add("macro", 20 if bmi >= 35 else 15 if bmi >= 30 else 5 if bmi >= 25 else 0,
        f"ІМТ {bmi:.1f} кг/м²")
    if trig_mmol is not None and np.isfinite(trig_mmol):
        add("macro", 10 if trig_mmol >= 2.3 else 5 if trig_mmol >= 1.7 else 0,
            f"тригліцериди {trig_mmol:.1f} ммоль/л")
    add("macro", 15 if age >= 75 else 10 if age >= 60 else 5 if age >= 45 else 0, f"вік {age} р.")

    micro, macro = min(micro, 100), min(macro, 100)
    overall = round(0.7 * max(micro, macro) + 0.3 * min(micro, macro))
    factors.sort(key=lambda f: f[0], reverse=True)
    return micro, macro, overall, [text for _, text in factors]


def _complication_assessment(hba1c, prev_hba1c, glucose_mmol, bp, bmi, trig_mmol, age,
                             predicted_hba1c, glucose_next, bp_next):
    micro, macro, score, factors = _complication_scores(
        hba1c, prev_hba1c, glucose_mmol, bp, bmi, trig_mmol, age, predicted_hba1c)
    # Прогноз на наступний візит: прогнозні HbA1c (ML), глюкоза й АТ (тренд) замість поточних
    _, _, forecast_score, _ = _complication_scores(
        predicted_hba1c, hba1c, glucose_next, bp_next, bmi, trig_mmol, age, None)

    if score < 25:
        level, rec = "Низький індекс судинного ризику", "Оцінку слід розглядати разом із лікарем; правила не валідовані."
    elif score < 50:
        level, rec = "Помірний індекс судинного ризику", "Обговоріть показники з лікарем; правила не визначають лікування."
    else:
        level = "Високий індекс судинного ризику"
        rec = ("Індекс судинного ризику високий. Обговоріть із лікарем потребу в обстеженні; "
               "правила не валідовані й не визначають лікування.")

    top = ", ".join(factors[:3]) if factors else "значущих факторів ризику не виявлено"
    summary = (f"<b>Індекс судинного ризику:</b> {score}% "
               f"(мікросудинний {micro}%, макросудинний {macro}%); "
               f"сценарний індекс на наступний візит — {forecast_score}%. "
               f"Це експертна шкала 0–100, не ймовірність події й не валідована шкала. "
               f"Тригліцериди {'виміряні' if trig_mmol is not None else 'не виміряні; їхній внесок не враховано'}. "
               f"Основні фактори: {top}.")
    return {"score": score, "micro": micro, "macro": macro, "forecast_score": forecast_score,
            "factors": factors[:3], "level": level, "recommendation": rec, "summary": summary}


# ───────────────────────── Ендпоїнти ─────────────────────────
@app.get("/")
def serve_ui():
    html_path = BASE_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Файл index.html не знайдено")
    return FileResponse(html_path, headers={"Cache-Control": "no-cache"})


@app.get("/api/patients/all")
def get_all_patients_for_lab():
    try:
        query = text(f"""
            SELECT patient_id, MAX(age) AS age, MAX(sex) AS sex,
                   MAX(dx_type2_diabetes) AS dx_type2_diabetes, MAX(test_date) AS test_date
            FROM {TABLE}
            GROUP BY patient_id
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, con=conn)

        patients = [{
            "patient_id": str(r.patient_id),
            "age": int(float(r.age)),
            "sex": "M" if pd.isna(r.sex) else str(r.sex),
            "dx_type2_diabetes": int(_num(r.dx_type2_diabetes, 0)),
            "last_visit": "" if pd.isna(r.test_date) else str(r.test_date),
        } for r in df.itertuples(index=False)]

        return {
            "patients": sorted(patients, key=lambda x: x["patient_id"]),
            "next_patient_id": _next_id_from(df["patient_id"]),
        }
    except Exception as e:
        logger.exception("patients/all")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/patients")
def get_patients_list(mode: str = "risk", q: str = ""):
    try:
        if mode not in ("risk", "complication"):
            raise HTTPException(422, "mode: допустимі значення risk або complication")
        eligible = _eligible_patient_rows(mode)
        query = q.strip().lower()
        if query:
            eligible = eligible[eligible["patient_id"].str.lower().str.contains(query, regex=False)]
        eligible = eligible.sort_values("patient_id").head(300)

        return [{
            "patient_id": str(r.patient_id),
            "age": int(float(r.age)),
            "sex": "M" if pd.isna(r.sex) else str(r.sex),
            "visits": int(r.visits),
            "status": "Повні аналізи · наступний візит " + r.next_visit_date,
            "next_visit_date": r.next_visit_date,
        } for r in eligible.itertuples(index=False)]
    except Exception as e:
        logger.exception("patients list")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/patient/add_record")
def add_patient_record(record: LabRecordInput):
    try:
        if record.sex not in ("M", "F"):
            raise HTTPException(422, "sex: допустимі значення M або F")
        life = _clean_lifestyle(record.smoking, record.alcohol, record.family_history)

        cols = _table_columns()
        required = {"patient_id", "test_date", "hba1c", "glucose_fasting"}
        if required - cols.keys():
            raise HTTPException(500, f"У таблиці {TABLE} немає обов'язкових колонок: "
                                     f"{', '.join(sorted(required - cols.keys()))}")
        if life:
            _ensure_lifestyle_table()

        with _id_lock, engine.begin() as conn:
            pid = record.patient_id.strip()
            exists = conn.execute(text(f"SELECT 1 FROM {TABLE} WHERE patient_id = :p LIMIT 1"),
                                  {"p": pid}).first()
            changed = False
            if record.patient_mode == "new" and exists:  # ID вже зайнятий — беремо наступний вільний
                pid, changed = _next_patient_id(conn), True
            elif record.patient_mode != "new" and not exists:
                raise HTTPException(404, f"Пацієнта {pid} не знайдено")

            values = {
                "patient_id": pid, "age": record.age, "sex": record.sex,
                "dx_type2_diabetes": record.dx_type2_diabetes, "test_date": record.test_date,
                "hba1c": record.hba1c,
                "glucose_fasting": record.glucose_fasting_mmol * GLUCOSE_MMOL_TO_MGDL,
                "bmi": record.bmi, "systolic_bp": record.systolic_bp,
                "diastolic_bp": record.diastolic_bp,
                "triglycerides": record.triglycerides * TRIGLYCERIDES_MMOL_TO_MGDL,
            }
            # Вставляємо лише в колонки, які реально існують у таблиці
            used = {k: v for k, v in values.items() if k in cols}
            if len(used) < len(values):
                logger.warning("Пропущено колонки, яких немає в БД: %s", sorted(set(values) - set(used)))
            conn.execute(text(
                f"INSERT INTO {TABLE} ({', '.join(f'`{cols[k]}`' for k in used)}) "
                f"VALUES ({', '.join(f':{k}' for k in used)})"), used)
            _save_lifestyle(conn, pid, life)

        _cohort_cache.clear()  # нові дані — оновлюємо прогноз і списки пацієнтів
        with _patient_rows_lock:
            _patient_rows_cache.clear()
        msg = f"Новий запис для пацієнта {pid} збережено!"
        if changed:
            msg += f" (ID {record.patient_id} уже зайнятий, призначено {pid})"
        return {"status": "success", "patient_id": pid, "message": msg}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("add_record")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/patient/{patient_id}/lifestyle")
def update_lifestyle(patient_id: str, data: LifestyleInput):
    try:
        life = _clean_lifestyle(data.smoking, data.alcohol, data.family_history)
        _ensure_lifestyle_table()
        with engine.begin() as conn:
            if not conn.execute(text(f"SELECT 1 FROM {TABLE} WHERE patient_id = :p LIMIT 1"),
                                {"p": patient_id}).first():
                raise HTTPException(404, "Пацієнта не знайдено")
            _save_lifestyle(conn, patient_id, life)
        return {"status": "success", "lifestyle": _get_lifestyle(patient_id)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("update lifestyle")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/patient/{patient_id}/history")
def get_patient_history(patient_id: str, mode: str = "risk"):
    try:
        query = text(f"SELECT * FROM {TABLE} WHERE patient_id = :pid ORDER BY test_date ASC")
        with engine.connect() as conn:
            p_df = pd.read_sql(query, con=conn, params={"pid": patient_id})
        p_df = _normalize_columns(p_df)

        if p_df.empty:
            raise HTTPException(status_code=404, detail="Пацієнта не знайдено")

        eligible_latest = _complete_latest_visits(p_df)
        if eligible_latest.empty:
            raise HTTPException(422, "Прогноз недоступний: немає повного набору аналізів і достатньої історії дат візитів.")
        latest = eligible_latest.iloc[0].to_dict()
        next_visit_date = latest["next_visit_date"]
        days_to_next_visit = int(latest["days_to_next_visit"])
        prev = p_df.iloc[-2].to_dict() if len(p_df) > 1 else latest

        model_features = pd.DataFrame([{**{k: latest[k] for k in REQUIRED_FORECAST_FIELDS},
                                         "days_to_next_visit": days_to_next_visit}])
        risk_raw = _t1_proba(model_features)[0] if mode == "risk" else None
        risk_prob = float(risk_raw) if risk_raw is not None and np.isfinite(risk_raw) else None
        risk_model_auc = (t1_forecast or {}).get("metrics", {}).get("roc_auc") if mode == "risk" else None
        risk_usable = risk_prob is not None
        predicted_hba1c = float(_t2_predict(model_features)[0])

        missing_cols = [c for c in ("HbA1c", "triglycerides") if c not in p_df.columns]
        if missing_cols:
            logger.warning("У таблиці %s немає колонок %s. Наявні колонки: %s", TABLE, missing_cols, list(p_df.columns))
        hba1c_val = float(latest["HbA1c"])
        prev_hba1c = _num(latest.get("previous_hba1c"), hba1c_val)
        glucose_mmol = float(latest["glucose_fasting"]) / GLUCOSE_MMOL_TO_MGDL
        bmi_val = float(latest["bmi"])
        bp_val = float(latest["systolic_bp"])
        age_val = int(float(latest["age"]))
        trig_mmol_real = _trig_to_mmol(float(latest["triglycerides"]))
        trig_mmol_profile = None if trig_mmol_real is None else round(trig_mmol_real, 2)

        quality = 100

        tail = p_df.tail(8)
        glucose_trend = _trend_forecast(tail["glucose_fasting"] / GLUCOSE_MMOL_TO_MGDL)
        # Blood pressure comes from the patient profile, not dated encounter
        # measurements, so it is intentionally not forecast as a time series.
        bp_trend = None

        lifestyle = _get_lifestyle(patient_id)
        conclusions, actions = _build_clinical_summary(
            hba1c_val, prev_hba1c, glucose_mmol, predicted_hba1c,
            bmi_val, bp_val, lifestyle["smoking"], diabetic=(mode != "risk"))
        risk_level, rec = _risk_assessment(risk_prob, hba1c_val, glucose_mmol) if mode == "risk" else ("", "")

        complication = None
        if mode != "risk":  # для підтвердженого ЦД2 оцінюємо ускладнення, а не ймовірність діабету
            complication = _complication_assessment(
                hba1c_val, prev_hba1c, glucose_mmol, bp_val, bmi_val,
                trig_mmol_real, age_val, predicted_hba1c,
                glucose_trend if glucose_trend is not None else glucose_mmol,
                bp_trend if bp_trend is not None else bp_val)
            risk_level, rec = complication["level"], complication["recommendation"]
            conclusions.append(complication["summary"])

        dates = pd.to_datetime(p_df["test_date"])
        p_df["days_from_start"] = (dates - dates.min()).dt.days
        if mode == "risk":
            p_df["formatted_date"] = dates.dt.strftime("%m.%Y")
        else:
            p_df["formatted_date"] = (dates.dt.strftime("%d.%m.%y") + " (+"
                                      + p_df["days_from_start"].astype(str) + "д)")

        recent = p_df.tail(8).copy()
        recent["glucose_fasting"] = recent["glucose_fasting"] / GLUCOSE_MMOL_TO_MGDL
        cols = ['test_date', 'formatted_date', 'days_from_start', 'HbA1c',
                'glucose_fasting']
        hist = recent[[c for c in cols if c in recent.columns]]
        history = [{k: (None if pd.isna(v) else v) for k, v in row.items()}  # пропуски = null, а не 0
                   for row in hist.to_dict(orient="records")]

        return {
            "patient_id": patient_id,
            "is_diabetic": int(_num(latest.get("dx_type2_diabetes"), 0)) == 1,
            "profile": {
                "age": age_val,
                "sex": str(latest.get("sex") or "M"),
                "bmi": bmi_val,
                "glucose_fasting": glucose_mmol,
                "HbA1c": hba1c_val,
                "systolic_bp": bp_val,
                "triglycerides": trig_mmol_profile,
            },
            "data_quality": quality,
            "next_visit_date": next_visit_date,
            "days_to_next_visit": days_to_next_visit,
            "lifestyle": lifestyle,
            "ml_predictions": {
                "risk_score": complication["score"] if complication else (round(risk_prob * 100, 2) if risk_usable else None),
                "tier1_probability": None if risk_prob is None else round(risk_prob * 100, 2),
                "risk_level": risk_level,
                "risk_threshold": round(RISK_LOW * 100, 1) if risk_usable else None,
                "risk_horizon_days": days_to_next_visit if risk_usable else None,
                "risk_model_auc": risk_model_auc,
                "predicted_hba1c": round(predicted_hba1c, 2),
                "hba1c_test_mae": (t2_next_visit or {}).get("metrics", {}).get("mae"),
                "hba1c_test_n": (t2_next_visit or {}).get("metrics", {}).get("n_test"),
                "model_version": "next_visit" if t1_forecast and t2_next_visit else ("v2" if (t1_v2 or t2_v2) else "v1"),
                "recommendation": rec,
                "complication": complication,
            },
            "trend_forecasts": {
                "glucose_fasting": None if glucose_trend is None else round(glucose_trend, 2),
                "systolic_bp": None if bp_trend is None else round(bp_trend),
            },
            "clinical_summary": {"conclusions": conclusions, "recommended_actions": actions},
            "history": history,
        }
    except HTTPException:  # 404 більше не перетворюється на 500
        raise
    except Exception as e:
        logger.exception("patient history")
        raise HTTPException(status_code=500, detail=str(e))


# ───────────────────────── Прогноз для всіх пацієнтів ─────────────────────────
COHORT_TTL_SEC = 1800  # кеш на 30 хв; скидається при додаванні запису, кнопка "Перерахувати" оновлює вручну
_cohort_cache: dict = {}
_cohort_lock = Lock()  # паралельні запити не рахують те саме двічі
PATIENT_ROWS_TTL_SEC = 300
_patient_rows_cache: dict = {}
_patient_rows_lock = Lock()
_LEVEL_GROUPS = ("Низький", "Помірний", "Високий", "Недостатньо даних", "Модель не пройшла перевірку")


def _risk_group(level: str) -> str:
    if "не пройшла перевірку" in level:
        return "Модель не пройшла перевірку"
    if "Недостатньо даних" in level:
        return "Недостатньо даних"
    if "Високий" in level:
        return "Високий"
    if "Понад поріг" in level or "Помірний" in level:
        return "Помірний"
    return "Низький"


_COHORT_COLS = {"patient_id", "test_date", "age", "sex", "HbA1c", "glucose_fasting",
                "bmi", "systolic_bp", "triglycerides"}


def _load_cohort_df(conn, mode: str) -> pd.DataFrame:
    """Читаємо лише потрібні колонки (а не SELECT * по 180 тис. рядків)."""
    real = [r for low, r in _table_columns().items() if _COLUMN_ALIASES.get(low) in _COHORT_COLS]
    cols = ", ".join(f"`{c}`" for c in real) or "*"
    return pd.read_sql(text(f"SELECT {cols} FROM {TABLE} WHERE dx_type2_diabetes = :dx"),
                       con=conn, params={"dx": 0 if mode == "risk" else 1})


def _eligible_patient_rows(mode: str, refresh: bool = False) -> pd.DataFrame:
    """Load and filter a cohort once; the selector search reuses this prepared list."""
    if mode not in ("risk", "complication"):
        raise HTTPException(422, "mode: допустимі значення risk або complication")
    with _patient_rows_lock:
        cached = _patient_rows_cache.get(mode)
        if not refresh and cached and time.time() - cached[0] < PATIENT_ROWS_TTL_SEC:
            return cached[1].copy()
        with engine.connect() as conn:
            df = _load_cohort_df(conn, mode)
        eligible = _complete_latest_visits(_normalize_columns(df))
        _patient_rows_cache[mode] = (time.time(), eligible)
        return eligible.copy()


def _cached_cohort(mode: str):
    cached = _cohort_cache.get(mode)
    if cached and time.time() - cached[0] < COHORT_TTL_SEC:
        return cached[1]
    return None


@app.get("/api/forecast/all")
def forecast_all_patients(mode: str = "risk", refresh: bool = False):
    """Прогноз HbA1c (Tier 2) та ризику для всіх пацієнтів вкладки — за останнім візитом кожного."""
    if mode not in ("risk", "complication"):
        raise HTTPException(422, "mode: допустимі значення risk або complication")
    if not refresh:
        hit = _cached_cohort(mode)
        if hit is not None:
            return hit
    with _cohort_lock:
        if not refresh:  # поки чекали на lock, інший запит (або прогрів) міг уже порахувати
            hit = _cached_cohort(mode)
            if hit is not None:
                return hit
        return _compute_cohort(mode)


def _compute_cohort(mode: str):
    try:
        started = time.time()
        df = _eligible_patient_rows(mode)
        auc = (t1_forecast or {}).get("metrics", {}).get("roc_auc") if mode == "risk" else None
        if df.empty:
            result = {"mode": mode, "total": 0, "patients": [], "risk_model_auc": auc,
                      "summary": {"mean_hba1c": None, "mean_forecast_hba1c": None,
                                  "worsening": 0, "improving": 0,
                                  "tier1_alerts": 0, "tier1_threshold": round(RISK_LOW * 100, 1),
                                  "levels": {k: 0 for k in _LEVEL_GROUPS}}}
            _cohort_cache[mode] = (time.time(), result)
            return result

        # Forecast both outcomes from the same six fully measured fields and personal horizon.
        features = pd.DataFrame([{**{k: row[k] for k in REQUIRED_FORECAST_FIELDS},
                                  "days_to_next_visit": row["days_to_next_visit"]}
                                 for _, row in df.iterrows()])
        probs = _t1_proba(features) if mode == "risk" else np.full(len(features), np.nan)
        predicted = _t2_predict(features)
        rows = []
        for i, row in df.reset_index(drop=True).iterrows():
            pid = str(row["patient_id"])
            hba1c = float(row["HbA1c"])
            glucose = float(row["glucose_fasting"]) / GLUCOSE_MMOL_TO_MGDL
            bp, bmi, age = float(row["systolic_bp"]), float(row["bmi"]), int(float(row["age"]))
            tg = float(row["triglycerides"])
            forecast = float(predicted[i])
            if mode == "risk":
                probability = float(probs[i])
                level, _ = _risk_assessment(probability, hba1c, glucose)
                risk = round(probability * 100, 1)
                risk_forecast = None
                tier1_alert = bool(probability >= RISK_LOW)
            else:
                comp = _complication_assessment(
                    hba1c, _num(row.get("previous_hba1c"), hba1c), glucose,
                    bp, bmi, _trig_to_mmol(tg), age, forecast, glucose, bp)
                level, risk = comp["level"], float(comp["score"])
                risk_forecast = float(comp["forecast_score"])
                tier1_alert = None
            rows.append({
                "patient_id": pid, "age": age,
                "sex": "M" if pd.isna(row.get("sex")) else str(row.get("sex")),
                "visits": int(row["visits"]),
                "next_visit_date": str(row["next_visit_date"]),
                "days_to_next_visit": int(row["days_to_next_visit"]),
                "hba1c": round(hba1c, 2),
                "forecast_hba1c": round(forecast, 2),
                "delta": round(forecast - hba1c, 2),
                "risk": round(risk, 1),
                "tier1_probability": (round(float(probs[i]) * 100, 1)
                                      if mode == "risk" and np.isfinite(probs[i]) else None),
                "tier1_alert": tier1_alert,
                "risk_forecast": None if risk_forecast is None else round(risk_forecast, 1),
                "level": level, "group": _risk_group(level),
            })

        # Спочатку за рівнем (лабораторні критерії можуть підвищити рівень над ML-відсотком), потім за відсотком
        rank = {"Високий": 2, "Помірний": 1, "Низький": 0,
                "Недостатньо даних": -1, "Модель не пройшла перевірку": -2}
        rows.sort(key=lambda r: (rank[r["group"]], r["risk"] or 0), reverse=True)
        n = len(rows)
        result = {
            "mode": mode,
            "total": n,
            "risk_model_auc": auc,
            "summary": {
                "mean_hba1c": round(sum(r["hba1c"] for r in rows) / n, 2),
                "mean_forecast_hba1c": round(sum(r["forecast_hba1c"] for r in rows) / n, 2),
                "worsening": sum(r["delta"] > 0.2 for r in rows),
                "improving": sum(r["delta"] < -0.2 for r in rows),
                "tier1_alerts": sum(r["tier1_alert"] is True for r in rows),
                "tier1_threshold": round(RISK_LOW * 100, 1),
                "levels": {k: sum(r["group"] == k for r in rows) for k in _LEVEL_GROUPS},
            },
            "patients": rows,
        }
        logger.info("forecast all (%s): %d пацієнтів за %.1f с", mode, n, time.time() - started)
        _cohort_cache[mode] = (time.time(), result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("forecast all")
        raise HTTPException(status_code=500, detail=str(e))


def _warm_cohort_cache():
    """Прогріваємо кеш одразу після старту, щоб перше відкриття панелі не чекало розрахунку."""
    for m in ("risk", "complication"):
        try:
            with _cohort_lock:
                _compute_cohort(m)
        except Exception:
            logger.exception("прогрів кешу (%s)", m)


@app.on_event("startup")
def _startup_warmup():
    Thread(target=_warm_cohort_cache, daemon=True).start()
