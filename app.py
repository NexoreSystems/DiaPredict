"""Endocrinology Decision Support System API (FastAPI + MySQL + ML-моделі)."""
import logging
import mimetypes
import os
import time
from functools import lru_cache
from datetime import date
from pathlib import Path
from threading import Lock
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL

try:  # .env необов'язковий: pip install python-dotenv
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("dss")

# Фікс для Windows/MIME-type
mimetypes.add_type("application/javascript", ".js", True)

# ───────────────────────── Конфігурація ─────────────────────────
BASE_DIR = Path(__file__).resolve().parent
TABLE = "patient_records"
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

class NoCacheStaticFiles(StaticFiles):
    """Браузер перевіряє файли при кожному запиті, тож зміни в JS підхоплюються одразу."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


js_dir = BASE_DIR / "js"
if js_dir.exists():
    app.mount("/js", NoCacheStaticFiles(directory=js_dir), name="js")

t1_model = joblib.load(BASE_DIR / "tier1_diabetes_risk.joblib")
t2_model = joblib.load(BASE_DIR / "tier2_hba1c_forecast.joblib")

T1_DEFAULTS = {"age": 50, "glucose_fasting": 100, "HbA1c": 6.0,
               "bmi": 25, "systolic_bp": 120, "triglycerides": 150}
T2_DEFAULTS = {"HbA1c": 6.0, "glucose_fasting": 100, "bmi": 25, "age": 50}


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
    triglycerides: float = Field(default=1.7, ge=0.1, le=50)
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


def _features(latest: dict, defaults: dict) -> pd.DataFrame:
    return pd.DataFrame([{k: _num(latest.get(k), d) for k, d in defaults.items()}])


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
        actions.append("Плановий профілактичний огляд через 12 місяців.")

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
    """Модуль 1. Лабораторні критерії діабету мають пріоритет над ML-ймовірністю."""
    if hba1c >= 6.5 or glucose_mmol >= 7.0:
        return ("Високий ризик (лабораторні ознаки ЦД)",
                "Показники в діапазоні діабету: потрібне підтверджувальне обстеження та консультація ендокринолога.")
    if risk_prob < 0.3:
        return "Низький ризик", "Показники в межах норми."
    if risk_prob < 0.6:
        return "Помірний ризик", "Модифікація способу життя."
    return "Високий ризик", "Високий ризик розвитку діабету!"


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
        level, rec = "Низький ризик ускладнень", "Стабільний стан, контроль через 6 міс."
    elif score < 50:
        level, rec = "Помірний ризик ангіопатії", "Потрібна корекція терапії для профілактики ускладнень."
    else:
        level = "Високий судинний ризик"
        rec = ("Високий ризик ускладнень: поглиблене обстеження (очне дно, альбумінурія/ШКФ, ЕКГ) "
               "та перегляд терапії.")

    top = ", ".join(factors[:3]) if factors else "значущих факторів ризику не виявлено"
    summary = (f"<b>Ризик ускладнень:</b> {score}% (мікросудинні {micro}%, макросудинні {macro}%); "
               f"прогноз на наступний візит — {forecast_score}%. Оцінка за клінічними правилами, не калібрована ймовірність. "
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
            "age": int(_num(r.age, 50)),
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
        query = text(f"""
            SELECT patient_id, MAX(age) AS age, MAX(sex) AS sex, COUNT(*) AS visits
            FROM {TABLE}
            WHERE dx_type2_diabetes = :dx_val AND patient_id LIKE :q
            GROUP BY patient_id
            ORDER BY patient_id
            LIMIT 300
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, con=conn, params={"dx_val": 0 if mode == "risk" else 1,
                                                 "q": f"%{q.strip()}%"})

        status = "Група ризику / Скринінг" if mode == "risk" else "Підтверджений ЦД 2 типу"
        return [{
            "patient_id": str(r.patient_id),
            "age": int(_num(r.age, 50)),
            "sex": "M" if pd.isna(r.sex) else str(r.sex),
            "visits": int(r.visits),
            "status": status,
        } for r in df.itertuples(index=False)]
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

        _cohort_cache.clear()  # нові дані — прогноз для всіх пацієнтів треба перерахувати
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

        if p_df.empty:
            raise HTTPException(status_code=404, detail="Пацієнта не знайдено")

        latest = p_df.iloc[-1].to_dict()
        prev = p_df.iloc[-2].to_dict() if len(p_df) > 1 else latest

        # Tier 1 навчена на скринінгу — для пацієнтів із ЦД2 (модуль 2) не застосовується
        risk_prob = float(t1_model.predict_proba(_features(latest, T1_DEFAULTS))[0][1]) if mode == "risk" else None
        predicted_hba1c = float(t2_model.predict(_features(latest, T2_DEFAULTS))[0])

        hba1c_val = _num(latest.get("HbA1c"), 6.0)
        prev_hba1c = _num(prev.get("HbA1c"), hba1c_val)
        glucose_mmol = _num(latest.get("glucose_fasting"), 100) / GLUCOSE_MMOL_TO_MGDL
        bmi_val = _num(latest.get("bmi"), 25)
        bp_val = _num(latest.get("systolic_bp"), 120)
        age_val = int(_num(latest.get("age"), 50))
        trig_val = _num(latest.get("triglycerides"), 0)
        trig_raw = _num(latest.get("triglycerides"), None)
        trig_mmol_profile = None if trig_raw is None else round(trig_raw / TRIGLYCERIDES_MMOL_TO_MGDL, 2)

        key_fields = ("age", "glucose_fasting", "HbA1c", "bmi", "systolic_bp", "triglycerides")
        quality = round(100 * sum(_num(latest.get(k), None) is not None for k in key_fields) / len(key_fields))

        tail = p_df.tail(8)
        glucose_trend = _trend_forecast(tail["glucose_fasting"] / GLUCOSE_MMOL_TO_MGDL)
        bp_trend = _trend_forecast(tail["systolic_bp"])

        lifestyle = _get_lifestyle(patient_id)
        conclusions, actions = _build_clinical_summary(
            hba1c_val, prev_hba1c, glucose_mmol, predicted_hba1c,
            bmi_val, bp_val, lifestyle["smoking"], diabetic=(mode != "risk"))
        risk_level, rec = _risk_assessment(risk_prob, hba1c_val, glucose_mmol) if mode == "risk" else ("", "")

        complication = None
        if mode != "risk":  # для підтвердженого ЦД2 оцінюємо ускладнення, а не ймовірність діабету
            complication = _complication_assessment(
                hba1c_val, prev_hba1c, glucose_mmol, bp_val, bmi_val,
                trig_val / TRIGLYCERIDES_MMOL_TO_MGDL, age_val, predicted_hba1c,
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
                'glucose_fasting', 'bmi', 'systolic_bp']
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
            "lifestyle": lifestyle,
            "ml_predictions": {
                "risk_score": complication["score"] if complication else round(risk_prob * 100, 2),
                "risk_level": risk_level,
                "predicted_hba1c": round(predicted_hba1c, 2),
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
COHORT_TTL_SEC = 120  # короткий кеш, щоб повторне відкриття панелі не перераховувало всю базу
_cohort_cache: dict = {}
_LEVEL_GROUPS = ("Низький", "Помірний", "Високий")


def _risk_group(level: str) -> str:
    return "Високий" if "Високий" in level else "Помірний" if "Помірний" in level else "Низький"


@app.get("/api/forecast/all")
def forecast_all_patients(mode: str = "risk", refresh: bool = False):
    """Прогноз HbA1c (Tier 2) та ризику для всіх пацієнтів вкладки — за останнім візитом кожного."""
    if mode not in ("risk", "complication"):
        raise HTTPException(422, "mode: допустимі значення risk або complication")
    cached = _cohort_cache.get(mode)
    if cached and not refresh and time.time() - cached[0] < COHORT_TTL_SEC:
        return cached[1]

    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(f"SELECT * FROM {TABLE} WHERE dx_type2_diabetes = :dx"),
                             con=conn, params={"dx": 0 if mode == "risk" else 1})

        if df.empty:
            result = {"mode": mode, "total": 0, "patients": [],
                      "summary": {"mean_hba1c": None, "mean_forecast_hba1c": None,
                                  "worsening": 0, "improving": 0, "levels": {k: 0 for k in _LEVEL_GROUPS}}}
            _cohort_cache[mode] = (time.time(), result)
            return result

        df = df.sort_values("test_date", kind="stable")
        groups = [(str(pid), g) for pid, g in df.groupby("patient_id", sort=False)]
        latest = [g.iloc[-1].to_dict() for _, g in groups]
        previous = [g.iloc[-2].to_dict() if len(g) > 1 else cur for (_, g), cur in zip(groups, latest)]

        # Моделі викликаються один раз для всієї вибірки, а не по пацієнту
        x2 = pd.DataFrame([{k: _num(r.get(k), d) for k, d in T2_DEFAULTS.items()} for r in latest])
        predicted = t2_model.predict(x2)
        probs = None
        if mode == "risk":
            x1 = pd.DataFrame([{k: _num(r.get(k), d) for k, d in T1_DEFAULTS.items()} for r in latest])
            probs = t1_model.predict_proba(x1)[:, 1]

        rows = []
        for i, ((pid, g), cur, prev) in enumerate(zip(groups, latest, previous)):
            hba1c = _num(cur.get("HbA1c"), 6.0)
            glucose = _num(cur.get("glucose_fasting"), 100) / GLUCOSE_MMOL_TO_MGDL
            forecast = float(predicted[i])

            if mode == "risk":
                level, _ = _risk_assessment(float(probs[i]), hba1c, glucose)
                risk, risk_forecast = float(probs[i]) * 100, None
            else:
                tail = g.tail(8)
                g_trend = _trend_forecast(tail["glucose_fasting"] / GLUCOSE_MMOL_TO_MGDL)
                bp = _num(cur.get("systolic_bp"), 120)
                bp_trend = _trend_forecast(tail["systolic_bp"])
                comp = _complication_assessment(
                    hba1c, _num(prev.get("HbA1c"), hba1c), glucose, bp, _num(cur.get("bmi"), 25),
                    _num(cur.get("triglycerides"), 0) / TRIGLYCERIDES_MMOL_TO_MGDL,
                    int(_num(cur.get("age"), 50)), forecast,
                    g_trend if g_trend is not None else glucose,
                    bp_trend if bp_trend is not None else bp)
                level, risk, risk_forecast = comp["level"], float(comp["score"]), float(comp["forecast_score"])

            rows.append({
                "patient_id": pid,
                "age": int(_num(cur.get("age"), 50)),
                "sex": "M" if pd.isna(cur.get("sex")) else str(cur.get("sex")),
                "visits": int(len(g)),
                "hba1c": round(hba1c, 2),
                "forecast_hba1c": round(forecast, 2),
                "delta": round(forecast - hba1c, 2),
                "risk": round(risk, 1),
                "risk_forecast": None if risk_forecast is None else round(risk_forecast, 1),
                "level": level,
                "group": _risk_group(level),
            })

        rows.sort(key=lambda r: r["risk"], reverse=True)
        n = len(rows)
        result = {
            "mode": mode,
            "total": n,
            "summary": {
                "mean_hba1c": round(sum(r["hba1c"] for r in rows) / n, 2),
                "mean_forecast_hba1c": round(sum(r["forecast_hba1c"] for r in rows) / n, 2),
                "worsening": sum(r["delta"] > 0.2 for r in rows),
                "improving": sum(r["delta"] < -0.2 for r in rows),
                "levels": {k: sum(r["group"] == k for r in rows) for k in _LEVEL_GROUPS},
            },
            "patients": rows,
        }
        _cohort_cache[mode] = (time.time(), result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("forecast all")
        raise HTTPException(status_code=500, detail=str(e))