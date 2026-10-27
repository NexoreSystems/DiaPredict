import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# 🔑 Налаштування підключення до MySQL
MYSQL_USER = "root"
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = "localhost"
MYSQL_PORT = "3306"
MYSQL_DB = "diabetes_db"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'data_prep' else BASE_DIR
csv_path = os.path.join(ROOT_DIR, "diabetes_master_longitudinal.csv")

def migrate():
    if not os.path.exists(csv_path):
        print(f"❌ Файл {csv_path} не знайдено!")
        return

    print("⏳ Читання CSV-файлу...")
    df = pd.read_csv(csv_path)

    if 'diastolic_bp' not in df.columns:
        df['diastolic_bp'] = 80.0
    if 'triglycerides' not in df.columns:
        df['triglycerides'] = 150.0

    print("⏳ Підключення до MySQL та імпорт даних...")
    connection_string = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
    engine = create_engine(connection_string)

    df.to_sql('patient_records', con=engine, if_exists='replace', index=False)

    print("⏳ Створення індексу для прискорення пошуку...")
    with engine.connect() as conn:
        try:
            conn.execute("CREATE INDEX idx_patient_id ON patient_records(patient_id);")
        except Exception as e:
            print("Примітка щодо індексу:", e)

    print(f"✅ Успішно перенесено {len(df)} записів у MySQL базу `{MYSQL_DB}`!")

if __name__ == "__main__":
    migrate()