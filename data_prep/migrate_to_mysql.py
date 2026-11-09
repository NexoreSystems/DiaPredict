"""Create a separate clean MySQL table from observed source measurements.

The legacy patient_records table is deliberately left untouched.
"""
import os
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL
from dotenv import load_dotenv

from build_master_dataset import build_master_dataset


load_dotenv(Path(__file__).resolve().parents[1] / ".env")
TARGET_TABLE = "patient_records_observed"


def migrate() -> None:
    password = os.getenv("MYSQL_PASSWORD")
    if password is None:
        raise RuntimeError("MYSQL_PASSWORD is not configured in the project .env file.")

    url = URL.create(
        "mysql+pymysql",
        username=os.getenv("MYSQL_USER", "root"),
        password=password,
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        database=os.getenv("MYSQL_DB", "diabetes_db"),
    )
    engine = create_engine(url, pool_pre_ping=True)
    if inspect(engine).has_table(TARGET_TABLE):
        index_names = {item["name"] for item in inspect(engine).get_indexes(TARGET_TABLE)}
        expected_index = f"idx_{TARGET_TABLE}_patient_date"
        if expected_index not in index_names:
            with engine.begin() as connection:
                connection.execute(text(
                    f"CREATE INDEX {expected_index} ON {TARGET_TABLE} (patient_id(32), test_date)"
                ))
        print(f"{TARGET_TABLE} already exists; data was not overwritten. Index is ready.")
        return

    print("Building observed encounter records from the raw dataset...")
    frame = build_master_dataset()
    print(f"Creating new table {TARGET_TABLE} ({len(frame):,} encounters)...")
    frame.to_sql(TARGET_TABLE, con=engine, if_exists="fail", index=False, chunksize=10_000)
    with engine.begin() as connection:
        connection.execute(text(
            f"CREATE INDEX idx_{TARGET_TABLE}_patient_date ON {TARGET_TABLE} (patient_id(32), test_date)"
        ))
    print(f"Created {TARGET_TABLE}. The legacy patient_records table was not changed.")


if __name__ == "__main__":
    migrate()
