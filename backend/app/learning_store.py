import json
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "aether_learning.db"

class LearningStore:
    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self):
        return sqlite3.connect(self.path)

    def _init(self):
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                citizen_id TEXT, citizen TEXT, state TEXT, action TEXT,
                reward REAL, next_state TEXT, result TEXT, tick INTEGER,
                timestamp TEXT)""")
            db.execute("""CREATE TABLE IF NOT EXISTS q_values (
                state TEXT, action TEXT, value REAL,
                PRIMARY KEY(state, action))""")
            db.commit()

    def add_experience(self, row: dict[str, Any]):
        with self._connect() as db:
            db.execute("""INSERT INTO experiences
                (citizen_id,citizen,state,action,reward,next_state,result,tick,timestamp)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (row["citizen_id"],row["citizen"],row["state"],row["action"],
                 row["reward"],row["next_state"],row["result"],row["tick"],row["timestamp"]))
            db.commit()

    def save_q(self, q: dict[tuple[str, str], float]):
        with self._connect() as db:
            db.executemany(
                "INSERT INTO q_values(state,action,value) VALUES(?,?,?) "
                "ON CONFLICT(state,action) DO UPDATE SET value=excluded.value",
                [(s,a,float(v)) for (s,a),v in q.items()])
            db.commit()

    def load_q(self) -> dict[tuple[str, str], float]:
        with self._connect() as db:
            return {(s,a):v for s,a,v in db.execute("SELECT state,action,value FROM q_values")}

    def recent(self, limit: int = 200):
        with self._connect() as db:
            rows = db.execute(
                "SELECT citizen_id,citizen,state,action,reward,next_state,result,tick,timestamp "
                "FROM experiences ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        keys=["citizen_id","citizen","state","action","reward","next_state","result","tick","timestamp"]
        return [dict(zip(keys,r)) for r in reversed(rows)]

    def count(self) -> int:
        with self._connect() as db:
            return db.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]

    def clear(self):
        with self._connect() as db:
            db.execute("DELETE FROM experiences")
            db.execute("DELETE FROM q_values")
            db.commit()

learning_store = LearningStore()
