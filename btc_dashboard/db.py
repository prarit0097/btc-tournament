import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def get_db_path() -> Path:
    return Path("data") / "btc_ohlcv.sqlite3"


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(get_db_path(), timeout=5)
    try:
        con.execute("PRAGMA busy_timeout = 5000")
        con.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        pass
    return con


def ensure_tables() -> None:
    with connect() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS tournament_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at TEXT,
                run_mode TEXT,
                candidate_count INTEGER
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS tournament_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                rank INTEGER,
                target TEXT,
                feature_set TEXT,
                model_name TEXT,
                family TEXT,
                final_score REAL,
                primary_metric_name TEXT,
                primary_metric_value REAL,
                trading_score REAL,
                stability_penalty REAL,
                is_champion INTEGER,
                run_at TEXT,
                FOREIGN KEY(run_id) REFERENCES tournament_runs(id)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS btc_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predicted_at TEXT,
                current_price REAL,
                predicted_return REAL,
                predicted_price REAL,
                actual_price_1h REAL,
                match_percent REAL,
                status TEXT,
                model_name TEXT,
                feature_set TEXT,
                run_id INTEGER,
                prediction_target TEXT,
                prediction_horizon_min INTEGER,
                timeframe TEXT,
                timeframe_minutes INTEGER
            )
            """
        )
        cols = {row[1] for row in con.execute("PRAGMA table_info(btc_predictions)").fetchall()}
        if "prediction_target" not in cols:
            con.execute("ALTER TABLE btc_predictions ADD COLUMN prediction_target TEXT")
        if "prediction_horizon_min" not in cols:
            con.execute("ALTER TABLE btc_predictions ADD COLUMN prediction_horizon_min INTEGER")
        if "timeframe" not in cols:
            con.execute("ALTER TABLE btc_predictions ADD COLUMN timeframe TEXT")
        if "timeframe_minutes" not in cols:
            con.execute("ALTER TABLE btc_predictions ADD COLUMN timeframe_minutes INTEGER")


def insert_run(run_at: str, run_mode: str, candidate_count: int) -> int:
    ensure_tables()
    with connect() as con:
        cur = con.execute(
            "INSERT INTO tournament_runs (run_at, run_mode, candidate_count) VALUES (?, ?, ?)",
            (run_at, run_mode, candidate_count),
        )
        return int(cur.lastrowid)


def insert_scores(run_id: int, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_tables()
    values = [
        (
            run_id,
            r.get("rank"),
            r.get("target"),
            r.get("feature_set"),
            r.get("model_name"),
            r.get("family"),
            r.get("final_score"),
            r.get("primary_metric_name"),
            r.get("primary_metric_value"),
            r.get("trading_score"),
            r.get("stability_penalty"),
            1 if r.get("is_champion") else 0,
            r.get("run_at"),
        )
        for r in rows
    ]
    with connect() as con:
        con.executemany(
            """
            INSERT INTO tournament_scores (
                run_id, rank, target, feature_set, model_name, family, final_score,
                primary_metric_name, primary_metric_value, trading_score, stability_penalty,
                is_champion, run_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )


def get_latest_run() -> Optional[Dict[str, Any]]:
    ensure_tables()
    with connect() as con:
        cur = con.execute("SELECT id, run_at, run_mode, candidate_count FROM tournament_runs ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
    if not row:
        return None
    return {"id": row[0], "run_at": row[1], "run_mode": row[2], "candidate_count": row[3]}


def get_scores(run_id: int, limit: int = 500) -> List[Dict[str, Any]]:
    ensure_tables()
    with connect() as con:
        cur = con.execute(
            """
            SELECT rank, target, feature_set, model_name, family, final_score,
                   primary_metric_name, primary_metric_value, trading_score,
                   stability_penalty, is_champion, run_at
            FROM tournament_scores
            WHERE run_id = ?
            ORDER BY target, rank
            LIMIT ?
            """,
            (run_id, limit),
        )
        rows = cur.fetchall()
    results = []
    for r in rows:
        results.append(
            {
                "rank": r[0],
                "target": r[1],
                "feature_set": r[2],
                "model_name": r[3],
                "family": r[4],
                "final_score": r[5],
                "primary_metric": {"name": r[6], "value": r[7]},
                "trading_score": r[8],
                "stability_penalty": r[9],
                "is_champion": bool(r[10]),
                "run_at": r[11],
            }
        )
    return results


def get_latest_prediction() -> Optional[Dict[str, Any]]:
    ensure_tables()
    with connect() as con:
        cur = con.execute(
            """
            SELECT id, predicted_at, current_price, predicted_return, predicted_price,
                   actual_price_1h, match_percent, status, model_name, feature_set, run_id,
                   prediction_target, prediction_horizon_min, timeframe, timeframe_minutes
            FROM btc_predictions
            ORDER BY id DESC LIMIT 1
            """
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "predicted_at": row[1],
        "current_price": row[2],
        "predicted_return": row[3],
        "predicted_price": row[4],
        "actual_price_1h": row[5],
        "match_percent": row[6],
        "status": row[7],
        "model_name": row[8],
        "feature_set": row[9],
        "run_id": row[10],
        "prediction_target": row[11],
        "prediction_horizon_min": row[12],
        "timeframe": row[13],
        "timeframe_minutes": row[14],
    }


def get_latest_ready_prediction() -> Optional[Dict[str, Any]]:
    ensure_tables()
    with connect() as con:
        cur = con.execute(
            """
            SELECT id, predicted_at, current_price, predicted_return, predicted_price,
                   actual_price_1h, match_percent, status, model_name, feature_set, run_id, prediction_target, prediction_horizon_min
            FROM btc_predictions
            WHERE status = 'ready'
            ORDER BY id DESC LIMIT 1
            """
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "predicted_at": row[1],
        "current_price": row[2],
        "predicted_return": row[3],
        "predicted_price": row[4],
        "actual_price_1h": row[5],
        "match_percent": row[6],
        "status": row[7],
        "model_name": row[8],
        "feature_set": row[9],
        "run_id": row[10],
        "prediction_target": row[11],
        "prediction_horizon_min": row[12],
    }


def insert_prediction(row: Dict[str, Any]) -> int:
    ensure_tables()
    with connect() as con:
        cur = con.execute(
            """
            INSERT INTO btc_predictions (
                predicted_at, current_price, predicted_return, predicted_price,
                actual_price_1h, match_percent, status, model_name, feature_set, run_id,
                prediction_target, prediction_horizon_min, timeframe, timeframe_minutes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("predicted_at"),
                row.get("current_price"),
                row.get("predicted_return"),
                row.get("predicted_price"),
                row.get("actual_price_1h"),
                row.get("match_percent"),
                row.get("status"),
                row.get("model_name"),
                row.get("feature_set"),
                row.get("run_id"),
                row.get("prediction_target"),
                row.get("prediction_horizon_min"),
                row.get("timeframe"),
                row.get("timeframe_minutes"),
            ),
        )
        return int(cur.lastrowid)


def update_prediction(pred_id: int, actual_price: float, match_percent: Optional[float], status: str) -> None:
    ensure_tables()
    with connect() as con:
        con.execute(
            """
            UPDATE btc_predictions
            SET actual_price_1h = ?, match_percent = ?, status = ?
            WHERE id = ?
            """,
            (actual_price, match_percent, status, pred_id),
        )


def list_pending_predictions(cutoff_iso: str) -> List[Dict[str, Any]]:
    ensure_tables()
    with connect() as con:
        cur = con.execute(
            """
            SELECT id, predicted_at, current_price, predicted_return, predicted_price,
                   prediction_horizon_min, timeframe, timeframe_minutes
            FROM btc_predictions
            WHERE status = 'pending' AND predicted_at <= ?
            ORDER BY predicted_at ASC
            """,
            (cutoff_iso,),
        )
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "predicted_at": r[1],
            "current_price": r[2],
            "predicted_return": r[3],
            "predicted_price": r[4],
            "prediction_horizon_min": r[5],
            "timeframe": r[6],
            "timeframe_minutes": r[7],
        }
        for r in rows
    ]


def get_latest_prediction_for_timeframe(timeframe: str) -> Optional[Dict[str, Any]]:
    ensure_tables()
    with connect() as con:
        cur = con.execute(
            """
            SELECT id, predicted_at, current_price, predicted_return, predicted_price,
                   actual_price_1h, match_percent, status, model_name, feature_set, run_id,
                   prediction_target, prediction_horizon_min, timeframe, timeframe_minutes
            FROM btc_predictions
            WHERE timeframe = ?
            ORDER BY id DESC LIMIT 1
            """,
            (timeframe,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "predicted_at": row[1],
        "current_price": row[2],
        "predicted_return": row[3],
        "predicted_price": row[4],
        "actual_price_1h": row[5],
        "match_percent": row[6],
        "status": row[7],
        "model_name": row[8],
        "feature_set": row[9],
        "run_id": row[10],
        "prediction_target": row[11],
        "prediction_horizon_min": row[12],
        "timeframe": row[13],
        "timeframe_minutes": row[14],
    }


def get_latest_ready_prediction_for_timeframe(timeframe: str) -> Optional[Dict[str, Any]]:
    ensure_tables()
    with connect() as con:
        cur = con.execute(
            """
            SELECT id, predicted_at, current_price, predicted_return, predicted_price,
                   actual_price_1h, match_percent, status, model_name, feature_set, run_id,
                   prediction_target, prediction_horizon_min, timeframe, timeframe_minutes
            FROM btc_predictions
            WHERE status = 'ready' AND timeframe = ?
            ORDER BY id DESC LIMIT 1
            """,
            (timeframe,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "predicted_at": row[1],
        "current_price": row[2],
        "predicted_return": row[3],
        "predicted_price": row[4],
        "actual_price_1h": row[5],
        "match_percent": row[6],
        "status": row[7],
        "model_name": row[8],
        "feature_set": row[9],
        "run_id": row[10],
        "prediction_target": row[11],
        "prediction_horizon_min": row[12],
        "timeframe": row[13],
        "timeframe_minutes": row[14],
    }


def get_ohlcv_close_at(timestamp_iso: str, table: str = "ohlcv") -> Optional[float]:
    try:
        with connect() as con:
            cur = con.execute(
                f"SELECT close FROM {table} WHERE timestamp_utc = ? LIMIT 1",
                (timestamp_iso,),
            )
            row = cur.fetchone()
    except sqlite3.OperationalError:
        return None
    if not row:
        return None
    return float(row[0])
