import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from btc_tournament.config import TournamentConfig, _timeframe_to_minutes
from btc_tournament.data_sources import fetch_and_stitch
from btc_tournament.features import make_supervised
from btc_tournament.storage import Storage

from .db import (
    ensure_tables,
    get_latest_prediction_for_timeframe,
    get_latest_ready_prediction_for_timeframe,
    get_latest_run,
    get_ohlcv_close_at,
    get_recent_ready_predictions,
    get_recent_runs,
    get_scores,
    insert_prediction,
    list_pending_predictions,
    update_prediction,
)

LOGGER = logging.getLogger(__name__)

BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
COINBASE_FX_URL = "https://api.coinbase.com/v2/exchange-rates?currency=USD"
FX_CACHE_SECONDS = int(os.getenv("FX_CACHE_SECONDS", "60"))
COINBASE_SPOT_URL = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
KRAKEN_TICKER_URL = "https://api.kraken.com/0/public/Ticker?pair=XXBTZUSD"
LEGACY_PREDICTION_HORIZON_MINUTES = 60
DEFAULT_TIMEFRAMES = ["1m", "3m", "5m", "10m", "15m", "30m", "1h", "2h", "4h"]
MATCH_EPS = 1e-6
MATCH_MAX_NONZERO = 99.9999

_RUN_LOCK = threading.Lock()
_RUN_STATE = {"running": False, "last_started_at": None}
_PRICE_CACHE: Dict[str, Any] = {}
_FX_CACHE: Dict[str, Any] = {}


def _load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf8") as f:
        return json.load(f)


def _fetch_binance_price() -> float:
    resp = requests.get(BINANCE_TICKER_URL, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    return float(data.get("price"))


def _fetch_coinbase_price() -> float:
    resp = requests.get(COINBASE_SPOT_URL, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    return float(data["data"]["amount"])


def _fetch_kraken_price() -> float:
    resp = requests.get(KRAKEN_TICKER_URL, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    price = data["result"]["XXBTZUSD"]["c"][0]
    return float(price)


def _fetch_fx_coinbase() -> float:
    resp = requests.get(COINBASE_FX_URL, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    return float(data["data"]["rates"]["INR"])


def _get_fx_rate() -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    cached = _FX_CACHE.copy()
    if cached:
        expires_at = cached.get("expires_at")
        if isinstance(expires_at, datetime) and expires_at > now:
            return {
                "rate": cached.get("rate"),
                "updated_at": cached.get("updated_at"),
                "source": cached.get("source"),
                "stale": False,
            }

    try:
        rate = _fetch_fx_coinbase()
        updated_at = now.isoformat()
        _FX_CACHE.update(
            {
                "rate": rate,
                "updated_at": updated_at,
                "source": "coinbase",
                "expires_at": now + timedelta(seconds=FX_CACHE_SECONDS),
            }
        )
        return {"rate": rate, "updated_at": updated_at, "source": "coinbase", "stale": False}
    except Exception:
        if cached.get("rate") is not None:
            return {
                "rate": cached.get("rate"),
                "updated_at": cached.get("updated_at"),
                "source": cached.get("source"),
                "stale": True,
            }
        raise


def get_live_price() -> Dict[str, Any]:
    sources = (
        ("binance", _fetch_binance_price),
        ("coinbase", _fetch_coinbase_price),
        ("kraken", _fetch_kraken_price),
    )
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    for name, fetch in sources:
        try:
            price = float(fetch())
            _PRICE_CACHE.update({"price": price, "updated_at": now_iso, "source": name})
            result: Dict[str, Any] = {"price": price, "updated_at": now_iso, "source": name}
            try:
                fx = _get_fx_rate()
                if fx.get("rate"):
                    result["price_inr"] = price * float(fx["rate"])
                    result["fx_rate"] = fx["rate"]
                    result["fx_updated_at"] = fx["updated_at"]
                    result["fx_source"] = fx["source"]
                    result["fx_stale"] = fx["stale"]
            except Exception:
                pass
            return result
        except Exception:
            continue

    cached = _PRICE_CACHE.copy()
    if cached:
        cached["stale"] = True
        try:
            fx = _get_fx_rate()
            if fx.get("rate"):
                cached["price_inr"] = cached["price"] * float(fx["rate"])
                cached["fx_rate"] = fx["rate"]
                cached["fx_updated_at"] = fx["updated_at"]
                cached["fx_source"] = fx["source"]
                cached["fx_stale"] = fx["stale"]
        except Exception:
            pass
        return cached
    raise RuntimeError("Price source unavailable")


def _parse_iso_utc(value: str) -> datetime:
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        if value.endswith("Z"):
            ts = datetime.fromisoformat(value[:-1])
        else:
            raise
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _parse_user_timestamp(value: str) -> datetime:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("empty timestamp")
    if raw.isdigit():
        epoch = int(raw)
        if epoch > 10_000_000_000:
            epoch = int(epoch / 1000)
        return datetime.fromtimestamp(epoch, tz=timezone.utc)
    try:
        ts = datetime.fromisoformat(raw)
    except ValueError:
        formats = (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%d %b %Y, %I:%M:%S %p",
            "%d %b %Y, %I:%M %p",
            "%d %b %Y %I:%M:%S %p",
            "%d %b %Y %I:%M %p",
        )
        for fmt in formats:
            try:
                ts = datetime.strptime(raw, fmt)
                break
            except ValueError:
                ts = None
        if ts is None:
            raise
    if ts.tzinfo is None:
        local_tz = datetime.now().astimezone().tzinfo
        ts = ts.replace(tzinfo=local_tz)
    return ts.astimezone(timezone.utc)


def _align_to_interval(ts: datetime, minutes: int) -> datetime:
    if minutes <= 0:
        return ts
    minute = (ts.minute // minutes) * minutes
    return ts.replace(minute=minute, second=0, microsecond=0)


def _resolve_feature_cols(model, fallback_cols: List[str]) -> List[str]:
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return list(fallback_cols)


def _load_latest_dataset(config: TournamentConfig) -> pd.DataFrame:
    storage = Storage(config.db_path, config.ohlcv_table)
    storage.init_db()
    df = storage.load()
    return df


def _ensure_recent_data(config: TournamentConfig, days: int = 14) -> pd.DataFrame:
    df = _load_latest_dataset(config)
    if not df.empty:
        return df
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        fetched, _ = fetch_and_stitch(
            config.symbol,
            config.yfinance_symbol,
            start,
            config.timeframe,
            config.candle_minutes,
        )
        if not fetched.empty:
            fetched = fetched.set_index("timestamp_utc")
            Storage(config.db_path, config.ohlcv_table).upsert(fetched)
        return _load_latest_dataset(config)
    except Exception:
        return df


def _parse_timeframes(value: Optional[str]) -> List[str]:
    if not value:
        return list(DEFAULT_TIMEFRAMES)
    tokens: List[str] = []
    for part in value.replace("|", ",").replace(";", ",").split(","):
        token = part.strip()
        if token:
            tokens.append(token)
    if not tokens:
        return list(DEFAULT_TIMEFRAMES)
    seen = set()
    ordered: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def get_timeframes(config: TournamentConfig) -> List[str]:
    env_list = os.getenv("BTC_TIMEFRAMES")
    if env_list:
        return _parse_timeframes(env_list)
    return list(DEFAULT_TIMEFRAMES)


def get_primary_timeframe(config: TournamentConfig) -> str:
    frames = get_timeframes(config)
    return frames[0] if frames else config.timeframe


def _config_for_timeframe(base: TournamentConfig, timeframe: str) -> TournamentConfig:
    cfg = TournamentConfig()
    cfg.__dict__.update(base.__dict__)
    cfg.timeframe = timeframe
    cfg.candle_minutes = _timeframe_to_minutes(timeframe, base.candle_minutes)
    if cfg.candle_minutes == 60:
        cfg.ohlcv_table = "ohlcv"
    else:
        cfg.ohlcv_table = f"ohlcv_{cfg.candle_minutes}m"
    cfg.registry_path = base.data_dir / f"registry_{cfg.candle_minutes}m.json"
    cfg.log_path = base.data_dir / f"tournament_{cfg.candle_minutes}m.log"
    return cfg


def get_price_at_timestamp(config: TournamentConfig, value: str) -> Dict[str, Any]:
    ts_utc = _parse_user_timestamp(value)
    primary_tf = get_primary_timeframe(config)
    tf_cfg = _config_for_timeframe(config, primary_tf)
    tf_minutes = max(1, int(tf_cfg.candle_minutes))
    anchor = _align_to_interval(ts_utc, tf_minutes)
    target_iso = anchor.isoformat()
    price = get_ohlcv_close_at(target_iso, table=tf_cfg.ohlcv_table)
    if price is None:
        raise LookupError("price not found for timestamp")
    result: Dict[str, Any] = {
        "requested_at": value,
        "timestamp_utc": ts_utc.isoformat(),
        "aligned_at": target_iso,
        "price": float(price),
        "timeframe": tf_cfg.timeframe,
        "table": tf_cfg.ohlcv_table,
        "aligned": True,
    }
    try:
        fx = _get_fx_rate()
        if fx.get("rate"):
            result["price_inr"] = float(price) * float(fx["rate"])
            result["fx_rate"] = fx["rate"]
            result["fx_updated_at"] = fx["updated_at"]
            result["fx_source"] = fx["source"]
            result["fx_stale"] = fx["stale"]
    except Exception:
        pass
    return result


def get_tournament_summary(config: TournamentConfig) -> Dict[str, Any]:
    primary_tf = get_primary_timeframe(config)
    tf_cfg = _config_for_timeframe(config, primary_tf)
    reg = _load_registry(tf_cfg.registry_path)
    latest_run = get_latest_run()
    candidate_count = latest_run["candidate_count"] if latest_run else 0
    eta_seconds = _estimate_eta_seconds(tf_cfg, candidate_count)
    return {
        "last_run_at": latest_run["run_at"] if latest_run else None,
        "run_mode": latest_run["run_mode"] if latest_run else None,
        "candidate_count": candidate_count,
        "champions": reg.get("champions", {}),
        "eta_seconds": eta_seconds,
    }


def get_scoreboard(limit: int = 500) -> List[Dict[str, Any]]:
    latest = get_latest_run()
    if not latest:
        return []
    return get_scores(latest["id"], limit)


def update_pending_predictions(config: TournamentConfig) -> None:
    ensure_tables()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=1)
    cutoff_iso = cutoff.isoformat()
    pending = list_pending_predictions(cutoff_iso)
    if not pending:
        return

    for p in pending:
        pred_at = _parse_iso_utc(p["predicted_at"])
        horizon_min = p.get("prediction_horizon_min") or p.get("timeframe_minutes") or LEGACY_PREDICTION_HORIZON_MINUTES
        if datetime.now(timezone.utc) - pred_at < timedelta(minutes=int(horizon_min)):
            continue
        tf_minutes = p.get("timeframe_minutes") or horizon_min
        anchor = _align_to_interval(pred_at, int(tf_minutes))
        target_ts = anchor + timedelta(minutes=int(horizon_min))
        target_iso = target_ts.isoformat()

        table = "ohlcv" if int(tf_minutes) == 60 else f"ohlcv_{int(tf_minutes)}m"
        actual = get_ohlcv_close_at(target_iso, table=table)
        if actual is None:
            try:
                actual = get_live_price()["price"]
            except Exception:
                continue
        metrics = _compute_match_metrics(p.get("predicted_price"), actual)
        update_prediction(p["id"], actual, metrics["match_percent"], "ready")


def _predict_return_from_champion(config: TournamentConfig, latest_row: pd.DataFrame) -> Optional[Dict[str, Any]]:
    reg = _load_registry(config.registry_path)
    champ = reg.get("champions", {}).get("return")
    if not champ:
        return None

    model_path = champ.get("model_path")
    if not model_path:
        return None

    import joblib

    model = joblib.load(model_path)
    feature_cols = _resolve_feature_cols(model, champ.get("feature_cols", []))
    X = latest_row.reindex(columns=feature_cols, fill_value=0.0)
    pred = float(model.predict(X)[0])
    return {
        "predicted_return": pred,
        "model_name": champ.get("model_id", "return_champion"),
        "feature_set": champ.get("feature_set_id"),
    }


def _predict_return_from_ensemble(config: TournamentConfig, latest_row: pd.DataFrame) -> Optional[Dict[str, Any]]:
    reg = _load_registry(config.registry_path)
    ensemble = reg.get("ensembles", {}).get("return")
    if not ensemble:
        return None
    members = ensemble.get("members") or []
    if not members:
        return None

    import joblib

    preds: List[float] = []
    weights: List[float] = []
    used_members: List[str] = []
    for member in members:
        model_path = member.get("model_path")
        if not model_path:
            continue
        if not Path(model_path).exists():
            continue
        model = joblib.load(model_path)
        feature_cols = _resolve_feature_cols(model, member.get("feature_cols", []))
        X = latest_row.reindex(columns=feature_cols, fill_value=0.0)
        pred_val = float(model.predict(X)[0])
        if not np.isfinite(pred_val):
            continue
        preds.append(pred_val)
        weight = member.get("final_score")
        try:
            weight_val = float(weight) if weight is not None else 1.0
        except (TypeError, ValueError):
            weight_val = 1.0
        weights.append(max(0.0, weight_val))
        used_members.append(member.get("model_id", "unknown"))

    if not preds:
        return None

    if sum(weights) > 0:
        predicted_return = float(np.average(preds, weights=weights))
    else:
        predicted_return = float(np.mean(preds))

    return {
        "predicted_return": predicted_return,
        "model_name": f"ensemble_top{len(preds)}",
        "feature_set": "ensemble",
        "ensemble_members": used_members,
        "ensemble_size": len(preds),
    }


def _predict_return_from_direction(config: TournamentConfig, latest_row: pd.DataFrame, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    reg = _load_registry(config.registry_path)
    champ = reg.get("champions", {}).get("direction")
    if not champ:
        return None

    model_path = champ.get("model_path")
    if not model_path:
        return None

    import joblib

    model = joblib.load(model_path)
    feature_cols = _resolve_feature_cols(model, champ.get("feature_cols", []))
    X = latest_row.reindex(columns=feature_cols, fill_value=0.0)
    direction = int(model.predict(X)[0])

    recent = df["close"].pct_change().dropna().tail(24)
    avg_move = float(np.abs(recent).mean()) if not recent.empty else 0.002
    sign = 1.0 if direction == 1 else -1.0
    predicted_return = np.log(1 + avg_move * sign)

    return {
        "predicted_return": float(predicted_return),
        "model_name": champ.get("model_id", "direction_champion"),
        "feature_set": champ.get("feature_set_id"),
    }


def _cooldown_minutes(horizon_min: int) -> int:
    return max(1, min(15, int(horizon_min / 2)))


def _compute_match_metrics(predicted: Optional[float], actual: Optional[float]) -> Dict[str, Optional[float]]:
    if predicted is None or actual is None:
        return {"abs_diff": None, "pct_error": None, "match_percent": None}
    try:
        predicted_val = float(predicted)
        actual_val = float(actual)
    except (TypeError, ValueError):
        return {"abs_diff": None, "pct_error": None, "match_percent": None}
    abs_diff = abs(predicted_val - actual_val)
    if actual_val == 0:
        return {"abs_diff": abs_diff, "pct_error": None, "match_percent": None}
    pct_error = (abs_diff / abs(actual_val)) * 100.0
    match = 100.0 - pct_error
    match = max(0.0, min(100.0, match))
    if abs_diff > MATCH_EPS:
        match = min(match, MATCH_MAX_NONZERO)
    return {"abs_diff": abs_diff, "pct_error": pct_error, "match_percent": match}


def _get_recent_bias(config: TournamentConfig, timeframe: str) -> float:
    limit = max(1, int(config.bias_window))
    rows = get_recent_ready_predictions(timeframe, limit)
    if not rows:
        return 0.0
    errors: List[float] = []
    for row in rows:
        predicted_ret = row.get("predicted_return")
        actual_price = row.get("actual_price_1h")
        current_price = row.get("current_price")
        if predicted_ret is None or actual_price is None or current_price is None:
            continue
        try:
            actual_val = float(actual_price)
            current_val = float(current_price)
            predicted_val = float(predicted_ret)
        except (TypeError, ValueError):
            continue
        if actual_val <= 0 or current_val <= 0:
            continue
        actual_ret = float(np.log(actual_val / current_val))
        err = actual_ret - predicted_val
        if np.isfinite(err):
            errors.append(err)
    if not errors:
        return 0.0
    bias = float(np.mean(errors))
    max_abs = float(config.bias_max_abs)
    if max_abs > 0:
        bias = max(-max_abs, min(max_abs, bias))
    return bias


def _estimate_eta_from_runs(
    runs: List[Dict[str, Any]], candidate_count: int, config: TournamentConfig
) -> Optional[int]:
    durations: List[float] = []
    counts: List[int] = []
    workers: List[int] = []
    train_days_list: List[int] = []
    val_hours_list: List[int] = []
    for run in runs:
        dur = run.get("duration_seconds")
        if dur is None:
            continue
        try:
            dur_val = float(dur)
        except (TypeError, ValueError):
            continue
        if dur_val <= 0:
            continue
        durations.append(dur_val)
        cnt = run.get("candidate_count")
        if isinstance(cnt, int) and cnt > 0:
            counts.append(cnt)
        wk = run.get("max_workers")
        if isinstance(wk, int) and wk > 0:
            workers.append(wk)
        td = run.get("train_days")
        if isinstance(td, int) and td > 0:
            train_days_list.append(td)
        vh = run.get("val_hours")
        if isinstance(vh, int) and vh > 0:
            val_hours_list.append(vh)
    if not durations:
        return None
    durations.sort()
    mid = len(durations) // 2
    median_duration = durations[mid] if len(durations) % 2 == 1 else (durations[mid - 1] + durations[mid]) / 2.0
    scale = 1.0
    if counts and candidate_count:
        avg_candidates = float(sum(counts) / len(counts))
        if avg_candidates > 0:
            scale *= float(candidate_count) / avg_candidates
    if workers and config.max_workers:
        avg_workers = float(sum(workers) / len(workers))
        if avg_workers > 0:
            scale *= avg_workers / float(config.max_workers)
    if train_days_list and getattr(config, "train_days", None):
        avg_train_days = float(sum(train_days_list) / len(train_days_list))
        if avg_train_days > 0:
            scale *= float(config.train_days) / avg_train_days
    if val_hours_list and getattr(config, "val_hours", None):
        avg_val_hours = float(sum(val_hours_list) / len(val_hours_list))
        if avg_val_hours > 0:
            scale *= float(config.val_hours) / avg_val_hours
    eta = max(30.0, median_duration * scale)
    return int(round(eta))


def _estimate_eta_seconds(config: TournamentConfig, candidate_count: int, limit: int = 20) -> Optional[int]:
    base_filters: Dict[str, Any] = {
        "run_mode": config.run_mode,
        "timeframe": config.timeframe,
        "candle_minutes": config.candle_minutes,
        "train_days": config.train_days,
        "val_hours": config.val_hours,
        "max_workers": config.max_workers,
        "max_candidates_total": config.max_candidates_total,
        "max_candidates_per_target": config.max_candidates_per_target,
        "enable_dl": config.enable_dl,
    }
    relax_steps = [
        {},
        {"max_candidates_total": None, "max_candidates_per_target": None},
        {"train_days": None, "val_hours": None},
        {"enable_dl": None},
        {"timeframe": None, "candle_minutes": None},
        {"max_workers": None},
        {"run_mode": None},
    ]
    for relax in relax_steps:
        filters = dict(base_filters)
        filters.update(relax)
        runs = get_recent_runs(limit=limit, **filters)
        eta = _estimate_eta_from_runs(runs, candidate_count, config)
        if eta is not None:
            return eta
    return None


def _apply_match_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    predicted = row.get("predicted_price")
    actual = row.get("actual_price_1h") if row.get("actual_price_1h") is not None else row.get("actual_price")
    metrics = _compute_match_metrics(predicted, actual)
    if metrics["abs_diff"] is not None:
        row["abs_diff_usd"] = round(float(metrics["abs_diff"]), 2)
    else:
        row["abs_diff_usd"] = None
    if metrics["pct_error"] is not None:
        row["pct_error"] = round(float(metrics["pct_error"]), 6)
    else:
        row["pct_error"] = None
    if metrics["match_percent"] is not None:
        row["match_percent_precise"] = round(float(metrics["match_percent"]), 4)
        row["match_percent"] = float(metrics["match_percent"])
    else:
        row["match_percent_precise"] = None
    return row


def _decorate_last_ready(last_ready: Optional[Dict[str, Any]], horizon_min: int) -> Optional[Dict[str, Any]]:
    if not last_ready:
        return None
    try:
        pred_at = _parse_iso_utc(last_ready["predicted_at"])
        delta_min = int(last_ready.get("prediction_horizon_min") or horizon_min)
        tf_minutes = int(last_ready.get("timeframe_minutes") or delta_min or horizon_min)
        tf_minutes = max(1, tf_minutes)
        anchor = _align_to_interval(pred_at, tf_minutes)
        target_ts = anchor + timedelta(minutes=delta_min)
        last_ready["actual_at"] = target_ts.isoformat()
        last_ready["target_iso"] = target_ts.isoformat()
        last_ready["target_aligned"] = True
    except Exception:
        last_ready["actual_at"] = None
        last_ready["target_iso"] = None
        last_ready["target_aligned"] = False
    last_ready["actual_price"] = last_ready.get("actual_price_1h")
    _apply_match_fields(last_ready)
    return last_ready


def refresh_prediction(config: TournamentConfig) -> Dict[str, Any]:
    if _RUN_STATE.get("running"):
        return latest_prediction(config, update_pending=False)
    ensure_tables()
    update_pending_predictions(config)
    predictions: List[Dict[str, Any]] = []

    try:
        price_info = get_live_price()
        live_price = price_info["price"]
    except Exception:
        live_price = None

    latest_run = get_latest_run()
    run_id = latest_run["id"] if latest_run else None

    for timeframe in get_timeframes(config):
        tf_cfg = _config_for_timeframe(config, timeframe)
        horizon_min = max(1, tf_cfg.candle_minutes)
        latest = get_latest_prediction_for_timeframe(timeframe)
        if latest:
            pred_at = _parse_iso_utc(latest["predicted_at"])
            cooldown = _cooldown_minutes(horizon_min)
            if datetime.now(timezone.utc) - pred_at < timedelta(minutes=cooldown):
                _apply_match_fields(latest)
                latest["last_ready"] = _decorate_last_ready(
                    get_latest_ready_prediction_for_timeframe(timeframe),
                    horizon_min,
                )
                predictions.append(latest)
                continue

        if live_price is None:
            predictions.append(
                {
                    "timeframe": timeframe,
                    "timeframe_minutes": horizon_min,
                    "prediction_horizon_min": horizon_min,
                    "status": "no_price",
                }
            )
            continue

        df = _ensure_recent_data(tf_cfg)
        sup = make_supervised(df, candle_minutes=tf_cfg.candle_minutes, feature_windows_hours=tf_cfg.feature_windows)
        if sup.empty:
            predictions.append(
                {
                    "timeframe": timeframe,
                    "timeframe_minutes": horizon_min,
                    "prediction_horizon_min": horizon_min,
                    "status": "no_data",
                }
            )
            continue
        latest_row = sup.iloc[-1:]

        pred = _predict_return_from_ensemble(tf_cfg, latest_row)
        prediction_target = "return"
        if pred is None:
            pred = _predict_return_from_champion(tf_cfg, latest_row)
        if pred is None:
            pred = _predict_return_from_direction(tf_cfg, latest_row, df)
            prediction_target = "direction"
        if pred is None:
            predictions.append(
                {
                    "timeframe": timeframe,
                    "timeframe_minutes": horizon_min,
                    "prediction_horizon_min": horizon_min,
                    "status": "no_champion",
                }
            )
            continue

        predicted_return = float(pred["predicted_return"])
        bias = _get_recent_bias(tf_cfg, timeframe)
        predicted_return = float(predicted_return + bias)
        predicted_price = float(live_price) * float(np.exp(predicted_return))

        record = {
            "predicted_at": datetime.now(timezone.utc).isoformat(),
            "current_price": live_price,
            "predicted_return": predicted_return,
            "predicted_price": predicted_price,
            "actual_price_1h": None,
            "match_percent": None,
            "status": "pending",
            "model_name": pred["model_name"],
            "feature_set": pred["feature_set"],
            "run_id": run_id,
            "prediction_target": prediction_target,
            "prediction_horizon_min": horizon_min,
            "timeframe": timeframe,
            "timeframe_minutes": horizon_min,
        }
        pred_id = insert_prediction(record)
        record["id"] = pred_id
        record["bias_correction"] = bias
        if pred.get("ensemble_members"):
            record["ensemble_members"] = pred.get("ensemble_members")
            record["ensemble_size"] = pred.get("ensemble_size")
        _apply_match_fields(record)
        predictions.append(record)

    return {"predictions": predictions}


def latest_prediction(config: TournamentConfig, update_pending: bool = True) -> Optional[Dict[str, Any]]:
    if update_pending and not _RUN_STATE.get("running"):
        update_pending_predictions(config)
    predictions: List[Dict[str, Any]] = []
    for timeframe in get_timeframes(config):
        latest = get_latest_prediction_for_timeframe(timeframe)
        horizon_min = _timeframe_to_minutes(timeframe, config.candle_minutes)
        if latest:
            _apply_match_fields(latest)
            latest["last_ready"] = _decorate_last_ready(
                get_latest_ready_prediction_for_timeframe(timeframe),
                horizon_min,
            )
            predictions.append(latest)
        else:
            predictions.append(
                {
                    "timeframe": timeframe,
                    "timeframe_minutes": horizon_min,
                    "prediction_horizon_min": horizon_min,
                    "status": "no_prediction",
                }
            )
    return {"predictions": predictions}


def run_status() -> Dict[str, Any]:
    return dict(_RUN_STATE)


def _run_tournament_thread(config: TournamentConfig) -> None:
    try:
        from btc_tournament.multi_timeframe import run_multi_timeframe_tournament
        run_multi_timeframe_tournament(config)
    finally:
        with _RUN_LOCK:
            _RUN_STATE["running"] = False


def run_tournament_async(config: TournamentConfig, run_mode: Optional[str]) -> Dict[str, Any]:
    with _RUN_LOCK:
        if _RUN_STATE["running"]:
            return {"status": "already_running", **_RUN_STATE}
        if run_mode:
            config.run_mode = run_mode
        _RUN_STATE["running"] = True
        _RUN_STATE["last_started_at"] = datetime.now(timezone.utc).isoformat()
        t = threading.Thread(target=_run_tournament_thread, args=(config,), daemon=True)
        t.start()
    return {"status": "started", **_RUN_STATE}
