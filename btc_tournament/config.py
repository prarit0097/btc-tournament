from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List


@dataclass
class TournamentConfig:
    base_dir: Path = Path(".")
    data_dir: Path = Path("data")
    db_path: Path = Path("data") / "btc_ohlcv.sqlite3"
    registry_path: Path = Path("data") / "registry.json"
    log_path: Path = Path("data") / "tournament.log"

    symbol: str = "BTC/USDT"
    yfinance_symbol: str = "BTC-USD"
    timeframe: str = "10m"
    candle_minutes: int = 10
    ohlcv_table: str = "ohlcv_10m"
    start_date_utc: str = "2015-01-01 00:00:00"
    data_lookback_years: int = 20
    update_every_hours: int = 1

    train_days: int = 90
    val_hours: int = 720
    test_hours: int = 24
    use_test: bool = False

    fee_slippage: float = 0.0008
    min_val_points: int = 500
    champion_margin: float = 0.02
    champion_margin_override: float = 0.05
    champion_cooldown_hours: int = 6

    max_candidates_total: int = 240
    max_candidates_per_target: int = 100
    max_workers: int = 4
    model_timeout_sec: int = 20
    random_seed: int = 42

    history_keep: int = 100
    stability_weight: float = 0.2

    run_mode: str = "hourly"
    enable_dl: bool = False

    feature_windows: List[int] = field(default_factory=lambda: [4, 12, 24, 48, 72])

    def __post_init__(self) -> None:
        env_timeframes = os.getenv("BTC_TIMEFRAMES")
        if env_timeframes:
            tokens = [t.strip() for t in env_timeframes.replace("|", ",").replace(";", ",").split(",") if t.strip()]
            if tokens:
                self.timeframe = tokens[0]
        env_tf = os.getenv("BTC_TIMEFRAME")
        if env_tf and not env_timeframes:
            self.timeframe = env_tf
        env_cm = os.getenv("BTC_CANDLE_MINUTES")
        if env_cm and env_cm.isdigit() and not env_timeframes:
            self.candle_minutes = int(env_cm)
        env_table = os.getenv("BTC_OHLCV_TABLE")
        if env_table and not env_timeframes:
            self.ohlcv_table = env_table

        self.candle_minutes = _timeframe_to_minutes(self.timeframe, self.candle_minutes)
        if not env_table or env_timeframes:
            if self.candle_minutes == 60:
                self.ohlcv_table = "ohlcv"
            else:
                self.ohlcv_table = f"ohlcv_{self.candle_minutes}m"

        env_total = os.getenv("MAX_CANDIDATES_TOTAL")
        if env_total and env_total.isdigit():
            self.max_candidates_total = int(env_total)
        env_per = os.getenv("MAX_CANDIDATES_PER_TARGET")
        if env_per and env_per.isdigit():
            self.max_candidates_per_target = int(env_per)
        env_workers = os.getenv("MAX_WORKERS")
        if env_workers and env_workers.isdigit():
            self.max_workers = int(env_workers)
        env_dl = os.getenv("ENABLE_DL")
        if env_dl is not None:
            self.enable_dl = env_dl.strip().lower() in {"1", "true", "yes", "on"}


def _timeframe_to_minutes(timeframe: str, fallback: int) -> int:
    tf = timeframe.strip().lower()
    if tf.endswith("m") and tf[:-1].isdigit():
        return int(tf[:-1])
    if tf.endswith("h") and tf[:-1].isdigit():
        return int(tf[:-1]) * 60
    return fallback
