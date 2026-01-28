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

    train_days: int = 180
    val_hours: int = 720
    test_hours: int = 24
    use_test: bool = False

    fee_slippage: float = 0.0008
    min_val_points: int = 500
    champion_margin: float = 0.02
    champion_margin_override: float = 0.05
    champion_cooldown_hours: int = 6

    max_candidates_total: int = 300
    max_candidates_per_target: int = 120
    max_workers: int = 4
    model_timeout_sec: int = 20
    random_seed: int = 42

    history_keep: int = 100
    stability_weight: float = 0.2

    run_mode: str = "hourly"
    enable_dl: bool = False

    ensemble_top_k: int = 3
    bias_window: int = 20
    bias_max_abs: float = 0.01

    feature_windows: List[int] = field(default_factory=lambda: [2, 4, 8, 12, 24, 48, 72, 96, 168])

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
        env_k = os.getenv("ENSEMBLE_TOP_K")
        if env_k and env_k.isdigit():
            self.ensemble_top_k = max(1, int(env_k))
        env_bias_window = os.getenv("BIAS_WINDOW")
        if env_bias_window and env_bias_window.isdigit():
            self.bias_window = max(1, int(env_bias_window))
        env_bias_max = os.getenv("BIAS_MAX_ABS")
        if env_bias_max:
            try:
                self.bias_max_abs = float(env_bias_max)
            except ValueError:
                pass
        env_windows = os.getenv("FEATURE_WINDOWS")
        if env_windows:
            tokens = [t.strip() for t in env_windows.replace("|", ",").replace(";", ",").split(",") if t.strip()]
            parsed = []
            for token in tokens:
                if token.isdigit():
                    parsed.append(int(token))
            if parsed:
                self.feature_windows = parsed
        env_train_days = os.getenv("TRAIN_DAYS")
        if env_train_days and env_train_days.isdigit():
            self.train_days = int(env_train_days)
        env_val_hours = os.getenv("VAL_HOURS")
        if env_val_hours and env_val_hours.isdigit():
            self.val_hours = int(env_val_hours)
        env_test_hours = os.getenv("TEST_HOURS")
        if env_test_hours and env_test_hours.isdigit():
            self.test_hours = int(env_test_hours)
        env_use_test = os.getenv("USE_TEST")
        if env_use_test is not None:
            self.use_test = env_use_test.strip().lower() in {"1", "true", "yes", "on"}


def _timeframe_to_minutes(timeframe: str, fallback: int) -> int:
    tf = timeframe.strip().lower()
    if tf.endswith("m") and tf[:-1].isdigit():
        return int(tf[:-1])
    if tf.endswith("h") and tf[:-1].isdigit():
        return int(tf[:-1]) * 60
    return fallback
