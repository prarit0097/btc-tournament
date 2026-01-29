from pathlib import Path
import json
import os

from .config import TournamentConfig
from .multi_timeframe import config_for_timeframe, run_multi_timeframe_tournament
from .tournament import run_tournament
from .env import load_env


def _is_running() -> bool:
    state_path = Path("data") / "run_state.json"
    if not state_path.exists():
        return False
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(data.get("running"))


def main():
    load_env()
    if _is_running():
        print("Tournament already running; skipping.")
        return
    config = TournamentConfig()
    config.base_dir = Path(".")
    config.data_dir = Path("data")
    config.db_path = config.data_dir / "btc_ohlcv.sqlite3"
    if os.getenv("BTC_TIMEFRAMES"):
        run_multi_timeframe_tournament(config)
    else:
        tf_cfg = config_for_timeframe(config, config.timeframe)
        run_tournament(tf_cfg)


if __name__ == "__main__":
    main()
