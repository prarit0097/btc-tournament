from pathlib import Path

from .config import TournamentConfig
from .tournament import run_tournament
from .env import load_env


def main():
    load_env()
    config = TournamentConfig()
    config.base_dir = Path(".")
    config.data_dir = Path("data")
    config.db_path = config.data_dir / "btc_ohlcv.sqlite3"
    config.registry_path = config.data_dir / "registry.json"
    config.log_path = config.data_dir / "tournament.log"
    run_tournament(config)


if __name__ == "__main__":
    main()
