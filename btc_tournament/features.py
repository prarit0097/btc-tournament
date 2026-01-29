from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _bars_for_hours(hours: int, candle_minutes: int) -> int:
    if candle_minutes <= 0:
        return max(1, hours)
    bars = int(round((hours * 60) / candle_minutes))
    return max(1, bars)


def compute_features(
    df: pd.DataFrame,
    candle_minutes: int = 60,
    feature_windows_hours: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    log_close = np.log(close)
    df["ret_1c"] = log_close.diff(1)
    bars_1h = _bars_for_hours(1, candle_minutes)
    df["ret_1h"] = log_close.diff(bars_1h)
    df["ret_4h"] = log_close.diff(_bars_for_hours(4, candle_minutes))
    df["ret_24h"] = log_close.diff(_bars_for_hours(24, candle_minutes))

    windows = list(feature_windows_hours) if feature_windows_hours is not None else [4, 12, 24, 48, 72]
    for w in windows:
        bars = _bars_for_hours(w, candle_minutes)
        df[f"roll_mean_{w}"] = df["ret_1c"].rolling(bars, min_periods=bars).mean()
        df[f"roll_std_{w}"] = df["ret_1c"].rolling(bars, min_periods=bars).std()
        df[f"z_{w}"] = df["ret_1c"] / (df[f"roll_std_{w}"] + 1e-9)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_window = _bars_for_hours(14, candle_minutes)
    df["atr_14"] = tr.rolling(atr_window, min_periods=atr_window).mean()
    vol_window = _bars_for_hours(24, candle_minutes)
    df["vol_24"] = df["ret_1c"].rolling(vol_window, min_periods=vol_window).std()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    for w in [14, 21]:
        bars = _bars_for_hours(w, candle_minutes)
        avg_gain = gain.rolling(bars, min_periods=bars).mean()
        avg_loss = loss.rolling(bars, min_periods=bars).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df[f"rsi_{w}"] = 100 - (100 / (1 + rs))

    ema12 = _ema(close, _bars_for_hours(12, candle_minutes))
    ema26 = _ema(close, _bars_for_hours(26, candle_minutes))
    df["macd"] = ema12 - ema26
    df["macd_signal"] = _ema(df["macd"], _bars_for_hours(9, candle_minutes))
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    bb_window = _bars_for_hours(20, candle_minutes)
    ma20 = close.rolling(bb_window, min_periods=bb_window).mean()
    std20 = close.rolling(bb_window, min_periods=bb_window).std()
    df["bb_upper"] = ma20 + 2 * std20
    df["bb_lower"] = ma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (ma20 + 1e-9)

    typical = (high + low + close) / 3
    vwap_window = _bars_for_hours(24, candle_minutes)
    vwap_num = (typical * volume).rolling(vwap_window, min_periods=vwap_window).sum()
    vwap_den = volume.rolling(vwap_window, min_periods=vwap_window).sum()
    df["vwap_24"] = vwap_num / (vwap_den + 1e-9)
    df["vwap_dist"] = (close - df["vwap_24"]) / (df["vwap_24"] + 1e-9)

    df["vol_chg"] = volume.pct_change(1, fill_method=None)
    df["vol_mean_24"] = volume.rolling(vwap_window, min_periods=vwap_window).mean()
    df["vol_std_24"] = volume.rolling(vwap_window, min_periods=vwap_window).std()

    fast = _ema(close, _bars_for_hours(12, candle_minutes))
    slow = _ema(close, _bars_for_hours(48, candle_minutes))
    trend_strength = (fast - slow).abs() / (close + 1e-9)
    df["trend_flag"] = (trend_strength > 0.002).astype(int)
    df["range_flag"] = (trend_strength <= 0.002).astype(int)

    vol_med_window = _bars_for_hours(168, candle_minutes)
    vol_med = df["vol_24"].rolling(vol_med_window, min_periods=vol_med_window).median()
    df["high_vol_flag"] = (df["vol_24"] > vol_med).astype(int)
    df["low_vol_flag"] = (df["vol_24"] <= vol_med).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def feature_sets(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    base = [
        "ret_1c",
        "ret_1h",
        "ret_4h",
        "ret_24h",
        "rsi_14",
        "rsi_21",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr_14",
        "vwap_24",
        "vwap_dist",
    ]
    vol = [
        "atr_14",
        "vol_24",
        "bb_width",
        "trend_flag",
        "range_flag",
        "high_vol_flag",
        "low_vol_flag",
    ]
    momentum = [
        "ret_1c",
        "ret_1h",
        "ret_4h",
        "ret_24h",
        "roll_mean_4",
        "roll_mean_12",
        "roll_mean_24",
        "z_4",
        "z_12",
        "z_24",
        "macd",
        "macd_hist",
    ]
    volume = [
        "vol_chg",
        "vol_mean_24",
        "vol_std_24",
        "vwap_24",
        "vwap_dist",
    ]
    trend = [
        "macd",
        "macd_signal",
        "macd_hist",
        "trend_flag",
        "range_flag",
        "bb_width",
        "vwap_dist",
        "atr_14",
    ]
    volatility = [
        "vol_24",
        "bb_width",
        "atr_14",
        "roll_std_4",
        "roll_std_12",
        "roll_std_24",
        "z_4",
        "z_12",
        "z_24",
        "high_vol_flag",
        "low_vol_flag",
    ]
    long = [
        "ret_24h",
        "roll_mean_24",
        "roll_mean_72",
        "roll_mean_168",
        "z_24",
        "z_72",
        "z_168",
        "vwap_24",
        "vwap_dist",
        "vol_24",
    ]
    signal = [
        "ret_1c",
        "ret_1h",
        "ret_4h",
        "ret_24h",
        "roll_mean_4",
        "roll_mean_12",
        "roll_mean_24",
        "macd",
        "macd_hist",
        "rsi_14",
        "rsi_21",
    ]
    minimal = ["ret_1c", "ret_1h", "ret_4h", "ret_24h"]

    sets = {
        "base": base,
        "vol": vol,
        "momentum": momentum,
        "volume": volume,
        "trend": trend,
        "volatility": volatility,
        "long": long,
        "signal": signal,
        "minimal": minimal,
    }

    cleaned = {}
    for k, v in sets.items():
        cleaned[k] = [c for c in v if c in cols]
    return cleaned


def make_supervised(
    df: pd.DataFrame,
    candle_minutes: int = 60,
    feature_windows_hours: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    df = compute_features(df, candle_minutes=candle_minutes, feature_windows_hours=feature_windows_hours)
    df["y_ret"] = df["ret_1c"].shift(-1)
    df["y_dir"] = (df["y_ret"] > 0).astype(int)
    df = df.dropna()
    return df
