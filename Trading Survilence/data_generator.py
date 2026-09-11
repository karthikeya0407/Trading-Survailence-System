"""
data_generator.py — Synthetic trading data generation.

Generates realistic labeled trade records representing 5 known
manipulation patterns plus normal trading behaviour.
Each suspicious trade also carries a `pattern_type` label for
multi-class analysis and reporting.
"""

import numpy as np
import pandas as pd
import sqlite3
import logging
from pathlib import Path
from typing import Tuple

from config import DATA_CONFIG, PATTERN_TYPES, DB_PATH, DATA_DIR

logger = logging.getLogger(__name__)


# ─── INDIVIDUAL PATTERN GENERATORS ───────────────────────────────────────────

def _normal(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "trade_volume"          : rng.lognormal(8, 1.2, n),
        "price"                 : rng.lognormal(4, 0.5, n),
        "trade_frequency"       : rng.poisson(5, n).astype(float),
        "time_between_trades"   : rng.exponential(120, n),
        "order_size_variance"   : rng.gamma(2, 50, n),
        "price_impact"          : rng.normal(0.001, 0.002, n),
        "volume_to_market_ratio": rng.beta(1.5, 8, n),
        "bid_ask_spread"        : rng.lognormal(-3, 0.5, n),
        "return_1min"           : rng.normal(0, 0.002, n),
        "return_5min"           : rng.normal(0, 0.005, n),
        "return_1hr"            : rng.normal(0, 0.012, n),
        "num_counterparties"    : rng.integers(3, 50, n).astype(float),
        "session_duration"      : rng.gamma(3, 20, n),
        "cancel_rate"           : rng.beta(1, 9, n),
        "same_price_repeat"     : rng.binomial(1, 0.05, n).astype(float),
        "cross_market_activity" : rng.beta(1, 10, n),
        "night_trading_ratio"   : rng.beta(0.5, 9, n),
        "order_amendment_rate"  : rng.beta(1, 8, n),
        "label"                 : 0,
        "pattern_type"          : 0,
    })


def _pump_and_dump(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "trade_volume"          : rng.lognormal(11, 0.8, n),
        "price"                 : rng.lognormal(4.5, 0.3, n),
        "trade_frequency"       : rng.poisson(25, n).astype(float),
        "time_between_trades"   : rng.exponential(15, n),
        "order_size_variance"   : rng.gamma(8, 200, n),
        "price_impact"          : rng.normal(0.05, 0.02, n),
        "volume_to_market_ratio": rng.beta(8, 2, n),
        "bid_ask_spread"        : rng.lognormal(-2, 0.3, n),
        "return_1min"           : rng.normal(0.01, 0.005, n),
        "return_5min"           : rng.normal(0.05, 0.01, n),
        "return_1hr"            : rng.normal(0.15, 0.03, n),
        "num_counterparties"    : rng.integers(1, 5, n).astype(float),
        "session_duration"      : rng.gamma(1, 10, n),
        "cancel_rate"           : rng.beta(1, 15, n),
        "same_price_repeat"     : rng.binomial(1, 0.15, n).astype(float),
        "cross_market_activity" : rng.beta(5, 3, n),
        "night_trading_ratio"   : rng.beta(2, 3, n),
        "order_amendment_rate"  : rng.beta(0.5, 5, n),
        "label"                 : 1,
        "pattern_type"          : 1,
    })


def _spoofing(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "trade_volume"          : rng.lognormal(10, 0.5, n),
        "price"                 : rng.lognormal(4, 0.4, n),
        "trade_frequency"       : rng.poisson(30, n).astype(float),
        "time_between_trades"   : rng.exponential(8, n),
        "order_size_variance"   : rng.gamma(10, 300, n),
        "price_impact"          : rng.normal(0.003, 0.001, n),
        "volume_to_market_ratio": rng.beta(5, 2, n),
        "bid_ask_spread"        : rng.lognormal(-4, 0.2, n),
        "return_1min"           : rng.normal(0, 0.001, n),
        "return_5min"           : rng.normal(0, 0.003, n),
        "return_1hr"            : rng.normal(0, 0.008, n),
        "num_counterparties"    : rng.integers(2, 8, n).astype(float),
        "session_duration"      : rng.gamma(2, 5, n),
        "cancel_rate"           : rng.beta(9, 1, n),
        "same_price_repeat"     : rng.binomial(1, 0.6, n).astype(float),
        "cross_market_activity" : rng.beta(1, 5, n),
        "night_trading_ratio"   : rng.beta(1, 5, n),
        "order_amendment_rate"  : rng.beta(9, 1, n),
        "label"                 : 1,
        "pattern_type"          : 2,
    })


def _layering(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "trade_volume"          : rng.lognormal(9, 0.6, n),
        "price"                 : rng.lognormal(4, 0.3, n),
        "trade_frequency"       : rng.poisson(40, n).astype(float),
        "time_between_trades"   : rng.exponential(5, n),
        "order_size_variance"   : rng.gamma(5, 150, n),
        "price_impact"          : rng.normal(0.002, 0.001, n),
        "volume_to_market_ratio": rng.beta(6, 2, n),
        "bid_ask_spread"        : rng.lognormal(-5, 0.3, n),
        "return_1min"           : rng.normal(0.002, 0.001, n),
        "return_5min"           : rng.normal(0.005, 0.003, n),
        "return_1hr"            : rng.normal(0.01, 0.005, n),
        "num_counterparties"    : rng.integers(1, 4, n).astype(float),
        "session_duration"      : rng.gamma(1.5, 8, n),
        "cancel_rate"           : rng.beta(8, 2, n),
        "same_price_repeat"     : rng.binomial(1, 0.8, n).astype(float),
        "cross_market_activity" : rng.beta(3, 2, n),
        "night_trading_ratio"   : rng.beta(1.5, 4, n),
        "order_amendment_rate"  : rng.beta(7, 2, n),
        "label"                 : 1,
        "pattern_type"          : 3,
    })


def _wash_trading(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "trade_volume"          : rng.lognormal(10, 0.4, n),
        "price"                 : rng.lognormal(4.2, 0.2, n),
        "trade_frequency"       : rng.poisson(15, n).astype(float),
        "time_between_trades"   : rng.exponential(30, n),
        "order_size_variance"   : rng.gamma(1, 20, n),
        "price_impact"          : rng.normal(0.0001, 0.0001, n),
        "volume_to_market_ratio": rng.beta(4, 2, n),
        "bid_ask_spread"        : rng.lognormal(-3, 0.2, n),
        "return_1min"           : rng.normal(0, 0.0005, n),
        "return_5min"           : rng.normal(0, 0.001, n),
        "return_1hr"            : rng.normal(0, 0.002, n),
        "num_counterparties"    : rng.integers(1, 2, n).astype(float),
        "session_duration"      : rng.gamma(4, 30, n),
        "cancel_rate"           : rng.beta(0.5, 5, n),
        "same_price_repeat"     : rng.binomial(1, 0.9, n).astype(float),
        "cross_market_activity" : rng.beta(0.5, 8, n),
        "night_trading_ratio"   : rng.beta(3, 2, n),
        "order_amendment_rate"  : rng.beta(0.5, 6, n),
        "label"                 : 1,
        "pattern_type"          : 4,
    })


def _front_running(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "trade_volume"          : rng.lognormal(7, 0.8, n),
        "price"                 : rng.lognormal(4, 0.5, n),
        "trade_frequency"       : rng.poisson(8, n).astype(float),
        "time_between_trades"   : rng.exponential(60, n),
        "order_size_variance"   : rng.gamma(3, 80, n),
        "price_impact"          : rng.normal(0.008, 0.003, n),
        "volume_to_market_ratio": rng.beta(3, 3, n),
        "bid_ask_spread"        : rng.lognormal(-3, 0.4, n),
        "return_1min"           : rng.normal(0.005, 0.002, n),
        "return_5min"           : rng.normal(0.012, 0.004, n),
        "return_1hr"            : rng.normal(0.02, 0.008, n),
        "num_counterparties"    : rng.integers(1, 3, n).astype(float),
        "session_duration"      : rng.gamma(2, 15, n),
        "cancel_rate"           : rng.beta(2, 6, n),
        "same_price_repeat"     : rng.binomial(1, 0.2, n).astype(float),
        "cross_market_activity" : rng.beta(6, 2, n),
        "night_trading_ratio"   : rng.beta(0.8, 8, n),
        "order_amendment_rate"  : rng.beta(2, 6, n),
        "label"                 : 1,
        "pattern_type"          : 5,
    })


# ─── MAIN GENERATOR ──────────────────────────────────────────────────────────

def generate_trading_data(
    n_normal: int = DATA_CONFIG["n_normal"],
    n_suspicious: int = DATA_CONFIG["n_suspicious"],
    seed: int = DATA_CONFIG["random_seed"],
) -> pd.DataFrame:
    """
    Generate a synthetic trading dataset with labelled manipulation patterns.

    Returns
    -------
    pd.DataFrame
        Shuffled dataset with raw features, binary `label`, and `pattern_type`.
    """
    rng = np.random.default_rng(seed)

    per_pattern = n_suspicious // 5
    remainder   = n_suspicious - per_pattern * 5          # absorb rounding in front-running

    pieces = [
        _normal(n_normal, rng),
        _pump_and_dump(per_pattern, rng),
        _spoofing(per_pattern, rng),
        _layering(per_pattern, rng),
        _wash_trading(per_pattern, rng),
        _front_running(per_pattern + remainder, rng),
    ]

    df = pd.concat(pieces, ignore_index=True)

    # ── Inject low-level noise to prevent perfect separation ───────────────
    noise_cols = [c for c in df.columns if c not in ("label", "pattern_type")]
    for col in noise_cols:
        df[col] = df[col].astype(float)

    noise_mask = rng.random(len(df)) < DATA_CONFIG["noise_rate"]
    for col in noise_cols:
        std = df[col].std()
        df.loc[noise_mask, col] += rng.normal(0, std * 0.08, noise_mask.sum())

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Add a unique trade ID and timestamp
    df.insert(0, "trade_id", [f"TRD{i:06d}" for i in range(len(df))])
    base_ts = pd.Timestamp("2024-01-01")
    df.insert(1, "timestamp", pd.date_range(base_ts, periods=len(df), freq="1min"))

    logger.info(
        "Generated %d trades | %d suspicious (%.1f%%)",
        len(df), df["label"].sum(), df["label"].mean() * 100,
    )
    return df


# ─── DATABASE STORAGE ────────────────────────────────────────────────────────

def save_to_database(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    """Persist the generated dataset to a local SQLite database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    df_db = df.copy()
    df_db["timestamp"] = df_db["timestamp"].astype(str)
    df_db.to_sql("trades", conn, if_exists="replace", index=False)
    conn.close()
    logger.info("Saved %d rows to %s", len(df), db_path)


def load_from_database(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load trades from the SQLite database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM trades", conn, parse_dates=["timestamp"])
    conn.close()
    return df


# ─── QUICK TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = generate_trading_data()
    save_to_database(df)
    print(df.head())
    print("\nPattern distribution:")
    print(df["pattern_type"].value_counts().sort_index())