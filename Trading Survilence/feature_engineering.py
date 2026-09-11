"""
feature_engineering.py — Feature engineering pipeline.

Transforms raw trade records into a rich feature set that maximises
the signal available to downstream ML models.
All transformations are deterministic and stateless so they can be
applied identically at training time and inference time.
"""

import numpy as np
import pandas as pd
import logging
from typing import List

from config import RAW_FEATURES, ENGINEERED_FEATURES

logger = logging.getLogger(__name__)

EPS = 1e-8   # numerical stability guard


# ─── DERIVED FEATURE BUILDERS ─────────────────────────────────────────────────

def _velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Capture how fast and aggressively trades arrive."""
    df["volume_price_impact_ratio"] = (
        df["trade_volume"] / (df["price_impact"].abs() + EPS)
    )
    df["frequency_time_ratio"] = (
        df["trade_frequency"] / (df["time_between_trades"] + 1)
    )
    df["cancel_to_execute_ratio"] = (
        df["cancel_rate"] / (1 - df["cancel_rate"] + EPS)
    )
    return df


def _manipulation_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite scores that directly encode known manipulation signals.
    Higher values → stronger suspicion for each pattern type.
    """
    # Spoofing: high cancels × high amendments × high frequency
    df["spoofing_score"] = (
        df["cancel_rate"]
        * df["order_amendment_rate"]
        * df["trade_frequency"]
    )

    # Wash trading: same-price repeats, very few counterparties
    df["wash_trade_score"] = (
        df["same_price_repeat"]
        / (df["num_counterparties"] + 1)
    )

    # Pump & dump: market-dominating volume + extreme returns
    df["pump_score"] = (
        df["volume_to_market_ratio"]
        * df["return_1hr"].abs()
        * df["trade_volume"]
    )
    return df


def _abnormality_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Capture deviation from expected statistical behaviour."""
    df["return_consistency"] = (
        df["return_1min"].abs() / (df["return_1hr"].abs() + EPS)
    )
    df["volume_anomaly"] = (
        df["trade_volume"] / (df["order_size_variance"] + 1)
    )
    df["market_dominance"] = (
        df["volume_to_market_ratio"] * df["trade_frequency"]
    )
    return df


def _interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-feature products that capture joint risk signals."""
    df["night_volume_product"]      = df["night_trading_ratio"] * df["trade_volume"]
    df["cross_cancel_interaction"]  = df["cross_market_activity"] * df["cancel_rate"]
    df["freq_counterparty_ratio"]   = df["trade_frequency"] / (df["num_counterparties"] + 1)
    return df


def _log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Log-compress highly skewed raw features."""
    for col in ["trade_volume", "price", "time_between_trades", "order_size_variance"]:
        df[f"log_{col}"] = np.log1p(df[col].abs())
    return df


# ─── PUBLIC PIPELINE ──────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full feature-engineering pipeline in-place.

    Parameters
    ----------
    df : pd.DataFrame
        Raw trade data (must contain all columns in RAW_FEATURES).

    Returns
    -------
    pd.DataFrame
        Original frame augmented with engineered features.
    """
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing raw feature columns: {missing}")

    df = df.copy()
    df = _velocity_features(df)
    df = _manipulation_scores(df)
    df = _abnormality_indicators(df)
    df = _interaction_features(df)
    df = _log_transforms(df)

    all_features = RAW_FEATURES + ENGINEERED_FEATURES
    logger.info("Feature engineering complete — %d total features", len(all_features))
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the ordered list of model-input columns present in df."""
    all_features = RAW_FEATURES + ENGINEERED_FEATURES
    return [c for c in all_features if c in df.columns]


# ─── QUICK TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from data_generator import generate_trading_data
    raw = generate_trading_data(500, 50)
    rich = engineer_features(raw)
    feat_cols = get_feature_columns(rich)
    print(f"Raw features      : {len(RAW_FEATURES)}")
    print(f"Engineered features: {len(ENGINEERED_FEATURES)}")
    print(f"Total model input : {len(feat_cols)}")
    print(rich[feat_cols].describe().T[["mean", "std", "min", "max"]].round(4))