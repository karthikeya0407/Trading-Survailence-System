"""
config.py — Central configuration for the Trading Surveillance System.
All constants, thresholds, and paths are defined here.
"""

import os
from pathlib import Path

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
DATA_DIR       = BASE_DIR / "data"
MODELS_DIR     = BASE_DIR / "models"
REPORTS_DIR    = BASE_DIR / "reports"
LOGS_DIR       = BASE_DIR / "logs"
DB_PATH        = DATA_DIR / "trades.db"

# ─── DATA GENERATION ──────────────────────────────────────────────────────────
DATA_CONFIG = {
    "n_normal"     : 6000,
    "n_suspicious" : 600,
    "random_seed"  : 42,
    "noise_rate"   : 0.02,
}

# ─── SUSPICIOUS PATTERN TYPES ─────────────────────────────────────────────────
PATTERN_TYPES = {
    0: "Normal",
    1: "Pump & Dump",
    2: "Spoofing",
    3: "Layering",
    4: "Wash Trading",
    5: "Front Running",
}

# ─── FEATURE GROUPS ───────────────────────────────────────────────────────────
RAW_FEATURES = [
    "trade_volume", "price", "trade_frequency", "time_between_trades",
    "order_size_variance", "price_impact", "volume_to_market_ratio",
    "bid_ask_spread", "return_1min", "return_5min", "return_1hr",
    "num_counterparties", "session_duration", "cancel_rate",
    "same_price_repeat", "cross_market_activity", "night_trading_ratio",
    "order_amendment_rate",
]

ENGINEERED_FEATURES = [
    "volume_price_impact_ratio", "frequency_time_ratio",
    "cancel_to_execute_ratio", "spoofing_score", "wash_trade_score",
    "pump_score", "return_consistency", "volume_anomaly",
    "market_dominance", "night_volume_product", "cross_cancel_interaction",
    "freq_counterparty_ratio", "log_trade_volume", "log_price",
    "log_time_between_trades", "log_order_size_variance",
]

# ─── MODEL HYPERPARAMETERS ────────────────────────────────────────────────────
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators"    : 300,
        "max_depth"       : 14,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "class_weight"    : "balanced",
        "random_state"    : 42,
        "n_jobs"          : -1,
    },
    "gradient_boosting": {
        "n_estimators"  : 200,
        "learning_rate" : 0.07,
        "max_depth"     : 5,
        "subsample"     : 0.8,
        "random_state"  : 42,
    },
    "isolation_forest": {
        "n_estimators"  : 250,
        "contamination" : 0.09,
        "random_state"  : 42,
        "n_jobs"        : -1,
    },
    "lof": {
        "n_neighbors"   : 25,
        "contamination" : 0.09,
        "n_jobs"        : -1,
    },
}

# ─── SCORING & THRESHOLDS ─────────────────────────────────────────────────────
SCORING_CONFIG = {
    "supervised_weight"  : 0.50,
    "isolation_weight"   : 0.25,
    "lof_weight"         : 0.25,
    "alert_threshold"    : 0.70,   # HIGH risk
    "review_threshold"   : 0.40,   # MEDIUM risk — send to review queue
    "cv_folds"           : 5,
}

# ─── RISK LEVELS ──────────────────────────────────────────────────────────────
RISK_LEVELS = {
    "HIGH"   : (0.70, 1.00),
    "MEDIUM" : (0.40, 0.70),
    "LOW"    : (0.00, 0.40),
}

# ─── ALERT CONFIG (placeholder for Phase 6) ───────────────────────────────────
ALERT_CONFIG = {
    "email_enabled" : False,
    "smtp_host"     : "smtp.gmail.com",
    "smtp_port"     : 587,
    "sender_email"  : os.getenv("ALERT_EMAIL", ""),
    "sender_pass"   : os.getenv("ALERT_PASS", ""),
    "recipient"     : os.getenv("ALERT_RECIPIENT", ""),
}