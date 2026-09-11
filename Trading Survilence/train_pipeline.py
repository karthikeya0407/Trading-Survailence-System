"""
train_pipeline.py — End-to-end training pipeline runner.

Run this script to:
  1. Generate (or load) trading data
  2. Engineer features
  3. Train the full model stack
  4. Evaluate and log all metrics
  5. Save model bundle to disk
  6. Render the analysis dashboard
  7. Export a JSON results summary

Usage
─────
    python train_pipeline.py            # full run, fresh data
    python train_pipeline.py --load-db  # load existing data from SQLite
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np

# ── local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR,
    PATTERN_TYPES, DATA_CONFIG,
)
from data_generator import (
    generate_trading_data, save_to_database, load_from_database,
)
from feature_engineering import engineer_features, get_feature_columns
from model import train, save_bundle
from visualizer import create_dashboard
from shap_explainer import run_shap_analysis

# ─── LOGGING ──────────────────────────────────────────────────────────────────
LOGS_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOGS_DIR / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("train_pipeline")


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    width = 60
    logger.info("─" * width)
    logger.info(f"  {text}")
    logger.info("─" * width)


def _export_summary(bundle: dict, output_path: Path) -> None:
    """Save a human-readable JSON metrics summary."""
    r = bundle["report"]
    summary = {
        "generated_at"    : datetime.now().isoformat(),
        "dataset": {
            "n_total"      : int(len(bundle["y_true"])),
            "n_suspicious" : int(bundle["y_true"].sum()),
            "suspicious_pct": float(bundle["y_true"].mean() * 100),
            "n_features"   : len(bundle["feature_names"]),
        },
        "performance": {
            "roc_auc"      : round(float(bundle["roc_auc"]), 4),
            "avg_precision": round(float(bundle["avg_precision"]), 4),
            "cv_auc_mean"  : round(float(bundle["cv_scores"].mean()), 4),
            "cv_auc_std"   : round(float(bundle["cv_scores"].std()), 4),
            "precision"    : round(float(r["1"]["precision"]), 4),
            "recall"       : round(float(r["1"]["recall"]), 4),
            "f1_score"     : round(float(r["1"]["f1-score"]), 4),
            "accuracy"     : round(float(r["accuracy"]), 4),
        },
        "models_used"     : [
            "RandomForest(n=300)", "GradientBoosting(n=200)",
            "IsolationForest(n=250)", "LocalOutlierFactor(k=25)",
        ],
        "patterns_modelled": list(PATTERN_TYPES.values()),
        "top_10_features"  : [
            bundle["feature_names"][i]
            for i in np.argsort(bundle["importances"])[-10:][::-1]
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results summary saved → %s", output_path)


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run(load_db: bool = False) -> None:
    t0 = time.perf_counter()

    # ── Step 1: Data ──────────────────────────────────────────────────────
    _banner("PHASE 1 — DATA")
    if load_db:
        logger.info("Loading data from SQLite …")
        df = load_from_database()
    else:
        logger.info("Generating fresh synthetic trading data …")
        df = generate_trading_data(
            n_normal=DATA_CONFIG["n_normal"],
            n_suspicious=DATA_CONFIG["n_suspicious"],
        )
        save_to_database(df)

    logger.info(
        "Dataset: %d trades | %d suspicious (%.1f%%)",
        len(df), df["label"].sum(), df["label"].mean() * 100,
    )
    logger.info("Pattern breakdown:")
    for pt, name in PATTERN_TYPES.items():
        n = (df["pattern_type"] == pt).sum()
        logger.info("  %-20s : %d", name, n)

    # ── Step 2: Feature engineering ───────────────────────────────────────
    _banner("PHASE 2 — FEATURE ENGINEERING")
    df = engineer_features(df)
    feature_cols = get_feature_columns(df)
    logger.info("Total feature columns: %d", len(feature_cols))

    X = df[feature_cols].values
    y = df["label"].values

    # ── Step 3: Train ─────────────────────────────────────────────────────
    _banner("PHASE 3 — MODEL TRAINING")
    bundle = train(X, y, feature_cols)

    # ── Step 4: Persist ───────────────────────────────────────────────────
    _banner("PHASE 4 — SAVING ARTIFACTS")
    save_bundle(bundle)
    _export_summary(bundle, REPORTS_DIR / "results_summary.json")

    # ── Step 5: Dashboard ─────────────────────────────────────────────────
    _banner("PHASE 5 — VISUALISATION")
    dashboard_path = create_dashboard(df, bundle, REPORTS_DIR / "dashboard.png")

    # ── Step 6: SHAP Analysis ────────────────────────────────────────────
    _banner("PHASE 6 — SHAP EXPLAINABILITY")
    shap_paths = run_shap_analysis(
        bundle, df, bundle["X_scaled"], feature_cols, REPORTS_DIR
    )
    for k, p in shap_paths.items():
        logger.info("SHAP %-12s → %s", k, p)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    _banner("PIPELINE COMPLETE")
    logger.info("Wall time           : %.1f s", elapsed)
    logger.info("ROC-AUC             : %.4f", bundle["roc_auc"])
    logger.info("Avg Precision       : %.4f", bundle["avg_precision"])
    logger.info("CV AUC (5-fold)     : %.4f ± %.4f",
                bundle["cv_scores"].mean(), bundle["cv_scores"].std())
    logger.info("Precision (susp)    : %.4f", bundle["report"]["1"]["precision"])
    logger.info("Recall    (susp)    : %.4f", bundle["report"]["1"]["recall"])
    logger.info("F1-Score  (susp)    : %.4f", bundle["report"]["1"]["f1-score"])
    logger.info("Dashboard           : %s", dashboard_path)
    logger.info("Model bundle        : %s", MODELS_DIR / "model_bundle.pkl")
    logger.info("Log                 : %s", log_file)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Surveillance — Training Pipeline")
    parser.add_argument(
        "--load-db", action="store_true",
        help="Load existing data from SQLite instead of generating fresh data",
    )
    args = parser.parse_args()
    run(load_db=args.load_db)