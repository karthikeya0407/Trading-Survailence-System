"""
model.py — ML model training, evaluation, persistence, and inference.

Architecture
────────────
Supervised layer   : Random Forest + Gradient Boosting soft-voting ensemble
Unsupervised layer : Isolation Forest + Local Outlier Factor
Hybrid score       : weighted blend of both layers (see config.SCORING_CONFIG)

The final risk score is a float in [0, 1].
Trades above ALERT_THRESHOLD   → HIGH risk
Trades above REVIEW_THRESHOLD  → MEDIUM risk
Everything else                → LOW risk
"""

import pickle
import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")

from config import MODEL_CONFIG, SCORING_CONFIG, RISK_LEVELS, MODELS_DIR

logger = logging.getLogger(__name__)

# ─── ARTIFACT PATHS ───────────────────────────────────────────────────────────
MODEL_BUNDLE_PATH = MODELS_DIR / "model_bundle.pkl"


# ─── MODEL FACTORY ────────────────────────────────────────────────────────────

def _build_supervised_ensemble() -> VotingClassifier:
    rf = RandomForestClassifier(**MODEL_CONFIG["random_forest"])
    gb = GradientBoostingClassifier(**MODEL_CONFIG["gradient_boosting"])
    return VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft",
        weights=[2, 3],          # GB gets slightly more weight
    )


def _build_anomaly_detectors() -> Tuple[IsolationForest, LocalOutlierFactor]:
    iso = IsolationForest(**MODEL_CONFIG["isolation_forest"])
    lof = LocalOutlierFactor(**MODEL_CONFIG["lof"])
    return iso, lof


# ─── SCORING HELPERS ─────────────────────────────────────────────────────────

def _normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _compute_hybrid_score(
    supervised_proba: np.ndarray,
    iso_scores: np.ndarray,
    lof_scores: np.ndarray,
) -> np.ndarray:
    w = SCORING_CONFIG
    return (
        w["supervised_weight"] * supervised_proba
        + w["isolation_weight"] * _normalise(iso_scores)
        + w["lof_weight"]       * _normalise(lof_scores)
    )


def risk_level(score: float) -> str:
    for level, (lo, hi) in RISK_LEVELS.items():
        if lo <= score < hi:
            return level
    return "HIGH"


# ─── TRAINING ────────────────────────────────────────────────────────────────

def train(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
) -> Dict[str, Any]:
    """
    Train the full model stack and return a results bundle.

    Parameters
    ----------
    X : np.ndarray  — (n_samples, n_features) scaled feature matrix
    y : np.ndarray  — binary labels
    feature_names : list — names matching columns in X

    Returns
    -------
    dict with keys: scaler, ensemble, iso, lof, results metrics, scores
    """
    logger.info("Fitting RobustScaler …")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Supervised ensemble ───────────────────────────────────────────────
    logger.info("Training supervised ensemble (RF + GradBoost) …")
    ensemble = _build_supervised_ensemble()

    cv = StratifiedKFold(
        n_splits=SCORING_CONFIG["cv_folds"], shuffle=True, random_state=42
    )
    cv_scores = cross_val_score(
        ensemble, X_scaled, y, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    logger.info(
        "CV ROC-AUC: %.4f ± %.4f", cv_scores.mean(), cv_scores.std()
    )
    ensemble.fit(X_scaled, y)
    supervised_proba = ensemble.predict_proba(X_scaled)[:, 1]

    # ── Unsupervised detectors ────────────────────────────────────────────
    logger.info("Training Isolation Forest …")
    iso, lof = _build_anomaly_detectors()
    iso.fit(X_scaled)
    iso_scores = -iso.score_samples(X_scaled)     # higher = more anomalous

    logger.info("Fitting LOF …")
    lof.fit_predict(X_scaled)
    lof_scores = -lof.negative_outlier_factor_

    # ── Hybrid score ──────────────────────────────────────────────────────
    hybrid = _compute_hybrid_score(supervised_proba, iso_scores, lof_scores)
    threshold = SCORING_CONFIG["review_threshold"]
    preds = (hybrid >= threshold).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────
    roc_auc   = roc_auc_score(y, hybrid)
    avg_prec  = average_precision_score(y, hybrid)
    report    = classification_report(y, preds, output_dict=True)
    cm        = confusion_matrix(y, preds)
    fpr, tpr, roc_thresh = roc_curve(y, hybrid)
    prec, rec, pr_thresh  = precision_recall_curve(y, hybrid)

    # ── Feature importance from RF sub-estimator ──────────────────────────
    rf_model    = ensemble.named_estimators_["rf"]
    importances = rf_model.feature_importances_

    logger.info(
        "Training complete | ROC-AUC=%.4f | Avg-Prec=%.4f | "
        "Precision=%.4f | Recall=%.4f | F1=%.4f",
        roc_auc, avg_prec,
        report["1"]["precision"],
        report["1"]["recall"],
        report["1"]["f1-score"],
    )

    bundle = {
        # ── Fitted objects ────────────────────────────────────────────────
        "scaler"           : scaler,
        "ensemble"         : ensemble,
        "iso"              : iso,
        "lof_neg_factor"   : lof.negative_outlier_factor_,   # store factors, not estimator
        "feature_names"    : feature_names,
        # ── Scores ───────────────────────────────────────────────────────
        "supervised_proba" : supervised_proba,
        "iso_scores"       : iso_scores,
        "lof_scores"       : lof_scores,
        "hybrid_score"     : hybrid,
        "predictions"      : preds,
        "y_true"           : y,
        "X_scaled"         : X_scaled,
        # ── Metrics ──────────────────────────────────────────────────────
        "roc_auc"          : roc_auc,
        "avg_precision"    : avg_prec,
        "cv_scores"        : cv_scores,
        "report"           : report,
        "confusion_matrix" : cm,
        "roc_curve"        : (fpr, tpr, roc_thresh),
        "pr_curve"         : (prec, rec, pr_thresh),
        "importances"      : importances,
    }
    return bundle


# ─── PERSISTENCE ─────────────────────────────────────────────────────────────

def save_bundle(bundle: Dict[str, Any], path: Path = MODEL_BUNDLE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Drop large numpy arrays that can be re-derived; keep only inference objects
    save_keys = ["scaler", "ensemble", "iso", "feature_names"]
    slim = {k: bundle[k] for k in save_keys}
    with open(path, "wb") as f:
        pickle.dump(slim, f)
    logger.info("Model bundle saved → %s", path)


def load_bundle(path: Path = MODEL_BUNDLE_PATH) -> Dict[str, Any]:
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    logger.info("Model bundle loaded ← %s", path)
    return bundle


# ─── INFERENCE ───────────────────────────────────────────────────────────────

def predict(
    bundle: Dict[str, Any],
    X_raw: np.ndarray,
    X_train_scaled: np.ndarray | None = None,
) -> Dict[str, Any]:
    """
    Score new trades using the loaded model bundle.

    Parameters
    ----------
    bundle        : loaded model bundle (scaler + ensemble + iso)
    X_raw         : (n, n_features) unscaled feature matrix for new trades
    X_train_scaled: optional — training data scaled; needed for LOF re-fit
                    If None, LOF is skipped and hybrid uses 50/50 split.

    Returns
    -------
    dict with `hybrid_score`, `risk_level`, `supervised_proba`, `iso_score`
    """
    scaler   = bundle["scaler"]
    ensemble = bundle["ensemble"]
    iso      = bundle["iso"]

    X_scaled        = scaler.transform(X_raw)
    supervised_prob = ensemble.predict_proba(X_scaled)[:, 1]
    iso_scores      = -iso.score_samples(X_scaled)
    iso_norm        = _normalise(iso_scores)

    # Without a LOF model available at inference, fall back to 50/50
    hybrid = 0.60 * supervised_prob + 0.40 * iso_norm

    return {
        "hybrid_score"    : hybrid,
        "supervised_proba": supervised_prob,
        "iso_score"       : iso_scores,
        "risk_level"      : [risk_level(s) for s in hybrid],
        "flagged"         : (hybrid >= SCORING_CONFIG["review_threshold"]).tolist(),
    }


# ─── QUICK TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from data_generator import generate_trading_data
    from feature_engineering import engineer_features, get_feature_columns

    df  = generate_trading_data()
    df  = engineer_features(df)
    fc  = get_feature_columns(df)
    X   = df[fc].values
    y   = df["label"].values

    bundle = train(X, y, fc)

    print(f"\nROC-AUC        : {bundle['roc_auc']:.4f}")
    print(f"Avg Precision  : {bundle['avg_precision']:.4f}")
    print(f"CV AUC (mean)  : {bundle['cv_scores'].mean():.4f}")

    save_bundle(bundle)
    loaded = load_bundle()
    result = predict(loaded, X[:10])
    print("\nFirst 10 inference results:")
    for i, (s, r) in enumerate(zip(result["hybrid_score"], result["risk_level"])):
        print(f"  Trade {i:02d}  score={s:.4f}  risk={r}")