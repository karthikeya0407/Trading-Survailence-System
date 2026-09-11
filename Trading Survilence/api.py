"""
api.py — REST API for the Trading Surveillance System.

Endpoints
─────────
GET  /                    → health check
GET  /health              → system status
POST /predict             → score a single trade
POST /predict/batch       → score multiple trades at once
GET  /trades/flagged      → get top flagged trades from DB
GET  /stats               → model performance statistics
GET  /patterns            → list all suspicious pattern types
"""

import sys
import logging
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List, Optional
from datetime import datetime

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

from config import PATTERN_TYPES, RISK_LEVELS, SCORING_CONFIG, MODELS_DIR
from feature_engineering import engineer_features, get_feature_columns
from model import load_bundle, predict

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("api")

# ─── APP SETUP ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Trading Surveillance API",
    description="ML-powered suspicious trading pattern detection system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow frontend dashboard to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── LOAD MODEL ON STARTUP ────────────────────────────────────────────────────
bundle = None

@app.on_event("startup")
async def load_model():
    global bundle
    try:
        bundle = load_bundle()
        logger.info("Model bundle loaded successfully!")
    except Exception as e:
        logger.error("Failed to load model bundle: %s", e)
        logger.error("Please run train_pipeline.py first!")


# ─── REQUEST / RESPONSE SCHEMAS ───────────────────────────────────────────────

class TradeInput(BaseModel):
    """Single trade feature input."""
    trade_id             : Optional[str]   = Field(None,  description="Unique trade ID")
    trade_volume         : float           = Field(...,   description="Total trade volume")
    price                : float           = Field(...,   description="Trade price")
    trade_frequency      : float           = Field(...,   description="Number of trades per session")
    time_between_trades  : float           = Field(...,   description="Seconds between trades")
    order_size_variance  : float           = Field(...,   description="Variance in order sizes")
    price_impact         : float           = Field(...,   description="Price impact of the trade")
    volume_to_market_ratio: float          = Field(...,   description="Trade volume / market volume")
    bid_ask_spread       : float           = Field(...,   description="Bid-ask spread")
    return_1min          : float           = Field(...,   description="1-minute return")
    return_5min          : float           = Field(...,   description="5-minute return")
    return_1hr           : float           = Field(...,   description="1-hour return")
    num_counterparties   : float           = Field(...,   description="Number of counterparties")
    session_duration     : float           = Field(...,   description="Session duration in minutes")
    cancel_rate          : float           = Field(...,   description="Order cancellation rate 0-1")
    same_price_repeat    : float           = Field(...,   description="Same price repeat flag 0 or 1")
    cross_market_activity: float           = Field(...,   description="Cross-market activity ratio")
    night_trading_ratio  : float           = Field(...,   description="Night trading ratio 0-1")
    order_amendment_rate : float           = Field(...,   description="Order amendment rate 0-1")

    class Config:
        json_schema_extra = {
            "example": {
                "trade_id"             : "TRD000001",
                "trade_volume"         : 150000,
                "price"                : 55.3,
                "trade_frequency"      : 30,
                "time_between_trades"  : 8,
                "order_size_variance"  : 3000,
                "price_impact"         : 0.003,
                "volume_to_market_ratio": 0.7,
                "bid_ask_spread"       : 0.02,
                "return_1min"          : 0.001,
                "return_5min"          : 0.003,
                "return_1hr"           : 0.008,
                "num_counterparties"   : 3,
                "session_duration"     : 15,
                "cancel_rate"          : 0.88,
                "same_price_repeat"    : 1,
                "cross_market_activity": 0.1,
                "night_trading_ratio"  : 0.2,
                "order_amendment_rate" : 0.91,
            }
        }


class TradeResult(BaseModel):
    """Prediction result for a single trade."""
    trade_id        : str
    risk_score      : float
    risk_level      : str
    flagged         : bool
    supervised_prob : float
    iso_score       : float
    top_reasons     : List[dict]
    timestamp       : str


class BatchInput(BaseModel):
    trades: List[TradeInput]


class BatchResult(BaseModel):
    total           : int
    flagged_count   : int
    high_risk_count : int
    results         : List[TradeResult]


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _trade_to_df(trade: TradeInput) -> pd.DataFrame:
    """Convert a TradeInput pydantic model to a DataFrame row."""
    row = trade.dict(exclude={"trade_id"})
    df  = pd.DataFrame([row])
    return df


def _get_top_reasons(
    X_scaled_row: np.ndarray,
    feature_names: list,
    importances: np.ndarray,
    n: int = 5,
) -> List[dict]:
    """
    Compute simple feature attribution for a single trade.
    Returns top N features pushing toward suspicious.
    """
    scaler      = bundle["scaler"]
    feat_means  = scaler.center_
    feat_scales = scaler.scale_

    # Standardised deviation from population centre
    deviations  = (X_scaled_row - 0) * importances
    top_idx     = np.argsort(np.abs(deviations))[-n:][::-1]

    reasons = []
    for i in top_idx:
        reasons.append({
            "feature"   : feature_names[i],
            "importance": round(float(importances[i]), 4),
            "deviation" : round(float(deviations[i]), 4),
            "direction" : "suspicious" if deviations[i] > 0 else "normal",
        })
    return reasons


def _score_dataframe(df_raw: pd.DataFrame) -> List[TradeResult]:
    """Run full pipeline on a raw dataframe and return results."""
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train_pipeline.py first."
        )

    # Feature engineering
    df_feat    = engineer_features(df_raw)
    feat_cols  = get_feature_columns(df_feat)
    X_raw      = df_feat[feat_cols].values

    # Predict
    preds      = predict(bundle, X_raw)
    rf_model   = bundle["ensemble"].named_estimators_["rf"]
    importances = rf_model.feature_importances_

    results = []
    for i in range(len(df_raw)):
        trade_id = df_raw.iloc[i].get("trade_id", f"TRD{i:06d}")
        if not isinstance(trade_id, str) or trade_id == "nan":
            trade_id = f"TRD{i:06d}"

        reasons = _get_top_reasons(
            preds["supervised_proba"][i:i+1][0]
            * importances,
            feat_cols,
            importances,
        )

        results.append(TradeResult(
            trade_id        = str(trade_id),
            risk_score      = round(float(preds["hybrid_score"][i]), 4),
            risk_level      = preds["risk_level"][i],
            flagged         = bool(preds["flagged"][i]),
            supervised_prob = round(float(preds["supervised_proba"][i]), 4),
            iso_score       = round(float(preds["iso_score"][i]), 4),
            top_reasons     = reasons,
            timestamp       = datetime.now().isoformat(),
        ))

    return results


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    """Health check endpoint."""
    return {
        "system"  : "Trading Surveillance API",
        "version" : "1.0.0",
        "status"  : "running",
        "model"   : "loaded" if bundle else "not loaded",
        "docs"    : "/docs",
    }


@app.get("/health", tags=["System"])
async def health():
    """Detailed system health check."""
    return {
        "status"          : "healthy" if bundle else "degraded",
        "model_loaded"    : bundle is not None,
        "model_path"      : str(MODELS_DIR / "model_bundle.pkl"),
        "timestamp"       : datetime.now().isoformat(),
        "thresholds"      : {
            "alert_threshold" : SCORING_CONFIG["alert_threshold"],
            "review_threshold": SCORING_CONFIG["review_threshold"],
        },
    }


@app.get("/patterns", tags=["Info"])
async def get_patterns():
    """List all suspicious trading patterns the model detects."""
    return {
        "patterns": [
            {
                "id"         : k,
                "name"       : v,
                "description": {
                    0: "Normal legitimate trading activity",
                    1: "Artificial price inflation followed by selling",
                    2: "Placing orders with intent to cancel to mislead",
                    3: "Multiple order layers to create false depth",
                    4: "Trading with yourself to inflate volume",
                    5: "Trading ahead of known client orders",
                }.get(k, ""),
            }
            for k, v in PATTERN_TYPES.items()
        ]
    }


@app.get("/stats", tags=["Info"])
async def get_stats():
    """Return model performance statistics."""
    import json
    stats_path = Path(__file__).parent / "reports" / "results_summary.json"
    if not stats_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Stats not found. Run train_pipeline.py first."
        )
    with open(stats_path) as f:
        return json.load(f)


@app.post("/predict", response_model=TradeResult, tags=["Prediction"])
async def predict_single(trade: TradeInput):
    """
    Score a single trade for suspicious activity.

    Returns risk score, risk level and top reasons why it was flagged.
    """
    logger.info("Scoring trade: %s", trade.trade_id or "unnamed")

    df_raw = _trade_to_df(trade)
    if trade.trade_id:
        df_raw["trade_id"] = trade.trade_id

    results = _score_dataframe(df_raw)
    return results[0]


@app.post("/predict/batch", response_model=BatchResult, tags=["Prediction"])
async def predict_batch(batch: BatchInput):
    """
    Score multiple trades at once.

    Returns summary counts and individual results for each trade.
    """
    logger.info("Batch scoring %d trades ...", len(batch.trades))

    rows = []
    for t in batch.trades:
        row = t.dict(exclude={"trade_id"})
        row["trade_id"] = t.trade_id or f"TRD{len(rows):06d}"
        rows.append(row)

    df_raw   = pd.DataFrame(rows)
    results  = _score_dataframe(df_raw)

    flagged  = [r for r in results if r.flagged]
    high     = [r for r in results if r.risk_level == "HIGH"]

    return BatchResult(
        total           = len(results),
        flagged_count   = len(flagged),
        high_risk_count = len(high),
        results         = results,
    )


@app.get("/trades/flagged", tags=["Trades"])
async def get_flagged_trades(limit: int = 20):
    """
    Return top flagged trades from the SHAP results JSON.
    """
    shap_path = Path(__file__).parent / "reports" / "shap_results.json"
    if not shap_path.exists():
        raise HTTPException(
            status_code=404,
            detail="SHAP results not found. Run train_pipeline.py first."
        )
    import json
    with open(shap_path) as f:
        data = json.load(f)

    trades = data.get("explanations", [])
    trades = sorted(trades, key=lambda x: x["risk_score"], reverse=True)
    return {
        "total"  : len(trades),
        "limit"  : limit,
        "trades" : trades[:limit],
    }


# ─── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("  Trading Surveillance API")
    print("  Starting server...")
    print("  API URL  : http://127.0.0.1:8000")
    print("  API Docs : http://127.0.0.1:8000/docs")
    print("="*50 + "\n")
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )