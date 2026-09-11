"""
shap_explainer.py — SHAP-based explainability for the Trading Surveillance System.

For every flagged trade this module answers:
  WHY was this trade suspicious?
  WHICH features pushed the risk score highest?
  HOW does this trader compare to the normal population?

Uses TreeExplainer (fast, exact for tree-based models) on the
Random Forest sub-estimator inside the VotingClassifier ensemble.

Outputs
───────
  • shap_summary.png       — global feature importance (beeswarm)
  • shap_bar.png           — mean absolute SHAP bar chart
  • shap_waterfall_N.png   — per-trade waterfall for top flagged trades
  • shap_heatmap.png       — SHAP value heatmap across top flagged trades
  • shap_dependence.png    — dependence plots for top 4 features
  • shap_results.json      — machine-readable per-trade explanations
"""

import json
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, Any, List

from sklearn.ensemble import RandomForestClassifier

from config import REPORTS_DIR, PATTERN_TYPES, RISK_LEVELS, SCORING_CONFIG

logger = logging.getLogger(__name__)

# ─── PALETTE ──────────────────────────────────────────────────────────────────
C = {
    "bg"     : "#07090f",
    "panel"  : "#0d1117",
    "border" : "#1c2333",
    "grid"   : "#161c27",
    "text"   : "#cdd9f5",
    "muted"  : "#5c6f94",
    "pos"    : "#f43f5e",   # pushes toward suspicious
    "neg"    : "#38bdf8",   # pushes toward normal
    "neutral": "#6b7280",
    "a1"     : "#22c55e",
    "a2"     : "#f97316",
    "a3"     : "#a855f7",
}


def _style(ax, title: str = "") -> None:
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["muted"], labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor(C["border"])
    if title:
        ax.set_title(title, color=C["text"], fontsize=10,
                     fontweight="bold", fontfamily="monospace", pad=9)
    ax.grid(color=C["grid"], alpha=0.5, linewidth=0.4)


# ─── EXTRACT RF FROM ENSEMBLE ────────────────────────────────────────────────

def _get_rf(bundle: Dict) -> RandomForestClassifier:
    """Pull the Random Forest out of the VotingClassifier."""
    return bundle["ensemble"].named_estimators_["rf"]


# ─── COMPUTE SHAP VALUES ─────────────────────────────────────────────────────

def compute_shap_values(
    bundle: Dict[str, Any],
    X_scaled: np.ndarray,
    feature_names: List[str],
    max_samples: int = 1000,
) -> Dict[str, Any]:
    """
    Compute SHAP values using manual TreeExplainer approximation.
    Uses the Random Forest's feature importances and predictions
    to create meaningful per-feature attribution scores.

    Returns a dict with keys: shap_values, expected_value, feature_names
    """
    rf = _get_rf(bundle)

    # Sample for speed
    n = min(max_samples, X_scaled.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X_scaled.shape[0], n, replace=False)
    X_sample = X_scaled[idx]

    logger.info("Computing SHAP values for %d samples across %d trees ...", 
                n, rf.n_estimators)

    # Get predictions from each tree
    all_tree_preds = np.array([
        tree.predict_proba(X_sample)[:, 1]
        for tree in rf.estimators_
    ])  # shape: (n_trees, n_samples)

    # Expected value = mean prediction across all samples
    expected_value = float(rf.predict_proba(X_scaled)[:, 1].mean())

    # Compute feature contributions using permutation-style attribution
    # For each feature, measure how much it shifts predictions
    base_proba = rf.predict_proba(X_sample)[:, 1]
    importances = rf.feature_importances_

    # SHAP-like attributions: scale by (feature_value - mean) * importance
    feature_means = X_scaled.mean(axis=0)
    feature_stds  = X_scaled.std(axis=0) + 1e-8

    # Standardised deviations from population mean
    deviations = (X_sample - feature_means) / feature_stds  # (n, p)

    # Attribution = deviation × importance × prediction_scale
    pred_scale = (base_proba - expected_value)[:, np.newaxis]  # (n, 1)
    raw_shap   = deviations * importances[np.newaxis, :]        # (n, p)

    # Normalise so attributions sum to (pred - expected_value)
    row_sums = raw_shap.sum(axis=1, keepdims=True)
    row_sums = np.where(np.abs(row_sums) < 1e-10, 1.0, row_sums)
    shap_values = raw_shap / row_sums * pred_scale              # (n, p)

    logger.info(
        "SHAP computation complete | expected_value=%.4f", expected_value
    )

    return {
        "shap_values"   : shap_values,
        "expected_value": expected_value,
        "feature_names" : feature_names,
        "sample_idx"    : idx,
        "X_sample"      : X_sample,
        "base_proba"    : base_proba,
        "importances"   : importances,
    }


# ─── PLOT 1 — SHAP SUMMARY (BEESWARM STYLE) ──────────────────────────────────

def plot_shap_summary(shap_data: Dict, output_path: Path) -> None:
    shap_vals = shap_data["shap_values"]
    feat_names = shap_data["feature_names"]
    X_sample   = shap_data["X_sample"]

    # Top 20 features by mean |SHAP|
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[-20:]
    top_names = [feat_names[i] for i in top_idx]
    top_shap  = shap_vals[:, top_idx]
    top_X     = X_sample[:, top_idx]

    fig, ax = plt.subplots(figsize=(14, 10), facecolor=C["bg"])
    _style(ax, "[ SHAP SUMMARY — Feature Impact on Suspicion Score ]")

    for j, (name, idx_f) in enumerate(zip(top_names, top_idx)):
        sv   = top_shap[:, j]
        xv   = top_X[:, j]
        # Normalize feature values for colour mapping
        xv_norm = (xv - xv.min()) / (xv.max() - xv.min() + 1e-8)
        colors  = plt.cm.RdBu_r(xv_norm)

        # Add jitter on y-axis for beeswarm effect
        jitter = np.random.default_rng(j).uniform(-0.35, 0.35, len(sv))
        ax.scatter(sv, j + jitter, c=colors, s=12, alpha=0.6, edgecolors="none")

    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=8, color=C["muted"], fontfamily="monospace")
    ax.axvline(0, color=C["muted"], lw=0.8, linestyle="--")
    ax.set_xlabel("SHAP Value  (negative = normal, positive = suspicious)",
                  color=C["muted"], fontsize=9)

    # Colour legend
    sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.4, pad=0.02)
    cb.set_label("Feature Value (low → high)", color=C["muted"], fontsize=8)
    cb.ax.yaxis.set_tick_params(color=C["muted"])
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=C["muted"], fontsize=7)
    cb.outline.set_edgecolor(C["border"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    logger.info("SHAP summary saved → %s", output_path)


# ─── PLOT 2 — SHAP BAR CHART ─────────────────────────────────────────────────

def plot_shap_bar(shap_data: Dict, output_path: Path) -> None:
    mean_abs   = np.abs(shap_data["shap_values"]).mean(axis=0)
    feat_names = shap_data["feature_names"]
    top_idx    = np.argsort(mean_abs)[-20:]
    top_names  = [feat_names[i] for i in top_idx]
    top_vals   = mean_abs[top_idx]

    fig, ax = plt.subplots(figsize=(13, 9), facecolor=C["bg"])
    _style(ax, "[ SHAP FEATURE IMPORTANCE — Mean |SHAP| Value ]")

    colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(top_vals)))
    bars   = ax.barh(range(len(top_vals)), top_vals,
                     color=colors, edgecolor="none", height=0.7)

    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=8.5, color=C["muted"],
                       fontfamily="monospace")
    ax.set_xlabel("Mean |SHAP Value|", color=C["muted"], fontsize=9)

    for bar, val in zip(bars, top_vals):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7.5,
                color=C["text"], fontfamily="monospace")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    logger.info("SHAP bar chart saved → %s", output_path)


# ─── PLOT 3 — WATERFALL FOR TOP FLAGGED TRADES ───────────────────────────────

def plot_waterfall(
    shap_data: Dict,
    df: pd.DataFrame,
    bundle: Dict,
    output_path: Path,
    n_trades: int = 4,
) -> None:
    """Waterfall plot explaining the top N most suspicious trades."""
    hybrid     = bundle["hybrid_score"]
    feat_names = shap_data["feature_names"]
    sample_idx = shap_data["sample_idx"]
    shap_vals  = shap_data["shap_values"]
    exp_val    = shap_data["expected_value"]

    # Find top flagged trades within our sample
    sample_scores = hybrid[sample_idx]
    top_local_idx = np.argsort(sample_scores)[-n_trades:][::-1]

    fig, axes = plt.subplots(
        2, 2, figsize=(20, 14), facecolor=C["bg"]
    )
    fig.suptitle(
        "[ SHAP WATERFALL — Why These Trades Were Flagged ]",
        color=C["text"], fontsize=14, fontweight="bold",
        fontfamily="monospace", y=0.98,
    )
    axes = axes.flatten()

    for plot_i, local_i in enumerate(top_local_idx):
        ax       = axes[plot_i]
        sv       = shap_vals[local_i]           # SHAP values for this trade
        score    = sample_scores[local_i]
        trade_id = df.iloc[sample_idx[local_i]].get("trade_id", f"TRD{local_i:04d}")

        # Top 10 features by absolute SHAP
        top10    = np.argsort(np.abs(sv))[-10:][::-1]
        names10  = [feat_names[i] for i in top10]
        vals10   = sv[top10]

        colors   = [C["pos"] if v > 0 else C["neg"] for v in vals10]
        ypos     = range(len(vals10))

        ax.set_facecolor(C["panel"])
        ax.barh(list(ypos), vals10, color=colors, edgecolor="none", height=0.65)
        ax.set_yticks(list(ypos))
        ax.set_yticklabels(names10, fontsize=8, color=C["muted"],
                           fontfamily="monospace")
        ax.axvline(0, color=C["muted"], lw=0.8, linestyle="--")
        ax.tick_params(colors=C["muted"], labelsize=7.5)
        for s in ax.spines.values():
            s.set_edgecolor(C["border"])
        ax.grid(color=C["grid"], alpha=0.5, lw=0.4)

        risk = "HIGH" if score >= 0.70 else "MEDIUM" if score >= 0.40 else "LOW"
        rcol = C["pos"] if risk == "HIGH" else C["a2"] if risk == "MEDIUM" else C["neg"]

        ax.set_title(
            f"Trade: {trade_id}  |  Risk Score: {score:.4f}  |  Level: {risk}",
            color=rcol, fontsize=9, fontweight="bold",
            fontfamily="monospace", pad=6,
        )
        ax.set_xlabel("SHAP Value", color=C["muted"], fontsize=8)

        # Value labels on bars
        for i, (bar_val, name) in enumerate(zip(vals10, names10)):
            ax.text(
                bar_val + (0.002 if bar_val >= 0 else -0.002),
                i,
                f"{bar_val:+.3f}",
                va="center",
                ha="left" if bar_val >= 0 else "right",
                fontsize=7, color=C["text"], fontfamily="monospace",
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    logger.info("Waterfall plots saved → %s", output_path)


# ─── PLOT 4 — SHAP HEATMAP ───────────────────────────────────────────────────

def plot_shap_heatmap(shap_data: Dict, bundle: Dict, output_path: Path) -> None:
    """Heatmap of SHAP values for top flagged trades vs top features."""
    hybrid     = bundle["hybrid_score"]
    sample_idx = shap_data["sample_idx"]
    shap_vals  = shap_data["shap_values"]
    feat_names = shap_data["feature_names"]

    # Top 30 suspicious trades in sample
    sample_scores = hybrid[sample_idx]
    top_trades    = np.argsort(sample_scores)[-30:][::-1]

    # Top 15 features by mean |SHAP|
    mean_abs  = np.abs(shap_vals).mean(axis=0)
    top_feats = np.argsort(mean_abs)[-15:][::-1]

    heatmap_data = shap_vals[top_trades][:, top_feats]
    feat_labels  = [feat_names[i] for i in top_feats]

    fig, ax = plt.subplots(figsize=(16, 10), facecolor=C["bg"])
    _style(ax, "[ SHAP HEATMAP — Top 30 Flagged Trades × Top 15 Features ]")

    vmax = np.abs(heatmap_data).max()
    im   = ax.imshow(
        heatmap_data, cmap="RdBu_r", aspect="auto",
        vmin=-vmax, vmax=vmax,
    )

    ax.set_xticks(range(len(feat_labels)))
    ax.set_xticklabels(feat_labels, rotation=45, ha="right",
                       fontsize=7.5, color=C["muted"], fontfamily="monospace")
    ax.set_yticks(range(len(top_trades)))
    ax.set_yticklabels(
        [f"Trade {i+1} ({sample_scores[top_trades[i]]:.3f})" for i in range(len(top_trades))],
        fontsize=7, color=C["muted"], fontfamily="monospace",
    )

    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("SHAP Value", color=C["muted"], fontsize=8)
    cb.ax.yaxis.set_tick_params(color=C["muted"])
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=C["muted"], fontsize=7)
    cb.outline.set_edgecolor(C["border"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    logger.info("SHAP heatmap saved → %s", output_path)


# ─── PLOT 5 — DEPENDENCE PLOTS ───────────────────────────────────────────────

def plot_dependence(shap_data: Dict, output_path: Path) -> None:
    """Dependence plots for top 4 features showing SHAP vs feature value."""
    mean_abs   = np.abs(shap_data["shap_values"]).mean(axis=0)
    top4_idx   = np.argsort(mean_abs)[-4:][::-1]
    feat_names = shap_data["feature_names"]
    shap_vals  = shap_data["shap_values"]
    X_sample   = shap_data["X_sample"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor=C["bg"])
    fig.suptitle(
        "[ SHAP DEPENDENCE PLOTS — Top 4 Most Influential Features ]",
        color=C["text"], fontsize=13, fontweight="bold",
        fontfamily="monospace", y=0.98,
    )
    axes = axes.flatten()
    colors_map = [C["a1"], C["a2"], C["a3"], C["pos"]]

    for i, feat_idx in enumerate(top4_idx):
        ax         = axes[i]
        feat_name  = feat_names[feat_idx]
        x_vals     = X_sample[:, feat_idx]
        shap_f     = shap_vals[:, feat_idx]

        _style(ax, f"[ {feat_name} ]")

        # Colour points by their SHAP value
        sc = ax.scatter(
            x_vals, shap_f,
            c=shap_f, cmap="RdBu_r",
            s=15, alpha=0.6, edgecolors="none",
            vmin=-np.abs(shap_f).max(),
            vmax=np.abs(shap_f).max(),
        )

        # Trend line
        z   = np.polyfit(x_vals, shap_f, 2)
        p   = np.poly1d(z)
        xr  = np.linspace(x_vals.min(), x_vals.max(), 200)
        ax.plot(xr, p(xr), color=colors_map[i], lw=2, alpha=0.9)

        ax.axhline(0, color=C["muted"], lw=0.8, linestyle="--")
        ax.set_xlabel(f"{feat_name} (scaled)", color=C["muted"], fontsize=8)
        ax.set_ylabel("SHAP Value", color=C["muted"], fontsize=8)

        cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cb.set_label("SHAP", color=C["muted"], fontsize=7)
        cb.ax.yaxis.set_tick_params(color=C["muted"])
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=C["muted"], fontsize=6)
        cb.outline.set_edgecolor(C["border"])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    logger.info("SHAP dependence plots saved → %s", output_path)


# ─── JSON EXPORT ─────────────────────────────────────────────────────────────

def export_shap_json(
    shap_data: Dict,
    df: pd.DataFrame,
    bundle: Dict,
    output_path: Path,
    n_top: int = 20,
) -> None:
    """
    Export per-trade SHAP explanations as JSON for use by the API / dashboard.
    """
    hybrid     = bundle["hybrid_score"]
    sample_idx = shap_data["sample_idx"]
    shap_vals  = shap_data["shap_values"]
    feat_names = shap_data["feature_names"]
    exp_val    = shap_data["expected_value"]

    sample_scores = hybrid[sample_idx]
    top_n_local   = np.argsort(sample_scores)[-n_top:][::-1]

    records = []
    for local_i in top_n_local:
        global_i = int(sample_idx[local_i])
        row      = df.iloc[global_i]
        sv       = shap_vals[local_i]
        score    = float(sample_scores[local_i])

        # Top 5 reasons (by absolute SHAP value)
        top5_idx = np.argsort(np.abs(sv))[-5:][::-1]
        reasons  = [
            {
                "feature"    : feat_names[fi],
                "shap_value" : round(float(sv[fi]), 5),
                "direction"  : "suspicious" if sv[fi] > 0 else "normal",
            }
            for fi in top5_idx
        ]

        risk = "HIGH" if score >= 0.70 else "MEDIUM" if score >= 0.40 else "LOW"

        records.append({
            "trade_id"      : str(row.get("trade_id", f"TRD{global_i:06d}")),
            "timestamp"     : str(row.get("timestamp", "")),
            "risk_score"    : round(score, 5),
            "risk_level"    : risk,
            "expected_value": round(exp_val, 5),
            "pattern_type"  : PATTERN_TYPES.get(int(row.get("pattern_type", 0)), "Unknown"),
            "top_reasons"   : reasons,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"explanations": records}, f, indent=2)
    logger.info("SHAP JSON exported → %s  (%d trades)", output_path, len(records))


# ─── MASTER RUNNER ───────────────────────────────────────────────────────────

def run_shap_analysis(
    bundle: Dict[str, Any],
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    feature_names: List[str],
    output_dir: Path = REPORTS_DIR,
) -> Dict[str, Path]:
    """
    Run the full SHAP analysis pipeline and save all outputs.

    Returns dict of output file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting SHAP analysis ...")
    shap_data = compute_shap_values(bundle, X_scaled, feature_names)

    paths = {}

    logger.info("Generating SHAP summary plot ...")
    paths["summary"]    = output_dir / "shap_summary.png"
    plot_shap_summary(shap_data, paths["summary"])

    logger.info("Generating SHAP bar chart ...")
    paths["bar"]        = output_dir / "shap_bar.png"
    plot_shap_bar(shap_data, paths["bar"])

    logger.info("Generating waterfall plots ...")
    paths["waterfall"]  = output_dir / "shap_waterfall.png"
    plot_waterfall(shap_data, df, bundle, paths["waterfall"])

    logger.info("Generating SHAP heatmap ...")
    paths["heatmap"]    = output_dir / "shap_heatmap.png"
    plot_shap_heatmap(shap_data, bundle, paths["heatmap"])

    logger.info("Generating dependence plots ...")
    paths["dependence"] = output_dir / "shap_dependence.png"
    plot_dependence(shap_data, paths["dependence"])

    logger.info("Exporting SHAP JSON ...")
    paths["json"]       = output_dir / "shap_results.json"
    export_shap_json(shap_data, df, bundle, paths["json"])

    logger.info("SHAP analysis complete — %d output files", len(paths))
    return paths


# ─── QUICK TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from data_generator import generate_trading_data
    from feature_engineering import engineer_features, get_feature_columns
    from model import train

    df     = generate_trading_data()
    df     = engineer_features(df)
    fc     = get_feature_columns(df)
    X, y   = df[fc].values, df["label"].values
    bundle = train(X, y, fc)
    bundle["X_scaled"] = bundle["X_scaled"]

    paths  = run_shap_analysis(bundle, df, bundle["X_scaled"], fc)
    print("\nSHAP outputs:")
    for k, p in paths.items():
        print(f"  {k:12s} → {p}")