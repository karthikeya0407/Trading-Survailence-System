"""
visualizer.py — Generate the full analysis dashboard as a high-res PNG.

Panels
──────
Row 0 : ROC curve | PR curve | Confusion matrix | Metrics scorecard
Row 1 : Feature importances (top-20) | Score distribution
Row 2 : PCA 2-D projection | IsoForest vs LOF scatter
Row 3 : Feature comparison by class | Top-10 flagged trades table
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from pathlib import Path
from typing import Dict, Any, List

from config import PATTERN_TYPES, REPORTS_DIR

logger = logging.getLogger(__name__)

# ─── PALETTE ─────────────────────────────────────────────────────────────────
C = {
    "bg"        : "#07090f",
    "panel"     : "#0d1117",
    "border"    : "#1c2333",
    "grid"      : "#161c27",
    "text"      : "#cdd9f5",
    "muted"     : "#5c6f94",
    "normal"    : "#38bdf8",
    "susp"      : "#f43f5e",
    "a1"        : "#38bdf8",
    "a2"        : "#f97316",
    "a3"        : "#22c55e",
    "a4"        : "#a855f7",
    "a5"        : "#eab308",
}


def _style(ax, title: str = "") -> None:
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["muted"], labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor(C["border"])
    if title:
        ax.set_title(
            title, color=C["text"], fontsize=10,
            fontweight="bold", fontfamily="monospace", pad=9,
        )
    ax.grid(color=C["grid"], alpha=0.6, linewidth=0.4)


def _label(ax, txt: str, color=None) -> None:
    ax.set_xlabel(txt, color=C["muted"], fontsize=8)


def _ylabel(ax, txt: str) -> None:
    ax.set_ylabel(txt, color=C["muted"], fontsize=8)


# ─── INDIVIDUAL PANELS ───────────────────────────────────────────────────────

def _panel_roc(ax, bundle: Dict) -> None:
    y = bundle["y_true"]
    pairs = [
        (bundle["hybrid_score"],     "Hybrid",      C["a3"]),
        (bundle["supervised_proba"], "Ensemble",    C["a1"]),
        (bundle["iso_scores"],       "IsoForest",   C["a2"]),
    ]
    for proba, name, col in pairs:
        fpr, tpr, _ = roc_curve(y, proba)
        auc = roc_auc_score(y, proba)
        ax.plot(fpr, tpr, color=col, lw=1.8, label=f"{name} {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=C["muted"], lw=0.8)
    _label(ax, "FPR"); _ylabel(ax, "TPR")
    ax.legend(fontsize=7.5, facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["text"])


def _panel_pr(ax, bundle: Dict) -> None:
    y = bundle["y_true"]
    pairs = [
        (bundle["hybrid_score"],     "Hybrid",   C["a3"]),
        (bundle["supervised_proba"], "Ensemble", C["a1"]),
    ]
    for proba, name, col in pairs:
        prec, rec, _ = precision_recall_curve(y, proba)
        ap = average_precision_score(y, proba)
        ax.plot(rec, prec, color=col, lw=1.8, label=f"{name} AP={ap:.3f}")
    ax.axhline(y.mean(), color=C["muted"], linestyle="--", lw=0.8, label="Baseline")
    _label(ax, "Recall"); _ylabel(ax, "Precision")
    ax.legend(fontsize=7.5, facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["text"])


def _panel_confusion(ax, bundle: Dict) -> None:
    cm = bundle["confusion_matrix"]
    ax.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white", fontsize=16, fontweight="bold", fontfamily="monospace")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Susp."], color=C["muted"], fontsize=9)
    ax.set_yticklabels(["Normal", "Susp."], color=C["muted"], fontsize=9)
    _label(ax, "Predicted"); _ylabel(ax, "Actual")


def _panel_metrics(ax, bundle: Dict) -> None:
    ax.set_facecolor(C["panel"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    r = bundle["report"]
    items = [
        ("ROC-AUC",        f"{bundle['roc_auc']:.4f}",                  C["a3"]),
        ("Avg Precision",  f"{bundle['avg_precision']:.4f}",             C["a1"]),
        ("Precision",      f"{r['1']['precision']:.4f}",                 C["a2"]),
        ("Recall",         f"{r['1']['recall']:.4f}",                    C["a4"]),
        ("F1-Score",       f"{r['1']['f1-score']:.4f}",                  C["a3"]),
        ("Accuracy",       f"{r['accuracy']:.4f}",                       C["a1"]),
        ("CV AUC mean",    f"{bundle['cv_scores'].mean():.4f}",          C["a2"]),
        ("CV AUC std",     f"± {bundle['cv_scores'].std():.4f}",         C["muted"]),
    ]
    for i, (label, val, col) in enumerate(items):
        yp = 0.91 - i * 0.11
        ax.text(0.05, yp, label, color=C["muted"], fontsize=8.5,
                fontfamily="monospace", va="center")
        ax.text(0.97, yp, val, color=col, fontsize=9,
                fontfamily="monospace", va="center", ha="right", fontweight="bold")
        ax.axhline(yp - 0.05, color=C["border"], lw=0.3)


def _panel_importances(ax, bundle: Dict) -> None:
    feat   = bundle["feature_names"]
    imps   = bundle["importances"]
    idx    = np.argsort(imps)[-20:]
    names  = [feat[i] for i in idx]
    vals   = imps[idx]
    colors = plt.cm.plasma(np.linspace(0.15, 0.95, len(vals)))
    bars   = ax.barh(range(len(vals)), vals, color=colors, height=0.72, edgecolor="none")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=7, color=C["muted"], fontfamily="monospace")
    _label(ax, "Importance")
    for b, v in zip(bars, vals):
        ax.text(v + 0.001, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=6.5, color=C["text"],
                fontfamily="monospace")


def _panel_score_dist(ax, bundle: Dict) -> None:
    y      = bundle["y_true"]
    score  = bundle["hybrid_score"]
    ax.hist(score[y == 0], bins=70, alpha=0.55, color=C["normal"],
            label="Normal", density=True, edgecolor="none")
    ax.hist(score[y == 1], bins=70, alpha=0.65, color=C["susp"],
            label="Suspicious", density=True, edgecolor="none")
    ax.axvline(0.40, color=C["a5"], linestyle="--", lw=1.8, label="Review (0.40)")
    ax.axvline(0.70, color=C["susp"], linestyle="--", lw=1.8, label="Alert (0.70)")
    _label(ax, "Hybrid Risk Score"); _ylabel(ax, "Density")
    ax.legend(fontsize=7.5, facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["text"])


def _panel_pca(ax, bundle: Dict) -> None:
    pca   = PCA(n_components=2, random_state=42)
    Xp    = pca.fit_transform(bundle["X_scaled"])
    y     = bundle["y_true"]
    idx   = np.random.default_rng(0).choice(len(y), min(2500, len(y)), replace=False)
    mask0 = y[idx] == 0
    ax.scatter(Xp[idx[mask0], 0],  Xp[idx[mask0], 1],
               c=C["normal"], alpha=0.25, s=7, edgecolors="none", label="Normal")
    ax.scatter(Xp[idx[~mask0], 0], Xp[idx[~mask0], 1],
               c=C["susp"],   alpha=0.70, s=10, edgecolors="none", label="Suspicious")
    _label(ax, f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    _ylabel(ax, f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(fontsize=7.5, facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["text"])


def _panel_iso_lof(ax, bundle: Dict) -> None:
    y     = bundle["y_true"]
    iso   = bundle["iso_scores"]
    lof   = bundle["lof_scores"]
    rng   = np.random.default_rng(1)
    idx   = rng.choice(len(y), min(1800, len(y)), replace=False)
    cols  = [C["susp"] if l == 1 else C["normal"] for l in y[idx]]
    sizes = [14 if l == 1 else 5 for l in y[idx]]
    alpha = [0.75 if l == 1 else 0.18 for l in y[idx]]
    for i in range(len(idx)):
        ax.scatter(iso[idx[i]], lof[idx[i]], c=cols[i],
                   s=sizes[i], alpha=alpha[i], edgecolors="none")
    h = [mpatches.Patch(color=C["normal"], label="Normal"),
         mpatches.Patch(color=C["susp"],   label="Suspicious")]
    ax.legend(handles=h, fontsize=7.5, facecolor=C["bg"],
              edgecolor=C["border"], labelcolor=C["text"])
    _label(ax, "IsoForest Anomaly Score"); _ylabel(ax, "LOF Score")


def _panel_feature_compare(ax, df: pd.DataFrame, bundle: Dict) -> None:
    feat  = bundle["feature_names"]
    imps  = bundle["importances"]
    top10 = [feat[i] for i in np.argsort(imps)[-10:]]
    n_means = df[df["label"] == 0][top10].mean()
    s_means = df[df["label"] == 1][top10].mean()
    mx    = np.maximum(n_means.values, s_means.values)
    mx    = np.where(mx == 0, 1, mx)
    w     = 0.35
    pos   = np.arange(len(top10))
    ax.bar(pos - w/2, n_means.values / mx, w, color=C["normal"], alpha=0.7,
           label="Normal", edgecolor="none")
    ax.bar(pos + w/2, s_means.values / mx, w, color=C["susp"],   alpha=0.7,
           label="Suspicious", edgecolor="none")
    ax.set_xticks(pos)
    ax.set_xticklabels([f.replace("_", "\n") for f in top10],
                       fontsize=6.5, color=C["muted"], fontfamily="monospace")
    _ylabel(ax, "Normalised Mean")
    ax.legend(fontsize=7.5, facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["text"])


def _panel_top_flagged(ax, df: pd.DataFrame, bundle: Dict) -> None:
    ax.set_facecolor(C["panel"]); ax.axis("off")
    df2           = df.copy()
    df2["score"]  = bundle["hybrid_score"]
    df2["pred"]   = bundle["predictions"]
    top           = df2[df2["pred"] == 1].nlargest(10, "score")
    header = f"{'#':<3} {'TRADE_ID':<12} {'VOLUME':>10} {'FREQ':>5} {'CANCEL':>7} {'RISK_SCORE':>11}"
    ax.text(0.5, 0.97, header, color=C["a1"], fontsize=7.8,
            fontfamily="monospace", transform=ax.transAxes,
            va="top", ha="center")
    ax.plot([0, 1], [0.93, 0.93], color=C["border"], lw=0.4, transform=ax.transAxes)
    for rank, (_, row) in enumerate(top.iterrows()):
        yp    = 0.89 - rank * 0.085
        score = row["score"]
        rcol  = C["susp"] if score >= 0.70 else C["a5"] if score >= 0.40 else C["a3"]
        line  = (f"{rank+1:<3} {row.get('trade_id','N/A'):<12} "
                 f"{row['trade_volume']:>10,.0f} {row['trade_frequency']:>5.0f} "
                 f"{row['cancel_rate']:>7.3f}")
        ax.text(0.04, yp, line, color=C["text"], fontsize=7.5,
                fontfamily="monospace", transform=ax.transAxes, va="top")
        ax.text(0.96, yp, f"{score:.4f}", color=rcol, fontsize=8,
                fontfamily="monospace", transform=ax.transAxes, va="top",
                ha="right", fontweight="bold")


# ─── MASTER DASHBOARD ────────────────────────────────────────────────────────

def create_dashboard(
    df: pd.DataFrame,
    bundle: Dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    """
    Render the full 10-panel surveillance dashboard and save to PNG.

    Returns the saved file path.
    """
    output_path = output_path or (REPORTS_DIR / "dashboard.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(26, 30), facecolor=C["bg"])
    fig.suptitle(
        "TRADING SURVEILLANCE SYSTEM  ·  ML ANALYSIS DASHBOARD",
        fontsize=20, fontweight="bold", color=C["text"],
        fontfamily="monospace", y=0.985,
    )

    gs = GridSpec(4, 4, figure=fig, hspace=0.48, wspace=0.42,
                  top=0.965, bottom=0.03, left=0.06, right=0.97)

    panels = [
        (gs[0, 0], "[ ROC CURVES ]",         _panel_roc,         (bundle,)),
        (gs[0, 1], "[ PRECISION-RECALL ]",   _panel_pr,          (bundle,)),
        (gs[0, 2], "[ CONFUSION MATRIX ]",   _panel_confusion,   (bundle,)),
        (gs[0, 3], "[ METRICS ]",             _panel_metrics,     (bundle,)),
        (gs[1, :2],"[ FEATURE IMPORTANCE ]", _panel_importances, (bundle,)),
        (gs[1, 2:],"[ RISK SCORE DIST. ]",   _panel_score_dist,  (bundle,)),
        (gs[2, :2],"[ PCA PROJECTION ]",     _panel_pca,         (bundle,)),
        (gs[2, 2:],"[ ISO FOREST vs LOF ]",  _panel_iso_lof,     (bundle,)),
        (gs[3, :2],"[ FEATURE COMPARISON ]", _panel_feature_compare, (df, bundle)),
        (gs[3, 2:],"[ TOP FLAGGED TRADES ]", _panel_top_flagged, (df, bundle)),
    ]

    for spec, title, fn, args in panels:
        ax = fig.add_subplot(spec)
        _style(ax, title)
        fn(ax, *args)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    logger.info("Dashboard saved → %s", output_path)
    return output_path


# ─── QUICK TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from data_generator import generate_trading_data
    from feature_engineering import engineer_features, get_feature_columns
    from model import train

    df  = generate_trading_data()
    df  = engineer_features(df)
    fc  = get_feature_columns(df)
    X, y = df[fc].values, df["label"].values
    bundle = train(X, y, fc)
    path = create_dashboard(df, bundle)
    print(f"Saved: {path}")