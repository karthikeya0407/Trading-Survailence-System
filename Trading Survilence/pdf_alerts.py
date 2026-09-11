"""
pdf_alerts.py — PDF Report Generator + Real-time Alert System.

PDF Reports
───────────
  Generates professional compliance-ready PDF reports containing:
  - Executive summary with risk statistics
  - Top flagged trades with SHAP explanations
  - Risk distribution analysis
  - Pattern breakdown
  - Recommendations

Alert System
────────────
  Monitors trades in real-time and fires alerts for:
  - HIGH risk trades (score >= 0.70)
  - MEDIUM risk trades (score >= 0.40)
  - Suspicious pattern clusters
  - Logs all alerts to alerts.log
"""

import json
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF

from config import REPORTS_DIR, LOGS_DIR, PATTERN_TYPES, SCORING_CONFIG

logger = logging.getLogger(__name__)


# ─── COLOUR PALETTE ───────────────────────────────────────────────────────────
class PDF_COLORS:
    BG          = colors.HexColor('#050810')
    PANEL       = colors.HexColor('#090d16')
    ACCENT      = colors.HexColor('#00d4ff')
    DANGER      = colors.HexColor('#ff2d55')
    WARN        = colors.HexColor('#ff9f0a')
    SAFE        = colors.HexColor('#30d158')
    TEXT        = colors.HexColor('#c8d8f0')
    MUTED       = colors.HexColor('#4a5a7a')
    WHITE       = colors.white
    BLACK       = colors.black
    DARK_PANEL  = colors.HexColor('#0d1117')


# ─── CUSTOM STYLES ────────────────────────────────────────────────────────────
def get_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name        = 'ReportTitle',
        fontSize    = 24,
        textColor   = PDF_COLORS.WHITE,
        fontName    = 'Helvetica-Bold',
        alignment   = TA_CENTER,
        spaceAfter  = 6,
    ))
    styles.add(ParagraphStyle(
        name        = 'ReportSubTitle',
        fontSize    = 11,
        textColor   = PDF_COLORS.MUTED,
        fontName    = 'Helvetica',
        alignment   = TA_CENTER,
        spaceAfter  = 4,
    ))
    styles.add(ParagraphStyle(
        name        = 'SectionHeader',
        fontSize    = 13,
        textColor   = PDF_COLORS.ACCENT,
        fontName    = 'Helvetica-Bold',
        spaceBefore = 14,
        spaceAfter  = 6,
    ))
    styles.add(ParagraphStyle(
        name        = 'BodyText2',
        fontSize    = 9,
        textColor   = PDF_COLORS.TEXT,
        fontName    = 'Helvetica',
        spaceAfter  = 4,
        leading     = 14,
    ))
    styles.add(ParagraphStyle(
        name        = 'MonoSmall',
        fontSize    = 8,
        textColor   = PDF_COLORS.MUTED,
        fontName    = 'Courier',
        spaceAfter  = 2,
    ))
    styles.add(ParagraphStyle(
        name        = 'RiskHigh',
        fontSize    = 9,
        textColor   = PDF_COLORS.DANGER,
        fontName    = 'Helvetica-Bold',
    ))
    styles.add(ParagraphStyle(
        name        = 'RiskMed',
        fontSize    = 9,
        textColor   = PDF_COLORS.WARN,
        fontName    = 'Helvetica-Bold',
    ))
    styles.add(ParagraphStyle(
        name        = 'RiskLow',
        fontSize    = 9,
        textColor   = PDF_COLORS.SAFE,
        fontName    = 'Helvetica-Bold',
    ))
    return styles


# ─── RISK COLOUR HELPER ───────────────────────────────────────────────────────
def risk_color(level: str) -> colors.Color:
    return (
        PDF_COLORS.DANGER if level == 'HIGH'   else
        PDF_COLORS.WARN   if level == 'MEDIUM' else
        PDF_COLORS.SAFE
    )


# ─── STAT BOX DRAWING ────────────────────────────────────────────────────────
def make_stat_box(label: str, value: str, color: colors.Color) -> Drawing:
    d = Drawing(110, 60)
    d.add(Rect(0, 0, 110, 60,
               fillColor=PDF_COLORS.DARK_PANEL,
               strokeColor=color, strokeWidth=1.5))
    d.add(String(55, 38, value,
                 fontName='Helvetica-Bold', fontSize=20,
                 fillColor=color, textAnchor='middle'))
    d.add(String(55, 18, label,
                 fontName='Helvetica', fontSize=7,
                 fillColor=PDF_COLORS.MUTED, textAnchor='middle'))
    return d


# ─── MINI BAR CHART ──────────────────────────────────────────────────────────
def make_bar_chart(data: List[float], labels: List[str],
                   bar_colors: List) -> Drawing:
    d   = Drawing(440, 130)
    bc  = VerticalBarChart()
    bc.x, bc.y, bc.width, bc.height = 30, 20, 390, 90
    bc.data = [data]
    bc.bars[0].fillColor = PDF_COLORS.ACCENT

    # Per-bar colors
    for i, c in enumerate(bar_colors):
        bc.bars[(0, i)].fillColor = c

    bc.categoryAxis.categoryNames       = labels
    bc.categoryAxis.labels.fontName     = 'Courier'
    bc.categoryAxis.labels.fontSize     = 6
    bc.categoryAxis.labels.fillColor    = PDF_COLORS.MUTED
    bc.categoryAxis.labels.angle        = 30
    bc.valueAxis.labels.fontName        = 'Courier'
    bc.valueAxis.labels.fontSize        = 6
    bc.valueAxis.labels.fillColor       = PDF_COLORS.MUTED
    bc.valueAxis.gridStrokeColor        = PDF_COLORS.PANEL
    bc.valueAxis.visibleGrid            = True
    bc.strokeColor                      = None
    bc.fillColor                        = PDF_COLORS.DARK_PANEL
    d.add(bc)
    return d


# ─── PDF REPORT GENERATOR ─────────────────────────────────────────────────────

class SurveillanceReportPDF:
    """Generates a professional PDF surveillance report."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.styles      = get_styles()
        self.story       = []
        self.doc         = SimpleDocTemplate(
            str(output_path),
            pagesize        = A4,
            leftMargin      = 1.8 * cm,
            rightMargin     = 1.8 * cm,
            topMargin       = 2.0 * cm,
            bottomMargin    = 2.0 * cm,
        )

    def _hr(self, color=None):
        self.story.append(HRFlowable(
            width='100%', thickness=0.5,
            color=color or PDF_COLORS.MUTED,
            spaceAfter=8, spaceBefore=4,
        ))

    def _space(self, h=8):
        self.story.append(Spacer(1, h))

    def _h(self, text: str, style='SectionHeader'):
        self.story.append(Paragraph(text, self.styles[style]))

    def _p(self, text: str, style='BodyText2'):
        self.story.append(Paragraph(text, self.styles[style]))

    # ── COVER PAGE ────────────────────────────────────────────
    def add_cover(self, stats: Dict):
        self._space(20)
        self._h('TRADING SURVEILLANCE SYSTEM', 'ReportTitle')
        self._h('SUSPICIOUS ACTIVITY DETECTION REPORT', 'ReportSubTitle')
        self._space(4)
        self._p(
            f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}',
            'MonoSmall'
        )
        self._hr(PDF_COLORS.ACCENT)
        self._space(16)

        # Stat boxes row
        p = stats.get('performance', {})
        d = stats.get('dataset', {})

        table_data = [[
            make_stat_box('TOTAL TRADES',    str(d.get('n_total', '--')),        PDF_COLORS.ACCENT),
            make_stat_box('SUSPICIOUS',      str(d.get('n_suspicious', '--')),   PDF_COLORS.DANGER),
            make_stat_box('ROC-AUC',         str(p.get('roc_auc', '--')),        PDF_COLORS.SAFE),
            make_stat_box('F1-SCORE',        str(p.get('f1_score', '--')),       PDF_COLORS.WARN),
        ]]
        t = Table(table_data, colWidths=[4.2*cm]*4)
        t.setStyle(TableStyle([
            ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
            ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
            ('BACKGROUND', (0,0), (-1,-1), PDF_COLORS.BG),
            ('GRID',       (0,0), (-1,-1), 0, PDF_COLORS.BG),
        ]))
        self.story.append(t)
        self._space(20)

        # Executive summary
        self._h('EXECUTIVE SUMMARY')
        susp_pct = d.get('suspicious_pct', 0)
        self._p(
            f'This report presents the findings of the ML-powered trading '
            f'surveillance system. Out of <b>{d.get("n_total","N/A"):,}</b> '
            f'trades analyzed, <b>{d.get("n_suspicious","N/A"):,}</b> '
            f'({susp_pct:.1f}%) were flagged as suspicious. '
            f'The model achieved a ROC-AUC of '
            f'<b>{p.get("roc_auc","N/A")}</b> with precision '
            f'<b>{p.get("precision","N/A")}</b> and recall '
            f'<b>{p.get("recall","N/A")}</b>.'
        )
        self._space(8)
        self._p(
            'The following manipulation patterns were detected and '
            'are detailed in subsequent sections: '
            '<b>Pump &amp; Dump, Spoofing, Layering, '
            'Wash Trading, and Front Running.</b>'
        )
        self.story.append(PageBreak())

    # ── MODEL PERFORMANCE ─────────────────────────────────────
    def add_model_performance(self, stats: Dict):
        self._h('MODEL PERFORMANCE METRICS')
        self._hr()

        p = stats.get('performance', {})
        metrics = [
            ('ROC-AUC Score',        p.get('roc_auc',       '--'), PDF_COLORS.SAFE),
            ('Average Precision',    p.get('avg_precision', '--'), PDF_COLORS.ACCENT),
            ('Precision (Susp.)',    p.get('precision',     '--'), PDF_COLORS.WARN),
            ('Recall (Susp.)',       p.get('recall',        '--'), PDF_COLORS.WARN),
            ('F1-Score (Susp.)',     p.get('f1_score',      '--'), PDF_COLORS.ACCENT),
            ('Accuracy',             p.get('accuracy',      '--'), PDF_COLORS.SAFE),
            ('CV AUC Mean',          p.get('cv_auc_mean',   '--'), PDF_COLORS.SAFE),
            ('CV AUC Std',           p.get('cv_auc_std',    '--'), PDF_COLORS.MUTED),
        ]

        table_data = [['METRIC', 'VALUE', 'ASSESSMENT']]
        for label, val, col in metrics:
            assessment = (
                'EXCELLENT' if isinstance(val, float) and val >= 0.95 else
                'GOOD'      if isinstance(val, float) and val >= 0.80 else
                'FAIR'
            )
            table_data.append([label, str(val), assessment])

        t = Table(table_data, colWidths=[7*cm, 4*cm, 5.5*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND',  (0,0), (-1,0),  PDF_COLORS.DARK_PANEL),
            ('TEXTCOLOR',   (0,0), (-1,0),  PDF_COLORS.ACCENT),
            ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
            ('FONTSIZE',    (0,0), (-1,-1), 8),
            ('FONTNAME',    (0,1), (-1,-1), 'Courier'),
            ('TEXTCOLOR',   (0,1), (-1,-1), PDF_COLORS.TEXT),
            ('BACKGROUND',  (0,1), (-1,-1), PDF_COLORS.PANEL),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[PDF_COLORS.PANEL, PDF_COLORS.DARK_PANEL]),
            ('GRID',        (0,0), (-1,-1), 0.3, PDF_COLORS.MUTED),
            ('ALIGN',       (1,0), (-1,-1), 'CENTER'),
            ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING',  (0,0), (-1,-1), 6),
            ('BOTTOMPADDING',(0,0),(-1,-1), 6),
        ]))
        self.story.append(t)
        self._space(12)

        # Models used
        self._h('MODELS USED IN ENSEMBLE')
        models = stats.get('models_used', [])
        for m in models:
            self._p(f'  • {m}', 'MonoSmall')

    # ── FLAGGED TRADES ────────────────────────────────────────
    def add_flagged_trades(self, trades: List[Dict]):
        self.story.append(PageBreak())
        self._h('TOP FLAGGED TRADES')
        self._hr(PDF_COLORS.DANGER)
        self._p(
            f'The following {len(trades)} trades were identified as '
            f'highest risk by the ML ensemble. Each entry includes '
            f'the primary SHAP feature attributions explaining '
            f'the detection.'
        )
        self._space(8)

        table_data = [['#', 'TRADE ID', 'RISK SCORE', 'LEVEL', 'PATTERN', 'TOP REASON']]

        for i, trade in enumerate(trades[:15]):
            level  = trade.get('risk_level', 'LOW')
            reason = ''
            if trade.get('top_reasons'):
                r      = trade['top_reasons'][0]
                reason = f"{r.get('feature','N/A')} ({r.get('direction','N/A')})"

            table_data.append([
                str(i + 1),
                trade.get('trade_id', '--'),
                f"{trade.get('risk_score', 0):.4f}",
                level,
                trade.get('pattern_type', '--'),
                reason[:32],
            ])

        col_w = [0.8*cm, 2.8*cm, 2.5*cm, 2.2*cm, 3.8*cm, 5.5*cm]
        t     = Table(table_data, colWidths=col_w)

        style = [
            ('BACKGROUND',  (0,0), (-1,0),  PDF_COLORS.DARK_PANEL),
            ('TEXTCOLOR',   (0,0), (-1,0),  PDF_COLORS.ACCENT),
            ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
            ('FONTSIZE',    (0,0), (-1,-1), 7.5),
            ('FONTNAME',    (0,1), (-1,-1), 'Courier'),
            ('TEXTCOLOR',   (0,1), (-1,-1), PDF_COLORS.TEXT),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[PDF_COLORS.PANEL, PDF_COLORS.DARK_PANEL]),
            ('GRID',        (0,0), (-1,-1), 0.3, PDF_COLORS.MUTED),
            ('ALIGN',       (0,0), (-1,-1), 'CENTER'),
            ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING',  (0,0), (-1,-1), 5),
            ('BOTTOMPADDING',(0,0),(-1,-1), 5),
        ]

        # Color risk level cells
        for row_i, trade in enumerate(trades[:15], start=1):
            level = trade.get('risk_level', 'LOW')
            col   = risk_color(level)
            style.append(('TEXTCOLOR', (3, row_i), (3, row_i), col))
            style.append(('FONTNAME',  (3, row_i), (3, row_i), 'Helvetica-Bold'))

        t.setStyle(TableStyle(style))
        self.story.append(t)

    # ── PATTERN ANALYSIS ─────────────────────────────────────
    def add_pattern_analysis(self):
        self.story.append(PageBreak())
        self._h('SUSPICIOUS PATTERN ANALYSIS')
        self._hr(PDF_COLORS.WARN)

        patterns_info = [
            ('Pump & Dump',    'Artificial price inflation through coordinated buying, followed by rapid selling at peak price.',       PDF_COLORS.DANGER),
            ('Spoofing',       'Placing large orders with no intention to execute, creating false market depth to mislead traders.',     PDF_COLORS.WARN),
            ('Layering',       'Multiple orders at different price levels, cancelled immediately after affecting other market participants.',PDF_COLORS.ACCENT),
            ('Wash Trading',   'Trading with oneself to create artificial volume and price movement without genuine market participation.',PDF_COLORS.SAFE),
            ('Front Running',  'Trading ahead of known pending large client orders to profit from the anticipated price movement.',      PDF_COLORS.MUTED),
        ]

        for name, desc, col in patterns_info:
            self._space(6)
            d = Drawing(460, 4)
            d.add(Rect(0, 0, 460, 4, fillColor=col, strokeColor=None))
            self.story.append(d)
            self._space(4)
            self._p(f'<b><font color="#{col.hexval()[2:]}">{name}</font></b>')
            self._p(desc)

    # ── RECOMMENDATIONS ──────────────────────────────────────
    def add_recommendations(self):
        self.story.append(PageBreak())
        self._h('REGULATORY RECOMMENDATIONS')
        self._hr(PDF_COLORS.SAFE)

        recs = [
            ('Immediate Review',   'All HIGH risk trades (score >= 0.70) should be reviewed by compliance officers within 24 hours.'),
            ('Pattern Monitoring', 'Traders with repeated spoofing or layering signals should be placed on enhanced monitoring watchlists.'),
            ('Regulatory Filing',  'Trades matching wash trading criteria should be reported to the relevant regulatory authority per applicable laws.'),
            ('Model Retraining',   'The ML model should be retrained monthly with newly labeled data to maintain detection accuracy.'),
            ('Alert Thresholds',   'Consider lowering the review threshold to 0.35 during periods of elevated market volatility.'),
            ('Network Analysis',   'Review trader network clusters flagged as coordinated manipulation rings for potential collusion.'),
        ]

        table_data = [['PRIORITY', 'ACTION', 'DESCRIPTION']]
        priorities = ['CRITICAL', 'HIGH', 'HIGH', 'MEDIUM', 'MEDIUM', 'MEDIUM']
        p_colors   = [PDF_COLORS.DANGER, PDF_COLORS.DANGER, PDF_COLORS.WARN,
                      PDF_COLORS.ACCENT, PDF_COLORS.ACCENT, PDF_COLORS.ACCENT]

        for i, (action, desc) in enumerate(recs):
            table_data.append([priorities[i], action, desc])

        t = Table(table_data, colWidths=[2.5*cm, 4*cm, 11*cm])
        style = [
            ('BACKGROUND',  (0,0), (-1,0),  PDF_COLORS.DARK_PANEL),
            ('TEXTCOLOR',   (0,0), (-1,0),  PDF_COLORS.ACCENT),
            ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
            ('FONTSIZE',    (0,0), (-1,-1), 8),
            ('FONTNAME',    (0,1), (-1,-1), 'Helvetica'),
            ('TEXTCOLOR',   (0,1), (-1,-1), PDF_COLORS.TEXT),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[PDF_COLORS.PANEL, PDF_COLORS.DARK_PANEL]),
            ('GRID',        (0,0), (-1,-1), 0.3, PDF_COLORS.MUTED),
            ('VALIGN',      (0,0), (-1,-1), 'TOP'),
            ('TOPPADDING',  (0,0), (-1,-1), 6),
            ('BOTTOMPADDING',(0,0),(-1,-1), 6),
            ('ALIGN',       (0,0), (0,-1),  'CENTER'),
        ]
        for i, col in enumerate(p_colors, start=1):
            style.append(('TEXTCOLOR', (0,i), (0,i), col))
            style.append(('FONTNAME',  (0,i), (0,i), 'Helvetica-Bold'))
        t.setStyle(TableStyle(style))
        self.story.append(t)

    # ── FOOTER PAGE ──────────────────────────────────────────
    def add_footer(self):
        self._space(20)
        self._hr(PDF_COLORS.MUTED)
        self._p(
            'CONFIDENTIAL — This report was generated by the Trading '
            'Surveillance System ML Pipeline. For regulatory and '
            'compliance use only.',
            'MonoSmall'
        )
        self._p(
            f'Report ID: TSS-{datetime.now().strftime("%Y%m%d%H%M%S")} | '
            f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'MonoSmall'
        )

    # ── BUILD ─────────────────────────────────────────────────
    def build(self, stats: Dict, trades: List[Dict]):
        """Assemble and write the PDF."""
        def on_page(canvas, doc):
            canvas.saveState()
            canvas.setFillColor(PDF_COLORS.BG)
            canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
            canvas.setStrokeColor(PDF_COLORS.MUTED)
            canvas.setLineWidth(0.3)
            canvas.line(1.8*cm, 1.5*cm, A4[0]-1.8*cm, 1.5*cm)
            canvas.setFont('Helvetica', 7)
            canvas.setFillColor(PDF_COLORS.MUTED)
            canvas.drawString(1.8*cm, 1.0*cm, 'TRADING SURVEILLANCE SYSTEM — CONFIDENTIAL')
            canvas.drawRightString(A4[0]-1.8*cm, 1.0*cm, f'Page {doc.page}')
            canvas.restoreState()

        self.add_cover(stats)
        self.add_model_performance(stats)
        self.add_flagged_trades(trades)
        self.add_pattern_analysis()
        self.add_recommendations()
        self.add_footer()

        self.doc.build(
            self.story,
            onFirstPage=on_page,
            onLaterPages=on_page,
        )
        logger.info("PDF report saved to %s", self.output_path)


# ─── ALERT SYSTEM ─────────────────────────────────────────────────────────────

class AlertSystem:
    """Real-time alert system for suspicious trade detection."""

    def __init__(self, log_path: Path = None):
        self.log_path  = log_path or (LOGS_DIR / "alerts.log")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.alerts    = []
        self._setup_logger()

    def _setup_logger(self):
        self.alert_logger = logging.getLogger("alerts")
        fh = logging.FileHandler(self.log_path, encoding='utf-8')
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s"
        ))
        self.alert_logger.addHandler(fh)
        self.alert_logger.setLevel(logging.INFO)

    def check_trade(self, trade_id: str, risk_score: float,
                    risk_level: str, top_reason: str = '') -> Dict:
        """Check a single trade and fire alert if threshold exceeded."""
        alert = None
        threshold = SCORING_CONFIG['review_threshold']
        alert_thr = SCORING_CONFIG['alert_threshold']

        if risk_score >= alert_thr:
            alert = {
                "alert_id"   : f"ALT{len(self.alerts)+1:06d}",
                "trade_id"   : trade_id,
                "risk_score" : round(risk_score, 4),
                "risk_level" : risk_level,
                "alert_type" : "HIGH_RISK_TRADE",
                "message"    : f"HIGH RISK trade detected! Score={risk_score:.4f} Reason={top_reason}",
                "timestamp"  : datetime.now().isoformat(),
                "severity"   : "CRITICAL",
            }
            self.alert_logger.warning(alert["message"])
            print(f"\n  *** ALERT *** {alert['message']}")

        elif risk_score >= threshold:
            alert = {
                "alert_id"   : f"ALT{len(self.alerts)+1:06d}",
                "trade_id"   : trade_id,
                "risk_score" : round(risk_score, 4),
                "risk_level" : risk_level,
                "alert_type" : "REVIEW_REQUIRED",
                "message"    : f"MEDIUM RISK trade flagged for review. Score={risk_score:.4f}",
                "timestamp"  : datetime.now().isoformat(),
                "severity"   : "WARNING",
            }
            self.alert_logger.info(alert["message"])

        if alert:
            self.alerts.append(alert)
        return alert

    def scan_trades(self, trades: List[Dict]) -> Dict:
        """Scan a list of trades and fire alerts for suspicious ones."""
        high_count   = 0
        medium_count = 0

        print(f"\n  Scanning {len(trades)} trades for alerts...")
        print("  " + "─" * 50)

        for trade in trades:
            score  = trade.get('risk_score', 0)
            level  = trade.get('risk_level', 'LOW')
            tid    = trade.get('trade_id', 'UNKNOWN')
            reason = ''
            if trade.get('top_reasons'):
                reason = trade['top_reasons'][0].get('feature', '')

            alert = self.check_trade(tid, score, level, reason)
            if alert:
                if level == 'HIGH':   high_count   += 1
                if level == 'MEDIUM': medium_count += 1

        print(f"\n  Scan complete!")
        print(f"  HIGH risk alerts   : {high_count}")
        print(f"  MEDIUM risk alerts : {medium_count}")
        print(f"  Alert log          : {self.log_path}")
        print("  " + "─" * 50)

        return {
            "total_scanned"  : len(trades),
            "high_alerts"    : high_count,
            "medium_alerts"  : medium_count,
            "total_alerts"   : len(self.alerts),
            "alerts"         : self.alerts,
        }

    def save_alert_json(self, output_path: Path = None) -> Path:
        """Save all alerts to JSON."""
        path = output_path or (REPORTS_DIR / "alerts.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                "generated_at" : datetime.now().isoformat(),
                "total_alerts" : len(self.alerts),
                "alerts"       : self.alerts,
            }, f, indent=2)
        logger.info("Alerts saved to %s", path)
        return path


# ─── MASTER RUNNER ────────────────────────────────────────────────────────────

def run_pdf_and_alerts(output_dir: Path = REPORTS_DIR) -> Dict[str, Path]:
    """Run the full PDF generation and alert scanning pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load stats
    stats_path = output_dir / "results_summary.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            "results_summary.json not found! Run train_pipeline.py first."
        )
    with open(stats_path) as f:
        stats = json.load(f)

    # Load SHAP flagged trades
    shap_path = output_dir / "shap_results.json"
    trades    = []
    if shap_path.exists():
        with open(shap_path) as f:
            trades = json.load(f).get("explanations", [])
    else:
        logger.warning("shap_results.json not found — PDF will have no trade details")

    # ── Generate PDF ──────────────────────────────────────────
    logger.info("Generating PDF report ...")
    pdf_path = output_dir / f"surveillance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    report   = SurveillanceReportPDF(pdf_path)
    report.build(stats, trades)

    # ── Run Alert System ──────────────────────────────────────
    logger.info("Running alert scanner ...")
    alert_sys  = AlertSystem()
    alert_data = alert_sys.scan_trades(trades)
    alert_path = alert_sys.save_alert_json(output_dir / "alerts.json")

    paths = {
        "pdf"    : pdf_path,
        "alerts" : alert_path,
    }

    logger.info("Phase 6 complete!")
    logger.info("PDF report   : %s", pdf_path)
    logger.info("Alert log    : %s", alert_sys.log_path)
    logger.info("Alerts JSON  : %s", alert_path)

    return paths, alert_data


# ─── QUICK TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s"
    )

    print("\n" + "="*55)
    print("  TRADING SURVEILLANCE — PDF REPORTS + ALERTS")
    print("="*55)

    paths, alert_data = run_pdf_and_alerts()

    print(f"\n  PDF Report     : {paths['pdf']}")
    print(f"  Alerts JSON    : {paths['alerts']}")
    print(f"  High Alerts    : {alert_data['high_alerts']}")
    print(f"  Medium Alerts  : {alert_data['medium_alerts']}")
    print(f"  Total Alerts   : {alert_data['total_alerts']}")
    print("\n  Phase 6 COMPLETE!")
    print("="*55)