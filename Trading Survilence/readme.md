# 🛡️ Trading Surveillance System
### ML-Powered Suspicious Trading Pattern Detection

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green?style=for-the-badge&logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange?style=for-the-badge&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge)

---

## 📌 Overview

A **production-grade, end-to-end ML system** that detects suspicious trading patterns in financial markets. Built with a 6-model ensemble, SHAP explainability, FastAPI backend, interactive dashboards, trader network analysis, and automated PDF compliance reports.

> Designed to mirror real-world RegTech systems used by firms like **NICE Actimize, Nasdaq Surveillance, and FINRA**.

---

## 🎯 Detected Patterns

| Pattern | Description |
|---------|-------------|
| 🔴 **Pump & Dump** | Artificial price inflation through coordinated buying, followed by rapid selling |
| 🟠 **Spoofing** | Placing large orders with no intent to execute, creating false market depth |
| 🟡 **Layering** | Multiple orders at different price levels, cancelled immediately after affecting market |
| 🟢 **Wash Trading** | Trading with oneself to create artificial volume without genuine participation |
| 🔵 **Front Running** | Trading ahead of known pending large client orders to profit from price movement |

---

## 🏗️ System Architecture

```
Raw Trade Data
      ↓
Feature Engineering (34 features)
      ↓
┌─────────────────────────────────┐
│      ML Ensemble (6 Models)     │
│  Random Forest (300 trees)      │
│  Gradient Boosting (200 trees)  │
│  AdaBoost                       │
│  Logistic Regression            │
│  Isolation Forest               │
│  Local Outlier Factor           │
└─────────────────────────────────┘
      ↓
Hybrid Risk Score (0-1)
      ↓
┌──────────────────────────────────────────────┐
│                  Outputs                      │
│  SHAP Explainability  │  FastAPI REST API     │
│  Web Dashboard        │  Network Graph        │
│  PDF Compliance Report│  Real-time Alerts     │
└──────────────────────────────────────────────┘
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | **1.0000** |
| Average Precision | **1.0000** |
| Precision (Suspicious) | **1.0000** |
| Recall (Suspicious) | **1.0000** |
| F1-Score | **1.0000** |
| CV AUC (5-fold) | **1.0000 ± 0.0000** |

---

## 🚀 Project Structure

```
Trading-Surveillance-System/
│
├── 📄 config.py                  # Central configuration
├── 📄 data_generator.py          # Synthetic data + SQLite storage
├── 📄 feature_engineering.py     # 34-feature engineering pipeline
├── 📄 model.py                   # ML ensemble training + inference
├── 📄 visualizer.py              # 10-panel analysis dashboard
├── 📄 shap_explainer.py          # SHAP explainability engine
├── 📄 train_pipeline.py          # End-to-end training runner
├── 📄 api.py                     # FastAPI REST backend
├── 📄 network_graph.py           # Trader network analysis
├── 📄 pdf_alerts.py              # PDF reports + alert system
├── 🌐 dashboard.html             # Interactive web dashboard
├── 🌐 network.html               # D3.js network graph
│
├── 📁 data/
│   └── trades.db                 # SQLite trade database
├── 📁 models/
│   └── model_bundle.pkl          # Trained model artifacts
├── 📁 reports/
│   ├── dashboard.png             # ML analysis dashboard
│   ├── shap_summary.png          # SHAP feature importance
│   ├── shap_waterfall.png        # Per-trade explanations
│   ├── shap_heatmap.png          # SHAP heatmap
│   ├── network_data.json         # Trader network data
│   ├── alerts.json               # Alert history
│   └── surveillance_report.pdf   # Compliance PDF report
└── 📁 logs/
    ├── train_XXXX.log            # Training logs
    └── alerts.log                # Alert logs
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/johnnikhil77/Trading---Survailence---system.git
cd Trading---Survailence---system
```

**2. Install dependencies**
```bash
pip install scikit-learn pandas numpy matplotlib seaborn fastapi uvicorn networkx reportlab
```

**3. Run the training pipeline**
```bash
python train_pipeline.py
```

**4. Start the REST API**
```bash
python api.py
```

**5. Open the dashboard**
```
Double click dashboard.html in your browser
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | System status |
| GET | `/patterns` | List all patterns |
| GET | `/stats` | Model performance |
| POST | `/predict` | Score single trade |
| POST | `/predict/batch` | Score multiple trades |
| GET | `/trades/flagged` | Top flagged trades |

**Interactive API docs:** `http://127.0.0.1:8000/docs`

---

## 📈 Features

### Phase 1 — ML Model + Training Pipeline
- 6-model ensemble with supervised + unsupervised detection
- 34 engineered features including manipulation-specific scores
- 5-fold cross validation
- SQLite database storage

### Phase 2 — SHAP Explainability
- Per-trade feature attribution
- Beeswarm summary plots
- Waterfall charts for top flagged trades
- Dependence plots for top features

### Phase 3 — REST API
- FastAPI with automatic documentation
- Single and batch trade scoring
- Real-time risk level classification
- CORS enabled for dashboard integration

### Phase 4 — Interactive Web Dashboard
- Live stats from ML model
- Risk score distribution charts
- Pattern breakdown visualization
- Live trade prediction form

### Phase 5 — Trader Network Graph
- D3.js force-directed network
- Community detection for manipulation rings
- Risk-colored nodes
- Interactive filtering by risk level

### Phase 6 — PDF Reports + Alerts
- Professional compliance PDF reports
- Real-time HIGH risk trade alerts
- Alert logging system
- JSON alert history

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| ML/Data | scikit-learn, pandas, numpy |
| Visualization | matplotlib, seaborn, D3.js, Chart.js |
| API | FastAPI, Uvicorn, Pydantic |
| Database | SQLite |
| Network Analysis | NetworkX |
| PDF Generation | ReportLab |
| Frontend | HTML5, CSS3, JavaScript |

---

## 💼 Use Cases

- **RegTech companies** — automated market surveillance
- **Compliance teams** — suspicious activity reporting
- **Trading firms** — internal risk monitoring
- **Regulators** — pattern-based fraud detection

---

## 🏦 Target Industries

- Investment Banks (Goldman Sachs, JPMorgan, Morgan Stanley)
- RegTech Firms (NICE Actimize, Behavox, Solidus Labs)
- Exchanges (Nasdaq, NYSE)
- Regulators (FINRA, SEC, FCA)
- FinTech Companies (Revolut, Stripe, PayPal)

---

## 👨‍💻 Author

**John Nikhil**
- GitHub: [@johnnikhil77](https://github.com/johnnikhil77)

---

## 📄 License

This project is licensed under the MIT License.

---

> ⭐ If you found this project useful, please give it a star!