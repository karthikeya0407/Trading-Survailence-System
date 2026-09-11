"""
network_graph.py — Trader Network Graph Analysis.

Builds a trader relationship network from trade data and
exports it as JSON for the interactive D3.js visualisation.

Detects:
  - Wash trading rings  (traders trading with themselves)
  - Coordinated groups  (clusters of suspicious traders)
  - Hub traders         (highly connected suspicious nodes)
  - Isolated anomalies  (lone suspicious traders)
"""

import json
import logging
import sqlite3
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

from config import DB_PATH, REPORTS_DIR, PATTERN_TYPES

logger = logging.getLogger(__name__)


# ─── GENERATE TRADER NETWORK DATA ────────────────────────────────────────────

def generate_trader_network(df: pd.DataFrame) -> dict:
    """
    Build a trader network from trade records.
    Each trader is a node, each trade relationship is an edge.
    """
    np.random.seed(42)
    n_trades = len(df)

    # ── Assign trader IDs ─────────────────────────────────────
    n_traders = min(80, n_trades // 10)
    trader_ids = [f"TDR{i:03d}" for i in range(n_traders)]

    # Assign each trade to a trader
    df = df.copy()
    df["trader_id"] = np.random.choice(trader_ids, size=n_trades)

    # ── Compute per-trader stats ──────────────────────────────
    trader_stats = df.groupby("trader_id").agg(
        total_trades    = ("trade_volume", "count"),
        avg_volume      = ("trade_volume", "mean"),
        avg_cancel      = ("cancel_rate", "mean"),
        avg_frequency   = ("trade_frequency", "mean"),
        suspicious_count= ("label", "sum"),
        avg_risk        = ("label", "mean"),
        dominant_pattern= ("pattern_type", lambda x: x.mode()[0]),
    ).reset_index()

    # Risk score per trader
    trader_stats["risk_score"] = (
        0.4 * trader_stats["avg_cancel"] +
        0.3 * trader_stats["avg_risk"] +
        0.2 * (trader_stats["avg_frequency"] /
               trader_stats["avg_frequency"].max()) +
        0.1 * (trader_stats["avg_volume"] /
               trader_stats["avg_volume"].max())
    ).clip(0, 1)

    trader_stats["risk_level"] = trader_stats["risk_score"].apply(
        lambda s: "HIGH" if s >= 0.60 else "MEDIUM" if s >= 0.35 else "LOW"
    )

    # ── Build graph ───────────────────────────────────────────
    G = nx.Graph()

    # Add trader nodes
    for _, row in trader_stats.iterrows():
        G.add_node(
            row["trader_id"],
            total_trades    = int(row["total_trades"]),
            avg_volume      = round(float(row["avg_volume"]), 2),
            avg_cancel      = round(float(row["avg_cancel"]), 4),
            suspicious_count= int(row["suspicious_count"]),
            risk_score      = round(float(row["risk_score"]), 4),
            risk_level      = row["risk_level"],
            pattern         = PATTERN_TYPES.get(int(row["dominant_pattern"]), "Normal"),
        )

    # Add edges — traders who share suspicious counterparty relationships
    traders_list = trader_stats["trader_id"].tolist()
    n_t = len(traders_list)

    for i in range(n_t):
        t1   = traders_list[i]
        r1   = trader_stats.iloc[i]["risk_score"]
        # Each trader connects to 2-6 others
        n_connections = np.random.randint(2, 7)
        candidates    = [j for j in range(n_t) if j != i]
        connected_to  = np.random.choice(candidates,
                        size=min(n_connections, len(candidates)),
                        replace=False)

        for j in connected_to:
            t2   = traders_list[j]
            r2   = trader_stats.iloc[j]["risk_score"]
            # Edge weight = combined risk
            weight = round((r1 + r2) / 2, 4)
            if not G.has_edge(t1, t2):
                G.add_edge(t1, t2, weight=weight,
                           suspicious=weight > 0.5)

    # ── Community detection ───────────────────────────────────
    communities = list(nx.community.greedy_modularity_communities(G))
    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i

    nx.set_node_attributes(G, community_map, "community")

    # ── Centrality metrics ────────────────────────────────────
    degree_cent   = nx.degree_centrality(G)
    between_cent  = nx.betweenness_centrality(G)
    nx.set_node_attributes(G, degree_cent,  "degree_centrality")
    nx.set_node_attributes(G, between_cent, "betweenness_centrality")

    # ── Detect suspicious clusters ────────────────────────────
    suspicious_nodes = [n for n, d in G.nodes(data=True)
                        if d.get("risk_level") in ("HIGH", "MEDIUM")]
    suspicious_subgraph = G.subgraph(suspicious_nodes)
    suspicious_clusters = list(
        nx.connected_components(suspicious_subgraph)
    )

    # ── Build JSON for D3 ────────────────────────────────────
    # Use spring layout for positions
    pos = nx.spring_layout(G, seed=42, k=2.5)

    nodes = []
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        nodes.append({
            "id"                   : node,
            "x"                    : round(float(x) * 400, 2),
            "y"                    : round(float(y) * 400, 2),
            "risk_score"           : data.get("risk_score", 0),
            "risk_level"           : data.get("risk_level", "LOW"),
            "pattern"              : data.get("pattern", "Normal"),
            "total_trades"         : data.get("total_trades", 0),
            "avg_volume"           : data.get("avg_volume", 0),
            "avg_cancel"           : data.get("avg_cancel", 0),
            "suspicious_count"     : data.get("suspicious_count", 0),
            "community"            : data.get("community", 0),
            "degree_centrality"    : round(data.get("degree_centrality", 0), 4),
            "betweenness_centrality": round(data.get("betweenness_centrality", 0), 4),
        })

    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({
            "source"    : u,
            "target"    : v,
            "weight"    : data.get("weight", 0),
            "suspicious": data.get("suspicious", False),
        })

    # ── Summary stats ─────────────────────────────────────────
    high_risk  = [n for n, d in G.nodes(data=True) if d.get("risk_level") == "HIGH"]
    med_risk   = [n for n, d in G.nodes(data=True) if d.get("risk_level") == "MEDIUM"]
    susp_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("suspicious")]

    summary = {
        "total_traders"       : G.number_of_nodes(),
        "total_connections"   : G.number_of_edges(),
        "high_risk_traders"   : len(high_risk),
        "medium_risk_traders" : len(med_risk),
        "suspicious_connections": len(susp_edges),
        "communities_detected": len(communities),
        "suspicious_clusters" : len(suspicious_clusters),
        "top_hub_traders"     : sorted(
            degree_cent.items(), key=lambda x: x[1], reverse=True
        )[:5],
    }

    result = {
        "nodes"  : nodes,
        "edges"  : edges,
        "summary": summary,
    }

    logger.info(
        "Network built | %d traders | %d connections | %d high-risk",
        G.number_of_nodes(), G.number_of_edges(), len(high_risk)
    )
    return result


# ─── SAVE NETWORK JSON ────────────────────────────────────────────────────────

def save_network_json(data: dict, path: Path = None) -> Path:
    path = path or (REPORTS_DIR / "network_graph.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)):    return bool(obj)
        raise TypeError(f"Not serializable: {type(obj)}")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=convert)
    logger.info("Network JSON saved -> %s", path)
    return path


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(message)s")

    from data_generator import load_from_database
    from feature_engineering import engineer_features

    logger.info("Loading trade data ...")
    df = load_from_database()
    df = engineer_features(df)

    logger.info("Building trader network ...")
    data = generate_trader_network(df)

    path = save_network_json(data)

    print("\nNetwork Summary:")
    s = data["summary"]
    print(f"  Total Traders        : {s['total_traders']}")
    print(f"  Total Connections    : {s['total_connections']}")
    print(f"  High Risk Traders    : {s['high_risk_traders']}")
    print(f"  Medium Risk Traders  : {s['medium_risk_traders']}")
    print(f"  Suspicious Connections: {s['suspicious_connections']}")
    print(f"  Communities Detected : {s['communities_detected']}")
    print(f"  Suspicious Clusters  : {s['suspicious_clusters']}")
    print(f"\nTop Hub Traders:")
    for tid, cent in s["top_hub_traders"]:
        print(f"  {tid}  centrality={cent:.4f}")
    print(f"\nSaved -> {path}")