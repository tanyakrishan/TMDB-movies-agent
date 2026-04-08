"""
Hypothesis tools (Step 3).

build_chart_data: transforms EDA findings into Chart.js visualization specs
for rendering on the frontend.
"""

import json

from google.adk.tools import FunctionTool

# Cinema color palette
_RED = "rgba(229, 9, 20, 0.85)"
_RED_BG = "rgba(229, 9, 20, 0.2)"
_GOLD = "rgba(255, 193, 7, 0.85)"
_GOLD_BG = "rgba(255, 193, 7, 0.2)"
_TEAL = "rgba(0, 188, 212, 0.85)"
_TEAL_BG = "rgba(0, 188, 212, 0.2)"
_MUTED = "rgba(160, 160, 160, 0.6)"


def _build_chart_data(eda_findings_json: str) -> str:
    """Transform EDA findings into Chart.js visualization specs.

    Takes the combined output of compute_stats and detect_anomalies and
    produces up to 3 chart specifications for the frontend.

    Args:
        eda_findings_json: JSON string with keys "summary_stats" and "anomalies"
            (the outputs from compute_stats and detect_anomalies).

    Returns:
        JSON string with chart_data containing specs for up to 3 charts:
          1. genre_chart: horizontal bar chart of top genres
          2. decade_chart: bar chart of metrics by decade
          3. scatter_chart: scatter plot of per-movie data points
    """
    try:
        findings = json.loads(eda_findings_json)
    except json.JSONDecodeError:
        # The model sometimes produces slightly malformed JSON.
        # Try to extract the summary_stats portion.
        try:
            # Attempt to fix common issues: trailing text, missing braces
            cleaned = eda_findings_json.strip()
            if not cleaned.endswith("}"):
                cleaned = cleaned + "}"
            findings = json.loads(cleaned)
        except json.JSONDecodeError:
            return json.dumps({"chart_data": {}})
    stats = findings.get("summary_stats", findings)
    charts = {}

    # --- Chart 1: Genre bar chart ---
    top_genres = stats.get("top_genres", [])
    if top_genres:
        labels = [g["genre"] for g in top_genres[:8]]
        counts = [g["count"] for g in top_genres[:8]]
        charts["genre_chart"] = {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": "Movie Count",
                    "data": counts,
                    "backgroundColor": _RED,
                    "borderColor": _RED,
                    "borderWidth": 1,
                }],
            },
            "options": {
                "indexAxis": "y",
                "responsive": True,
                "plugins": {"legend": {"display": False}},
                "scales": {
                    "x": {"title": {"display": True, "text": "Number of Movies"}},
                },
            },
        }

    # --- Chart 2: Decade bar chart ---
    decade_profiles = stats.get("decade_profiles", [])
    if decade_profiles:
        d_labels = [d["decade"] for d in decade_profiles]
        d_ratings = [d["avg_rating"] for d in decade_profiles]
        d_counts = [d["movie_count"] for d in decade_profiles]

        datasets = [
            {
                "label": "Avg Rating",
                "data": d_ratings,
                "backgroundColor": _GOLD,
                "borderColor": _GOLD,
                "borderWidth": 1,
                "yAxisID": "y",
            },
            {
                "label": "Movie Count",
                "data": d_counts,
                "backgroundColor": _TEAL,
                "borderColor": _TEAL,
                "borderWidth": 1,
                "yAxisID": "y1",
            },
        ]

        charts["decade_chart"] = {
            "type": "bar",
            "data": {"labels": d_labels, "datasets": datasets},
            "options": {
                "responsive": True,
                "plugins": {"legend": {"display": True}},
                "scales": {
                    "y": {
                        "type": "linear",
                        "position": "left",
                        "title": {"display": True, "text": "Avg Rating"},
                        "min": 0,
                        "max": 10,
                    },
                    "y1": {
                        "type": "linear",
                        "position": "right",
                        "title": {"display": True, "text": "Count"},
                        "grid": {"drawOnChartArea": False},
                    },
                },
            },
        }

    # --- Chart 3: Scatter plot ---
    scatter_points = stats.get("scatter_points", [])
    if scatter_points:
        scatter_data = [
            {"x": p["x"], "y": p["y"]}
            for p in scatter_points[:80]
        ]
        charts["scatter_chart"] = {
            "type": "scatter",
            "data": {
                "datasets": [{
                    "label": "Rating by Year",
                    "data": scatter_data,
                    "backgroundColor": _RED,
                    "borderColor": _RED_BG,
                    "pointRadius": 5,
                    "pointHoverRadius": 8,
                }],
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "legend": {"display": False},
                    "tooltip": {
                        "callbacks": {}
                    },
                },
                "scales": {
                    "x": {"title": {"display": True, "text": "Year"}},
                    "y": {
                        "title": {"display": True, "text": "Rating"},
                        "min": 0,
                        "max": 10,
                    },
                },
            },
        }

    return json.dumps({"chart_data": charts})


build_chart_data = FunctionTool(func=_build_chart_data)
