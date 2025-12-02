# visuals.py
"""
Plotly-based clinical-grade visual components used by app.py.
These functions return plotly.graph_objects.Figure objects ready for Streamlit.
"""
from typing import Dict, Any
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils
import pandas as pd

def _base_layout(fig: go.Figure, height=260, title=None) -> go.Figure:
    fig.update_layout(template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=8,r=8,t=30 if title else 6,b=6))
    if title:
        fig.update_layout(title=title)
    return fig

def sparkline(df: pd.DataFrame, col: str, color: str, low: float, high: float) -> go.Figure:
    data = df[col].iloc[-60:]
    fig = go.Figure()
    if data.empty:
        return _base_layout(fig, height=48)
    fig.add_shape(type="rect", x0=data.index[0], x1=data.index[-1], y0=low, y1=high, fillcolor="rgba(255,255,255,0.03)", line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=data.index, y=data.values, mode="lines", line=dict(color=color, width=2), fill="tozeroy", fillcolor="rgba(255,255,255,0.03)"))
    _base_layout(fig, height=48)
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return fig

def predictive_compass(df: pd.DataFrame, preds: Dict[str, Any], curr_idx: int) -> go.Figure:
    start = max(0, curr_idx - 60)
    window = df.iloc[start:curr_idx]
    cur = window.iloc[-1] if len(window) else df.iloc[-1]
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=2.5, y1=utils.CLIN_THRESH["MAP_TARGET"], fillcolor=utils.THEME["crit"], layer="below", line_width=0)
    fig.add_shape(type="rect", x0=2.5, y0=utils.CLIN_THRESH["MAP_TARGET"], x1=6.0, y1=180, fillcolor=utils.THEME["ok"], layer="below", line_width=0)
    if len(window)>0:
        fig.add_trace(go.Scatter(x=window["CI"], y=window["MAP"], mode="lines", line=dict(color="#555", dash="dot"), name="history"))
    fig.add_trace(go.Scatter(x=[cur["CI"]], y=[cur["MAP"]], mode="markers", marker=dict(color="white", size=12, symbol="x"), name="current"))
    # predicted endpoints
    for label, key, color in [("fluid","fluid",utils.THEME["ci"] if "ci" in utils.THEME else "#00ff33"),
                              ("press","press",utils.THEME["map"]),
                              ("inot","inot",utils.THEME["do2"])]:
        val_map = float(preds[key]["MAP"][-1])
        base_ci = float(max(0.2, cur["CI"]))
        dx = 0.2 if label=="press" else (0.45 if label=="inot" else 0.6)
        fig.add_annotation(x=base_ci+dx, y=val_map, ax=cur["CI"], ay=cur["MAP"], showarrow=True, arrowhead=2, arrowcolor=color, text=label.upper(), font=dict(color=color))
    _base_layout(fig, height=340, title="<b>Predictive Compass</b>")
    fig.update_xaxes(title_text="CI (L/min/m²)", range=[0.5,6.0])
    fig.update_yaxes(title_text="MAP (mmHg)", range=[30,160])
    fig.update_layout(showlegend=False)
    return fig

def multi_scenario_horizon(preds: Dict[str, Any]) -> go.Figure:
    h = len(preds["time"])
    t = np.arange(h)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=preds["nat"]["MAP"], line=dict(color="#666", dash="dot"), name="natural"))
    fig.add_trace(go.Scatter(x=t, y=preds["fluid"]["MAP"], line=dict(color=utils.THEME["ci"]), name="fluid"))
    fig.add_trace(go.Scatter(x=t, y=preds["press"]["MAP"], line=dict(color=utils.THEME["map"]), name="pressor"))
    fig.add_trace(go.Scatter(x=t, y=preds["inot"]["MAP"], line=dict(color=utils.THEME["do2"]), name="inotrope"))
    fig.add_hline(y=utils.CLIN_THRESH["MAP_TARGET"], line_color="red", line_dash="dot")
    _base_layout(fig, height=300, title="<b>Intervention Horizon - MAP</b>")
    fig.update_xaxes(title_text="Minutes ahead")
    fig.update_yaxes(title_text="MAP (mmHg)")
    return fig

def organ_radar(df: pd.DataFrame, curr_idx: int) -> go.Figure:
    cur = df.iloc[max(0, curr_idx-1)]
    r_renal = float(np.clip((utils.CLIN_THRESH["MAP_TARGET"] - cur["MAP"]) / 25.0, 0, 1))
    r_card = float(np.clip((cur["HR"] - 100)/60.0, 0, 1))
    r_meta = float(np.clip((cur["Lactate"] - 1.5)/4.0, 0, 1))
    r_perf = float(np.clip((2.2 - cur["CI"]) / 1.5, 0, 1))
    R = [r_renal, r_card, r_meta, r_perf, r_renal]
    theta = ["Renal","Cardiac","Metabolic","Perfusion","Renal"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[0.2]*5, theta=theta, fill="toself", fillcolor=utils.THEME["ok"], line=dict(width=0)))
    fig.add_trace(go.Scatterpolar(r=R, theta=theta, fill="toself", fillcolor=utils.THEME["crit"], line=dict(color="red", width=2)))
    _base_layout(fig, height=280, title="<b>Organ Risk Topology</b>")
    fig.update_polars(radialaxis=dict(visible=False, range=[0,1]))
    fig.update_layout(showlegend=False)
    return fig
