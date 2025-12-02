# app.py
"""
TITAN Streamlit front-end (commercial-grade layout).
Depends on: utils.py, clinical_logic.py, visuals.py, themes.py
"""
import streamlit as st
import utils
import visuals
import clinical_logic
import themes
import pandas as pd

st.set_page_config(page_title="TITAN | ICU (Clinical)", layout="wide", initial_sidebar_state="expanded")
st.markdown(themes.THEME_CSS, unsafe_allow_html=True)

# Load or simulate data (fast cached)
@st.cache_data
def load(mins=720):
    df, preds = utils.simulate_titan_data(mins=mins, seed=42)
    return df, preds

df, preds = load(720)

# Sidebar: patient context & controls
with st.sidebar:
    st.header("TITAN Controls")
    st.write("Inspect timeline and adjust model knobs for 'what-if' scenarios.")
    minute = st.slider("Minute (1-based)", 1, len(df), len(df), help="Select minute index")
    # quick simulation knobs (re-run)
    st.markdown("**Model knobs**")
    preload = st.number_input("Preload", value=12.0, step=0.5)
    contractility = st.number_input("Contractility", value=1.0, step=0.05, format="%.2f")
    afterload = st.number_input("Afterload", value=1.0, step=0.05, format="%.2f")
    if st.button("Re-run sim with knobs"):
        twin = utils.DigitalTwin(preload=preload, contractility=contractility, afterload=afterload)
        df, preds = utils.simulate_titan_data(mins=720, twin=twin, seed=42)
        st.experimental_rerun()

# index
idx = max(0, minute - 1)
row = df.iloc[idx]

# compute alerts + action plan
alerts = utils.compute_alerts(df, lookback_min=15)
plan = clinical_logic.build_action_plan(df, alerts)

# top header: status
def compute_header(row):
    status = "STABLE"
    color = "var(--accent)"
    action = "Monitor"
    rationale = "Vitals in acceptable range."
    if row["MAP"] < utils.CLIN_THRESH["MAP_TARGET"]:
        status = "HYPOTENSION"
        action = "Fluids ± Vasopressor" if row["CI"] >= utils.CLIN_THRESH["CI_LOW"] else "Vasopressor ± Inotrope"
        color = "var(--danger)" if row["MAP"] < utils.CLIN_THRESH["MAP_CRIT"] else "var(--warn)"
        rationale = f"MAP {row['MAP']:.0f} mmHg. CI {row['CI']:.2f}."
    if row["Lactate"] >= utils.CLIN_THRESH["LACTATE_ELEVATED"]:
        status = "PERFUSION RISK" if status=="STABLE" else status
        rationale += f" Lactate {row['Lactate']:.2f} mmol/L."
    return status, action, color, rationale

status, action, color, rationale = compute_header(row)
st.markdown(f"<div class='alert-header' style='border-color:{color}'><div style='font-size:1.1rem;font-weight:800;color:{color}'>{status}</div><div style='font-weight:700;color:#dde'>ACTION: {action}</div></div>", unsafe_allow_html=True)
st.markdown(f"<div class='small'>{rationale}</div>", unsafe_allow_html=True)

# KPI row
k1, k2, k3, k4, k5, k6 = st.columns(6, gap="small")
def kpi(col, label, val, unit, color, dfcol, low, high):
    with col:
        st.markdown(f"<div class='titan-card' style='border-top:3px solid {color}'><div class='kpi-lbl'>{label}</div><div class='kpi-val' style='color:{color}'>{val}<span style='font-size:0.72rem;color:#bbb'> {unit}</span></div></div>", unsafe_allow_html=True)
        fig = visuals.sparkline(df, dfcol, color, low, high)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

kpi(k1, "MAP", f"{row['MAP']:.0f}", "mmHg", utils.THEME["map"], "MAP", 55, 110)
kpi(k2, "CI", f"{row['CI']:.2f}", "L/min/m²", utils.THEME["ci"], "CI", 1.0, 4.5)
kpi(k3, "SVR", f"{row['SVRI']:.0f}", "dyn", utils.THEME["svr"], "SVRI", 400, 1400)
kpi(k4, "SV", f"{row['SV']:.0f}", "mL", utils.THEME["ci"], "SV", 30, 120)
kpi(k5, "Lactate", f"{row['Lactate']:.2f}", "mmol/L", utils.THEME["do2"], "Lactate", 0.4, 6.0)
kpi(k6, "Entropy", f"{row['Entropy']:.2f}", "σ", "#ffffff", "Entropy", 0.0, 2.0)

# main panels: predictive compass + multiverse
st.markdown('<div class="section-header">🔮 Predictive Hemodynamics & Forecasts</div>', unsafe_allow_html=True)
left, right = st.columns((1,1), gap="large")
with left:
    st.markdown("<div class='titan-card'>", unsafe_allow_html=True)
    fig_c = visuals.predictive_compass(df, preds, idx)
    st.plotly_chart(fig_c, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)
with right:
    st.markdown("<div class='titan-card'>", unsafe_allow_html=True)
    fig_m = visuals.multi_scenario_horizon(preds)
    st.plotly_chart(fig_m, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# organ topology row
st.markdown('<div class="section-header">🫀 Organ Mechanics & Risk</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4, gap="large")
with c1:
    st.plotly_chart(visuals.organ_radar(df, idx), use_container_width=True, config={"displayModeBar": False})
with c2:
    st.plotly_chart(visuals.sparkline(df, "SV", utils.THEME["ci"], 30, 120), use_container_width=True, config={"displayModeBar": False})
with c3:
    st.plotly_chart(visuals.sparkline(df, "DO2", utils.THEME["do2"], 200, 700), use_container_width=True, config={"displayModeBar": False})
with c4:
    st.plotly_chart(visuals.sparkline(df, "Urine", "#ffffff", 0.0, 2.0), use_container_width=True, config={"displayModeBar": False})

# Action panel: prioritized plan
st.markdown('<div class="section-header">🩺 Suggested Actions (Prioritized)</div>', unsafe_allow_html=True)
for p in plan:
    pr = p["priority"].upper()
    st.markdown(f"- **{pr}** — {p['action']}  \n  _{p['rationale']}_")

# debug
with st.expander("Raw data snapshot / debug", expanded=False):
    st.write(f"Minute index: {minute}")
    st.dataframe(df.iloc[max(0, idx-10):idx+1])
