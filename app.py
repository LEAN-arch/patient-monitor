import streamlit as st
import pandas as pd
import numpy as np
import utils

# --- CONFIG ---
st.set_page_config(page_title="TITAN | ICU", layout="wide", initial_sidebar_state="collapsed")

# --- CSS: TITAN UI ---
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #e0e0e0; }
    .titan-card {
        background: #111111; border: 1px solid #333; border-radius: 4px; padding: 5px 10px;
        margin-bottom: 5px;
    }
    .kpi-lbl { font-size: 0.7rem; color: #888; font-weight: 700; text-transform: uppercase; }
    .kpi-val { font-size: 1.5rem; font-weight: 700; font-family: 'Roboto Mono', monospace; }
    
    .alert-header {
        background: #222; border-left: 5px solid; padding: 10px; margin-bottom: 10px;
        display: flex; justify-content: space-between; align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD ---
@st.cache_data
def load(): return utils.simulate_titan_data()
df, preds = load()

# --- SIDEBAR ---
with st.sidebar:
    st.header("TITAN Control")
    curr_time = st.slider("Time", 200, 720, 720)

cur = df.iloc[curr_time-1]

# --- 1. INTELLIGENT HEADER ---
status = "STABLE"
action = "MONITOR"
col = "#00ff33"
if cur['MAP'] < 65:
    status = "CRITICAL: SHOCK"
    action = "VASOPRESSORS" if cur['CI'] > 2.5 else "FLUIDS"
    col = "#ff2975"
elif cur['Lactate'] > 2.0:
    status = "OCCULT HYPOPERFUSION"
    action = "OPTIMIZE FLOW"
    col = "#ff9900"

st.markdown(f"""
<div class="alert-header" style="border-color:{col}">
    <div style="font-size:1.2rem; font-weight:bold; color:{col}">{status}</div>
    <div style="font-weight:bold;">PROTOCOL: {action}</div>
</div>
""", unsafe_allow_html=True)

# --- 2. ROW 1: SPARKLINE KPI STRIP ---
cols = st.columns(6)
def kpi(col, lbl, val, unit, color, df_col, l, h):
    with col:
        st.markdown(f"<div class='titan-card' style='border-top:2px solid {color}'><div class='kpi-lbl'>{lbl}</div><div class='kpi-val' style='color:{color}'>{val}<span style='font-size:0.8rem;color:#666'>{unit}</span></div></div>", unsafe_allow_html=True)
        st.plotly_chart(utils.plot_spark_spc(df, df_col, color, l, h), use_container_width=True, config={'displayModeBar': False})

kpi(cols[0], "MAP", f"{cur['MAP']:.0f}", "mmHg", "#ff2975", "MAP", 65, 100)
kpi(cols[1], "C. INDEX", f"{cur['CI']:.1f}", "L/min", "#00ff33", "CI", 2.5, 4.0)
kpi(cols[2], "SVR", f"{cur['SVRI']:.0f}", "dyn", "#ff9900", "SVRI", 800, 1200)
kpi(cols[3], "STROKE VOL", f"{cur['SV']:.0f}", "mL", "#00e5ff", "SV", 60, 100)
kpi(cols[4], "LACTATE", f"{cur['Lactate']:.1f}", "mM", "#8c1eff", "Lactate", 0, 2.0)
kpi(cols[5], "ENTROPY", f"{cur['Entropy']:.2f}", "σ", "#ffffff", "Entropy", 0.5, 2.0)

# --- 3. ROW 2: PREDICTION & DECISION (60% Height) ---
c_dec1, c_dec2, c_dec3 = st.columns([2, 2, 1])

with c_dec1:
    st.markdown('<div class="titan-card">', unsafe_allow_html=True)
    st.markdown("<div class='kpi-lbl'>1. PREDICTIVE COMPASS</div>", unsafe_allow_html=True)
    st.plotly_chart(utils.plot_predictive_compass(df, preds, curr_time), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c_dec2:
    st.markdown('<div class="titan-card">', unsafe_allow_html=True)
    st.markdown("<div class='kpi-lbl'>2. INTERVENTION MULTIVERSE</div>", unsafe_allow_html=True)
    st.plotly_chart(utils.plot_multiverse(df, preds, curr_time), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c_dec3:
    st.markdown('<div class="titan-card">', unsafe_allow_html=True)
    st.markdown("<div class='kpi-lbl'>3. GLOBAL STABILITY (PCA)</div>", unsafe_allow_html=True)
    st.plotly_chart(utils.plot_pca_space(df, curr_time), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 4. ROW 3: ORGAN MECHANICS (30% Height) ---
c_org1, c_org2, c_org3, c_org4 = st.columns(4)

with c_org1:
    st.markdown('<div class="titan-card">', unsafe_allow_html=True)
    st.markdown("<div class='kpi-lbl'>4. STARLING VECTOR</div>", unsafe_allow_html=True)
    st.plotly_chart(utils.plot_starling_vector(df, curr_time), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c_org2:
    st.markdown('<div class="titan-card">', unsafe_allow_html=True)
    st.markdown("<div class='kpi-lbl'>5. O2 DEBT LEDGER</div>", unsafe_allow_html=True)
    st.plotly_chart(utils.plot_oxygen_debt(df, curr_time), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c_org3:
    st.markdown('<div class="titan-card">', unsafe_allow_html=True)
    st.markdown("<div class='kpi-lbl'>6. RENAL CLIFF</div>", unsafe_allow_html=True)
    st.plotly_chart(utils.plot_renal_cliff(df, curr_time), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c_org4:
    st.markdown('<div class="titan-card">', unsafe_allow_html=True)
    st.markdown("<div class='kpi-lbl'>7. NEURO SPECTRUM</div>", unsafe_allow_html=True)
    st.plotly_chart(utils.plot_spectrum(df, curr_time), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
