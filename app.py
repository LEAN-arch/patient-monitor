import streamlit as st
import pandas as pd
import numpy as np
import utils

# --- CONFIG ---
st.set_page_config(page_title="SENTINEL | GDT", layout="wide", initial_sidebar_state="collapsed")

# --- CSS ---
st.markdown("""
<style>
    .stApp { background-color: #f1f5f9; }
    
    .card {
        background: white; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 10px;
    }
    
    .kpi-val { font-size: 2rem; font-weight: 800; color: #0f172a; line-height: 1; }
    .kpi-lbl { font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; }
    .kpi-sub { font-size: 0.8rem; font-weight: 600; margin-top: 5px; }
    
    .txt-crit { color: #ef4444; }
    .txt-warn { color: #f59e0b; }
    .txt-ok { color: #10b981; }
    
    .header-box {
        background: #1e293b; color: white; padding: 15px; border-radius: 8px;
        display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD ---
@st.cache_data
def get_data(): return utils.simulate_gdt_data()
df = get_data()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Sentinel Control")
    curr_time = st.slider("Time", 200, 720, 720)
    st.info("Scenario: Progressive Septic Shock")

# --- STATE ---
cur = df.iloc[curr_time-1]
prv = df.iloc[curr_time-15]

# --- 1. HEADER ---
# Clinical Logic Engine
status = "HEMODYNAMICALLY STABLE"
action = "CONTINUE MONITORING"
bg_col = "#10b981"

if cur['Lactate'] > 2.0:
    status = "METABOLIC DISTRESS"
    action = "EVALUATE PERFUSION"
    bg_col = "#f59e0b"

if cur['MAP'] < 65:
    status = "CIRCULATORY SHOCK"
    bg_col = "#ef4444"
    if cur['CI'] > 3.0:
        action = "START VASOPRESSORS (High Flow)"
    else:
        action = "FLUID RESUSCITATION (Low Flow)"

st.markdown(f"""
<div class="header-box" style="background:{bg_col}">
    <div style="font-size:1.2rem; font-weight:bold;">STATUS: {status}</div>
    <div style="font-size:1.2rem; font-weight:bold;">RECOMMENDATION: {action}</div>
</div>
""", unsafe_allow_html=True)

# --- 2. KPI STRIP ---
cols = st.columns(6)

def kpi(col, label, val, unit, delta, thresh, invert=False, fmt="{:.0f}"):
    bad = val < thresh if not invert else val > thresh
    color = "txt-crit" if bad else "txt-ok"
    col.markdown(f"""
    <div class="card">
        <div class="kpi-lbl">{label}</div>
        <div class="kpi-val">{fmt.format(val)} <span style="font-size:0.8rem; color:#94a3b8">{unit}</span></div>
        <div class="kpi-sub {color}">{delta}</div>
    </div>
    """, unsafe_allow_html=True)

d_map = cur['MAP'] - prv['MAP']
kpi(cols[0], "MAP", cur['MAP'], "mmHg", f"{d_map:+.0f}", 65)

d_ci = cur['CI'] - prv['CI']
kpi(cols[1], "Cardiac Index", cur['CI'], "L/min", f"{d_ci:+.1f}", 2.2, fmt="{:.1f}")

d_svr = cur['SVRI'] - prv['SVRI']
kpi(cols[2], "SVR Index", cur['SVRI'], "dyn", f"{d_svr:+.0f}", 800)

d_do2 = cur['DO2I'] - prv['DO2I']
kpi(cols[3], "DO2 Index", cur['DO2I'], "mL/m2", f"{d_do2:+.0f}", 400)

d_lac = cur['Lactate'] - prv['Lactate']
kpi(cols[4], "Lactate", cur['Lactate'], "mmol", f"{d_lac:+.1f}", 2.0, invert=True, fmt="{:.1f}")

kpi(cols[5], "Urine Out", cur['Urine'], "mL/kg", "Oliguric" if cur['Urine']<0.5 else "OK", 0.5, fmt="{:.1f}")

# --- 3. DIAGNOSTIC ROW (The Compass & The Ledger) ---
c1, c2 = st.columns([1, 1])

with c1:
    st.markdown("**1. HEMODYNAMIC COMPASS (Diagnosis)**")
    st.caption("Background zones identify the shock phenotype instantly.")
    fig_comp = utils.plot_shock_compass(df, curr_time)
    st.plotly_chart(fig_comp, use_container_width=True)

with c2:
    st.markdown("**2. OXYGEN LEDGER (Supply vs Debt)**")
    st.caption("Red fill indicates accumulating oxygen debt (Lactate driver).")
    fig_ox = utils.plot_oxygen_mismatch(df, curr_time)
    st.plotly_chart(fig_ox, use_container_width=True)

# --- 4. MECHANICS ROW ---
c3, c4, c5 = st.columns(3)

with c3:
    st.markdown("**3. RENAL AUTOREGULATION**")
    st.caption("Is perfusion pressure sufficient for filtration?")
    fig_ren = utils.plot_renal_autoregulation(df, curr_time)
    st.plotly_chart(fig_ren, use_container_width=True)

with c4:
    st.markdown("**4. STARLING VECTOR (Responsiveness)**")
    st.caption("Arrow length/direction shows response to recent preload change.")
    fig_vec = utils.plot_fluid_vector(df, curr_time)
    st.plotly_chart(fig_vec, use_container_width=True)

with c5:
    st.markdown("**5. NEURO-AUTONOMIC STATE**")
    st.caption("Heart Rate Variability Spectrum (Stress Indicator).")
    fig_spec = utils.plot_autonomic_spectrum(df, curr_time)
    st.plotly_chart(fig_spec, use_container_width=True)
