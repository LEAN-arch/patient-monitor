import streamlit as st
import pandas as pd
import numpy as np
import utils

# --- PAGE CONFIG ---
st.set_page_config(page_title="CDS | Hemodynamic AI", layout="wide", initial_sidebar_state="collapsed")

# --- MEDICAL UI CSS ---
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    
    /* The "Vital Card" */
    .vital-card {
        background-color: white;
        border-left: 5px solid #e5e7eb;
        padding: 15px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .vital-label { font-size: 0.75rem; font-weight: 700; color: #6b7280; text-transform: uppercase; }
    .vital-value { font-size: 2rem; font-weight: 800; color: #111827; line-height: 1.1; }
    .vital-unit { font-size: 1rem; color: #9ca3af; font-weight: 500; }
    .vital-delta { font-size: 0.85rem; font-weight: 600; margin-top: 5px; }
    
    /* Status Colors */
    .status-crit { border-left-color: #ef4444 !important; }
    .status-warn { border-left-color: #f59e0b !important; }
    .status-ok { border-left-color: #10b981 !important; }
    .txt-crit { color: #ef4444; }
    .txt-ok { color: #10b981; }
    
    /* Diagnosis Box */
    .dx-box {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        padding: 15px;
        border-radius: 8px;
        color: #1e40af;
        font-weight: 500;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def get_data(): return utils.simulate_shock_progression()
df = get_data()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Simulation Timeline")
    curr_time = st.slider("Time (min)", 200, 720, 720)
    st.markdown("---")
    st.info("**Instructions:** Move slider to T=500 to see 'Compensated Shock' (Narrow PP, Low PI). Move to T=650 to see Decompensation.")

# --- ANALYTICS ENGINE ---
current = df.iloc[curr_time-1]
prev = df.iloc[curr_time-15]

# Hemodynamic Logic
pp = current['PP']
si = current['SI']
map_val = current['MAP']

# Diagnosis Logic
dx_status = "STABLE"
if pp < 35 and si > 0.8:
    dx_status = "COMPENSATED SHOCK (Low Stroke Volume)"
if map_val < 65:
    dx_status = "DECOMPENSATED SHOCK (Hypotension)"

# --- HEADER ---
c1, c2 = st.columns([3, 1])
with c1:
    st.title("Hemodynamic CDS")
    st.markdown(f"**AI Assessment:** <span class='dx-box'>{dx_status}</span>", unsafe_allow_html=True)
with c2:
    risk_score = min(100, int((si * 40) + (60 - pp)))
    st.metric("Instability Index", f"{risk_score}/100", "High" if risk_score > 50 else "Low", delta_color="inverse")

st.markdown("---")

# --- ROW 1: PRIMARY VITALS (The "What") ---
col1, col2, col3, col4 = st.columns(4)

def vital_box(col, label, val, unit, delta, threshold, invert=False):
    is_bad = val < threshold if not invert else val > threshold
    status = "status-crit" if is_bad else "status-ok"
    delta_color = "txt-crit" if is_bad else "txt-ok"
    
    col.markdown(f"""
    <div class="vital-card {status}">
        <div class="vital-label">{label}</div>
        <div class="vital-value">{val} <span class="vital-unit">{unit}</span></div>
        <div class="vital-delta {delta_color}">{delta}</div>
    </div>
    """, unsafe_allow_html=True)

# 1. MAP (Perfusion)
d_map = map_val - prev['MAP']
vital_box(col1, "Mean Art. Pressure", f"{int(map_val)}", "mmHg", f"{d_map:+.0f}", 65)

# 2. Pulse Pressure (Stroke Volume)
d_pp = pp - prev['PP']
vital_box(col2, "Pulse Pressure", f"{int(pp)}", "mmHg", f"{d_pp:+.0f} (Narrowing)" if d_pp < -2 else "Stable", 30)

# 3. Shock Index (Stability)
d_si = si - prev['SI']
vital_box(col3, "Shock Index", f"{si:.2f}", "", f"{d_si:+.2f}", 0.9, invert=True)

# 4. Perfusion Index (Micro-circ)
pi = current['PI']
vital_box(col4, "Perfusion Index", f"{pi:.1f}", "%", "Vasoconstriction" if pi < 1.0 else "Normal", 1.0)


# --- ROW 2: CLINICAL DECISION SUPPORT (The "Why" & "When") ---
c_left, c_mid, c_right = st.columns([1.5, 1, 1])

with c_left:
    # 1. Prognosis
    fig_prog = utils.plot_prognostic_horizon(df, curr_time)
    st.plotly_chart(fig_prog, use_container_width=True)
    st.caption("ℹ️ **Action:** Projecting MAP trends to estimate 'Time to Crash'. Prepare vasopressors if trajectory intersects Red Line.")

with c_mid:
    # 2. Diagnosis
    fig_pheno = utils.plot_shock_phenotype(df, curr_time)
    st.plotly_chart(fig_pheno, use_container_width=True)
    st.caption("ℹ️ **Diagnosis:** Movement into Orange/Red zones indicates low stroke volume compensation. Consider fluid challenge.")

with c_right:
    # 3. Early Warning
    fig_auto = utils.plot_autonomic_str
