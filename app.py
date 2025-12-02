import streamlit as st
import pandas as pd
import numpy as np
import utils

# --- PAGE CONFIG ---
st.set_page_config(page_title="Aether | Advanced ICU", layout="wide", initial_sidebar_state="collapsed")

# --- COMMERCIAL STYLING (Light Mode) ---
st.markdown("""
<style>
    /* Global Cleanliness */
    .stApp { background-color: #f8fafc; }
    
    /* The Prognostic Card */
    .metric-card {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: column;
    }
    .metric-title { font-size: 0.75rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-val { font-size: 1.8rem; font-weight: 700; color: #0f172a; font-feature-settings: "tnum"; margin-top: 4px; }
    .metric-delta { font-size: 0.85rem; font-weight: 500; margin-top: 4px; }
    
    /* Logic Colors */
    .text-crit { color: #dc2626; }
    .text-warn { color: #d97706; }
    .text-safe { color: #059669; }
    .border-crit { border-left: 4px solid #dc2626 !important; }
    .border-safe { border-left: 4px solid #059669 !important; }
    
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data(): return utils.simulate_comprehensive_patient()
df = load_data()

# --- SIDEBAR CONTROL ---
with st.sidebar:
    st.header("Simulation Timeline")
    curr_time = st.slider("Time (min)", 150, 720, 720)
    st.info("""
    **Event:** Septic Cardiomyopathy (T=250)
    **Dynamics:** 
    - PI Drops (Vasoconstriction)
    - Pulse Pressure Narrows
    - Entropy Loss (Autonomic Failure)
    """)

# --- ANALYTICS ENGINE ---
current = df.iloc[curr_time-1]
prev = df.iloc[curr_time-15]

# Risk Logic
cei = current['CEI']
risk_status = "STABLE"
status_color = "text-safe"
if cei > 2.5: 
    risk_status = "COMPENSATED SHOCK" 
    status_color = "text-warn"
if current['MAP'] < 65 or cei > 4.0: 
    risk_status = "DECOMPENSATION" 
    status_color = "text-crit"

# --- HEADER (HUD) ---
c1, c2 = st.columns([3, 1])
with c1:
    st.title("Aether™ Clinical Command")
    st.markdown(f"**Patient State:** <span class='{status_color}' style='font-weight:bold'>{risk_status}</span>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div style='text-align:right; font-size:0.9rem; color:#64748b'>Kalman Confidence: <b>98.2%</b><br>Signal Entropy: <b>{current['Entropy']:.2f}</b></div>", unsafe_allow_html=True)

st.markdown("---")

# --- ROW 1: PROGNOSTIC METRICS (The "Secret Sauce") ---
# Customized HTML Cards for Light Mode
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

def kpi_card(col, title, val, unit, delta, status="safe"):
    color_class = "text-safe" if status == "safe" else "text-crit"
    border_class = "border-safe" if status == "safe" else "border-crit"
    
    col.markdown(f"""
    <div class="metric-card {border_class}">
        <div class="metric-title">{title}</div>
        <div class="metric-val">{val}<span style="font-size:1rem; color:#94a3b8; margin-left:4px">{unit}</span></div>
        <div class="metric-delta {color_class}">{delta}</div>
    </div>
    """, unsafe_allow_html=True)

# 1. Cardiac Effort (CEI)
d_cei = cei - prev['CEI']
kpi_card(col_m1, "Cardiac Effort Index", f"{cei:.1f}", "pts", f"{d_cei:+.1f} (Increasing)", "crit" if cei > 3 else "safe")

# 2. Pulse Pressure
pp = current['PP']
d_pp = pp - prev['PP']
kpi_card(col_m2, "Pulse Pressure (SV)", f"{int(pp)}", "mmHg", f"{d_pp:+.0f} (Narrowing)" if d_pp < 0 else "Stable", "crit" if pp < 30 else "safe")

# 3. True HR (Kalman)
hr = current['Kalman_HR']
d_hr = hr - prev['Kalman_HR']
kpi_card(col_m3, "True Heart Rate", f"{int(hr)}", "bpm", f"{d_hr:+.0f} (Trend)", "crit" if hr > 100 else "safe")

# 4. MAP
map_val = current['MAP']
kpi_card(col_m4, "Mean Art. Pressure", f"{int(map_val)}", "mmHg", "Target > 65", "crit" if map_val < 65 else "safe")


# --- ROW 2: COMMAND CENTER STRIP ---
st.subheader("1. Hemodynamic Timeline (Kalman Filtered)")
fig_strip = utils.plot_command_strip(df, curr_time)
st.plotly_chart(fig_strip, use_container_width=True)

# --- ROW 3: ADVANCED DIAGNOSTICS ---
st.subheader("2. Advanced Phenotyping")
c_left, c_mid, c_right = st.columns([1, 1, 1])

with c_left:
    # The Loop
    fig_loop = utils.plot_hemo_loop(df, curr_time)
    st.plotly_chart(fig_loop, use_container_width=True)
    st.info("**Frank-Starling Proxy:** Bottom-Right shift indicates cardiac exhaustion (High Rate, Low Output).")

with c_mid:
    # 3D Attractor
    fig_3d = utils.plot_3d_attractor(df, curr_time)
    st.plotly_chart(fig_3d, use_container_width=True)
    st.info("**Attractor Geometry:** Visualizes stability. A collapse into a flat loop indicates loss of reserve.")

with c_right:
    # The Fingerprint
    fig_heat = utils.plot_heatmap_fingerprint(df, curr_time)
    st.plotly_chart(fig_heat, use_container_width=True)
    # Deep Metrics
    fig_adv = utils.plot_advanced_metrics(df, curr_time)
    st.plotly_chart(fig_adv, use_container_width=True)
