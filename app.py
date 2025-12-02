import streamlit as st
import pandas as pd
import numpy as np
import utils # Importing the new robust utils

# --- Config ---
st.set_page_config(page_title="ICU Sentinel AI", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for clinical dashboard look
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    h1, h2, h3 {margin-bottom: 0rem;}
    .stMetric {background-color: #1E1E1E; padding: 10px; border-radius: 5px; border: 1px solid #333;}
    .big-font {font-size:20px !important; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- 1. Data Loading & Processing ---
@st.cache_data
def get_sim_data():
    return utils.simulate_patient(mins_total=720)

df_full = get_sim_data()
VARS = ['HR','SBP','SpO2','RR','PI']

with st.spinner("Analyzing hemodynamic stability..."):
    # Z-Score Normalization (Standardization against patient baseline)
    z_full, cov_est = utils.fit_var_and_residuals_full(df_full[VARS], baseline_window=120)
    # Mahalanobis Risk Score
    risk_full, cov_inv = utils.compute_mahalanobis_risk(z_full, cov_est)
    # Calculate Shock Index
    df_full['SI'] = df_full['HR'] / df_full['SBP']

# --- 2. Sidebar Controls ---
with st.sidebar:
    st.header("Simulation Timeline")
    curr_time = st.slider("Time (minutes)", 60, 720, 720)
    view_window = st.selectbox("Lookback Window", [60, 120, 240], index=1)
    
    st.markdown("### Clinical Scenario")
    st.info("""
    **Patient ID:** 8392-A
    **Admit:** Post-Op Abdominal
    **Event:** Occult Bleeding
    
    **Instructions:**
    1. Move slider to **min 350** (Stable).
    2. Move to **min 450** (Compensated Shock: HR up, PI down, BP normal).
    3. Move to **min 600** (Decompensation: Hypotension).
    """)

# --- 3. Data Slicing ---
start_idx = max(0, curr_time - view_window)
end_idx = curr_time

# Slicing for plots
df_view = df_full.iloc[start_idx:end_idx]
z_view = z_full.iloc[start_idx:end_idx]
t_axis = df_view.index

# Current Instant values
current_vals = df_full.iloc[curr_time-1]
current_risk = risk_full[curr_time-1]
current_si = current_vals['SI']

# --- 4. Dashboard Header (Heads Up Display) ---
col_h1, col_h2, col_h3, col_h4 = st.columns([2, 1, 1, 2])

with col_h1:
    st.title("🏥 ICU Sentinel")
    st.caption(f"Real-time Hemodynamic Monitoring | T={curr_time}min")

with col_h2:
    st.metric("Risk Score", f"{current_risk:.1f}", 
              delta="High Risk" if current_risk > 25 else "Stable", 
              delta_color="inverse")

with col_h3:
    st.metric("Shock Index", f"{current_si:.2f}", 
              delta="Warning" if current_si > 0.9 else "Normal", 
              delta_color="inverse")

with col_h4:
    # Clinical Action Logic
    if current_risk < 15:
        st.success("✅ **PATIENT STABLE**\n\nContinue routine monitoring.")
    elif current_risk < 25:
        st.warning("⚠️ **EARLY WARNING: COMPENSATORY EFFORT**\n\nCheck perfusion (PI). Assess fluid responsiveness.")
    else:
        st.error("🚨 **CRITICAL: DECOMPENSATION**\n\nImmediate assessment required. Evaluate for Shock Protocol.")

st.divider()

# --- 5. Main Clinical Workspace ---
# Left Column: Timeline Trends (The "Story")
# Right Column: Instantaneous Profile (The "Status")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("#### 1. Hemodynamic & Perfusion Trends")
    # Improved Vitals Plot (Coupled HR/BP + SI + PI)
    fig_vitals = utils.plot_combined_vitals(df_view, t_axis, df_view['SI'])
    st.plotly_chart(fig_vitals, use_container_width=True)
    
    st.markdown("#### 2. Anatomy of Instability (Contribution Heatmap)")
    # Heatmap is better than bar chart because it shows the SEQUENCE of deterioration
    fig_heat = utils.plot_temporal_contribution(z_view, VARS)
    st.plotly_chart(fig_heat, use_container_width=True)

with col_right:
    st.markdown("#### 3. Current Status")
    
    # A. Risk Gauge
    fig_gauge = utils.plot_risk_gauge(current_risk)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # B. Physiological Footprint (Radar)
    # This helps distinguish 'why' the risk is high (Resp vs Hemo)
    fig_radar = utils.plot_clinical_radar(current_vals, df_full.iloc[:120].mean())
    st.plotly_chart(fig_radar, use_container_width=True)

    # C. Raw Vitals Table (Quick Glance)
    st.markdown("#### Live Vitals")
    cols = st.columns(2)
    cols[0].metric("HR", f"{int(current_vals['HR'])}")
    cols[1].metric("SBP", f"{int(current_vals['SBP'])}")
    cols[0].metric("SpO2", f"{int(current_vals['SpO2'])}%")
    cols[1].metric("RR", f"{int(current_vals['RR'])}")
    st.metric("Perfusion Index (PI)", f"{current_vals['PI']:.2f}")
    if current_vals['PI'] < 1.0:
        st.caption("🔴 Poor Peripheral Perfusion")
