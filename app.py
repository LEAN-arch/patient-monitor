import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go  
import utils

# --- Config ---
st.set_page_config(page_title="Smart ICU Monitor", layout="wide", initial_sidebar_state="expanded")

st.title("🏥 Smart ICU Monitor: Early Warning System")
st.markdown("""
**Status**: Monitoring hemodynamic stability via Multivariate SPC.
**Goal**: Detect decompensation *before* vitals breach standard alarm limits.
""")

# --- 1. Load Data ---
@st.cache_data
def get_data():
    return utils.simulate_patient(mins_total=720)

df_full = get_data()
VARS = ['HR','SBP','SpO2','RR','PI']

# --- 2. Compute Analytics (Full History) ---
with st.spinner("Processing physiological signals..."):
    # Fit model on first 2 hours (assumed baseline)
    residuals_full, cov_est, lag = utils.fit_var_and_residuals_full(df_full[VARS], baseline_window=120)
    # Compute Risk
    risk_full, cov_inv = utils.compute_mahalanobis_risk(residuals_full, cov_est)

# --- 3. Controls ---
with st.sidebar:
    st.header("Simulation Control")
    curr_time = st.slider("Time Elapsed (min)", 60, 720, 720)
    view_window = st.selectbox("Zoom Window (min)", [60, 120, 240, 720], index=2)
    st.info("Tip: Slide 'Time Elapsed' back to min 400 to see the stable state, then forward to 600 to watch the shock develop.")

# --- 4. Data Slicing ---
start_idx = max(0, curr_time - view_window)
end_idx = curr_time
df_view = df_full.iloc[start_idx:end_idx].reset_index(drop=True)
risk_view = risk_full[start_idx:end_idx]
resid_view = residuals_full[start_idx:end_idx]
t_axis = np.arange(start_idx, end_idx)

# --- 5. Alert Banner ---
current_risk = risk_full[curr_time-1]
if current_risk < 15:
    st.success(f"✅ PATIENT STABLE (Risk Score: {current_risk:.1f})")
elif current_risk < 30:
    st.warning(f"⚠️ WARNING: PHYSIOLOGICAL DEVIATION (Risk Score: {current_risk:.1f})")
else:
    st.error(f"🚨 CRITICAL INSTABILITY DETECTED (Risk Score: {current_risk:.1f})")

# --- 6. Vitals & Risk (The "What") ---
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("1. Hemodynamic Trends")
    fig_vitals = utils.plot_vitals(df_view, t_axis)
    st.plotly_chart(fig_vitals, use_container_width=True)

with c2:
    st.subheader("2. Integrated Instability")
    fig_risk = utils.plot_risk_enhanced(risk_view, t_axis)
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Mini-KPIs
    last = df_full.iloc[curr_time-1]
    c2a, c2b = st.columns(2)
    c2a.metric("Shock Index", f"{(last['HR']/last['SBP']):.2f}", delta_color="inverse")
    c2b.metric("Perfusion Index", f"{last['PI']:.2f}", delta_color="normal")

# --- 7. Root Cause Analysis (The "Why") ---
st.divider()
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("3. Root Cause Analysis (Contribution)")
    # Calculate contributions for the LAST minute only
    r_t = residuals_full[curr_time-1]
    # Contribution = residual * weighted_residual
    contrib = r_t * (cov_inv @ r_t)
    # Convert to meaningful percentage
    contrib_abs = np.abs(contrib)
    contrib_norm = (contrib_abs / np.sum(contrib_abs)) * 100
    
    # Color code: Red if residual was positive (High), Blue if negative (Low)
    # This tells us: "HR contributing 40% because it is HIGH"
    colors = ['#ef553b' if r_t[i] > 0 else '#636efa' for i in range(5)]
    
    fig_bar = go.Figure(go.Bar(
        x=VARS, 
        y=contrib_norm,
        marker_color=colors,
        text=[f"{val:.1f}%" for val in contrib_norm],
        textposition='auto'
    ))
    fig_bar.update_layout(
        title=f"Which vitals are driving the alarm? (t={curr_time})",
        yaxis_title="Contribution to Risk (%)",
        height=350
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("Red = Variable is higher than predicted. Blue = Variable is lower than predicted.")

with col_right:
    st.subheader("4. State Space Trajectory")
    fig_pca = utils.plot_pca_clinical(df_full, curr_time, baseline_window=120)
    st.plotly_chart(fig_pca, use_container_width=True)

# --- 8. Heatmap (Deep Dive) ---
st.subheader("5. Clinical Deviation Matrix (Deep Dive)")
fig_heat = utils.plot_deviation_matrix(resid_view, VARS)
st.plotly_chart(fig_heat, use_container_width=True)
