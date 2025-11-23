import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  # <--- This was missing
import utils

# --- Config ---
st.set_page_config(page_title="Clinical SPC Monitor", layout="wide")
st.title("🏥 Real-Time Clinical SPC Monitor")
st.markdown("Monitoring **Multivariate Physiological Stability** using Vector Autoregression & Mahalanobis Distance.")

# --- 1. Load & Simulate Data ---
@st.cache_data
def get_data():
    # Generate 12 hours (720 mins) of data
    return utils.simulate_patient(mins_total=720)

df_full = get_data()
VARS = ['HR','SBP','SpO2','RR','PI']

# --- 2. Global Analytics (Pre-computation) ---
# We compute this on the FULL dataset so the history is valid 
# even if we only view the last 30 mins.

with st.spinner("Analyzing patient history..."):
    # A. Fit VAR model on first 120 mins (stable baseline)
    residuals_full, cov_est, lag = utils.fit_var_and_residuals_full(df_full[VARS], baseline_window=120)
    
    # B. Compute Risk Scores (Mahalanobis)
    risk_full, cov_inv = utils.compute_mahalanobis_risk(residuals_full, cov_est)

# --- 3. Dashboard Controls ---
with st.sidebar:
    st.header("Monitor Settings")
    # Slider to control "Current Time" simulation
    curr_time = st.slider("Simulation Time (min)", min_value=60, max_value=720, value=720)
    view_window = st.selectbox("View Window", [60, 120, 240, 720], index=1)
    
    st.divider()
    show_pca = st.checkbox("Show PCA State Space", value=True)
    show_heat = st.checkbox("Show Residual Heatmap", value=True)
    
    if st.button("Reset Simulation"):
        st.cache_data.clear()
        st.rerun()

# --- 4. Slice Data for View ---
# Determine start/end indices
start_idx = max(0, curr_time - view_window)
end_idx = curr_time

# Slice the Dataframes and Arrays
df_view = df_full.iloc[start_idx:end_idx].reset_index(drop=True)
risk_view = risk_full[start_idx:end_idx]
resid_view = residuals_full[start_idx:end_idx]
t_axis = np.arange(start_idx, end_idx)

# --- 5. KPI Header ---
# Get the very last value from the selected time
last_row = df_full.iloc[curr_time-1]
prev_row = df_full.iloc[curr_time-2] if curr_time > 1 else last_row

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Heart Rate", f"{last_row['HR']:.0f}", f"{last_row['HR']-prev_row['HR']:.1f}")
c2.metric("SBP", f"{last_row['SBP']:.0f}", f"{last_row['SBP']-prev_row['SBP']:.1f}", delta_color="inverse")
c3.metric("SpO2", f"{last_row['SpO2']:.1f}%", f"{last_row['SpO2']-prev_row['SpO2']:.1f}")
c4.metric("Resp Rate", f"{last_row['RR']:.0f}", f"{last_row['RR']-prev_row['RR']:.1f}")
c5.metric("Risk Score", f"{risk_full[curr_time-1]:.1f}", delta_color="off" if risk_full[curr_time-1] < 20 else "inverse")

# --- 6. Main Plots ---
st.subheader("Hemodynamic Trends")
fig_vitals = utils.plot_vitals(df_view, t_axis)
st.plotly_chart(fig_vitals, use_container_width=True)

st.subheader("Integrated Risk Analysis")
fig_risk = utils.plot_risk(risk_view, t_axis)
st.plotly_chart(fig_risk, use_container_width=True)

# --- 7. Advanced Analytics ---
col1, col2 = st.columns(2)

if show_pca:
    with col1:
        # Show PCA trajectory up to current time
        fig_pca = utils.plot_pca(df_full.iloc[:curr_time])
        st.plotly_chart(fig_pca, use_container_width=True)

if show_heat:
    with col2:
        fig_heat = utils.plot_heatmap(resid_view, VARS)
        st.plotly_chart(fig_heat, use_container_width=True)

# --- 8. Contribution Analysis (Why is risk high?) ---
if risk_full[curr_time-1] > 10:
    st.warning(f"High Anomaly Score Detected at min {curr_time}")
    st.markdown("**Root Cause Contribution (Last Minute):**")
    
    # Calculate contribution: w_i = r_i * (S^-1 * r)_i
    # Only for the last point
    r_t = residuals_full[curr_time-1]
    # Element-wise multiply residual by (InvCov dot Residual)
    contrib = r_t * (cov_inv @ r_t)
    # Normalize
    contrib_pct = (contrib / np.sum(np.abs(contrib))) * 100
    
    contrib_df = pd.DataFrame({
        'Metric': VARS,
        'Contribution (%)': contrib_pct
    })
    
    fig_bar = px.bar(contrib_df, x='Metric', y='Contribution (%)', 
                     color='Contribution (%)', color_continuous_scale='Redor')
    st.plotly_chart(fig_bar, use_container_width=True)
