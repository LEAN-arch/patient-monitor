import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import chi2
import math

# Import local utilities
import utils

# Configuration
st.set_page_config(page_title="SPC Patient Monitor", layout="wide")
VARS = ['HR','SBP','SpO2','RR','PI']
BASELINE_WINDOW = 180

st.title("Streamlined SPC Patient Monitor — Analytics Dashboard")

# --- Data Loading / Generation ---
@st.cache_data
def load_data():
    return utils.simulate_patient(mins_total=720, seed=42)

df = load_data()

# --- Sidebar ---
st.sidebar.header("Controls")
window_size = st.sidebar.slider("Display window (minutes)", 60, 720, 240, step=30)
show_pca = st.sidebar.checkbox("Show PCA trajectory", value=True)
show_heatmap = st.sidebar.checkbox("Show residual heatmap", value=True)
show_contrib = st.sidebar.checkbox("Show contrib bar", value=True)

if st.sidebar.button("Regenerate Simulation"):
    st.cache_data.clear()
    st.rerun()

# Prepare View Data
df_view = df.tail(window_size).reset_index(drop=True)
t = np.arange(len(df_view))

# --- KPI Row ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Heart Rate", f"{df_view['HR'].iloc[-1]:.0f} bpm", delta=f"{df_view['HR'].diff().iloc[-1]:.1f}")
c2.metric("SpO2", f"{df_view['SpO2'].iloc[-1]:.1f} %", delta=f"{df_view['SpO2'].diff().iloc[-1]:.1f}")
c3.metric("Shock Index", f"{(df_view['HR'].iloc[-1]/df_view['SBP'].iloc[-1]):.2f}")
c4.metric("Perfusion Index", f"{df_view['PI'].iloc[-1]:.2f}")

# --- Analytics Calculation (On the Fly) ---
# 1. VAR Residuals & Mahalanobis
try:
    # Fit on baseline of current view (or logical baseline)
    calc_vars = df_view[VARS]
    train_n = min(BASELINE_WINDOW, len(calc_vars))
    residuals_full, cov_est = utils.fit_var_and_residuals(calc_vars, train_end=train_n)
    D2, zD, cov_inv = utils.mahalanobis_ewma(residuals_full, cov_est)
except Exception as e:
    st.error(f"Analytics Error: {e}")
    residuals_full, D2, zD = None, None, None

# 2. MEWMA T2
try:
    T2 = utils.mewma_T2(df_view[VARS], baseline_end=min(BASELINE_WINDOW, len(df_view)))
    T2_thresh = chi2.ppf(1-0.0005, df=len(VARS))
except:
    T2, T2_thresh = None, None

# --- Main Visualizations ---

# 1. Vitals Timeline
st.subheader("Vitals Timeline")
fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(x=t, y=df_view['HR'], mode='lines', name='HR', line=dict(color='firebrick')))
fig_ts.add_trace(go.Scatter(x=t, y=df_view['SpO2'], mode='lines', name='SpO2', yaxis='y2', line=dict(color='royalblue')))
fig_ts.update_layout(
    height=350,
    yaxis=dict(title='HR (bpm)'), 
    yaxis2=dict(title='SpO2 (%)', overlaying='y', side='right', range=[80,100]),
    margin=dict(l=20, r=20, t=20, b=20)
)
st.plotly_chart(fig_ts, use_container_width=True)

# 2. Multivariate Risk (Mahalanobis EWMA)
if zD is not None:
    st.subheader("Multivariate Anomaly Score (Mahalanobis EWMA)")
    fig_risk = go.Figure()
    fig_risk.add_trace(go.Scatter(x=t, y=zD, mode='lines', name='Risk Score', fill='tozeroy'))
    fig_risk.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Anomaly Score (zD)")
    st.plotly_chart(fig_risk, use_container_width=True)

# 3. Advanced Analytics Layout
col_left, col_right = st.columns(2)

with col_left:
    if show_pca:
        st.subheader("PCA Trajectory")
        fig_pca = utils.plot_pca_trajectory(df_view[VARS])
        st.plotly_chart(fig_pca, use_container_width=True)

with col_right:
    if show_heatmap and residuals_full is not None:
        st.subheader("Residual Heatmap")
        fig_heat = utils.plot_residuals_heatmap(residuals_full, VARS)
        st.plotly_chart(fig_heat, use_container_width=True)

# 4. Contributions (Root Cause Analysis)
if show_contrib and residuals_full is not None:
    st.subheader("Metric Contribution to Risk (Last Minute)")
    try:
        last_resid = residuals_full[-1]
        contrib = last_resid * (cov_inv.dot(last_resid))
        contrib_norm = contrib / (np.sum(np.abs(contrib)) + 1e-12)
        dfc = pd.DataFrame({'Variable': VARS, 'Contribution': contrib_norm})
        fig_contrib = px.bar(dfc, x='Variable', y='Contribution', 
                             title='Which vital sign is driving the anomaly?')
        st.plotly_chart(fig_contrib, use_container_width=True)
    except Exception as e:
        st.info("Insufficient data for contribution analysis.")
