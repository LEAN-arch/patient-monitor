import streamlit as st
import pandas as pd
import numpy as np
import utils

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ICU Sentinel Pro", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS (The "Commercial" Skin) ---
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Hardware Monitor Card Style */
    .metric-card {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-label {
        color: #888;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-family: 'Courier New', Courier, monospace;
        font-size: 2.2rem;
        font-weight: bold;
    }
    .metric-unit {
        font-size: 0.9rem;
        color: #666;
    }
    
    /* Neon Colors for specific metrics */
    .color-hr { color: #00ff00; }
    .color-sbp { color: #ff3333; }
    .color-spo2 { color: #00ccff; }
    .color-pi { color: #d142f5; }
    .color-si { color: #ffa500; }
    
    /* Alert Banners */
    .alert-box {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        font-weight: bold;
    }
    .alert-safe { background-color: rgba(0, 255, 0, 0.1); border: 1px solid #00ff00; color: #00ff00; }
    .alert-warn { background-color: rgba(255, 165, 0, 0.1); border: 1px solid #ffa500; color: #ffa500; }
    .alert-crit { background-color: rgba(255, 0, 0, 0.2); border: 1px solid #ff0000; color: #ff0000; }
    
</style>
""", unsafe_allow_html=True)

# --- DATA PREP ---
@st.cache_data
def load_data():
    df = utils.simulate_patient()
    df['SI'] = df['HR'] / df['SBP']
    return df

df_full = load_data()
VARS = ['HR','SBP','SpO2','RR','PI']

# Analytics
z_full, cov_est = utils.fit_var_and_residuals_full(df_full[VARS])
risk_full, _ = utils.compute_mahalanobis_risk(z_full, cov_est)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Playback Control")
    curr_time = st.slider("Timeline (mins)", 120, 720, 720)
    view_window = st.selectbox("Zoom", [60, 180, 360], index=1)
    st.markdown("---")
    st.markdown("**Simulation Guide:**")
    st.markdown("- **< 400m:** Stable Baseline")
    st.markdown("- **400-550m:** Compensated Shock (High HR, Low PI)")
    st.markdown("- **> 550m:** Decompensated Shock (Low BP)")

# --- SLICING ---
start_idx = max(0, curr_time - view_window)
df_view = df_full.iloc[start_idx:curr_time]
z_view = z_full.iloc[start_idx:curr_time]
current_vals = df_full.iloc[curr_time-1]
current_risk = risk_full[curr_time-1]

# --- TOP BAR: ALERT SYSTEM ---
status_col, spacer = st.columns([3, 1])
with status_col:
    if current_risk < 15:
        st.markdown(f'<div class="alert-box alert-safe">✅ MONITORING ACTIVE - PATIENT STABLE (Risk: {current_risk:.1f})</div>', unsafe_allow_html=True)
    elif current_risk < 25:
        st.markdown(f'<div class="alert-box alert-warn">⚠️ EARLY WARNING - HEMODYNAMIC COMPENSATION DETECTED (Risk: {current_risk:.1f})</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-box alert-crit">🚨 CRITICAL ALARM - DECOMPENSATION / SHOCK STATE (Risk: {current_risk:.1f})</div>', unsafe_allow_html=True)

# --- MAIN DASHBOARD GRID ---
# Layout: Left (Live Monitors) - Right (Advanced Analytics)
c_left, c_right = st.columns([1, 3])

with c_left:
    st.markdown("### LIVE VITALS")
    
    # Custom HTML Metric Cards
    def metric_card(label, value, unit, color_class):
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {color_class}">{value}</div>
            <div class="metric-unit">{unit}</div>
        </div>
        """, unsafe_allow_html=True)

    metric_card("Heart Rate", int(current_vals['HR']), "bpm", "color-hr")
    metric_card("Invasive BP", f"{int(current_vals['SBP'])}/80", "mmHg", "color-sbp")
    metric_card("Shock Index", f"{current_vals['SI']:.2f}", "ratio", "color-si")
    metric_card("Perfusion Idx", f"{current_vals['PI']:.2f}", "%", "color-pi")
    metric_card("SpO2", int(current_vals['SpO2']), "%", "color-spo2")

    st.markdown("### RISK PROFILE")
    fig_gauge = utils.plot_gauge_commercial(current_risk)
    st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("### CLINICAL AXIS")
    fig_radar = utils.plot_radar_commercial(current_vals)
    st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

with c_right:
    # 1. Main Monitor Strip
    st.markdown("### HEMODYNAMIC TRENDS")
    fig_strip = utils.plot_monitor_strip(df_view, df_view.index, df_view['SI'])
    st.plotly_chart(fig_strip, use_container_width=True, config={'displayModeBar': False})
    
    # 2. Deviation Heatmap
    st.markdown("### PHYSIOLOGICAL DEVIATION MATRIX")
    fig_heat = utils.plot_heatmap_commercial(z_view, VARS)
    st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': False})
