import streamlit as st
import pandas as pd
import numpy as np
import utils

# --- 1. UX: PAGE CONFIG & STYLING ---
st.set_page_config(page_title="Hemodynamic AI", layout="wide")

st.markdown("""
<style>
    /* CSS Variables for Consistency */
    :root {
        --primary: #2563eb;
        --danger: #dc2626;
        --bg-card: #ffffff;
        --text-meta: #6b7280;
    }
    
    /* Clean Card Layout */
    .metric-container {
        background: var(--bg-card);
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        display: flex;
        flex-direction: column;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-meta);
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
        font-feature-settings: "tnum";
    }
    .metric-sub {
        font-size: 0.75rem;
        margin-top: 4px;
        font-weight: 500;
    }
    
    /* Status Indicators */
    .trend-up { color: #dc2626; }   /* Bad trend up (HR) */
    .trend-down { color: #dc2626; } /* Bad trend down (BP) */
    .trend-ok { color: #059669; }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA LAYER ---
@st.cache_data
def get_data(): return utils.simulate_hemodynamic_shock()
df = get_data()

# --- 3. CONTROLS ---
with st.sidebar:
    st.header("Simulation Control")
    curr_time = st.slider("Time Elapsed (min)", 120, 720, 720)
    
    st.markdown("---")
    st.markdown("#### 📖 Scenario Context")
    st.info("""
    **Occult Hemorrhage Model**
    
    1. **T=0-250 (Stable):** Normal Physiology.
    2. **T=250-550 (Compensated Shock):** 
       - Volume drops. 
       - **PI drops** (Vasoconstriction).
       - **HR rises** to maintain MAP.
       - **Pulse Pressure narrows**.
    3. **T=550+ (Decompensation):**
       - Reserve exhausted. MAP crashes.
    """)

# --- 4. LOGIC ENGINE ---
current = df.iloc[curr_time-1]
prev = df.iloc[curr_time-10] # 10 min lookback for delta

# Clinical Logic
shock_index = current['HR'] / current['SBP']
pulse_pressure = current['SBP'] - current['DBP']

risk_status = "STABLE"
risk_color = "green"
if current['PI'] < 1.0 or shock_index > 0.8:
    risk_status = "COMPENSATING"
    risk_color = "orange"
if current['MAP'] < 65:
    risk_status = "CRITICAL SHOCK"
    risk_color = "red"

# --- 5. UI LAYOUT ---

# Header
c1, c2 = st.columns([3, 1])
with c1:
    st.title("Hemodynamic Command Center")
    st.markdown(f"**Patient Status:** :{risk_color}[{risk_status}] | **Protocol:** SEPSIS-3")
with c2:
    # Top-level Risk Score (0-100)
    # Simple heuristic: normalized average of bad metrics
    risk_score = min(100, int((shock_index * 50) + (10/current['PI'] * 2)))
    st.metric("Aggregate Risk Score", f"{risk_score}/100", 
              delta="High" if risk_score > 50 else "Normal", delta_color="inverse")

st.markdown("---")

# METRIC CARDS (The "Dashboard" View)
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

def html_metric(col, label, value, sub_val, sub_color="gray"):
    col.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub" style="color:{sub_color}">{sub_val}</div>
    </div>
    """, unsafe_allow_html=True)

# 1. MAP (The Organ Perfusion Goal)
delta_map = current['MAP'] - prev['MAP']
html_metric(col_m1, "MAP (Mean Art. Press)", f"{int(current['MAP'])}", 
            f"{delta_map:+.1f} mmHg (10m)", 
            "red" if current['MAP'] < 65 else "green")

# 2. Heart Rate (The Compensation Engine)
delta_hr = current['HR'] - prev['HR']
html_metric(col_m2, "Heart Rate", f"{int(current['HR'])}", 
            f"{delta_hr:+.1f} bpm (Trend)", 
            "red" if current['HR'] > 100 else "gray")

# 3. Pulse Pressure (The Stroke Volume Proxy)
# *Crucial Value Add*
html_metric(col_m3, "Pulse Pressure", f"{int(pulse_pressure)}", 
            "Narrowing < 30" if pulse_pressure < 30 else "Normal",
            "red" if pulse_pressure < 30 else "green")

# 4. Shock Index (The Occult Indicator)
html_metric(col_m4, "Shock Index (HR/SBP)", f"{shock_index:.2f}", 
            "Limit > 0.9", 
            "red" if shock_index > 0.9 else "gray")

# 5. PI (The Early Warning)
html_metric(col_m5, "Perfusion Index (PI)", f"{current['PI']:.2f}%", 
            "Vasoconstriction < 1.0", 
            "red" if current['PI'] < 1.0 else "green")

st.markdown("<br>", unsafe_allow_html=True)

# MAIN VISUALIZATION ROW
c_main, c_side = st.columns([2.5, 1])

with c_main:
    st.subheader("Hemodynamic Profile")
    st.caption("Visualizing the 'Crossover' (HR rising / BP stable) and Pulse Pressure narrowing.")
    fig_profile = utils.plot_hemodynamic_profile(df, curr_time)
    st.plotly_chart(fig_profile, use_container_width=True)

with c_side:
    st.subheader("Early Warning System")
    
    # 1. PI Trend (SPC)
    fig_pi = utils.plot_causal_spc(df, curr_time)
    st.plotly_chart(fig_pi, use_container_width=True)
    
    # 2. Radar Diagnosis
    st.markdown("**Stability Fingerprint**")
    fig_radar = utils.plot_perfusion_radar(current)
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # 3. AI Insight Box
    if risk_status == "COMPENSATING":
        st.warning("""
        **AI Analysis:** High Probability of Compensated Shock.
        - HR is rising to defend MAP.
        - Pulse Pressure is narrowing (Low Stroke Volume).
        - **Action:** Assess Fluid Responsiveness.
        """)
    elif risk_status == "CRITICAL SHOCK":
        st.error("**AI Analysis:** Hemodynamic Decompensation. Vasopressors required.")
    else:
        st.success("**AI Analysis:** Hemodynamics Stable. No micro-circulatory failure detected.")
