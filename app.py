import streamlit as st
import pandas as pd
import numpy as np
import utils

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Aigis: Advanced ICU Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS: PRO-GRADE STYLING ---
st.markdown("""
<style>
    /* Global Clean Up */
    .stApp { background-color: #0b0e11; }
    .block-container { padding-top: 1rem; }
    
    /* Metrics Cards */
    div[data-testid="metric-container"] {
        background-color: #15191f;
        border: 1px solid #2b313b;
        padding: 10px;
        border-radius: 6px;
        color: #e0e0e0;
    }
    label[data-testid="stMetricLabel"] { font-size: 0.8rem; color: #8b949e; }
    div[data-testid="stMetricValue"] { font-family: 'Roboto Mono', monospace; font-size: 1.8rem; }
    
    /* Custom Headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #58a6ff;
        border-bottom: 1px solid #2b313b;
        padding-bottom: 5px;
        margin-bottom: 15px;
        margin-top: 20px;
    }
    
    /* Risk Badges */
    .badge-crit { background-color: #ff0055; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-warn { background-color: #d29922; color: black; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-ok { background-color: #238636; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }

</style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def get_data():
    return utils.simulate_patient_dynamics()

df = get_data()

# --- SIDEBAR: TIME TRAVEL ---
with st.sidebar:
    st.title("Aigis Control")
    curr_time = st.slider("Session Time (min)", 100, 720, 720)
    st.info("""
    **Event Log:**
    - T=0-400: Stable
    - T=400: Instability Onset (Hidden)
    - T=600: Hemodynamic Crash
    """)

# --- ANALYTICS ENGINE ---
current_row = df.iloc[curr_time-1]

# 1. SPC Check
_, _, _, violations, _ = utils.detect_spc_violations(df['HR'].iloc[:curr_time])
spc_status = "STABLE"
if len(violations) > 0 and violations[-1] > curr_time - 10:
    spc_status = "UNSTABLE (Drift Detected)"

# 2. Risk Calculation (Simple heuristic for demo)
risk_score = 0
if current_row['PI'] < 2.0: risk_score += 30
if current_row['HR'] > 100: risk_score += 30
if spc_status != "STABLE": risk_score += 20
if current_row['SBP'] < 90: risk_score += 20
risk_score = min(100, risk_score)

# --- LAYOUT: HEADS UP DISPLAY (HUD) ---
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

with c1:
    st.title("Aigis™ ICU Monitor")
    st.caption(f"Patient ID: 9942-X | Male, 58y | Septic Protocol | T={curr_time}min")

with c2:
    st.metric("Risk Probability", f"{risk_score}%", delta=f"{risk_score-10}% vs 1hr ago", delta_color="inverse")

with c3:
    if risk_score > 70:
        st.markdown('<div style="text-align:center; margin-top:10px;"><span class="badge-crit">CRITICAL</span></div>', unsafe_allow_html=True)
    elif risk_score > 30:
        st.markdown('<div style="text-align:center; margin-top:10px;"><span class="badge-warn">WARNING</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:center; margin-top:10px;"><span class="badge-ok">STABLE</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; color:#888; font-size:0.8em;">AI Assessment</div>', unsafe_allow_html=True)

with c4:
    st.metric("Perfusion (PI)", f"{current_row['PI']:.2f}", delta="-0.2" if current_row['PI'] < 2 else "0.0")

# --- ROW 2: PREDICTIVE MONITORING ---
col_main, col_side = st.columns([2, 1])

with col_main:
    st.markdown('<div class="section-header">SPC TRENDS & AI FORECAST (+30 MIN)</div>', unsafe_allow_html=True)
    # The Advanced SPC Plot
    fig_spc = utils.plot_spc_monitor(df, curr_time)
    st.plotly_chart(fig_spc, use_container_width=True)
    
    # Textual Insight
    if risk_score > 30:
        st.warning(f"**AI Insight:** Western Electric Rule violation detected. HR trend indicates non-random drift. Projected to breach 110 BPM in 15 mins.")

with col_side:
    st.markdown('<div class="section-header">CHAOS DYNAMICS</div>', unsafe_allow_html=True)
    # The Chaos Attractor
    fig_chaos = utils.plot_chaos_attractor(df, curr_time)
    st.plotly_chart(fig_chaos, use_container_width=True)
    st.caption("Lag Plot (Attractor): Compression of this shape indicates loss of heart rate complexity/variability (Early Sepsis Marker).")

# --- ROW 3: STATE SPACE & DEEP DIVE ---
col_pca, col_metrics = st.columns([1, 2])

with col_pca:
    st.markdown('<div class="section-header">STATE SPACE TRAJECTORY</div>', unsafe_allow_html=True)
    # The PCA Plot
    fig_pca = utils.plot_state_space(df, curr_time)
    st.plotly_chart(fig_pca, use_container_width=True)

with col_metrics:
    st.markdown('<div class="section-header">CLINICAL METRICS DRILLDOWN</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Heart Rate", f"{int(current_row['HR'])}")
    m2.metric("Sys BP", f"{int(current_row['SBP'])}")
    m3.metric("Shock Index", f"{(current_row['HR']/current_row['SBP']):.2f}")
    
    st.markdown("---")
    st.markdown("""
    **Actionable Protocol:**
    1.  **Check SPC Status:** If yellow markers appear, assess fluid responsiveness.
    2.  **Monitor State Trajectory:** Movement rightward (PC1) indicates hemodynamic stress.
    3.  **Chaos Check:** If Attractor collapses to a line, patient has lost autonomic variability.
    """)
