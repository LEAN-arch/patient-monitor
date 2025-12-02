import streamlit as st
import pandas as pd
import numpy as np
import utils

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Vector | Advanced ICU Math", layout="wide")

st.markdown("""
<style>
    /* Dark Mode Medical UI */
    .stApp { background-color: #0f172a; color: #f8fafc; }
    
    /* Metrics */
    .math-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .math-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .math-val { font-size: 2.0rem; font-weight: 300; font-family: monospace; color: #38bdf8; }
    
    /* Statuses */
    .status-box { padding: 10px; border-radius: 4px; font-weight: bold; text-align: center; }
    .crit { background-color: rgba(220, 38, 38, 0.2); color: #f87171; border: 1px solid #ef4444; }
    .safe { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }

</style>
""", unsafe_allow_html=True)

# --- 2. INITIALIZE MATH ENGINE ---
@st.cache_data
def run_simulation():
    raw_df = utils.simulate_advanced_dynamics()
    # Apply Math Pipeline
    processed_df = utils.run_kalman_smoothing(raw_df)
    final_df = utils.run_complexity_analysis(processed_df)
    return final_df

df = run_simulation()

# --- 3. CONTROLS ---
with st.sidebar:
    st.header("Vector Engine")
    curr_time = st.slider("Session Time (min)", 120, 720, 720)
    st.markdown("### 🧮 Model Explained")
    st.info("""
    **1. Kalman Filter:** 
    Removes sensor noise ($z_t$) to find true state ($x_t$).
    
    **2. Sample Entropy:**
    Calculates signal complexity. Low Entropy = Sepsis.
    
    **3. Attractor Reconstruction:**
    3D visualization of hemodynamic stability.
    """)

# --- 4. HUD (Heads Up Display) ---
current = df.iloc[curr_time-1]
entropy = current['Entropy']
kalman_hr = current['Kalman_HR']

# Prognosis Logic
system_state = "HEALTHY CHAOS"
state_class = "safe"

if entropy < 0.5:
    system_state = "PATHOLOGICAL RIGIDITY (SEPSIS)"
    state_class = "crit"
elif entropy < 0.8:
    system_state = "COMPLEXITY LOSS (WARNING)"
    state_class = "crit" # using red for high visual impact

c1, c2 = st.columns([3, 1])
with c1:
    st.title("Vector™ | Computational ICU")
    st.markdown(f"**System State Analysis:** <span class='status-box {state_class}'>{system_state}</span>", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="math-card">
        <div class="math-label">Complexity Index</div>
        <div class="math-val">{entropy:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- 5. VISUALIZATION GRID ---
col_left, col_right = st.columns([2, 1])

with col_left:
    # 1. Kalman Filter Plot
    st.markdown("### 1. Signal Separation (Kalman Filter)")
    st.caption("Gray: Raw Noisy Sensor Data. Blue: Probabilistic State Estimation.")
    fig_kalman = utils.plot_kalman_separation(df, curr_time)
    st.plotly_chart(fig_kalman, use_container_width=True)
    
    # 2. Entropy Monitor
    st.markdown("### 2. Non-Linear Dynamics (Entropy)")
    st.caption("A drop in entropy indicates the heart rate is becoming 'too regular' - a hallmark of Autonomic Failure.")
    fig_ent = utils.plot_entropy_monitor(df, curr_time)
    st.plotly_chart(fig_ent, use_container_width=True)

with col_right:
    # 3. 3D Attractor
    st.markdown("### 3. Phase Space Geometry")
    st.caption("Time-Delay Embedding (t, t-1, t-2). Visualizes the shape of the attractor.")
    fig_3d = utils.plot_phase_space_3d(df, curr_time)
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Technical Data Box
    st.markdown("""
    <div style="background:#1e293b; padding:15px; border-radius:8px; font-size:0.85rem; border:1px solid #334155;">
        <strong>MATH TELEMETRY:</strong><br>
        • <strong>Model:</strong> Linear Gaussian State Space<br>
        • <strong>Process Var (Q):</strong> 1e-5<br>
        • <strong>Measure Var (R):</strong> 0.1<br>
        • <strong>Embedding Dim (m):</strong> 3<br>
        • <strong>Time Lag (τ):</strong> 2 samples
    </div>
    """, unsafe_allow_html=True)
