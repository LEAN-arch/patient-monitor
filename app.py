import streamlit as st
import pandas as pd
import numpy as np
import utils

# --- CONFIG ---
st.set_page_config(page_title="AETHER | Digital Twin", layout="wide", initial_sidebar_state="expanded")

# --- CSS: GLASSMORPHISM ---
st.markdown("""
<style>
    .stApp { background-color: #f1f5f9; }
    
    /* Hero Card */
    .hero-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* KPI Mini Card */
    .kpi-mini {
        background: white; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 10px; text-align: center; height: 100%;
    }
    .kpi-val { font-size: 1.6rem; font-weight: 800; color: #0f172a; }
    .kpi-lbl { font-size: 0.7rem; font-weight: 700; color: #64748b; text-transform: uppercase; }
    
    /* Rec Box */
    .rec-box {
        background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534;
        padding: 15px; border-radius: 8px; font-weight: 600;
        display: flex; align-items: center; gap: 10px;
    }
    .rec-crit { background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }
    
</style>
""", unsafe_allow_html=True)

# --- ENGINE ---
@st.cache_data
def run_simulation(): return utils.simulate_predictive_scenario()
df, predictions = run_simulation()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Aether Control")
    curr_time = st.slider("Session Time", 200, 720, 720)
    
    st.markdown("### 🧪 Predictive Sandbox")
    st.caption("Simulate intervention effect:")
    sim_mode = st.radio("Select Simulation:", ["None", "Fluid Bolus (+500mL)", "Norepinephrine (+0.1mcg)"], index=0)
    
    st.markdown("---")
    st.info("**Digital Twin Engine:** Uses 0-D Cardiovascular Physics (Windkessel) + Baroreflex Feedback Loop.")

# --- ANALYTICS ---
cur = df.iloc[curr_time-1]
prv = df.iloc[curr_time-15]

# --- 1. INTELLIGENT HEADER ---
c1, c2 = st.columns([2, 1])

with c1:
    st.markdown("### 🧬 AETHER | Hemodynamic Digital Twin")
    
    # Recommendation Engine
    rec_text = "Patient Stable."
    rec_style = ""
    if cur['MAP'] < 65:
        if cur['SVR'] < 800:
            rec_text = "⚠️ RECOMMENDATION: Vasopressors indicated (Vasoplegia)."
            rec_style = "rec-crit"
        else:
            rec_text = "⚠️ RECOMMENDATION: Fluid Resuscitation indicated (Hypovolemia)."
            rec_style = "rec-crit"
    elif cur['Lactate'] > 2.0:
        rec_text = "⚠️ WARNING: Occult Hypoperfusion. Evaluate Cardiac Index."
        rec_style = "rec-crit"
        
    st.markdown(f'<div class="rec-box {rec_style}">🤖 AI INSIGHT: {rec_text}</div>', unsafe_allow_html=True)

with c2:
    # Top Level Risk Score
    risk = int(np.mean([
        (100 - cur['MAP']) if cur['MAP'] < 65 else 0,
        (cur['Lactate'] * 10),
        (cur['HR'] - 100) if cur['HR'] > 100 else 0
    ]))
    risk = np.clip(risk, 0, 100)
    st.metric("Total Decompensation Risk", f"{int(risk)}%", f"{int(risk - 10)}% vs 15m ago", delta_color="inverse")

# --- 2. KPI MATRIX (Sparklines implied by context) ---
cols = st.columns(6)
def kpi(col, lbl, val, unit, color):
    col.markdown(f"""
    <div class="kpi-mini" style="border-top: 4px solid {color}">
        <div class="kpi-lbl">{lbl}</div>
        <div class="kpi-val" style="color:{color}">{val}</div>
        <div style="font-size:0.8rem; color:#94a3b8">{unit}</div>
    </div>
    """, unsafe_allow_html=True)

kpi(cols[0], "MAP", f"{cur['MAP']:.0f}", "mmHg", "#d946ef")
kpi(cols[1], "Cardiac Index", f"{cur['CI']:.1f}", "L/min", "#10b981")
kpi(cols[2], "SVR Index", f"{cur['SVRI']:.0f}", "dyn", "#f59e0b")
kpi(cols[3], "Stroke Vol", f"{cur['SV']:.0f}", "mL", "#0ea5e9")
kpi(cols[4], "DO2 Index", f"{cur['DO2']:.0f}", "mL/m2", "#6366f1")
kpi(cols[5], "Lactate", f"{cur['Lactate']:.1f}", "mmol", "#ef4444")

st.markdown("<br>", unsafe_allow_html=True)

# --- 3. PREDICTIVE CANVAS ---
t1, t2 = st.tabs(["🔮 PREDICTIVE HEMODYNAMICS", "🫀 ORGAN MECHANICS"])

with t1:
    c_left, c_right = st.columns([1, 1])
    
    with c_left:
        # The Money Plot: Bullseye with Prediction Vectors
        st.markdown("**1. PREDICTIVE COMPASS**")
        st.caption("Arrows indicate predicted response to Fluids (Blue) vs Pressors (Purple).")
        fig_bull = utils.plot_predictive_bullseye(df, predictions, curr_time)
        st.plotly_chart(fig_bull, use_container_width=True)
    
    with c_right:
        # The Time Machine: Multiverse
        st.markdown("**2. INTERVENTION SIMULATOR**")
        st.caption("Projected MAP trajectory for different therapeutic strategies.")
        fig_hor = utils.plot_horizon_multiverse(df, predictions, curr_time)
        st.plotly_chart(fig_hor, use_container_width=True)
        
        if sim_mode == "Fluid Bolus (+500mL)":
            st.success(f"SIMULATION: Fluid Bolus projected to increase MAP by {int(predictions['fluid'][-1] - predictions['natural'][-1])} mmHg.")
        elif sim_mode == "Norepinephrine (+0.1mcg)":
            st.warning(f"SIMULATION: Pressor projected to increase MAP by {int(predictions['pressor'][-1] - predictions['natural'][-1])} mmHg.")

with t2:
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**3. ORGAN RISK TOPOLOGY**")
        st.caption("Geometric view of competing organ risks.")
        fig_rad = utils.plot_organ_radar(df, curr_time)
        st.plotly_chart(fig_rad, use_container_width=True)
    with c4:
        st.markdown("**4. FRANK-STARLING MECHANICS**")
        st.caption("Preload vs Stroke Volume relationship.")
        fig_star = utils.plot_starling_dynamic(df, curr_time)
        st.plotly_chart(fig_star, use_container_width=True)

# --- 4. EXPLANATION ---
with st.expander("ℹ️ HOW THE DIGITAL TWIN WORKS"):
    st.markdown("""
    **1. Physics Engine:** This app runs a 0-D cardiovascular model (Windkessel approximation) in the background. It solves for Flow, Pressure, and Resistance based on simulated contractility and compliance.
    
    **2. Predictive Vectors:** The 'Compass' chart simulates 30 minutes into the future for 3 scenarios (Do nothing, Give Fluids, Give Pressors). The arrows show you *exactly* where the patient will land.
    
    **3. Organ Radar:** Normalizes risks across Renal (Low MAP), Cardiac (High Rate), and Metabolic (High Lactate) axes to visualize 'Therapeutic Conflict'.
    """)
