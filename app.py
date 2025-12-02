import streamlit as st
import pandas as pd
import numpy as np
import utils

# --- CONFIG ---
st.set_page_config(page_title="OMNI | Command Center", layout="wide", initial_sidebar_state="collapsed")

# --- DARK MATTER CSS ---
st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e2e8f0; }
    
    /* Neon Card */
    .neon-card {
        background: rgba(20, 25, 35, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        padding: 10px;
        backdrop-filter: blur(10px);
        margin-bottom: 10px;
    }
    
    /* Sparkline KPI */
    .spark-val { font-size: 1.8rem; font-weight: 800; font-family: monospace; text-shadow: 0 0 10px rgba(0,0,0,0.5); }
    .spark-lbl { font-size: 0.7rem; font-weight: 700; color: #64748b; letter-spacing: 1px; }
    
    /* Alert Header */
    .alert-banner {
        padding: 15px; border-radius: 6px; border: 1px solid; margin-bottom: 20px;
        display: flex; justify-content: space-between; align-items: center;
        background: rgba(255, 0, 85, 0.1); border-color: #ff0055; color: #ff0055;
    }
    .safe-banner { background: rgba(0, 255, 159, 0.1); border-color: #00ff9f; color: #00ff9f; }
    
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load(): return utils.simulate_omni_scenario()
df, preds = load()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Omni Control")
    curr_time = st.slider("Time", 200, 720, 720)
    st.info("Simulation: Sepsis w/ Predictive Digital Twin")

# --- CALC ---
cur = df.iloc[curr_time-1]
prv = df.iloc[curr_time-15]

# --- 1. INTELLIGENT HEADER ---
# AI Logic
if cur['MAP'] < 65:
    status = "CRITICAL: CIRCULATORY FAILURE"
    rec = "RECOMMENDATION: START VASOPRESSORS (Septic Shock Profile)"
    style = "alert-banner"
elif cur['Lactate'] > 2.0:
    status = "WARNING: OCCULT HYPOPERFUSION"
    rec = "RECOMMENDATION: OPTIMIZE FLOW (Fluid/Inotrope)"
    style = "alert-banner"
else:
    status = "STABLE: HEMODYNAMICS OPTIMIZED"
    rec = "CONTINUE MONITORING"
    style = "safe-banner"

st.markdown(f"""
<div class="{style}">
    <div style="font-size:1.2rem; font-weight:bold;">{status}</div>
    <div style="font-weight:600;">{rec}</div>
</div>
""", unsafe_allow_html=True)

# --- 2. ROW 1: SPARKLINE KPI STRIP (The Vital 6) ---
cols = st.columns(6)

def spark_card(col, label, val, unit, color, df_col, low, high):
    with col:
        st.markdown(f"""
        <div class="neon-card" style="border-top: 3px solid {color}">
            <div class="spark-lbl">{label}</div>
            <div class="spark-val" style="color:{color}">{val}</div>
            <div style="font-size:0.7rem; color:#64748b">{unit}</div>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(utils.plot_spark_spc(df, df_col, color, low, high), use_container_width=True, config={'displayModeBar': False})

spark_card(cols[0], "MAP", f"{cur['MAP']:.0f}", "mmHg", "#ff0055", "MAP", 65, 100)
spark_card(cols[1], "CARD. INDEX", f"{cur['CI']:.1f}", "L/min", "#00ff9f", "CI", 2.5, 4.0)
spark_card(cols[2], "SVR", f"{cur['SVR']:.0f}", "dyn", "#ffaa00", "SVR", 800, 1200)
spark_card(cols[3], "STROKE VOL", f"{cur['SV']:.0f}", "mL", "#00f2ea", "SV", 60, 100)
spark_card(cols[4], "LACTATE", f"{cur['Lactate']:.1f}", "mmol", "#b026ff", "Lactate", 0, 2.0)
spark_card(cols[5], "ENTROPY", f"{cur['Entropy']:.2f}", "σ", "#ffffff", "Entropy", 0.5, 2.0)

# --- 3. ROW 2: THE PREDICTIVE CORE (Bullseye & Multiverse) ---
c_left, c_right = st.columns([1, 1])

with c_left:
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    st.markdown("**1. PREDICTIVE COMPASS (Guidance)**")
    st.plotly_chart(utils.plot_predictive_compass(df, preds, curr_time), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c_right:
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    st.markdown("**2. INTERVENTION MULTIVERSE (Horizon)**")
    st.plotly_chart(utils.plot_multiverse(df, preds, curr_time), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 4. ROW 3: DEEP PHYSIOLOGY (Physics & Mechanics) ---
c3, c4, c5 = st.columns(3)

with c3:
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    st.markdown("**3. STARLING VECTOR**")
    st.plotly_chart(utils.plot_starling_vector(df, curr_time), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c4:
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    st.markdown("**4. OXYGEN DEBT LEDGER**")
    st.plotly_chart(utils.plot_oxygen_debt(df, curr_time), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c5:
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    st.markdown("**5. RENAL AUTOREGULATION**")
    st.plotly_chart(utils.plot_renal_cliff(df, curr_time), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 5. ROW 4: ADVANCED MATH (The "Black Box" Output) ---
c6, c7 = st.columns(2)

with c6:
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    st.markdown("**6. GLOBAL STATE SPACE (PCA)**")
    st.caption("Visualizing Total System Stability.")
    st.plotly_chart(utils.plot_pca_space(df, curr_time), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c7:
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    st.markdown("**7. NEURO-AUTONOMIC SPECTRUM**")
    st.caption("Heart Rate Variability Power Spectral Density.")
    st.plotly_chart(utils.plot_spectrum(df, curr_time), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
