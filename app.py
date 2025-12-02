import streamlit as st
import pandas as pd
import numpy as np
import utils

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Vitals | Precision Monitor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. COMMERCIAL CSS (The "Linear/Stripe" Look) ---
st.markdown("""
<style>
    /* 1. Reset & Background */
    .stApp { background-color: #f3f4f6; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }

    /* 2. The 'Card' Component */
    .viz-card {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 16px;
    }

    /* 3. Typography */
    h1, h2, h3 { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; letter-spacing: -0.5px; color: #111827; }
    .card-header { font-size: 14px; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px; }
    
    /* 4. KPI Box Styling */
    .kpi-container { display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid #f3f4f6; padding-bottom: 8px; margin-bottom: 8px; }
    .kpi-label { font-size: 13px; font-weight: 500; color: #6b7280; }
    .kpi-value { font-size: 28px; font-weight: 700; color: #111827; font-feature-settings: "tnum"; }
    .kpi-unit { font-size: 14px; color: #9ca3af; margin-left: 4px; }
    
    /* 5. Alert Badges */
    .badge { padding: 4px 12px; border-radius: 9999px; font-size: 12px; font-weight: 700; display: inline-block; }
    .badge-ok { background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
    .badge-warn { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .badge-crit { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }

</style>
""", unsafe_allow_html=True)

# --- 3. DATA LOADING ---
@st.cache_data
def get_sim_data():
    return utils.simulate_clinical_course()

df = get_sim_data()

# --- 4. CONTROLS (Top Bar) ---
with st.sidebar:
    st.header("Simulation Control")
    curr_time = st.slider("Timeline (minutes)", 120, 720, 720)
    st.caption("Try T=350 (Early Warning) vs T=600 (Crash)")

# --- 5. LOGIC ENGINE ---
current = df.iloc[curr_time-1]
spc_stats = utils.run_western_electric_spc(df['HR'].iloc[:curr_time])
coords, loadings = utils.run_pca_compass(df, curr_time)

# Heuristic Risk Assessment
risk_level = "STABLE"
if current['PI'] < 2.0 or (current['HR'] > spc_stats['u2'].iloc[-1]):
    risk_level = "WARNING"
if current['SI'] > 0.9 or current['SBP'] < 90:
    risk_level = "CRITICAL"

# --- 6. UI LAYOUT ---

# A. Header (HUD)
c_hud1, c_hud2 = st.columns([3, 1])
with c_hud1:
    st.markdown(f"### 🏥 ICU Monitor | Bed 04 | {risk_level} PROTOCOL")
with c_hud2:
    if risk_level == "STABLE":
        st.markdown('<div style="text-align:right"><span class="badge badge-ok">SYSTEM STABLE</span></div>', unsafe_allow_html=True)
    elif risk_level == "WARNING":
        st.markdown('<div style="text-align:right"><span class="badge badge-warn">⚠️ EARLY DRIFT DETECTED</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:right"><span class="badge badge-crit">🚨 HEMODYNAMIC CRASH</span></div>', unsafe_allow_html=True)

# B. Main Grid
col_left, col_right = st.columns([2.8, 1.2])

# LEFT COLUMN: The "Command Center"
with col_left:
    st.markdown('<div class="viz-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Real-time Hemodynamics & Forecast</div>', unsafe_allow_html=True)
    
    # 1. The Master Plot
    fig_master = utils.plot_command_center(df, curr_time)
    st.plotly_chart(fig_master, use_container_width=True, config={'displayModeBar': False})
    
    # 2. The Fingerprint (Heatmap)
    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True) # Spacer
    fig_heat = utils.plot_causal_strip(df, curr_time)
    st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': False})
    
    st.caption("Graph 1 Analysis: Shaded orange regions indicate statistical deviation (SPC). Purple area is AI Confidence Interval.")
    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT COLUMN: Metrics & State Vector
with col_right:
    # 1. KPI Cards (HTML/CSS for perfect layout)
    st.markdown('<div class="viz-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Key Vitals</div>', unsafe_allow_html=True)
    
    def kpi(label, val, unit, color="#111827"):
        st.markdown(f"""
        <div class="kpi-container">
            <span class="kpi-label">{label}</span>
            <span>
                <span class="kpi-value" style="color:{color}">{val}</span>
                <span class="kpi-unit">{unit}</span>
            </span>
        </div>
        """, unsafe_allow_html=True)

    kpi("Heart Rate", int(current['HR']), "bpm", "#2563eb")
    kpi("Blood Pressure", f"{int(current['SBP'])}/80", "mmHg", "#db2777")
    kpi("Shock Index", f"{current['SI']:.2f}", "idx", "#d97706" if current['SI']>0.8 else "#111827")
    kpi("Perfusion (PI)", f"{current['PI']:.2f}", "%", "#059669")
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. The Compass (PCA)
    st.markdown('<div class="viz-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Stability Compass (PCA)</div>', unsafe_allow_html=True)
    
    fig_pca = utils.plot_pca_compass(coords, loadings)
    st.plotly_chart(fig_pca, use_container_width=True, config={'displayModeBar': False})
    
    # Dynamic Explanation
    driver = loadings['x'].idxmax() if abs(coords[-1,0]) > abs(coords[-1,1]) else loadings['y'].idxmax()
    st.info(f"**Interpretation:** Patient is drifting from center. Primary vector: **{driver}**.")
    st.markdown('</div>', unsafe_allow_html=True)
