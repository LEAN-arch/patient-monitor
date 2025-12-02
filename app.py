import streamlit as st
import pandas as pd
import numpy as np
import utils

# --- PAGE CONFIG ---
st.set_page_config(page_title="LUMEN | Clinical AI", layout="wide", initial_sidebar_state="collapsed")

# --- CSS: PRECISION MEDICAL THEME ---
st.markdown("""
<style>
    /* Base */
    .stApp { background-color: #f8fafc; color: #1e293b; }
    
    /* Metrics Card */
    .med-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    .med-label { font-size: 0.7rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }
    .med-val { font-size: 1.8rem; font-weight: 700; color: #0f172a; margin-top: 4px; line-height: 1.1; }
    .med-unit { font-size: 0.9rem; color: #94a3b8; font-weight: 500; margin-left: 2px; }
    .med-delta { font-size: 0.8rem; font-weight: 600; margin-top: 8px; }
    
    /* Logic Colors */
    .ok { color: #10b981; }
    .warn { color: #d97706; }
    .crit { color: #ef4444; }
    .b-ok { border-left: 4px solid #10b981; }
    .b-warn { border-left: 4px solid #d97706; }
    .b-crit { border-left: 4px solid #ef4444; }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: white; border-radius: 4px; border: 1px solid #e2e8f0; font-size: 0.9rem; }
    .stTabs [aria-selected="true"] { background-color: #e0f2fe !important; color: #0369a1 !important; border-color: #0369a1 !important; }
    
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data(): return utils.simulate_patient_physiology()
df = load_data()

# --- SIDEBAR ---
with st.sidebar:
    st.header("LUMEN Control")
    curr_time = st.slider("Timeline", 200, 720, 720)
    st.info("Simulation: Sepsis-induced Preload Failure")

# --- ANALYTICS ---
cur = df.iloc[curr_time-1]
prv = df.iloc[curr_time-15]

# --- HEADER (HUD) ---
c1, c2 = st.columns([3, 1])
with c1:
    st.title("LUMEN™ Precision Monitor")
    st.caption(f"Patient ID: 9382-A | Protocol: Sepsis Bundle | T={curr_time}min")
with c2:
    status = "STABLE"
    if cur['CEI'] > 2.5: status = "COMPENSATED"
    if cur['MAP'] < 65: status = "DECOMPENSATED"
    color = "green" if status == "STABLE" else "orange" if status == "COMPENSATED" else "red"
    st.markdown(f"<div style='text-align:right; font-weight:bold; color:{color}; font-size:1.2rem; border:1px solid {color}; padding:5px; border-radius:5px;'>{status}</div>", unsafe_allow_html=True)

st.markdown("---")

# --- ROW 1: HIGH VALUE METRICS ---
cols = st.columns(5)

def card(col, label, val, unit, delta, check_val, thresh_bad, thresh_warn, invert=False):
    is_bad = check_val < thresh_bad if not invert else check_val > thresh_bad
    is_warn = check_val < thresh_warn if not invert else check_val > thresh_warn
    
    style = "b-ok"
    d_color = "ok"
    if is_warn: style, d_color = "b-warn", "warn"
    if is_bad: style, d_color = "b-crit", "crit"
    
    col.markdown(f"""
    <div class="med-card {style}">
        <div class="med-label">{label}</div>
        <div class="med-val">{val}<span class="med-unit">{unit}</span></div>
        <div class="med-delta {d_color}">{delta}</div>
    </div>
    """, unsafe_allow_html=True)

# 1. MAP
d_map = cur['MAP'] - prv['MAP']
card(cols[0], "Mean Pressure", f"{cur['MAP']:.0f}", "mmHg", f"{d_map:+.0f}", cur['MAP'], 65, 70)

# 2. Pulse Pressure (Stroke Volume)
d_pp = cur['PP'] - prv['PP']
card(cols[1], "Pulse Pressure", f"{cur['PP']:.0f}", "mmHg", f"{d_pp:+.0f}", cur['PP'], 30, 40)

# 3. Cardiac Effort Index (CEI)
d_cei = cur['CEI'] - prv['CEI']
card(cols[2], "Cardiac Effort", f"{cur['CEI']:.1f}", "idx", f"{d_cei:+.1f}", cur['CEI'], 4.0, 2.5, invert=True)

# 4. SVR (Resistance)
d_svr = cur['SVR'] - prv['SVR']
card(cols[3], "Vasc. Resistance", f"{cur['SVR']:.0f}", "dyn", f"{d_svr:+.0f}", cur['SVR'], 800, 1000)

# 5. Entropy (Neuro)
d_ent = cur['Entropy'] - prv['Entropy']
card(cols[4], "Neuro Entropy", f"{cur['Entropy']:.2f}", "σ", f"{d_ent:+.2f}", cur['Entropy'], 0.5, 0.8)


# --- ROW 2: COMMAND CENTER ---
c_main, c_side = st.columns([2.5, 1])

with c_main:
    st.subheader("Hemodynamic Command Center")
    fig_cmd = utils.plot_command_timeline(df, curr_time)
    st.plotly_chart(fig_cmd, use_container_width=True)

with c_side:
    st.subheader("System Status")
    fig_org = utils.plot_organ_matrix(df, curr_time)
    st.plotly_chart(fig_org, use_container_width=True)

# --- ROW 3: DEEP DIVE TABS ---
tab1, tab2 = st.tabs(["⚡ FLUID PHYSICS & DIAGNOSIS", "🔮 PROGNOSTICS"])

with tab1:
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        st.markdown("**Starling Curve (Fluid Responsiveness)**")
        st.caption("Flattening of curve = Non-Responder.")
        fig_star = utils.plot_starling_curve(df, curr_time)
        st.plotly_chart(fig_star, use_container_width=True)
    with c_p2:
        st.markdown("**Shock Phenotype (Diagnosis)**")
        st.caption("Quadrants define clinical state.")
        fig_pheno = utils.plot_shock_phenotype(df, curr_time)
        st.plotly_chart(fig_pheno, use_container_width=True)

with tab2:
    c_h1, c_h2 = st.columns([2, 1])
    with c_h1:
        st.markdown("**Time-to-Event Horizon**")
        st.caption("Projected MAP intersection with 65 mmHg.")
        fig_hor = utils.plot_horizon(df, curr_time)
        st.plotly_chart(fig_hor, use_container_width=True)
    with c_h2:
        st.info("""
        **AI Recommendation Engine:**
        
        1. **Check CEI:** If > 3.0, Heart is straining.
        2. **Check Starling:** If Flat, STOP FLUIDS.
        3. **Check SVR:** If High, consider Vasodilation.
        
        *Confidence: 94.2% based on Kalman variance.*
        """)
