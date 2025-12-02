import streamlit as st
import pandas as pd
import numpy as np
import utils

# --- PAGE CONFIG ---
st.set_page_config(page_title="NEXUS | ICU", layout="wide", initial_sidebar_state="collapsed")

# --- FUTURISTIC CSS ---
st.markdown("""
<style>
    /* Dark Matter Theme */
    .stApp { background-color: #050505; color: #e2e8f0; }
    
    /* Neon Cards */
    .nexus-card {
        background: rgba(20, 20, 30, 0.6);
        border: 1px solid rgba(100, 100, 100, 0.2);
        border-radius: 8px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .nexus-card:hover {
        border-color: rgba(0, 242, 234, 0.5);
        box-shadow: 0 0 15px rgba(0, 242, 234, 0.2);
    }
    
    /* Typography */
    .label-xs { font-size: 0.7rem; color: #94a3b8; letter-spacing: 1.5px; text-transform: uppercase; }
    .val-xl { font-size: 2.5rem; font-weight: 200; font-family: 'Courier New', monospace; color: white; text-shadow: 0 0 10px rgba(255,255,255,0.3); }
    .unit { font-size: 0.9rem; color: #64748b; }
    
    /* Status Dots */
    .dot { height: 10px; width: 10px; border-radius: 50%; display: inline-block; margin-right: 5px; }
    .dot-ok { background-color: #ccff00; box-shadow: 0 0 8px #ccff00; }
    .dot-warn { background-color: #ffae00; box-shadow: 0 0 8px #ffae00; }
    .dot-crit { background-color: #ff0055; box-shadow: 0 0 8px #ff0055; }
    
    /* Tabs styling override */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] { background-color: rgba(255,255,255,0.05); border-radius: 4px; border: none; color: white; }
    .stTabs [aria-selected="true"] { background-color: rgba(0, 242, 234, 0.1) !important; color: #00f2ea !important; border: 1px solid #00f2ea !important; }

</style>
""", unsafe_allow_html=True)

# --- LOAD ENGINE ---
@st.cache_data
def load_nexus(): return utils.simulate_comprehensive_shock()
df = load_nexus()

# --- SIDEBAR CONTROL ---
with st.sidebar:
    st.title("NEXUS CONTROL")
    curr_time = st.slider("Timeline Sync", 200, 720, 720)
    st.markdown("---")
    st.info("System: Hypovolemic -> Cardiogenic Cascade")

# --- CALC ---
current = df.iloc[curr_time-1]
prev = df.iloc[curr_time-15]

# --- TOP BAR (HUD) ---
c1, c2, c3, c4, c5 = st.columns(5)

def hud_metric(col, label, val, unit, delta, color_class="dot-ok"):
    col.markdown(f"""
    <div class="nexus-card" style="padding: 15px; text-align: center;">
        <div class="label-xs"><span class="dot {color_class}"></span>{label}</div>
        <div class="val-xl">{val}</div>
        <div class="unit">{unit}</div>
        <div style="font-size: 0.8rem; color: #64748b; margin-top:5px;">{delta}</div>
    </div>
    """, unsafe_allow_html=True)

# 1. MAP
d_map = current['MAP'] - prev['MAP']
status_map = "dot-crit" if current['MAP'] < 65 else "dot-ok"
hud_metric(c1, "Mean Pressure", f"{int(current['MAP'])}", "mmHg", f"{d_map:+.0f}", status_map)

# 2. Cardiac Output
d_co = current['CO'] - prev['CO']
status_co = "dot-crit" if current['CO'] < 4.0 else "dot-ok"
hud_metric(c2, "Cardiac Output", f"{current['CO']:.1f}", "L/min", f"{d_co:+.1f}", status_co)

# 3. SVR
d_svr = current['SVR'] - prev['SVR']
hud_metric(c3, "Vasc. Resistance", f"{int(current['SVR'])}", "dyn", f"{d_svr:+.0f}")

# 4. Stroke Volume (PP)
d_pp = current['PP'] - prev['PP']
status_pp = "dot-crit" if current['PP'] < 30 else "dot-warn"
hud_metric(c4, "Stroke Volume Proxy", f"{int(current['PP'])}", "mL/beat", f"{d_pp:+.0f}", status_pp)

# 5. Instability (SI)
si = current['SI']
status_si = "dot-crit" if si > 1.0 else "dot-ok"
hud_metric(c5, "Shock Index", f"{si:.2f}", "", "Stable" if si < 0.8 else "Unstable", status_si)


# --- MAIN INTERFACE (TABS) ---
tab_hemo, tab_phys, tab_organ = st.tabs(["1. HEMODYNAMICS", "2. FLUID PHYSICS", "3. ORGAN SYSTEMS"])

# --- TAB 1: HEMODYNAMICS ---
with tab_hemo:
    col_main, col_detail = st.columns([3, 1])
    with col_main:
        st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
        st.markdown('<div class="label-xs">REAL-TIME PERFUSION TRACKING</div>', unsafe_allow_html=True)
        fig_timeline = utils.plot_neon_timeline(df, curr_time)
        st.plotly_chart(fig_timeline, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_detail:
        st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
        st.markdown('<div class="label-xs">AI DIAGNOSIS</div><br>', unsafe_allow_html=True)
        if current['CO'] < 4.0 and current['SVR'] > 20:
            st.markdown("### 🥶 COLD SHOCK")
            st.markdown("Low Flow, High Resistance.")
            st.error("Protocol: INOTROPES")
        elif current['MAP'] < 65:
            st.markdown("### 🩸 HYPOTENSION")
            st.warning("Protocol: FLUIDS")
        else:
            st.markdown("### ✅ STABLE")
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: FLUID PHYSICS ---
with tab_phys:
    c_starling, c_coupling = st.columns(2)
    
    with c_starling:
        st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
        st.markdown('<div class="label-xs">STARLING CURVE (PRELOAD RESPONSIVENESS)</div>', unsafe_allow_html=True)
        # The chart
        fig_starling = utils.plot_starling_curve(df, curr_time)
        st.plotly_chart(fig_starling, use_container_width=True)
        
        # Actionable Insight
        slope = (current['PP'] - prev['PP']) / (current['DBP'] - prev['DBP'] + 0.01)
        if slope > 0.5:
            st.success("🌊 RESPONDER: Fluids will increase Stroke Volume.")
        else:
            st.error("🛑 NON-RESPONDER: Fluids will cause congestion.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c_coupling:
        st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
        st.markdown('<div class="label-xs">V-A COUPLING (PUMP vs PIPES)</div>', unsafe_allow_html=True)
        fig_svr = utils.plot_svr_co_coupling(df, curr_time)
        st.plotly_chart(fig_svr, use_container_width=True)
        st.info("Top Left = Vasoconstricted. Bottom Right = Vasodilated.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: ORGAN SYSTEMS ---
with tab_organ:
    st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
    st.markdown('<div class="label-xs">SYSTEMIC IMPACT MATRIX</div>', unsafe_allow_html=True)
    fig_matrix = utils.plot_organ_matrix(df, curr_time)
    st.plotly_chart(fig_matrix, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
