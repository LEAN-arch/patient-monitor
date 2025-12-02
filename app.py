import streamlit as st
import pandas as pd
import numpy as np
import utils

# --- PAGE CONFIG ---
st.set_page_config(page_title="Sentient | Clinical Analytics", layout="wide", initial_sidebar_state="collapsed")

# --- CSS: LIGHT MODE / MEDICAL PAPER STYLE ---
st.markdown("""
<style>
    /* Global Background */
    .stApp { background-color: #f4f6f9; color: #2c3e50; }
    
    /* Card Container */
    .css-card {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        margin-bottom: 20px;
    }
    
    /* Metrics */
    .metric-label { font-size: 0.85rem; font-weight: 600; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-val { font-size: 2.2rem; font-weight: 700; color: #2c3e50; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
    .metric-unit { font-size: 0.9rem; color: #adb5bd; font-weight: 500; }
    
    /* Status Badges */
    .status-ok { background-color: #d1e7dd; color: #0f5132; padding: 5px 10px; border-radius: 4px; font-weight: bold; font-size: 0.8rem; }
    .status-warn { background-color: #fff3cd; color: #664d03; padding: 5px 10px; border-radius: 4px; font-weight: bold; font-size: 0.8rem; }
    .status-crit { background-color: #f8d7da; color: #842029; padding: 5px 10px; border-radius: 4px; font-weight: bold; font-size: 0.8rem; }

    /* Headers */
    h1, h2, h3 { color: #2c3e50; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_sim(): return utils.simulate_patient_dynamics()
df = load_sim()

# --- SIDEBAR CONTROL ---
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/50/000000/heart-monitor.png", width=50)
    st.markdown("### Clinical Playback")
    curr_time = st.slider("Timeline (min)", 100, 720, 720)
    st.info("💡 **Demo:** Move slider to min 350 (Stable) and then to 450 (Drift). Observe how the 2σ Warning band is breached long before standard alarms.")

# --- ANALYTICS ---
current = df.iloc[curr_time-1]
spc_data = utils.detect_sensitive_spc(df['HR'].iloc[:curr_time])
last_violations = [v for v in spc_data['violations'] if v[0] > curr_time - 15]

# --- DASHBOARD HEADER ---
c1, c2, c3 = st.columns([3, 1, 1])
with c1:
    st.title("Sentient™ Clinical Analytics")
    st.markdown("**Unit:** ICU-4 | **Bed:** 12 | **Patient:** 49204-B")

with c3:
    if len(last_violations) > 0:
        st.markdown(f'<div class="status-warn" style="text-align:center">⚠ SPC DEVIATION</div>', unsafe_allow_html=True)
    elif current['HR'] > 100:
         st.markdown(f'<div class="status-crit" style="text-align:center">🚨 TACHYCARDIA</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-ok" style="text-align:center">✓ STABLE PROTOCOL</div>', unsafe_allow_html=True)

st.markdown("---")

# --- ROW 1: PRIMARY SPC VISUALIZATION ---
# This is the hero visual: White background, subtle bands, actionable triggers.
col_spc, col_kpi = st.columns([3, 1])

with col_spc:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("##### 📈 Statistical Process Control (HR Trend)")
    fig_spc = utils.plot_sensitive_spc(df, curr_time)
    st.plotly_chart(fig_spc, use_container_width=True, config={'displayModeBar': False})
    
    if len(last_violations) > 0:
        st.warning(f"**Action Required:** {last_violations[-1][2]} detected at T={last_violations[-1][0]}. Inspect fluid balance.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_kpi:
    # Custom HTML Metrics for "Light Mode"
    def kpi_card(label, val, unit, trend="neutral"):
        color = "#2c3e50"
        if trend == "up": color = "#dc3545"
        if trend == "down": color = "#198754"
        
        st.markdown(f"""
        <div class="css-card" style="padding: 15px; margin-bottom: 15px;">
            <div class="metric-label">{label}</div>
            <div class="metric-val" style="color: {color}">{val}</div>
            <div class="metric-unit">{unit}</div>
        </div>
        """, unsafe_allow_html=True)
    
    kpi_card("Heart Rate", f"{int(current['HR'])}", "BPM", "up" if current['HR']>90 else "neutral")
    kpi_card("Systolic BP", f"{int(current['SBP'])}", "mmHg", "down" if current['SBP']<100 else "neutral")
    kpi_card("Perfusion (PI)", f"{current['PI']:.2f}", "%", "down" if current['PI']<2.0 else "neutral")

# --- ROW 2: ADVANCED MATH (State Space & Chaos) ---
c_pca, c_chaos = st.columns(2)

with c_pca:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("##### 💠 Multivariate State Space")
    st.caption("Monitoring global physiological stability vector (PCA)")
    fig_pca = utils.plot_state_space_light(df, curr_time)
    st.plotly_chart(fig_pca, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with c_chaos:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("##### 🌀 Complexity Analysis (Chaos Theory)")
    st.caption("Poincaré Plot: Tight clustering indicates reduced HRV (Stress/Sepsis)")
    c_sub1, c_sub2 = st.columns([2, 1])
    with c_sub1:
        fig_chaos = utils.plot_chaos_light(df, curr_time)
        st.plotly_chart(fig_chaos, use_container_width=True, config={'displayModeBar': False})
    with c_sub2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("**Interpretation:**\nA 'Cloud' shape is healthy.\nA 'Cigar' shape implies autonomic failure.")
    st.markdown('</div>', unsafe_allow_html=True)
