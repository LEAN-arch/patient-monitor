import streamlit as st
import pandas as pd
import numpy as np
import utils

# --- PAGE SETUP ---
st.set_page_config(page_title="BioSentry | Advanced Analytics", layout="wide")

# --- CSS STYLING (Medical Report Style) ---
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; color: #212529; }
    .card { background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 20px; border: 1px solid #e9ecef; }
    .metric-box { text-align: center; border-right: 1px solid #eee; }
    .big-num { font-size: 24px; font-weight: bold; color: #2c3e50; }
    .label { font-size: 12px; text-transform: uppercase; color: #6c757d; }
    .alert-high { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; font-weight: bold; }
    .alert-med { background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; font-weight: bold; }
    .alert-low { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load(): return utils.simulate_patient_dynamics()
df = load()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Playback")
    curr_time = st.slider("Time (min)", 150, 720, 720)
    st.info("**Scenario:**\nPatient stable until T=300.\nAt T=300, 'Silent Drift' begins.\nWatch how SPC and Heatmap detect this before standard alarms.")

# --- CALCULATIONS ---
current = df.iloc[curr_time-1]
spc_data = utils.detect_western_electric_spc(df['HR'].iloc[:curr_time])
risk = utils.calculate_risk_score(current, spc_data['violations'])
coords, loadings, variance = utils.run_pca_analysis(df, curr_time)

# --- HEADER: HUD ---
st.markdown(f"### 🏥 BioSentry: Clinical Surveillance (T={curr_time})")

# Top Alert Banner
if risk > 60:
    st.markdown(f'<div class="alert-high">🚨 CRITICAL RISK (Score: {risk}) - Decompensation Imminent. Check Fluid Status.</div>', unsafe_allow_html=True)
elif risk > 30:
    st.markdown(f'<div class="alert-med">⚠️ WARNING (Score: {risk}) - Physiological Drift Detected. Monitor Closely.</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="alert-low">✅ STABLE (Score: {risk}) - Within Baseline Limits.</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- SECTION 1: TIMELINE & TRENDS (The "When") ---
c1, c2 = st.columns([3, 1])

with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 1. High-Sensitivity SPC (Western Electric Rules)")
    fig_spc = utils.plot_sensitive_spc_with_forecast(df, curr_time)
    st.plotly_chart(fig_spc, use_container_width=True)
    
    with st.expander("ℹ️ Clinical Significance & Actionability"):
        st.markdown("""
        **What is this?** A Statistical Process Control (SPC) chart that visualizes stability zones ($1\sigma, 2\sigma, 3\sigma$) rather than just raw limits.
        
        **Why it matters:** Standard monitors only alarm at fixed thresholds (e.g., HR > 100). This chart detects **"Drift"** (points clustering in the yellow/warning zone) 30-60 minutes *before* the threshold is breached.
        
        **Action:** If you see markers in the **Yellow Zone**, assess the patient immediately. Do not wait for the alarm.
        """)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Key Metrics")
    
    def metric(label, val, color="black"):
        st.markdown(f"<div class='label'>{label}</div><div class='big-num' style='color:{color}'>{val}</div><hr style='margin:5px 0'>", unsafe_allow_html=True)
    
    metric("Heart Rate", int(current['HR']), "#0056b3")
    metric("Systolic BP", int(current['SBP']), "#d63384")
    metric("Shock Index", f"{current['SI']:.2f}", "red" if current['SI']>0.9 else "black")
    metric("Perfusion (PI)", f"{current['PI']:.2f}", "orange" if current['PI']<1.0 else "green")
    st.markdown('</div>', unsafe_allow_html=True)

# --- SECTION 2: ROOT CAUSE ANALYSIS (The "What") ---
c3, c4 = st.columns(2)

with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 2. Deviation Heatmap")
    fig_heat = utils.plot_heatmap_restored(df, curr_time)
    st.plotly_chart(fig_heat, use_container_width=True)
    
    with st.expander("ℹ️ Meaning & Interpretation"):
        st.markdown("""
        **Meaning:** This heatmap uses Z-scores to show how far each vital is from the patient's *own* baseline. Red = Higher than baseline, Blue = Lower.
        
        **Clinical Value:** It reveals the **Cascade of Instability**.
        1.  First, you might see **Blue in PI** (Vasoconstriction).
        2.  Then **Red in HR** (Compensation).
        3.  Finally **Blue in SBP** (Decompensation).
        """)
    st.markdown('</div>', unsafe_allow_html=True)

with c4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 3. Physiological Footprint")
    fig_radar = utils.plot_radar_snapshot(current)
    st.plotly_chart(fig_radar, use_container_width=True)
    
    with st.expander("ℹ️ Diagnosis Aid"):
        st.markdown("""
        **Actionability:** Quick pattern recognition.
        *   **Hypovolemic Pattern:** Spikes in Tachycardia + Hypoperfusion.
        *   **Respiratory Pattern:** Spikes in Tachypnea + Desaturation.
        *   **Septic Pattern:** Hypoperfusion + Hypotension often precede Tachycardia.
        """)
    st.markdown('</div>', unsafe_allow_html=True)

# --- SECTION 3: ADVANCED DYNAMICS (The "Why") ---
c5, c6 = st.columns([2, 1])

with c5:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 4. PCA State Space Trajectory")
    fig_pca = utils.plot_pca_explained(coords, loadings)
    st.plotly_chart(fig_pca, use_container_width=True)
    
    with st.expander("ℹ️ DEEP DIVE: Understanding PCA"):
        st.markdown(f"""
        **What is PCA?** Principal Component Analysis takes all 5 vitals and compresses them into a "Global Stability Map."
        
        **Interpretation:**
        *   **The Green Circle:** This is Homeostasis (Safe Zone).
        *   **The Dot:** The patient's current state.
        *   **The Arrows (Vectors):** These tell you *why* the dot is moving.
        
        **Scenario Analysis:** 
        Notice how the **HR Vector** points Right and **SBP Vector** points Left? 
        If the dot moves **Right**, it means HR is driving the instability (Compensation). 
        If it moves **Left/Down**, SBP is driving it (Crash).
        
        **Current Status:** PC1 explains {variance[0]*100:.1f}% of the variance. The patient is drifting along the {loadings['PC1'].idxmax()} axis.
        """)
    st.markdown('</div>', unsafe_allow_html=True)

with c6:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 5. Chaos Theory (HRV)")
    fig_chaos = utils.plot_chaos_attractor(df, curr_time)
    st.plotly_chart(fig_chaos, use_container_width=True)
    
    with st.expander("ℹ️ Autonomic Health"):
        st.markdown("""
        **Lag Plot:** Plots $HR(t)$ vs $HR(t+1)$.
        
        **Healthy:** A fuzzy "Cloud" or ball. This indicates a responsive Autonomic Nervous System.
        
        **Unhealthy:** A tight diagonal line or "Cigar." This indicates **Loss of Complexity**, often a precursor to sepsis or sudden cardiac arrest.
        """)
    st.markdown('</div>', unsafe_allow_html=True)
