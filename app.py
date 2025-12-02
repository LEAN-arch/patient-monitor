import streamlit as st
import pandas as pd
import numpy as np
import utils

# --- PAGE CONFIG ---
st.set_page_config(page_title="PANOPTICON | ICU", layout="wide", initial_sidebar_state="collapsed")

# --- CSS: DENSE GRID LAYOUT ---
st.markdown("""
<style>
    .stApp { background-color: #f1f5f9; }
    
    /* Card Container */
    .pan-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        height: 100%;
    }
    
    /* Headers */
    .pan-header { font-size: 0.8rem; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 8px; border-bottom: 1px solid #f1f5f9; padding-bottom: 4px; }
    
    /* Sparkline Row */
    .spark-box { text-align: center; border-right: 1px solid #f1f5f9; padding: 0 10px; }
    .spark-val { font-size: 1.6rem; font-weight: 800; color: #0f172a; line-height: 1; }
    .spark-label { font-size: 0.7rem; color: #94a3b8; font-weight: 600; }
    
    /* Status Badges */
    .badge { padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
    .b-crit { background: #fee2e2; color: #991b1b; }
    .b-warn { background: #fef3c7; color: #92400e; }
    .b-ok { background: #dcfce7; color: #166534; }
    
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data(): return utils.simulate_panopticon_data()
df = load_data()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Timeline")
    curr_time = st.slider("Min", 200, 720, 720)
    st.info("Scenario: Distributive Shock (Sepsis)")

# --- CALC ---
cur = df.iloc[curr_time-1]
prv = df.iloc[curr_time-15]

# --- 1. HEADER (HUD) ---
c1, c2, c3 = st.columns([2, 5, 1])
with c1:
    st.markdown("### 👁️ PANOPTICON")
    st.caption("Advanced Physiological Telemetry")
with c3:
    risk = int(cur['HR']/cur['MAP'] * 100)
    st.markdown(f"<div style='text-align:right; font-weight:bold; font-size:1.5rem; color:{'red' if risk>100 else 'green'}'>RISK: {risk}</div>", unsafe_allow_html=True)

# --- 2. ROW 1: SPARKLINE KPI STRIP (All key vitals) ---
# Create a container for the sparkline strip
st.markdown('<div class="pan-card" style="margin-bottom:15px; padding: 5px;">', unsafe_allow_html=True)
sp_cols = st.columns(6)

def spark_kpi(col, label, val, unit, color_hex, df_col):
    with col:
        st.markdown(f"<div class='spark-box'><div class='spark-label'>{label}</div><div class='spark-val' style='color:{color_hex}'>{val}</div><div class='spark-label'>{unit}</div></div>", unsafe_allow_html=True)
        fig = utils.plot_sparkline(df, df_col, color_hex)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

spark_kpi(sp_cols[0], "HEART RATE", int(cur['HR']), "bpm", "#0369a1", "HR")
spark_kpi(sp_cols[1], "MAP", int(cur['MAP']), "mmHg", "#be185d", "MAP")
spark_kpi(sp_cols[2], "CARDIAC OUTPUT", f"{cur['CO']:.1f}", "L/min", "#059669", "CO")
spark_kpi(sp_cols[3], "SVR", int(cur['SVR']), "dyn", "#d97706", "SVR")
spark_kpi(sp_cols[4], "SpO2", int(cur['SpO2']), "%", "#7c3aed", "SpO2")
spark_kpi(sp_cols[5], "LACTATE", f"{cur['Lactate']:.1f}", "mmol/L", "#dc2626", "Lactate")
st.markdown('</div>', unsafe_allow_html=True)

# --- 3. ROW 2: PRIMARY MONITORS & DIAGNOSTICS ---
r2_1, r2_2, r2_3 = st.columns([2, 1, 1])

with r2_1:
    st.markdown('<div class="pan-card">', unsafe_allow_html=True)
    st.markdown('<div class="pan-header">1. HEMODYNAMIC COMMAND (HR vs MAP)</div>', unsafe_allow_html=True)
    fig_cmd = utils.plot_command_center(df, curr_time)
    st.plotly_chart(fig_cmd, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with r2_2:
    st.markdown('<div class="pan-card">', unsafe_allow_html=True)
    st.markdown('<div class="pan-header">2. OXYGEN SUPPLY/DEMAND</div>', unsafe_allow_html=True)
    fig_do2 = utils.plot_oxygen_delivery(df, curr_time)
    st.plotly_chart(fig_do2, use_container_width=True)
    st.caption("Gap = Oxygen Debt (Lactate)")
    st.markdown('</div>', unsafe_allow_html=True)

with r2_3:
    st.markdown('<div class="pan-card">', unsafe_allow_html=True)
    st.markdown('<div class="pan-header">3. RENAL PERFUSION</div>', unsafe_allow_html=True)
    fig_ren = utils.plot_renal_curve(df, curr_time)
    st.plotly_chart(fig_ren, use_container_width=True)
    st.caption("Left of Red Line = Acute Kidney Injury")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 4. ROW 3: MECHANICS & PHYSICS ---
r3_1, r3_2, r3_3, r3_4 = st.columns(4)

with r3_1:
    st.markdown('<div class="pan-card">', unsafe_allow_html=True)
    st.markdown('<div class="pan-header">4. STARLING (PRELOAD)</div>', unsafe_allow_html=True)
    st.plotly_chart(utils.plot_starling(df, curr_time), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with r3_2:
    st.markdown('<div class="pan-card">', unsafe_allow_html=True)
    st.markdown('<div class="pan-header">5. V-A COUPLING</div>', unsafe_allow_html=True)
    st.plotly_chart(utils.plot_svr_co(df, curr_time), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with r3_3:
    st.markdown('<div class="pan-card">', unsafe_allow_html=True)
    st.markdown('<div class="pan-header">6. WORK LOOP (PV)</div>', unsafe_allow_html=True)
    st.plotly_chart(utils.plot_pv_proxy(df, curr_time), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with r3_4:
    st.markdown('<div class="pan-card">', unsafe_allow_html=True)
    st.markdown('<div class="pan-header">7. HEMO-COHERENCE</div>', unsafe_allow_html=True)
    st.plotly_chart(utils.plot_coherence(df, curr_time), use_container_width=True)
    st.caption("Red = Uncoupling (Instability)")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 5. ROW 4: ADVANCED COMPUTATIONAL ---
r4_1, r4_2 = st.columns([1, 3])

with r4_1:
    st.markdown('<div class="pan-card">', unsafe_allow_html=True)
    st.markdown('<div class="pan-header">8. AUTONOMIC SPECTRUM</div>', unsafe_allow_html=True)
    st.plotly_chart(utils.plot_spectral_density(df, curr_time), use_container_width=True)
    st.caption("Low Power = Autonomic Failure")
    st.markdown('</div>', unsafe_allow_html=True)

with r4_2:
    st.markdown('<div class="pan-card">', unsafe_allow_html=True)
    st.markdown('<div class="pan-header">AI SUMMARY & PROTOCOL</div>', unsafe_allow_html=True)
    
    # Logic Engine
    dx = "STABLE"
    plan = "Monitor"
    if cur['Lactate'] > 4.0:
        dx = "SEVERE SEPTIC SHOCK"
        plan = "1. FLUIDS (30mL/kg) 2. NOREPINEPHRINE 3. ANTIBIOTICS"
    elif cur['SVR'] < 800:
        dx = "DISTRIBUTIVE SHOCK (Early)"
        plan = "PRESSORS (Target MAP > 65)"
        
    st.markdown(f"#### DIAGNOSIS: <span class='badge b-crit'>{dx}</span>", unsafe_allow_html=True)
    st.markdown(f"**PROTOCOL:** {plan}")
    st.markdown(f"""
    - **Renal Status:** {'Oliguric (<0.5 mL/kg/hr)' if cur['UrineOutput'] < 0.5 else 'Adequate'}
    - **O2 Supply/Demand:** {'Mismatch (High Lactate)' if cur['Lactate'] > 2 else 'Balanced'}
    - **Coupling:** {'Vasodilation Dominant' if cur['SVR'] < 800 else 'Normal'}
    """)
    st.markdown('</div>', unsafe_allow_html=True)
