import streamlit as st
import pandas as pd
import numpy as np
import utils

# --- CONFIG ---
st.set_page_config(page_title="APEX | Hemodynamic AI", layout="wide", initial_sidebar_state="collapsed")

# --- CSS: PRO MEDICAL UI ---
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; color: #0f172a; }
    
    /* KPI Card */
    .kpi-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        display: flex; flex-direction: column; justify-content: space-between;
        height: 100px;
    }
    .kpi-title { font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; }
    .kpi-val { font-size: 1.8rem; font-weight: 800; color: #0f172a; line-height: 1.1; }
    .kpi-unit { font-size: 0.8rem; color: #94a3b8; font-weight: 500; }
    
    /* Alerts */
    .alert-box { padding: 15px; border-radius: 6px; margin-bottom: 20px; font-weight: 600; display: flex; justify-content: space-between; }
    .alert-crit { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    .alert-warn { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .alert-safe { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    
</style>
""", unsafe_allow_html=True)

# --- LOAD ENGINE ---
@st.cache_data
def load(): return utils.simulate_coupled_physiology()
df = load()

# --- CONTROLS ---
with st.sidebar:
    st.header("Timeline Sync")
    curr_time = st.slider("Session Minute", 100, 720, 720)
    st.markdown("### 🧬 Physiology Engine")
    st.info("Simulation: Coupled Baroreflex Feedback Loop responding to Progressive Septic Vasoplegia.")

# --- ANALYTICS ---
cur = df.iloc[curr_time-1]
prv = df.iloc[curr_time-15]

# --- 1. INTELLIGENT TRIAGE HEADER ---
# Logic: MAP < 65 is Critical. MAP > 65 but High Lactate is Occult Shock.
status = "STABLE"
action = "Routine Monitoring"
style = "alert-safe"

if cur['Lactate'] > 2.0:
    status = "OCCULT SHOCK (Metabolic Mismatch)"
    action = "Evaluate DO2 (Fluids/Blood)"
    style = "alert-warn"

if cur['MAP'] < 65:
    status = "CIRCULATORY FAILURE"
    if cur['SVR'] < 800:
        action = "PROTOCOL: NOREPINEPHRINE (Target MAP > 65)"
    elif cur['CO'] < 4.0:
        action = "PROTOCOL: INOTROPE / VOLUME"
    style = "alert-crit"

st.markdown(f"""
<div class="alert-box {style}">
    <div><span>STATUS:</span> {status}</div>
    <div><span>RECOMMENDATION:</span> {action}</div>
</div>
""", unsafe_allow_html=True)

# --- 2. KPI GRID WITH SPARKLINES ---
# Using Plotly sparklines inside the grid
cols = st.columns(6)

def kpi(col, label, val, unit, color, col_name, thresh_bad, invert=False):
    is_bad = val < thresh_bad if not invert else val > thresh_bad
    txt_color = "#ef4444" if is_bad else "#0f172a"
    
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid {color}">
            <div class="kpi-title">{label}</div>
            <div class="kpi-val" style="color:{txt_color}">{val} <span class="kpi-unit">{unit}</span></div>
        </div>
        """, unsafe_allow_html=True)
        # Sparkline below
        st.plotly_chart(utils.plot_kpi_spark(df, col_name, color), use_container_width=True, config={'displayModeBar': False})

kpi(cols[0], "MAP", f"{cur['MAP']:.0f}", "mmHg", "#be185d", "MAP", 65)
kpi(cols[1], "Cardiac Index", f"{cur['CI']:.1f}", "L/min", "#0ea5e9", "CI", 2.2)
kpi(cols[2], "SVR Index", f"{cur['SVRI']:.0f}", "dyn", "#d97706", "SVRI", 800)
kpi(cols[3], "Heart Rate", f"{cur['HR']:.0f}", "bpm", "#0369a1", "HR", 100, invert=True)
kpi(cols[4], "Lactate", f"{cur['Lactate']:.1f}", "mmol", "#8b5cf6", "Lactate", 2.0, invert=True)
kpi(cols[5], "Ea_dyn", f"{cur['Eadyn']:.2f}", "idx", "#10b981", "Eadyn", 1.2, invert=True)

# --- 3. DIAGNOSTIC PANEL ---
c_left, c_right = st.columns([1, 1])

with c_left:
    st.markdown("**1. SHOCK PHENOTYPE (GDT Compass)**")
    st.caption("Visualizes the hemodynamic diagnosis. Target is the Green Zone.")
    fig_bull = utils.plot_gdt_bullseye(df, curr_time)
    st.plotly_chart(fig_bull, use_container_width=True)

with c_right:
    st.markdown("**2. V-A COUPLING (Efficiency)**")
    st.caption("Relationship between Cardiac Work (Pump) and Afterload (Pipes).")
    fig_coup = utils.plot_coupling_curve(df, curr_time)
    st.plotly_chart(fig_coup, use_container_width=True)

# --- 4. ORGAN PROTECTION ---
st.markdown("**3. ORGAN PERFUSION & RISK**")
fig_renal = utils.plot_renal_cliff(df, curr_time)
st.plotly_chart(fig_renal, use_container_width=True)

# --- 5. SME FOOTNOTE ---
with st.expander("ℹ️ CLINICAL LOGIC EXPLAINED"):
    st.markdown("""
    **Physiology Engine:** This simulation uses a coupled Baroreflex Feedback Loop.
    *   **Phase 1 (Compensated):** SVR drops due to sepsis. Baroreflex increases Sympathetic Tone -> HR & Contractility Rise. MAP is maintained.
    *   **Phase 2 (Decompensated):** Vasoplegia exceeds cardiac reserve. MAP crashes. Lactate rises due to supply/demand mismatch ($DO_2 < 400$).
    
    **Decision Logic:**
    *   **Vasoplegia:** Low SVR + High CI = Needs Vasopressors (Norepinephrine).
    *   **Hypovolemia:** Low CI + High SVR = Needs Volume/Blood.
    *   **Cardiogenic:** Low CI + High SVR + High Filling Pressures = Needs Inotropes (Dobutamine).
    """)
