import streamlit as st
import pandas as pd
import numpy as np
import utils

# --- PAGE CONFIG ---
st.set_page_config(page_title="PRECISION | GDT", layout="wide", initial_sidebar_state="collapsed")

# --- CSS: CLINICAL DASHBOARD ---
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    
    /* Tiles */
    .metric-tile {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        text-align: center;
    }
    .tile-label { font-size: 0.7rem; font-weight: 700; color: #64748b; text-transform: uppercase; }
    .tile-val { font-size: 1.8rem; font-weight: 800; color: #0f172a; line-height: 1.1; }
    .tile-unit { font-size: 0.8rem; color: #94a3b8; }
    .tile-context { font-size: 0.75rem; font-weight: 600; margin-top: 4px; }
    
    /* Context Colors */
    .c-crit { color: #ef4444; }
    .c-warn { color: #d97706; }
    .c-ok { color: #10b981; }
    
    /* Header */
    .protocol-header {
        background: #1e293b;
        color: white;
        padding: 10px 20px;
        border-radius: 6px;
        margin-bottom: 15px;
        display: flex;
        justify_content: space-between;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD ---
@st.cache_data
def load(): return utils.simulate_gdt_data()
df = load()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Simulation Control")
    curr_time = st.slider("Time (min)", 200, 720, 720)
    st.info("Scenario: Warm Sepsis -> Cold Shock")

# --- CALC ---
cur = df.iloc[curr_time-1]
prv = df.iloc[curr_time-15]

# --- 1. PROTOCOL HEADER ---
# Heuristic Logic
status = "STABLE"
action = "MONITOR"
bg_col = "#10b981" # Green

if cur['Lactate'] > 2.0:
    status = "TISSUE HYPOXIA"
    action = "CHECK DO2 TARGETS"
    bg_col = "#d97706" # Orange

if cur['MAP'] < 65:
    status = "CIRCULATORY FAILURE"
    if cur['CI'] > 2.5:
        action = "START VASOPRESSORS (Septic)"
    else:
        action = "FLUIDS / INOTROPES"
    bg_col = "#ef4444" # Red

st.markdown(f"""
<div class="protocol-header" style="background:{bg_col}">
    <div><strong>STATUS:</strong> {status}</div>
    <div><strong>ACTION:</strong> {action}</div>
</div>
""", unsafe_allow_html=True)

# --- 2. RESUSCITATION TARGETS (KPIs) ---
cols = st.columns(6)

def tile(col, label, val, unit, delta, target_val, invert=False, target_range=None, fmt="{:.0f}"):
    """
    Displays a metric tile.
    val: Raw numeric value (for logic comparison).
    fmt: Format string for display (e.g. "{:.1f}").
    """
    # Logic Check (Using raw number)
    is_bad = val < target_range[0] if target_range else (val > target_val if invert else val < target_val)
    color = "c-crit" if is_bad else "c-ok"
    
    # Format for display
    display_val = fmt.format(val)
    
    col.markdown(f"""
    <div class="metric-tile">
        <div class="tile-label">{label}</div>
        <div class="tile-val">{display_val}</div>
        <div class="tile-unit">{unit}</div>
        <div class="tile-context {color}">{delta}</div>
    </div>
    """, unsafe_allow_html=True)

# 1. MAP
d_map = cur['MAP'] - prv['MAP']
tile(cols[0], "MAP (Pressure)", cur['MAP'], "mmHg", f"{d_map:+.0f}", 65, fmt="{:.0f}")

# 2. CI
d_ci = cur['CI'] - prv['CI']
tile(cols[1], "Cardiac Index", cur['CI'], "L/min/m2", f"{d_ci:+.1f}", 2.5, fmt="{:.1f}")

# 3. SVRI
d_svr = cur['SVRI'] - prv['SVRI']
tile(cols[2], "SVRI (Resist)", cur['SVRI'], "dyn", f"{d_svr:+.0f}", 800, fmt="{:.0f}")

# 4. DO2I
d_do2 = cur['DO2I'] - prv['DO2I']
tile(cols[3], "DO2I (Delivery)", cur['DO2I'], "mL/m2", f"{d_do2:+.0f}", 400, fmt="{:.0f}")

# 5. Lactate
d_lac = cur['Lactate'] - prv['Lactate']
tile(cols[4], "Lactate", cur['Lactate'], "mmol/L", f"{d_lac:+.1f}", 2.0, invert=True, fmt="{:.1f}")

# 6. Urine
tile(cols[5], "Urine Output", cur['Urine'], "mL/kg/hr", "Oliguric" if cur['Urine']<0.5 else "Adequate", 0.5, fmt="{:.1f}")

# --- 3. ROW 1: MACRO-HEMODYNAMICS (The "Bullseye") ---
c_left, c_right = st.columns([1, 1])

with c_left:
    st.markdown("**1. HEMODYNAMIC TARGETING**")
    st.caption("Identify Shock Type: High Flow/Low Pressure vs Low Flow/Low Pressure.")
    fig_bull = utils.plot_hemodynamic_bullseye(df, curr_time)
    st.plotly_chart(fig_bull, use_container_width=True)

with c_right:
    st.markdown("**2. OXYGEN SUPPLY VS DEMAND**")
    st.caption("The root cause of shock. Ensure Delivery (Green) > Debt (Red).")
    fig_ox = utils.plot_oxygen_ledger(df, curr_time)
    st.plotly_chart(fig_ox, use_container_width=True)

# --- 4. ROW 2: ORGAN PROTECTION & MECHANICS ---
c3, c4, c5 = st.columns(3)

with c3:
    st.markdown("**3. RENAL TRAJECTORY**")
    st.caption("Preventing AKI.")
    fig_ren = utils.plot_renal_trajectory(df, curr_time)
    st.plotly_chart(fig_ren, use_container_width=True)

with c4:
    st.markdown("**4. FLUID RESPONSIVENESS**")
    st.caption("Frank-Starling Vector.")
    fig_star = utils.plot_frank_starling_vector(df, curr_time)
    st.plotly_chart(fig_star, use_container_width=True)
import streamlit as st
import pandas as pd
import numpy as np
import utils

# --- PAGE CONFIG ---
st.set_page_config(page_title="PRECISION | GDT", layout="wide", initial_sidebar_state="collapsed")

# --- CSS: CLINICAL DASHBOARD ---
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    
    /* Tiles */
    .metric-tile {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        text-align: center;
    }
    .tile-label { font-size: 0.7rem; font-weight: 700; color: #64748b; text-transform: uppercase; }
    .tile-val { font-size: 1.8rem; font-weight: 800; color: #0f172a; line-height: 1.1; }
    .tile-unit { font-size: 0.8rem; color: #94a3b8; }
    .tile-context { font-size: 0.75rem; font-weight: 600; margin-top: 4px; }
    
    /* Context Colors */
    .c-crit { color: #ef4444; }
    .c-warn { color: #d97706; }
    .c-ok { color: #10b981; }
    
    /* Header */
    .protocol-header {
        background: #1e293b;
        color: white;
        padding: 10px 20px;
        border-radius: 6px;
        margin-bottom: 15px;
        display: flex;
        justify_content: space-between;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD ---
@st.cache_data
def load(): return utils.simulate_gdt_data()
df = load()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Simulation Control")
    curr_time = st.slider("Time (min)", 200, 720, 720)
    st.info("Scenario: Warm Sepsis -> Cold Shock")

# --- CALC ---
cur = df.iloc[curr_time-1]
prv = df.iloc[curr_time-15]

# --- 1. PROTOCOL HEADER ---
# Heuristic Logic
status = "STABLE"
action = "MONITOR"
bg_col = "#10b981" # Green

if cur['Lactate'] > 2.0:
    status = "TISSUE HYPOXIA"
    action = "CHECK DO2 TARGETS"
    bg_col = "#d97706" # Orange

if cur['MAP'] < 65:
    status = "CIRCULATORY FAILURE"
    if cur['CI'] > 2.5:
        action = "START VASOPRESSORS (Septic)"
    else:
        action = "FLUIDS / INOTROPES"
    bg_col = "#ef4444" # Red

st.markdown(f"""
<div class="protocol-header" style="background:{bg_col}">
    <div><strong>STATUS:</strong> {status}</div>
    <div><strong>ACTION:</strong> {action}</div>
</div>
""", unsafe_allow_html=True)

# --- 2. RESUSCITATION TARGETS (KPIs) ---
cols = st.columns(6)

def tile(col, label, val, unit, delta, target_val, invert=False, target_range=None, fmt="{:.0f}"):
    """
    Displays a metric tile.
    val: Raw numeric value (for logic comparison).
    fmt: Format string for display (e.g. "{:.1f}").
    """
    # Logic Check (Using raw number)
    is_bad = val < target_range[0] if target_range else (val > target_val if invert else val < target_val)
    color = "c-crit" if is_bad else "c-ok"
    
    # Format for display
    display_val = fmt.format(val)
    
    col.markdown(f"""
    <div class="metric-tile">
        <div class="tile-label">{label}</div>
        <div class="tile-val">{display_val}</div>
        <div class="tile-unit">{unit}</div>
        <div class="tile-context {color}">{delta}</div>
    </div>
    """, unsafe_allow_html=True)

# 1. MAP
d_map = cur['MAP'] - prv['MAP']
tile(cols[0], "MAP (Pressure)", cur['MAP'], "mmHg", f"{d_map:+.0f}", 65, fmt="{:.0f}")

# 2. CI
d_ci = cur['CI'] - prv['CI']
tile(cols[1], "Cardiac Index", cur['CI'], "L/min/m2", f"{d_ci:+.1f}", 2.5, fmt="{:.1f}")

# 3. SVRI
d_svr = cur['SVRI'] - prv['SVRI']
tile(cols[2], "SVRI (Resist)", cur['SVRI'], "dyn", f"{d_svr:+.0f}", 800, fmt="{:.0f}")

# 4. DO2I
d_do2 = cur['DO2I'] - prv['DO2I']
tile(cols[3], "DO2I (Delivery)", cur['DO2I'], "mL/m2", f"{d_do2:+.0f}", 400, fmt="{:.0f}")

# 5. Lactate
d_lac = cur['Lactate'] - prv['Lactate']
tile(cols[4], "Lactate", cur['Lactate'], "mmol/L", f"{d_lac:+.1f}", 2.0, invert=True, fmt="{:.1f}")

# 6. Urine
tile(cols[5], "Urine Output", cur['Urine'], "mL/kg/hr", "Oliguric" if cur['Urine']<0.5 else "Adequate", 0.5, fmt="{:.1f}")

# --- 3. ROW 1: MACRO-HEMODYNAMICS (The "Bullseye") ---
c_left, c_right = st.columns([1, 1])

with c_left:
    st.markdown("**1. HEMODYNAMIC TARGETING**")
    st.caption("Identify Shock Type: High Flow/Low Pressure vs Low Flow/Low Pressure.")
    fig_bull = utils.plot_hemodynamic_bullseye(df, curr_time)
    st.plotly_chart(fig_bull, use_container_width=True)

with c_right:
    st.markdown("**2. OXYGEN SUPPLY VS DEMAND**")
    st.caption("The root cause of shock. Ensure Delivery (Green) > Debt (Red).")
    fig_ox = utils.plot_oxygen_ledger(df, curr_time)
    st.plotly_chart(fig_ox, use_container_width=True)

# --- 4. ROW 2: ORGAN PROTECTION & MECHANICS ---
c3, c4, c5 = st.columns(3)

with c3:
    st.markdown("**3. RENAL TRAJECTORY**")
    st.caption("Preventing AKI.")
    fig_ren = utils.plot_renal_trajectory(df, curr_time)
    st.plotly_chart(fig_ren, use_container_width=True)

with c4:
    st.markdown("**4. FLUID RESPONSIVENESS**")
    st.caption("Frank-Starling Vector.")
    fig_star = utils.plot_frank_starling_vector(df, curr_time)
    st.plotly_chart(fig_star, use_container_width=True)

with c5:
    st.markdown("**5. AUTONOMIC STRESS**")
    st.caption("Spectral Analysis (PSD).")
    fig_auto = utils.plot_autonomic_psd(df, curr_time)
    st.plotly_chart(fig_auto, use_container_width=True)

# --- 5. TEXT SUMMARY ---
st.info(f"""
**CLINICAL SUMMARY (T={curr_time}):**
The patient demonstrates **{'Hyperdynamic' if cur['CI'] > 3.0 else 'Hypodynamic'}** physiology. 
{'SVR is Critically Low (Vasoplegia)' if cur['SVRI'] < 800 else 'SVR is maintained'}.
Lactate is {'Rising (Metabolic Failure)' if d_lac > 0 else 'Stable'}.
**Priority:** {'Restore Perfusion Pressure (Vasopressors)' if cur['MAP'] < 65 else 'Optimize Flow'}.
""")
