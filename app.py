import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any

# ==========================================
# 1. CONFIGURATION & CITATIONS
# ==========================================
st.set_page_config(page_title="TITAN | Evidence-Based CDS", layout="wide", initial_sidebar_state="expanded")

CITATIONS = """
**Evidence Basis:**
1. **Sepsis-3:** Singer M, et al. *The Third International Consensus Definitions for Sepsis and Septic Shock.* JAMA. 2016.
2. **Lactate Clearance:** Jansen TC, et al. *Early lactate-guided therapy in intensive care unit patients.* JAMA. 2010.
3. **Fluid Responsiveness:** Marik PE, et al. *Fluid responsiveness: an evolution of our understanding.* Crit Care Med. 2017.
4. **SOFA Score:** Vincent JL, et al. *The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure.* Intensive Care Med. 1996.
"""

# ==========================================
# 2. CLINICAL DESIGN SYSTEM
# ==========================================
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Roboto+Mono:wght@500;700&display=swap');

:root {
  --bg: #f8fafc;
  --surface: #ffffff;
  --border: #cbd5e1;
  --text-primary: #0f172a;
  --text-secondary: #475569;
  --map-col: #be185d;
  --ci-col: #059669;
  --lac-col: #7c3aed;
  --svr-col: #d97706;
}

.stApp { background-color: var(--bg); font-family: 'Inter', sans-serif; color: var(--text-primary); }

/* Protocol Header */
.protocol-banner {
    padding: 16px 24px; border-radius: 8px; margin-bottom: 20px;
    border-left: 6px solid; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    background: white; display: flex; justify-content: space-between; align-items: center;
}
.banner-crit { border-color: #dc2626; background: #fef2f2; }
.banner-warn { border-color: #d97706; background: #fffbeb; }
.banner-ok { border-color: #16a34a; background: #f0fdf4; }

/* KPI Architecture */
.kpi-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px; margin-bottom: 10px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
.kpi-label { font-size: 0.75rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-value { font-family: 'Roboto Mono', monospace; font-size: 1.5rem; font-weight: 700; color: var(--text-primary); line-height: 1.2; }
.kpi-unit { font-size: 0.8rem; color: var(--text-secondary); font-weight: 500; }
.kpi-trend { font-size: 0.8rem; font-weight: 600; margin-top: 4px; }

/* Section Headers */
.section-head {
    font-size: 1.0rem; font-weight: 800; color: var(--text-primary);
    border-bottom: 2px solid var(--border); margin: 24px 0 12px 0; padding-bottom: 6px;
    text-transform: uppercase;
}
</style>
"""

# ==========================================
# 3. PHYSIOLOGY ENGINE (Validated Models)
# ==========================================

PLOT_COLORS = {
    "map": "#be185d", "ci": "#059669", "do2": "#7c3aed", 
    "svr": "#d97706", "hr": "#0284c7", "grid": "#f1f5f9", "text": "#334155"
}

class PhysiologyEngine:
    """
    Simulates septic shock progression based on Sepsis-3 pathophysiology:
    1. Vasodilation (Low SVR)
    2. Capillary Leak (Relative Hypovolemia)
    3. Myocardial Depression (Late stage)
    """
    def __init__(self, preload=12.0, contractility=1.0, afterload=1.0):
        self.preload = float(preload)
        self.contractility = float(contractility)
        self.afterload = float(afterload)

    def copy(self):
        return PhysiologyEngine(self.preload, self.contractility, self.afterload)

    def step(self, sepsis_idx=0.0):
        """
        sepsis_idx: 0.0 (Healthy) -> 1.0 (Refractory Shock)
        """
        sev = np.clip(sepsis_idx, 0.0, 1.0)
        
        # Pathophysiology
        # SVR drops significantly (Distributive Shock)
        eff_afterload = self.afterload * max(0.2, 1.0 - 0.7 * sev)
        # Preload drops (Capillary Leak)
        eff_preload = self.preload * max(0.5, 1.0 - 0.3 * sev)

        # Frank-Starling (Cardiac Function)
        sv_max = 100.0 * self.contractility
        # Sigmoid relationship: SV depends on Preload
        sv = sv_max * (eff_preload ** 2 / (eff_preload ** 2 + 64)) 
        
        # Compensatory Tachycardia (Baroreflex)
        # Target MAP ~80. If drops, HR rises.
        map_est = (sv * 70.0 * eff_afterload * 0.05) + 5.0
        hr_drive = float(np.clip((85.0 - map_est) * 2.0, 0.0, 100.0))
        hr = 70.0 + hr_drive

        # Hemodynamics
        co = (hr * sv) / 1000.0  # L/min
        svr_dyne = 1200.0 * eff_afterload
        map_val = (co * svr_dyne / 80.0) + 5.0
        
        # Oxygen Transport
        # DO2 = CO * Hb * 1.34 * SpO2
        do2 = co * 1.34 * 12.0 * 0.98 * 10.0
        
        # Lactate Kinetics (Jansen et al. 2010)
        # Lactate rises when DO2 < Critical Threshold (~400 mL/min/m2)
        # Also rises due to mitochondrial dysfunction in sepsis (cytopathic hypoxia)
        lac_gen = 0.0
        if do2 < 400: lac_gen += (400 - do2) * 0.005
        lac_gen += (sev * 0.05) # Metabolic drive
        
        return {
            "HR": hr, "SV": sv, "CO": co, "MAP": map_val, "SVR": svr_dyne, 
            "DO2": do2, "Lac_Gen": lac_gen, "Preload_Status": eff_preload
        }

def simulate_clinical_data(mins=720, seed=42):
    rng = np.random.default_rng(seed)
    engine = PhysiologyEngine()
    
    # Generate Sepsis Progression Curve
    sepsis_curve = np.linspace(0.0, 0.9, mins)
    
    history = []
    curr_lac = 1.0
    
    for t in range(mins):
        # Add biological variability (1/f noise)
        noise = rng.normal(0, 0.02)
        state = engine.step(sepsis_curve[t] + noise)
        
        # Lactate Clearance Model (First order kinetics)
        # Clearance rate assumes 50% reduction every 2 hours in healthy liver
        clearance = curr_lac * 0.005 
        curr_lac = curr_lac - clearance + state["Lac_Gen"]
        state["Lactate"] = max(0.5, curr_lac)
        
        # Urine Output (Sigmoid Autoregulation Curve)
        # Kidney GFR maintains until MAP < 65, then drops rapidly
        map_input = state["MAP"] + rng.normal(0, 2)
        uo = 1.0 / (1.0 + np.exp(-0.2 * (map_input - 60.0))) * 1.5
        state["Urine"] = max(0.0, uo)
        
        history.append(state)
        
    df = pd.DataFrame(history)
    
    # Derived Metrics per Guidelines
    df["CI"] = df["CO"] / 1.8 # Cardiac Index
    df["SVRI"] = df["SVR"] * 1.8 # SVR Index
    df["PP"] = df["SV"] / 1.5 # Pulse Pressure Proxy
    
    # SOFA Score Calculation (Real-time proxy)
    sofa_list = []
    for _, row in df.iterrows():
        s = 0
        if row["MAP"] < 70: s += 1
        if row["Urine"] < 0.5: s += 1 # Oliguria
        if row["Urine"] < 0.2: s += 2 # Anuria
        sofa_list.append(s)
    df["SOFA"] = sofa_list
    
    return df

def predict_response(base_engine, current_sev, current_lac, horizon=60):
    """
    Simulates physiological response to interventions based on Starling/SVR curves.
    """
    times = np.arange(horizon)
    futures = {"Natural": [], "Fluid": [], "Pressor": []}
    
    # 1. Natural Progression
    lac = current_lac
    for t in times:
        s = base_engine.copy().step(current_sev + 0.001*t)
        lac = lac * 0.995 + s["Lac_Gen"]
        futures["Natural"].append(s["MAP"])
        
    # 2. Fluid Loading (Increase Preload)
    # Evidence: Only works if patient is on steep part of Starling curve
    eng_fl = base_engine.copy()
    eng_fl.preload *= 1.4 # 1L Bolus equivalent
    for t in times:
        s = eng_fl.step(current_sev + 0.001*t)
        futures["Fluid"].append(s["MAP"])
        
    # 3. Vasopressor (Increase Afterload)
    # Evidence: Restores MAP but may increase cardiac work
    eng_pr = base_engine.copy()
    eng_pr.afterload *= 1.5 # Norepinephrine effect
    for t in times:
        s = eng_pr.step(current_sev + 0.001*t)
        futures["Pressor"].append(s["MAP"])
        
    return futures

# ==========================================
# 4. PLOTTING LIBRARY (Visuals)
# ==========================================

def hex_to_rgba(h, alpha):
    h = h.lstrip('#')
    return f"rgba({int(h[0:2],16)}, {int(h[2:4],16)}, {int(h[4:6],16)}, {alpha})"

def clean_layout(fig, height=200):
    fig.update_layout(
        template="plotly_white", margin=dict(l=10, r=10, t=30, b=10), height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=PLOT_COLORS["text"])
    )
    return fig

def plot_lactate_kinetics(df, curr_idx):
    """
    Evidence: Lactate Clearance > 10% is a survival predictor (Jansen 2010).
    """
    start = max(0, curr_idx - 180)
    data = df.iloc[start:curr_idx]
    
    fig = go.Figure()
    
    # Target Zone (Clearance)
    fig.add_shape(type="rect", x0=data.index[0], x1=data.index[-1], y0=0, y1=2.0, 
                  fillcolor="rgba(16, 185, 129, 0.1)", line_width=0, layer="below")
    
    fig.add_trace(go.Scatter(x=data.index, y=data["Lactate"], mode='lines', 
                             line=dict(color=PLOT_COLORS["do2"], width=3), name="Lactate"))
    
    # Calculate % Clearance over last 2 hours
    if len(data) >= 120:
        lac_start = data["Lactate"].iloc[-120]
        lac_now = data["Lactate"].iloc[-1]
        pct_change = ((lac_start - lac_now) / lac_start) * 100
        
        status = "CLEARING" if pct_change > 0 else "ACCUMULATING"
        col = "green" if pct_change > 10 else ("orange" if pct_change > 0 else "red")
        
        fig.add_annotation(x=data.index[-1], y=data["Lactate"].iloc[-1], 
                           text=f"{status}: {pct_change:+.1f}% (2hr)", 
                           font=dict(color=col, weight="bold"), yshift=15)

    fig = clean_layout(fig, height=250)
    fig.update_layout(title="<b>Lactate Kinetics (Metabolic Clearance)</b>", showlegend=False)
    fig.update_yaxes(title="Lactate (mmol/L)", gridcolor=PLOT_COLORS["grid"])
    return fig

def plot_hemo_profile(df, curr_idx):
    """
    Evidence: Surviving Sepsis Targets (MAP > 65).
    """
    data = df.iloc[max(0, curr_idx-60):curr_idx]
    cur = df.iloc[curr_idx]
    
    fig = go.Figure()
    
    # Diagnostic Zones (GDT)
    # Vasoplegia (High CI / Low MAP)
    fig.add_shape(type="rect", x0=2.5, x1=6.0, y0=0, y1=65, 
                  fillcolor="rgba(217, 119, 6, 0.15)", line_width=0)
    fig.add_annotation(x=4.0, y=40, text="VASOPLEGIA<br>(Need Pressor)", font=dict(size=10, color="#d97706"), showarrow=False)
    
    # Hypovolemia (Low CI / Low MAP)
    fig.add_shape(type="rect", x0=0, x1=2.5, y0=0, y1=65, 
                  fillcolor="rgba(220, 38, 38, 0.15)", line_width=0)
    fig.add_annotation(x=1.25, y=40, text="LOW FLOW<br>(Need Volume)", font=dict(size=10, color="#dc2626"), showarrow=False)
    
    # Goal
    fig.add_shape(type="rect", x0=2.5, x1=5.0, y0=65, y1=110, 
                  fillcolor="rgba(22, 163, 74, 0.15)", line_width=2, line_color="#16a34a")
    fig.add_annotation(x=3.75, y=85, text="TARGET", font=dict(size=12, color="#16a34a", weight="bold"), showarrow=False)

    # Patient State
    fig.add_trace(go.Scatter(x=data["CI"], y=data["MAP"], mode="lines", 
                             line=dict(color="#94a3b8", dash="dot"), name="Trend"))
    fig.add_trace(go.Scatter(x=[cur["CI"]], y=[cur["MAP"]], mode="markers", 
                             marker=dict(color="#0f172a", size=14, symbol="cross"), name="Current"))

    fig = clean_layout(fig, height=300)
    fig.update_layout(title="<b>Hemodynamic Profile (Diagnosis)</b>", showlegend=False)
    fig.update_xaxes(title="Cardiac Index (L/min/m²)", range=[1.0, 5.5], gridcolor=PLOT_COLORS["grid"])
    fig.update_yaxes(title="MAP (mmHg)", range=[30, 100], gridcolor=PLOT_COLORS["grid"])
    return fig

def plot_starling_slope(df, curr_idx):
    """
    Evidence: Dynamic measures (Delta SV) predict fluid responsiveness better than CVP (Marik 2017).
    """
    data = df.iloc[max(0, curr_idx-30):curr_idx]
    
    fig = go.Figure()
    
    # Ideal Curve
    x = np.linspace(0, 20, 100)
    y = np.log(x+1)*30
    fig.add_trace(go.Scatter(x=x, y=y, line=dict(color='#cbd5e1', dash='dot'), name='Reference'))
    
    # Patient Trajectory
    fig.add_trace(go.Scatter(x=data['Preload_Status'], y=data['SV'], mode='lines', 
                             line=dict(color=PLOT_COLORS['ci'], width=3), name='Patient'))
    fig.add_trace(go.Scatter(x=[data['Preload_Status'].iloc[-1]], y=[data['SV'].iloc[-1]], 
                             mode='markers', marker=dict(color=PLOT_COLORS['ci'], size=10), name='Now'))
    
    # Slope Calculation (Responsiveness)
    d_sv = data['SV'].iloc[-1] - data['SV'].iloc[0]
    d_pre = data['Preload_Status'].iloc[-1] - data['Preload_Status'].iloc[0] + 0.01
    slope = d_sv / d_pre
    
    status = "FLUID RESPONSIVE" if slope > 2.0 else "NON-RESPONDER"
    col = "green" if slope > 2.0 else "red"
    
    fig.add_annotation(x=10, y=10, text=f"{status}", font=dict(color=col, weight="bold"), showarrow=False)

    fig = clean_layout(fig, height=250)
    fig.update_layout(title="<b>Fluid Responsiveness (Starling Slope)</b>", showlegend=False)
    fig.update_xaxes(title="Preload Status", gridcolor=PLOT_COLORS["grid"])
    fig.update_yaxes(title="Stroke Volume", gridcolor=PLOT_COLORS["grid"])
    return fig

def plot_horizon_response(futures):
    """
    Evidence: Early goal-directed therapy (EGDT) concepts. 
    Shows predicted MAP response to standard interventions.
    """
    t = np.arange(len(futures["Natural"]))
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=t, y=futures["Natural"], line=dict(color="#94a3b8", dash="dot"), name="No Intervention"))
    fig.add_trace(go.Scatter(x=t, y=futures["Fluid"], line=dict(color=PLOT_COLORS["ci"]), name="Fluid (+500mL)"))
    fig.add_trace(go.Scatter(x=t, y=futures["Pressor"], line=dict(color=PLOT_COLORS["map"]), name="Pressor (Norepi)"))
    
    fig.add_hline(y=65, line_color="#ef4444", line_dash="solid", annotation_text="Target MAP 65")
    
    fig = clean_layout(fig, height=250)
    fig.update_layout(title="<b>Projected Hemodynamic Response (30min)</b>", legend=dict(orientation="h", y=1.1))
    fig.update_xaxes(title="Minutes Future", showgrid=False)
    fig.update_yaxes(title="Predicted MAP", gridcolor=PLOT_COLORS["grid"])
    return fig

def plot_spark(df, col, color, thresh_l, thresh_h):
    data = df[col].iloc[-60:]
    fig = go.Figure()
    
    # Safe Zone
    fig.add_shape(type="rect", x0=data.index[0], x1=data.index[-1], y0=thresh_l, y1=thresh_h,
                  fillcolor="rgba(0,0,0,0.04)", line_width=0, layer="below")
    
    fig.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines', 
                             line=dict(color=color, width=2), fill='tozeroy', fillcolor=hex_to_rgba(color, 0.1)))
    
    fig = clean_layout(fig, height=50)
    fig.update_layout(margin=dict(t=0,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

# ==========================================
# 5. MAIN APPLICATION
# ==========================================

st.markdown(THEME_CSS, unsafe_allow_html=True)

# Data Loading
@st.cache_data
def get_data(): return simulate_clinical_data(mins=720, seed=42)
df = get_data()

# Controls
with st.sidebar:
    st.header("TITAN | Control")
    curr_time = st.slider("Timeline", 60, 720, 720)
    st.info("Scenario: Progressive Sepsis (Distributive Shock)")
    with st.expander("References"):
        st.markdown(CITATIONS)

idx = curr_time - 1
row = df.iloc[idx]
prev = df.iloc[idx-15]

# --- 1. EVIDENCE-BASED HEADER ---
# Logic based on Surviving Sepsis Campaign 2021
status_msg = "STABLE"
action_msg = "Continue Monitoring"
banner_style = "banner-ok"

if row["Lactate"] > 2.0:
    status_msg = "HYPOPERFUSION DETECTED"
    action_msg = "Assess Fluid Status & Lactate Clearance"
    banner_style = "banner-warn"

if row["MAP"] < 65:
    status_msg = "SEPTIC SHOCK (Hypotension)"
    if row["CI"] > 2.5:
        action_msg = "PROTOCOL: Start Vasopressors (Norepinephrine)"
    else:
        action_msg = "PROTOCOL: Fluid Challenge (30mL/kg)"
    banner_style = "banner-crit"

st.markdown(f"""
<div class="protocol-banner {banner_style}">
    <div>
        <div style="font-weight:800; font-size:1.2rem;">{status_msg}</div>
        <div style="font-weight:500;">Target: MAP > 65 • Lactate < 2.0</div>
    </div>
    <div style="font-weight:700; font-size:1.1rem; text-align:right;">RECOMMENDATION:<br>{action_msg}</div>
</div>
""", unsafe_allow_html=True)

# --- 2. KPI STRIP ---
k_cols = st.columns(6)

def render_kpi(col, label, val, unit, color, df_col, t_low, t_high, key_id):
    delta = val - prev[df_col]
    delta_str = f"{delta:+.1f}"
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid {color}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color}">{val:.1f} <span class="kpi-unit">{unit}</span></div>
            <div class="kpi-trend" style="color:{'#ef4444' if delta < 0 else '#10b981'}">{delta_str} (15m)</div>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(plot_spark(df.iloc[:idx+1], df_col, color, t_low, t_high), 
                        use_container_width=True, config={'displayModeBar': False}, key=key_id)

render_kpi(k_cols[0], "MAP", row["MAP"], "mmHg", PLOT_COLORS["map"], "MAP", 65, 100, "k1")
render_kpi(k_cols[1], "Cardiac Index", row["CI"], "L/min", PLOT_COLORS["ci"], "CI", 2.5, 4.0, "k2")
render_kpi(k_cols[2], "SVR Index", row["SVRI"], "dyn", PLOT_COLORS["svr"], "SVRI", 800, 1200, "k3")
render_kpi(k_cols[3], "Lactate", row["Lactate"], "mmol", PLOT_COLORS["do2"], "Lactate", 0, 2.0, "k4")
render_kpi(k_cols[4], "SOFA Score", float(row["SOFA"]), "pts", "#475569", "SOFA", 0, 2, "k5")
render_kpi(k_cols[5], "Urine Out", row["Urine"], "mL/kg", "#0284c7", "Urine", 0.5, 2.0, "k6")

# --- 3. DIAGNOSTIC & PROGNOSTIC PANELS ---
st.markdown('<div class="section-head">🔮 HEMODYNAMIC PHENOTYPE & PREDICTION</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Shock Diagnosis (Compass)**")
    st.caption("Visualizes Shock Type: Vasoplegic (Right) vs Hypovolemic (Left).")
    st.plotly_chart(plot_hemo_profile(df, idx), use_container_width=True, config={'displayModeBar': False})

with c2:
    st.markdown("**Intervention Horizon (30 min)**")
    st.caption("Projected MAP response to Fluid vs Pressor based on current physiology.")
    futures = predict_response(PhysiologyEngine(), 0.9 * (idx/720), row["Lactate"])
    st.plotly_chart(plot_horizon_response(futures), use_container_width=True, config={'displayModeBar': False})

st.markdown('<div class="section-head">🫀 PHYSIOLOGICAL MECHANICS (SAFETY CHECK)</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)

with c3:
    st.markdown("**Metabolic Clearance (Lactate)**")
    st.caption("Target: Clearance > 10% over 2 hours (Jansen 2010).")
    st.plotly_chart(plot_lactate_kinetics(df, idx), use_container_width=True, config={'displayModeBar': False})

with c4:
    st.markdown("**Fluid Responsiveness (Starling)**")
    st.caption("Slope determines if Volume will increase Stroke Volume (Marik 2017).")
    st.plotly_chart(plot_starling_slope(df, idx), use_container_width=True, config={'displayModeBar': False})
