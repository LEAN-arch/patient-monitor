import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any, Optional, List

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="TITAN | Clinical Command", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. ENHANCED CLINICAL CSS
# ==========================================
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Roboto+Mono:wght@500&display=swap');

:root {
  --bg: #f8fafc;
  --surface: #ffffff;
  --border: #e2e8f0;
  --text-primary: #0f172a;
  --text-secondary: #64748b;
  --accent-blue: #0284c7;
  --accent-success: #10b981;
  --accent-warn: #f59e0b;
  --accent-crit: #ef4444;
}

.stApp { background-color: var(--bg); font-family: 'Inter', sans-serif; }

/* KPI Card Architecture */
.titan-card {
  background: var(--surface);
  border-radius: 12px;
  padding: 16px;
  border: 1px solid var(--border);
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  margin-bottom: 12px;
}
.titan-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
}

/* Typography */
.kpi-lbl { 
    font-size: 0.75rem; 
    color: var(--text-secondary); 
    font-weight: 700; 
    text-transform: uppercase; 
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.kpi-val { 
    font-size: 1.75rem; 
    font-weight: 800; 
    font-family: 'Roboto Mono', monospace; 
    color: var(--text-primary);
    line-height: 1.1;
}
.kpi-unit {
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-weight: 500;
    margin-left: 2px;
}

/* Alert Header System */
.alert-container {
    background: var(--surface);
    border-radius: 12px;
    border-left: 8px solid;
    padding: 20px;
    margin-bottom: 24px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.alert-status { font-size: 1.4rem; font-weight: 800; letter-spacing: -0.02em; }
.alert-action { font-size: 1.1rem; font-weight: 600; color: var(--text-primary); }
.alert-sub { font-size: 0.9rem; color: var(--text-secondary); margin-top: 4px; }

/* Section Dividers */
.section-header { 
    font-size: 1.1rem; 
    font-weight: 700; 
    color: var(--text-primary); 
    display: flex; 
    align-items: center; 
    gap: 8px;
    margin-top: 32px; 
    margin-bottom: 16px; 
    padding-bottom: 8px;
    border-bottom: 2px solid var(--border);
}
</style>
"""

# ==========================================
# 3. UTILS & PHYSIOLOGY ENGINE
# ==========================================

# High-Contrast Clinical Palette
PLOT_COLORS = {
    "map": "#be185d",      # Magenta (Pressure)
    "ci": "#059669",       # Emerald (Flow)
    "do2": "#7c3aed",      # Violet (Oxygen)
    "svr": "#d97706",      # Amber (Resistance)
    "hr": "#0284c7",       # Blue (Rate)
    "grid": "#f1f5f9",
    "text": "#334155",
    "bg": "#ffffff"
}

# Evidence-Based Thresholds
THRESHOLDS = {
    "MAP_TARGET": 65.0,
    "MAP_CRIT": 55.0,
    "LACTATE_HIGH": 2.0,
    "CI_LOW": 2.2,
    "SI_WARN": 0.9
}

class DigitalTwin:
    """
    0-D Cardiovascular Model simulating Sepsis Physiology.
    """
    def __init__(self, preload=12.0, contractility=1.0, afterload=1.0, lactate_hl=120.0):
        self.preload = float(preload)
        self.contractility = float(contractility)
        self.afterload = float(afterload)
        self.lactate_hl = float(lactate_hl)

    def copy(self):
        return DigitalTwin(self.preload, self.contractility, self.afterload, self.lactate_hl)

    def step(self, sepsis_severity=0.0, dt_min=1.0):
        sev = float(np.clip(sepsis_severity, 0.0, 1.0))
        
        # Pathophysiology: Vasodilation + Capillary Leak
        eff_afterload = self.afterload * max(0.01, 1.0 - 0.7 * sev)
        eff_preload = self.preload * max(0.01, 1.0 - 0.4 * sev)

        # Frank-Starling Law
        k_preload = 8.0
        sv_max = 100.0 * self.contractility
        sv = sv_max * (eff_preload ** 2 / (eff_preload ** 2 + k_preload ** 2))

        # Baroreceptor Reflex
        map_est = (sv * 75.0 * eff_afterload * 0.05) + 5.0
        hr_drive = float(np.clip((90.0 - map_est) * 1.8, 0.0, 160.0))
        hr = 70.0 + hr_drive

        # Hemodynamics
        co = (hr * sv) / 1000.0  # L/min
        svr = 1200.0 * eff_afterload
        map_val = (co * svr / 80.0) + 5.0

        # Metabolic
        do2 = co * 1.34 * 12.0 * 0.98 * 10.0
        lac_gen = max(0.0, (400.0 - do2) * 0.01) + 0.01
        
        # Pharmacokinetics (Clearance)
        clearance_k = np.log(2) / max(1.0, self.lactate_hl)
        lac_clear_frac = clearance_k * dt_min

        return {
            "HR": float(hr), "SV": float(sv), "CO": float(co),
            "MAP": float(map_val), "SVR": float(svr), "DO2": float(do2),
            "Lac_Gen": float(lac_gen), "Lac_Clear_frac": float(lac_clear_frac),
            "Preload": float(eff_preload)
        }

def predict_horizon(base_twin: DigitalTwin, last_sev: float, horizon: int = 30) -> Dict[str, Any]:
    """Generates 'Multiverse' predictions for intervention support."""
    nat, fluid, press, inot = [], [], [], []
    
    for i in range(horizon):
        sev = float(np.clip(last_sev + 0.001 * i, 0.0, 1.0))

        # 1. Natural Course (Do Nothing)
        s_nat = base_twin.copy().step(sev)
        nat.append(s_nat)

        # 2. Fluid Bolus (+500mL equivalent)
        t_fl = base_twin.copy(); t_fl.preload *= 1.30
        s_fl = t_fl.step(sev)
        fluid.append(s_fl)

        # 3. Vasopressor (Norepinephrine)
        t_pr = base_twin.copy(); t_pr.afterload *= 1.40
        s_pr = t_pr.step(sev)
        press.append(s_pr)

        # 4. Inotrope (Dobutamine)
        t_in = base_twin.copy(); t_in.contractility *= 1.15
        s_in = t_in.step(sev)
        inot.append(s_in)

    def extract(lst, key): return np.array([d[key] for d in lst])
    
    return {
        "time": np.arange(horizon),
        "nat": {"MAP": extract(nat,"MAP"), "CI": extract(nat,"CO")/1.8},
        "fluid": {"MAP": extract(fluid,"MAP"), "CI": extract(fluid,"CO")/1.8},
        "press": {"MAP": extract(press,"MAP"), "CI": extract(press,"CO")/1.8},
        "inot": {"MAP": extract(inot,"MAP"), "CI": extract(inot,"CO")/1.8}
    }

def simulate_data(mins=720, twin=None, seed=None):
    rng = np.random.default_rng(seed)
    base = twin.copy() if twin else DigitalTwin()
    sepsis = np.linspace(0.0, 0.85, mins)
    noise = rng.normal(0.0, 0.02, size=mins)

    rows = []
    curr_lac = 1.0
    for i in range(mins):
        sev = float(np.clip(sepsis[i] + noise[i], 0.0, 1.0))
        s = base.step(sev)
        curr_lac = curr_lac * (1.0 - s["Lac_Clear_frac"]) + s["Lac_Gen"]
        s["Lactate"] = float(max(0.4, curr_lac))
        rows.append(s)

    df = pd.DataFrame(rows)
    df.index.name = "minute"
    
    # Derivations
    df["CI"] = df["CO"] / 1.8
    df["SVRI"] = df["SVR"] * 1.8
    df["PP"] = df["SV"] / 1.5
    df["MAP"] = df["MAP"].fillna(method="ffill").fillna(65.0)
    df["SI"] = df["HR"] / df["MAP"]
    
    # Entropy (Complexity)
    df["Entropy"] = df["HR"].diff().rolling(60).std().fillna(0.0)
    
    # Urine Output (Sigmoid proxy)
    uo_curve = 1.0 / (1.0 + np.exp(-0.15 * (df["MAP"] - 65.0))) * 2.0
    df["Urine"] = (uo_curve + rng.normal(0, 0.05, len(df))).clip(lower=0)

    preds = predict_horizon(base, sepsis[-1])
    return df, preds

def compute_alerts(df, lookback=15):
    alerts = []
    if len(df) < lookback: return {"alerts": []}
    
    window = df.iloc[-lookback:]
    curr = window.iloc[-1]
    
    # MAP
    if (window["MAP"] < THRESHOLDS["MAP_TARGET"]).sum() > 5:
        sev = "critical" if curr["MAP"] < THRESHOLDS["MAP_CRIT"] else "warning"
        alerts.append({"type": "HYPOTENSION", "severity": sev, "message": f"MAP < {THRESHOLDS['MAP_TARGET']} mmHg sustained.", "action": "Assess Volume status first, then Pressors."})
        
    # Lactate
    if curr["Lactate"] > THRESHOLDS["LACTATE_HIGH"]:
        alerts.append({"type": "LACTATE", "severity": "critical", "message": f"Lactate {curr['Lactate']:.1f} indicates tissue hypoxia.", "action": "Maximize DO2 (Cardiac Output/Hgb)."})
        
    return {"alerts": alerts}

# ==========================================
# 4. VISUALIZATION ENGINE (SME Optimized)
# ==========================================

def hex_to_rgba(h, alpha):
    h = h.lstrip('#')
    return f"rgba({int(h[0:2],16)}, {int(h[2:4],16)}, {int(h[4:6],16)}, {alpha})"

def _clean_layout(fig, height=200):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=10),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=PLOT_COLORS["text"])
    )
    return fig

def plot_sparkline_context(df, col, color, safe_low, safe_high, label):
    """
    SME Upgrade: Adds a 'Safety Corridor' background to the sparkline.
    Visualizes deviation from 'Normal' instantly.
    """
    data = df[col].iloc[-60:]
    fig = go.Figure()
    
    # 1. Safety Corridor (Grey Background)
    fig.add_shape(type="rect", 
                  x0=data.index[0], x1=data.index[-1], 
                  y0=safe_low, y1=safe_high,
                  fillcolor="rgba(0,0,0,0.04)", line_width=0, layer="below")
    
    # 2. Main Trend Line (Gradient Fill)
    fig.add_trace(go.Scatter(
        x=data.index, y=data.values, 
        mode='lines', 
        line=dict(color=color, width=3),
        fill='tozeroy',
        fillcolor=hex_to_rgba(color, 0.1),
        hoverinfo='y'
    ))
    
    # 3. Endpoint Indicator
    fig.add_trace(go.Scatter(
        x=[data.index[-1]], y=[data.values[-1]],
        mode='markers',
        marker=dict(color=color, size=8, line=dict(color='white', width=2))
    ))

    fig = _clean_layout(fig, height=50)
    fig.update_layout(margin=dict(t=0, b=0))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, range=[min(data.min(), safe_low)*0.9, max(data.max(), safe_high)*1.1])
    return fig

def plot_predictive_compass(df, preds, curr_idx):
    """
    SME Upgrade: Replaces 'Scatter' with 'Contour-like' zones for clearer diagnosis.
    Uses bolder vectors for treatment recommendation.
    """
    start = max(0, curr_idx - 60)
    hist = df.iloc[start:curr_idx]
    cur = df.iloc[curr_idx]
    
    fig = go.Figure()
    
    # 1. Diagnostic Zones (Rectangles)
    # Vasoplegia (High Flow, Low Press)
    fig.add_shape(type="rect", x0=2.5, x1=6.0, y0=0, y1=65, 
                  fillcolor="rgba(217, 119, 6, 0.1)", line_width=0)
    fig.add_annotation(x=4.25, y=40, text="VASOPLEGIA<br>(Needs Pressor)", 
                       font=dict(size=10, color="#d97706"), showarrow=False)
    
    # Hypoperfusion (Low Flow, Low Press)
    fig.add_shape(type="rect", x0=0, x1=2.5, y0=0, y1=65, 
                  fillcolor="rgba(220, 38, 38, 0.1)", line_width=0)
    fig.add_annotation(x=1.25, y=40, text="CRITICAL SHOCK<br>(Needs Flow)", 
                       font=dict(size=10, color="#dc2626"), showarrow=False)
    
    # Goal Zone
    fig.add_shape(type="rect", x0=2.5, x1=5.0, y0=65, y1=120, 
                  fillcolor="rgba(22, 163, 74, 0.1)", line_width=0)
    fig.add_annotation(x=3.75, y=90, text="GOAL", 
                       font=dict(size=12, color="#16a34a", weight="bold"), showarrow=False)

    # 2. History Trail
    if not hist.empty:
        fig.add_trace(go.Scatter(x=hist["CI"], y=hist["MAP"], mode="lines", 
                                 line=dict(color="#cbd5e1", width=2, dash="dot"), name="History"))
    
    # 3. Current Position
    fig.add_trace(go.Scatter(x=[cur["CI"]], y=[cur["MAP"]], mode="markers", 
                             marker=dict(color="#0f172a", size=14, symbol="cross", line=dict(width=2, color="white")), 
                             name="Current"))

    # 4. Intervention Vectors (Arrows)
    # Fluid Vector
    f_ci = preds["fluid"]["CI"][-1]
    f_map = preds["fluid"]["MAP"][-1]
    fig.add_annotation(x=f_ci, y=f_map, ax=cur["CI"], ay=cur["MAP"], xref="x", yref="y", axref="x", ayref="y",
                       arrowhead=3, arrowsize=1.5, arrowwidth=2, arrowcolor=PLOT_COLORS["ci"],
                       text="FLUID", font=dict(color=PLOT_COLORS["ci"], weight="bold"))
    
    # Pressor Vector
    p_ci = preds["press"]["CI"][-1]
    p_map = preds["press"]["MAP"][-1]
    fig.add_annotation(x=p_ci, y=p_map, ax=cur["CI"], ay=cur["MAP"], xref="x", yref="y", axref="x", ayref="y",
                       arrowhead=3, arrowsize=1.5, arrowwidth=2, arrowcolor=PLOT_COLORS["map"],
                       text="PRESSOR", font=dict(color=PLOT_COLORS["map"], weight="bold"))

    fig = _clean_layout(fig, height=320)
    fig.update_layout(title="<b>Hemodynamic Compass (Decision Support)</b>", showlegend=False)
    fig.update_xaxes(title="Cardiac Index (L/min/m²)", range=[1.0, 5.0], showgrid=True, gridcolor=PLOT_COLORS["grid"])
    fig.update_yaxes(title="MAP (mmHg)", range=[30, 110], showgrid=True, gridcolor=PLOT_COLORS["grid"])
    return fig

def plot_horizon_benefit(preds):
    """
    SME Upgrade: Highlights the 'Therapeutic Gain' (Area between curves).
    """
    t = preds["time"]
    fig = go.Figure()
    
    # Natural Course (Baseline Risk)
    fig.add_trace(go.Scatter(x=t, y=preds["nat"]["MAP"], 
                             line=dict(color="#94a3b8", dash="dot"), name="Natural Course"))
    
    # Intervention (Best Option - e.g., Pressor for Sepsis)
    # Filling the area to show 'Benefit'
    fig.add_trace(go.Scatter(x=t, y=preds["press"]["MAP"], 
                             line=dict(color=PLOT_COLORS["map"], width=3), 
                             fill='tonexty', fillcolor=hex_to_rgba(PLOT_COLORS["map"], 0.1),
                             name="With Vasopressor"))
    
    # Target Line
    fig.add_hline(y=65, line_dash="solid", line_color="#ef4444", annotation_text="Safety Threshold (65)", annotation_position="top right")
    
    fig = _clean_layout(fig, height=320)
    fig.update_layout(title="<b>Predicted Response (30min Horizon)</b>", legend=dict(orientation="h", y=1.1))
    fig.update_xaxes(title="Minutes into Future", showgrid=False)
    fig.update_yaxes(title="Projected MAP (mmHg)", gridcolor=PLOT_COLORS["grid"])
    return fig

def plot_organ_radar(df, curr_idx):
    """
    SME Upgrade: Adds a 'Baseline' wireframe for comparison.
    """
    cur = df.iloc[curr_idx]
    
    # Risk Calculations (0=Safe, 1=Crit)
    r_kidney = np.clip((65 - cur["MAP"])/20, 0, 1)
    r_heart = np.clip((cur["HR"] - 100)/60, 0, 1)
    r_meta = np.clip((cur["Lactate"] - 1.5)/4, 0, 1)
    r_flow = np.clip((2.2 - cur["CI"])/1.5, 0, 1)
    
    vals = [r_kidney, r_heart, r_meta, r_flow, r_kidney]
    cats = ["Kidney (AKI)", "Heart (Strain)", "Metabolic (Acid)", "Perfusion (Shock)", "Kidney (AKI)"]
    
    fig = go.Figure()
    
    # Baseline (Healthy Reference)
    fig.add_trace(go.Scatterpolar(r=[0.2]*5, theta=cats, fill='none', 
                                  line=dict(color='lightgrey', dash='dot'), name='Healthy Baseline'))
    
    # Current Risk
    fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill='toself', 
                                  fillcolor=hex_to_rgba("#ef4444", 0.2), 
                                  line=dict(color="#ef4444", width=2), name='Current Risk'))
    
    fig = _clean_layout(fig, height=250)
    fig.update_layout(title="<b>Organ Risk Topology</b>", showlegend=False)
    fig.update_polars(radialaxis=dict(visible=False, range=[0, 1]))
    return fig

# ==========================================
# 5. MAIN APP EXECUTION
# ==========================================

# A. Styles
st.markdown(THEME_CSS, unsafe_allow_html=True)

# B. Data Management
@st.cache_data
def get_sim_data():
    return simulate_titan_data(mins=720, seed=42)

df, preds = get_sim_data()

# C. Sidebar
with st.sidebar:
    st.header("TITAN | Controls")
    st.write("Simulating Septic Shock Progression")
    minute = st.slider("Time Elapsed (min)", 1, 720, 720)
    
    st.markdown("### 🧪 Physiology Knobs")
    p_mod = st.number_input("Preload Modifier", 0.5, 2.0, 1.0, 0.1)
    c_mod = st.number_input("Contractility Modifier", 0.5, 2.0, 1.0, 0.1)
    
    if st.button("Refresh Simulation"):
        st.cache_data.clear()
        st.rerun()

# D. App Logic
idx = max(0, minute - 1)
row = df.iloc[idx]
alerts = compute_alerts(df.iloc[:idx+1])

# Header Logic
def get_status(r):
    if r["MAP"] < 60: return "CRITICAL SHOCK", "Start Norepinephrine immediately", "var(--accent-crit)", "var(--surface)"
    if r["MAP"] < 65: return "HYPOTENSION", "Assess Fluid Responsiveness", "var(--accent-warn)", "var(--surface)"
    if r["Lactate"] > 2.0: return "OCCULT SHOCK", "Check SVR & Cardiac Output", "var(--accent-warn)", "var(--surface)"
    return "STABLE", "Continue Standard Monitoring", "var(--accent-success)", "var(--surface)"

stat_txt, act_txt, color, bg = get_status(row)

# E. UI Layout

# 1. Header
st.markdown(f"""
<div class="alert-container" style="border-color: {color};">
    <div>
        <div class="alert-status" style="color: {color};">{stat_txt}</div>
        <div class="alert-sub">Lactate: {row['Lactate']:.1f} mmol/L • MAP: {row['MAP']:.0f} mmHg</div>
    </div>
    <div class="alert-action">{act_txt}</div>
</div>
""", unsafe_allow_html=True)

# 2. KPI Sparkline Grid
cols = st.columns(6, gap="small")

def render_kpi(col, label, val, unit, color, df_col, safe_l, safe_h, key):
    with col:
        st.markdown(f"""
        <div class="titan-card">
            <div class="kpi-lbl">{label}</div>
            <div class="kpi-val" style="color:{color}">{val}<span class="kpi-unit">{unit}</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(plot_sparkline_context(df.iloc[:idx+1], df_col, color, safe_l, safe_h, label), 
                        use_container_width=True, config={'displayModeBar': False}, key=key)

render_kpi(cols[0], "MAP", f"{row['MAP']:.0f}", "mmHg", PLOT_COLORS["map"], "MAP", 65, 100, "k1")
render_kpi(cols[1], "C. INDEX", f"{row['CI']:.1f}", "L/min", PLOT_COLORS["ci"], "CI", 2.5, 4.2, "k2")
render_kpi(cols[2], "SVR", f"{row['SVRI']:.0f}", "dyn", PLOT_COLORS["svr"], "SVRI", 800, 1200, "k3")
render_kpi(cols[3], "STROKE VOL", f"{row['PP']:.0f}", "mL", PLOT_COLORS["hr"], "PP", 40, 80, "k4")
render_kpi(cols[4], "DO2 Index", f"{row['DO2']:.0f}", "mL/m²", PLOT_COLORS["do2"], "DO2", 400, 600, "k5")
render_kpi(cols[5], "Urine Out", f"{row['Urine']:.1f}", "mL/kg", "#64748b", "Urine", 0.5, 2.0, "k6")

# 3. Decision Support Section
st.markdown('<div class="section-header">🔮 PREDICTIVE DECISION SUPPORT</div>', unsafe_allow_html=True)
c_left, c_right = st.columns([1, 1], gap="medium")

with c_left:
    st.markdown("**Hemodynamic Compass (Diagnosis)**")
    st.caption("Visualizes patient state (Cross) and predicted response to therapy (Arrows).")
    st.plotly_chart(plot_predictive_compass(df, preds, idx), use_container_width=True, config={'displayModeBar': False})

with c_right:
    st.markdown("**Therapeutic Horizon (Prognosis)**")
    st.caption("Shaded area represents the 'Benefit Gap' of intervention vs. natural decline.")
    st.plotly_chart(plot_horizon_benefit(preds), use_container_width=True, config={'displayModeBar': False})

# 4. Organ Safety Section
st.markdown('<div class="section-header">🫀 ORGAN SYSTEM SAFETY</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4, gap="medium")

with c1:
    st.plotly_chart(plot_organ_radar(df, idx), use_container_width=True, config={'displayModeBar': False})
# Placeholders for other organ mechanics if needed, or keeping it clean with just radar + context
with c2:
    st.info("**Renal Status:**\nPerfusion Pressure is critical. Autoregulation likely lost at MAP < 65.")
with c3:
    st.info("**Cardiac Status:**\nContractility is maintained, but afterload is critically low (Vasoplegia).")
with c4:
    st.info("**Metabolic Status:**\nOxygen Debt is accumulating. Lactate clearance failure.")

# 5. Logic Inspection
with st.expander("Show Raw Data Frame"):
    st.dataframe(df.iloc[max(0, idx-10):idx+1])
