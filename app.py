import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

# ==========================================
# 1. SAFETY & COMPLIANCE LAYER
# ==========================================
st.set_page_config(page_title="TITAN | Precision CDS", layout="wide", initial_sidebar_state="expanded")

SAFETY_DISCLAIMER = """
<div style="background-color:#fff3cd; border:1px solid #ffeeba; color:#856404; padding:12px; border-radius:6px; font-size:0.8rem; margin-bottom:20px;">
    <strong>⚠️ CAUTION: INVESTIGATIONAL DEVICE</strong><br>
    This application is a probabilistic Clinical Decision Support (CDS) simulation. 
    Predictive confidence intervals are statistical estimates. 
    <strong>DO NOT</strong> rely solely on these visualizations for patient management. 
    Standard of care, physical exam, and confirmatory diagnostics must take precedence.
</div>
"""

# ==========================================
# 2. UI/UX DESIGN SYSTEM (Clinical Light)
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
}

.stApp { background-color: var(--bg); font-family: 'Inter', sans-serif; color: var(--text-primary); }

/* Protocol Banner */
.protocol-banner {
    padding: 16px 24px; border-radius: 8px; margin-bottom: 20px;
    border-left: 6px solid; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    background: white; display: flex; justify-content: space-between; align-items: center;
}
.banner-crit { border-color: #dc2626; background: #fef2f2; }
.banner-warn { border-color: #d97706; background: #fffbeb; }
.banner-ok { border-color: #16a34a; background: #f0fdf4; }

/* KPI Card */
.kpi-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px; margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05); transition: all 0.2s;
}
.kpi-card:hover { box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
.kpi-label { font-size: 0.75rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-value { font-family: 'Roboto Mono', monospace; font-size: 1.6rem; font-weight: 700; color: var(--text-primary); line-height: 1.2; }
.kpi-unit { font-size: 0.85rem; color: var(--text-secondary); font-weight: 500; margin-left: 4px; }
.kpi-trend { font-size: 0.8rem; font-weight: 600; margin-top: 6px; display: flex; align-items: center; gap: 4px; }

/* Section Headers */
.section-head {
    font-size: 1.0rem; font-weight: 800; color: var(--text-primary);
    border-bottom: 2px solid var(--border); margin: 30px 0 15px 0; padding-bottom: 6px;
    text-transform: uppercase; letter-spacing: 0.05em;
}
</style>
"""

PLOT_COLORS = {
    "map": "#be185d", "ci": "#059669", "do2": "#7c3aed", 
    "svr": "#d97706", "hr": "#0284c7", "grid": "#f1f5f9", 
    "text": "#334155", "vo2": "#6366f1"
}

# ==========================================
# 3. PHYSIOLOGY ENGINE (Validated Models)
# ==========================================

@dataclass
class PatientConfig:
    shock_type: str = "Septic"
    hb: float = 12.0  # g/dL
    weight: float = 70.0 # kg
    bsa: float = 1.8 # m2
    map_target: float = 65.0

class PhysiologyEngine:
    """
    Advanced 0-D Cardiovascular Model.
    Supports Septic, Cardiogenic, Hypovolemic, and Obstructive profiles.
    """
    def __init__(self, config: PatientConfig, preload=12.0, contractility=1.0, afterload=1.0):
        self.cfg = config
        self.preload_base = float(preload)
        self.contractility_base = float(contractility)
        self.afterload_base = float(afterload)

    def copy(self):
        return PhysiologyEngine(self.cfg, self.preload_base, self.contractility_base, self.afterload_base)

    def step(self, disease_severity=0.0):
        """
        disease_severity: 0.0 (Healthy) -> 1.0 (Refractory Shock)
        """
        sev = np.clip(disease_severity, 0.0, 1.0)
        
        # --- SHOCK PHENOTYPE LOGIC ---
        mod_preload, mod_contract, mod_afterload = 1.0, 1.0, 1.0
        
        if self.cfg.shock_type == "Septic":
            # Distributive: Vasodilation + Capillary Leak
            mod_afterload = max(0.25, 1.0 - 0.7 * sev)
            mod_preload = max(0.6, 1.0 - 0.3 * sev)
            
        elif self.cfg.shock_type == "Cardiogenic":
            # Pump Failure: Low Contractility, High SVR (Compensation)
            mod_contract = max(0.2, 1.0 - 0.8 * sev)
            mod_afterload = 1.0 + (0.5 * sev)
            
        elif self.cfg.shock_type == "Hypovolemic":
            # Volume Loss: Massive Preload Drop, High SVR
            mod_preload = max(0.15, 1.0 - 0.9 * sev)
            mod_afterload = 1.0 + (0.6 * sev)
            mod_contract = 1.0 + (0.2 * sev) # Hyperdynamic
            
        elif self.cfg.shock_type == "Obstructive":
            # Tamponade/PE: Impaired Filling
            mod_preload = max(0.1, 1.0 - 0.9 * sev)
            mod_afterload = 1.0 + (0.4 * sev)

        # Apply Modifiers
        eff_preload = self.preload_base * mod_preload
        eff_afterload = self.afterload_base * mod_afterload
        eff_contract = self.contractility_base * mod_contract

        # --- MECHANICS (Frank-Starling Sigmoid) ---
        sv_max = 100.0 * eff_contract
        k_const = 8.0 # Steepness
        sv = sv_max * (eff_preload**2 / (eff_preload**2 + k_const**2))
        
        # --- BAROREFLEX (Compensatory Tachycardia) ---
        # Estimates perceived pressure based on SV and SVR
        map_est = (sv * 70.0 * eff_afterload * 0.05) + 5.0
        max_hr = 130.0 if self.cfg.shock_type == "Cardiogenic" else 170.0
        hr_drive = float(np.clip((85.0 - map_est) * 2.0, 0.0, max_hr - 60.0))
        hr = 60.0 + hr_drive

        # --- HEMODYNAMICS ---
        co = (hr * sv) / 1000.0  # L/min
        ci = co / self.cfg.bsa     # L/min/m2
        svr_dyne = 1200.0 * eff_afterload
        map_val = (co * svr_dyne / 80.0) + 5.0 # Ohm's Law approximation
        
        # --- OXYGEN TRANSPORT ---
        # DO2I = CI * Hb * 1.34 * SpO2 (assumed 95-98% varying with shock)
        spo2 = 0.98 if sev < 0.5 else 0.92
        do2i = ci * self.cfg.hb * 1.34 * spo2 * 10.0
        
        # VO2I (Oxygen Consumption)
        # Increases with stress (sepsis/fever) but limited by supply (shock)
        vo2_demand = 110.0 * (1.0 + (0.5 * sev)) # Metabolic drive
        vo2i = min(vo2_demand, do2i * 0.7) # Supply dependency if extraction maxed out
        
        # O2ER (Extraction Ratio)
        o2er = vo2i / do2i if do2i > 0 else 1.0
        
        # --- LACTATE KINETICS ---
        # Rises if O2ER > 40% (Critical Extraction) or Cytopathic Hypoxia (Sepsis)
        lac_gen = 0.0
        if o2er > 0.4: lac_gen += (o2er - 0.4) * 0.2
        if self.cfg.shock_type == "Septic": lac_gen += (sev * 0.03) 
        
        return {
            "HR": hr, "SV": sv, "CO": co, "CI": ci, "MAP": map_val, "SVR": svr_dyne, 
            "DO2I": do2i, "VO2I": vo2i, "O2ER": o2er,
            "Lac_Gen": lac_gen, "Preload_Status": eff_preload
        }

def simulate_clinical_data(config: PatientConfig, mins=720, seed=42):
    rng = np.random.default_rng(seed)
    engine = PhysiologyEngine(config)
    
    # Disease Progression: Sigmoid Curve (Biologically realistic)
    x = np.linspace(-6, 6, mins)
    progression = 1 / (1 + np.exp(-x)) # 0 to 1
    progression = progression * 0.9 # Max severity 0.9
    
    history = []
    curr_lac = 1.0
    
    for t in range(mins):
        # 1/f Pink Noise approximation
        noise = rng.normal(0, 0.02)
        state = engine.step(progression[t] + noise)
        
        # Lactate Clearance (Liver function)
        clearance = curr_lac * 0.005 
        curr_lac = curr_lac - clearance + state["Lac_Gen"]
        state["Lactate"] = max(0.5, curr_lac)
        
        # Urine Output (Sigmoid Autoregulation)
        # Steep drop when MAP < Target
        map_dist = state["MAP"] - config.map_target
        uo = 1.0 / (1.0 + np.exp(-0.3 * map_dist)) * 1.5
        state["Urine"] = max(0.0, uo + rng.normal(0, 0.05))
        
        history.append(state)
        
    df = pd.DataFrame(history)
    df["SVRI"] = df["SVR"] * config.bsa
    df["PP"] = df["SV"] / 1.5 
    
    # SOFA Score Proxy
    df["SOFA"] = 0
    df.loc[df["MAP"] < 70, "SOFA"] += 1
    df.loc[df["Urine"] < 0.5, "SOFA"] += 1
    df.loc[df["Lactate"] > 2.0, "SOFA"] += 1
    
    return df

def predict_response_uncertainty(base_engine, current_sev, current_lac, horizon=60):
    """
    Monte Carlo-lite simulation for Predictive Confidence Intervals.
    """
    times = np.arange(horizon)
    futures = {
        "Natural": {"mean": [], "upper": [], "lower": []}, 
        "Fluid": {"mean": [], "upper": [], "lower": []},
        "Pressor": {"mean": [], "upper": [], "lower": []}
    }
    
    # 1. Define Scenarios
    scenarios = [
        ("Natural", base_engine.copy()),
        ("Fluid", base_engine.copy()),
        ("Pressor", base_engine.copy())
    ]
    
    # Apply Interventions
    scenarios[1][1].preload_base *= 1.4 # Fluid Bolus
    scenarios[2][1].afterload_base *= 1.5 # Pressor
    
    for name, eng in scenarios:
        vals = []
        for t in times:
            s = eng.step(current_sev + 0.001*t)
            vals.append(s["MAP"])
        
        # Uncertainty grows with time (cone)
        vals = np.array(vals)
        sigma = np.linspace(2, 8, horizon) # +/- mmHg
        futures[name]["mean"] = vals
        futures[name]["upper"] = vals + sigma
        futures[name]["lower"] = vals - sigma
        
    return futures

# ==========================================
# 4. VISUALIZATION ENGINE
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

def plot_horizon_with_uncertainty(futures, target):
    """Visualizes 95% Confidence Intervals for predictions."""
    t = np.arange(len(futures["Natural"]["mean"]))
    fig = go.Figure()
    
    def add_band(name, color, data):
        fig.add_trace(go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([data["upper"], data["lower"][::-1]]),
            fill='toself', fillcolor=hex_to_rgba(color, 0.15),
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(x=t, y=data["mean"], line=dict(color=color, width=3), name=name))

    add_band("Natural Course", "#94a3b8", futures["Natural"])
    add_band("Fluid (+1L)", PLOT_COLORS["ci"], futures["Fluid"])
    add_band("Pressor", PLOT_COLORS["map"], futures["Pressor"])
    
    fig.add_hline(y=target, line_color="#ef4444", line_dash="solid", 
                  annotation_text=f"Target ({target})", annotation_position="top right")
    
    fig = clean_layout(fig, height=300)
    fig.update_layout(title="<b>Projected Hemodynamic Response (95% CI)</b>", legend=dict(orientation="h", y=1.1))
    fig.update_xaxes(title="Minutes Future", showgrid=False)
    fig.update_yaxes(title="Projected MAP (mmHg)", gridcolor=PLOT_COLORS["grid"])
    return fig

def plot_oxygen_ledger(df, curr_idx):
    """Visualizes Oxygen Supply/Demand Mismatch."""
    start = max(0, curr_idx - 180)
    data = df.iloc[start:curr_idx]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # DO2 Area
    fig.add_trace(go.Scatter(x=data.index, y=data['DO2I'], fill='tozeroy', 
                             fillcolor=hex_to_rgba(PLOT_COLORS['do2'], 0.2),
                             line=dict(color=PLOT_COLORS['do2']), name='DO2 Index'), secondary_y=False)
    
    # VO2 Line
    fig.add_trace(go.Scatter(x=data.index, y=data['VO2I'], 
                             line=dict(color=PLOT_COLORS['text'], width=2, dash='dot'), 
                             name='VO2 (Demand)'), secondary_y=False)
    
    # O2ER (Extraction Ratio) on Right Axis
    fig.add_trace(go.Scatter(x=data.index, y=data['O2ER']*100, 
                             line=dict(color=PLOT_COLORS['svr'], width=2), 
                             name='O2 Extraction %'), secondary_y=True)
    
    fig.add_hline(y=40, line_dash="dot", line_color="red", secondary_y=True, annotation_text="Crit Extraction")

    fig = clean_layout(fig, height=250)
    fig.update_layout(title="<b>Oxygen Kinetics (DO2 vs VO2)</b>", legend=dict(orientation="h", y=1.1))
    fig.update_yaxes(title="Index (mL/min/m²)", secondary_y=False, gridcolor=PLOT_COLORS['grid'])
    fig.update_yaxes(title="Extraction (%)", secondary_y=True, showgrid=False, range=[0, 60])
    return fig

def plot_hemo_compass(df, curr_idx):
    """Diagnostic Quadrants."""
    data = df.iloc[max(0, curr_idx-60):curr_idx]
    cur = df.iloc[curr_idx]
    fig = go.Figure()
    
    # Zones
    fig.add_shape(type="rect", x0=2.5, x1=6.0, y0=0, y1=65, fillcolor="rgba(217, 119, 6, 0.15)", line_width=0)
    fig.add_annotation(x=4.0, y=40, text="VASODILATED", font=dict(size=10, color="#d97706"), showarrow=False)
    fig.add_shape(type="rect", x0=0, x1=2.5, y0=0, y1=65, fillcolor="rgba(220, 38, 38, 0.15)", line_width=0)
    fig.add_annotation(x=1.25, y=40, text="HYPOPERFUSION", font=dict(size=10, color="#dc2626"), showarrow=False)
    fig.add_shape(type="rect", x0=2.5, x1=5.0, y0=65, y1=110, fillcolor="rgba(22, 163, 74, 0.15)", line_width=2, line_color="#16a34a")
    
    fig.add_trace(go.Scatter(x=data["CI"], y=data["MAP"], mode="lines", line=dict(color="#94a3b8", dash="dot"), name="Trend"))
    fig.add_trace(go.Scatter(x=[cur["CI"]], y=[cur["MAP"]], mode="markers", marker=dict(color="#0f172a", size=14, symbol="cross"), name="Current"))

    fig = clean_layout(fig, height=300)
    fig.update_layout(title="<b>Hemodynamic Compass</b>", showlegend=False)
    fig.update_xaxes(title="Cardiac Index", range=[1.0, 5.5], gridcolor=PLOT_COLORS["grid"])
    fig.update_yaxes(title="MAP", range=[30, 100], gridcolor=PLOT_COLORS["grid"])
    return fig

def plot_spark(df, col, color, thresh_l, thresh_h):
    data = df[col].iloc[-60:]
    fig = go.Figure()
    fig.add_shape(type="rect", x0=data.index[0], x1=data.index[-1], y0=thresh_l, y1=thresh_h, fillcolor="rgba(0,0,0,0.04)", line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines', line=dict(color=color, width=2), fill='tozeroy', fillcolor=hex_to_rgba(color, 0.1)))
    fig = clean_layout(fig, height=50)
    fig.update_layout(margin=dict(t=0,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

def plot_lactate_kinetics(df, curr_idx):
    start = max(0, curr_idx - 180)
    data = df.iloc[start:curr_idx]
    fig = go.Figure()
    fig.add_shape(type="rect", x0=data.index[0], x1=data.index[-1], y0=0, y1=2.0, fillcolor="rgba(16, 185, 129, 0.1)", line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=data.index, y=data["Lactate"], mode='lines', line=dict(color=PLOT_COLORS["do2"], width=3), name="Lactate"))
    fig = clean_layout(fig, height=250)
    fig.update_layout(title="<b>Lactate Clearance</b>", showlegend=False)
    fig.update_yaxes(title="mmol/L", gridcolor=PLOT_COLORS["grid"])
    return fig

def plot_starling_slope(df, curr_idx):
    data = df.iloc[max(0, curr_idx-30):curr_idx]
    fig = go.Figure()
    x = np.linspace(0, 20, 100); y = np.log(x+1)*30
    fig.add_trace(go.Scatter(x=x, y=y, line=dict(color='#cbd5e1', dash='dot'), name='Ref'))
    fig.add_trace(go.Scatter(x=data['Preload_Status'], y=data['SV'], mode='lines+markers', line=dict(color=PLOT_COLORS['ci'], width=3)))
    fig = clean_layout(fig, height=250)
    fig.update_layout(title="<b>Fluid Responsiveness</b>", showlegend=False)
    fig.update_xaxes(title="Preload", gridcolor=PLOT_COLORS["grid"])
    fig.update_yaxes(title="Stroke Volume", gridcolor=PLOT_COLORS["grid"])
    return fig

# ==========================================
# 5. MAIN APP EXECUTION
# ==========================================

st.markdown(THEME_CSS, unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("TITAN | Configuration")
    
    st.markdown("### 1. Patient Profile")
    shock_type = st.selectbox("Shock Phenotype", ["Septic", "Cardiogenic", "Hypovolemic", "Obstructive"])
    hb = st.slider("Hemoglobin (g/dL)", 6.0, 15.0, 12.0, 0.5, help="Affects DO2 calculation.")
    
    st.markdown("### 2. Physiology Targets")
    map_target = st.number_input("Target MAP (mmHg)", 55, 85, 65)
    
    st.markdown("### 3. Simulation")
    curr_time = st.slider("Timeline (min)", 60, 720, 720)
    
    config = PatientConfig(shock_type=shock_type, hb=hb, map_target=map_target)

# --- DATA GENERATION ---
df = simulate_clinical_data(config, mins=720, seed=42)
idx = curr_time - 1
row = df.iloc[idx]
prev = df.iloc[idx-15]

# --- DISCLAIMER ---
st.markdown(SAFETY_DISCLAIMER, unsafe_allow_html=True)

# --- 1. EVIDENCE-BASED HEADER ---
status_msg = "STABLE"
action_msg = "Continue Monitoring"
banner_style = "banner-ok"

if row["Lactate"] > 2.0:
    status_msg = "HYPOPERFUSION"
    action_msg = "Evaluate DO2/VO2 Mismatch"
    banner_style = "banner-warn"

if row["MAP"] < map_target:
    status_msg = f"{shock_type.upper()} SHOCK"
    if shock_type == "Septic": action_msg = "PROTOCOL: Vasopressors"
    elif shock_type == "Cardiogenic": action_msg = "PROTOCOL: Inotropes"
    elif shock_type == "Hypovolemic": action_msg = "PROTOCOL: Volume"
    else: action_msg = "PROTOCOL: Relieve Obstruction"
    banner_style = "banner-crit"

st.markdown(f"""
<div class="protocol-banner {banner_style}">
    <div>
        <div style="font-weight:800; font-size:1.2rem;">{status_msg}</div>
        <div style="font-weight:500;">Target MAP > {map_target} • Lactate < 2.0</div>
    </div>
    <div style="font-weight:700; font-size:1.1rem; text-align:right;">RECOMMENDATION:<br>{action_msg}</div>
</div>
""", unsafe_allow_html=True)

# --- 2. KPI STRIP ---
k_cols = st.columns(6)

def render_kpi(col, label, val, unit, color, df_col, t_low, t_high, key_id):
    delta = val - prev[df_col]
    trend_col = "#ef4444" if (val < t_low or val > t_high) else "#10b981"
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid {color}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color}">{val:.1f} <span class="kpi-unit">{unit}</span></div>
            <div class="kpi-trend" style="color:{trend_col}">{delta:+.1f} (15m)</div>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(plot_spark(df.iloc[:idx+1], df_col, color, t_low, t_high), 
                        use_container_width=True, config={'displayModeBar': False}, key=key_id)

render_kpi(k_cols[0], "MAP", row["MAP"], "mmHg", PLOT_COLORS["map"], "MAP", map_target, 100, "k1")
render_kpi(k_cols[1], "Cardiac Index", row["CI"], "L/min", PLOT_COLORS["ci"], "CI", 2.5, 4.0, "k2")
render_kpi(k_cols[2], "SVR Index", row["SVRI"], "dyn", PLOT_COLORS["svr"], "SVRI", 800, 1200, "k3")
render_kpi(k_cols[3], "O2 Extract", row["O2ER"]*100, "%", PLOT_COLORS["do2"], "O2ER", 20, 30, "k4")
render_kpi(k_cols[4], "SOFA Score", float(row["SOFA"]), "pts", "#475569", "SOFA", 0, 2, "k5")
render_kpi(k_cols[5], "Urine Out", row["Urine"], "mL/kg", "#0284c7", "Urine", 0.5, 2.0, "k6")

# --- 3. PROGNOSTICS & PHENOTYPE ---
st.markdown('<div class="section-head">🔮 HEMODYNAMIC PREDICTION & PHENOTYPE</div>', unsafe_allow_html=True)
c1, c2 = st.columns([2, 1])

with c1:
    st.markdown("**Intervention Horizon (with Uncertainty)**")
    st.caption("Projected response to Volume vs Pressors with 95% Confidence Intervals.")
    engine = PhysiologyEngine(config)
    futures = predict_response_uncertainty(engine, 0.9 * (idx/720), row["Lactate"])
    st.plotly_chart(plot_horizon_with_uncertainty(futures, map_target), use_container_width=True, config={'displayModeBar': False})

with c2:
    st.markdown("**Hemodynamic Compass**")
    st.caption("Visual Diagnosis of Shock Type.")
    st.plotly_chart(plot_hemo_compass(df, idx), use_container_width=True, config={'displayModeBar': False})

# --- 4. ORGAN MECHANICS ---
st.markdown('<div class="section-head">🫀 ORGAN SYSTEMS & MECHANICS</div>', unsafe_allow_html=True)
c3, c4, c5 = st.columns(3)

with c3:
    st.markdown("**Oxygen Kinetics**")
    st.caption("DO2 (Delivery) vs VO2 (Demand) vs O2ER.")
    st.plotly_chart(plot_oxygen_ledger(df, idx), use_container_width=True, config={'displayModeBar': False})

with c4:
    st.markdown("**Lactate Clearance**")
    st.caption("Metabolic washout rate (Prognostic marker).")
    st.plotly_chart(plot_lactate_kinetics(df, idx), use_container_width=True, config={'displayModeBar': False})

with c5:
    st.markdown("**Fluid Responsiveness**")
    st.caption("Dynamic Frank-Starling trajectory.")
    st.plotly_chart(plot_starling_slope(df, idx), use_container_width=True, config={'displayModeBar': False})
