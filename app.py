import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

# ==========================================
# 1. UX CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="TITAN | Precision CDS", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. DESIGN SYSTEM (Clinical Light & Clean)
# ==========================================
THEME = {
    # Semantic Colors
    "primary": "#0f172a",    # Dark Slate (Text)
    "secondary": "#64748b",  # Muted Slate (Labels)
    "accent": "#3b82f6",     # Brand Blue
    "success": "#10b981",    # Emerald
    "warning": "#f59e0b",    # Amber
    "danger": "#ef4444",     # Red
    "neutral": "#f1f5f9",    # Light Grey (Backgrounds)
    
    # Data Visualization Palette
    "vis_map": "#be185d",    # Pink/Magenta
    "vis_ci": "#059669",     # Green
    "vis_do2": "#7c3aed",    # Purple
    "vis_svr": "#d97706",    # Orange
    "vis_hr": "#0284c7",     # Blue
    "vis_ref": "#cbd5e1"     # Grey (Reference lines)
}

STYLING = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

:root { --bg: #f8fafc; --card: #ffffff; }

.stApp { background-color: var(--bg); font-family: 'Inter', sans-serif; color: #0f172a; }

/* Dashboard Cards */
.kpi-card {
    background: var(--card); border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: all 0.2s ease;
}
.kpi-card:hover { box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); transform: translateY(-1px); }

.kpi-label { 
    font-size: 0.75rem; font-weight: 600; color: #64748b; 
    text-transform: uppercase; letter-spacing: 0.05em; 
}
.kpi-value { 
    font-family: 'JetBrains Mono', monospace; font-size: 1.75rem; 
    font-weight: 700; color: #0f172a; letter-spacing: -0.03em; margin: 4px 0;
}
.kpi-unit { font-size: 0.85rem; color: #94a3b8; font-weight: 500; }

/* Status Banners */
.banner {
    padding: 12px 20px; border-radius: 8px; margin-bottom: 24px;
    border-left: 4px solid; display: flex; align-items: center; justify-content: space-between;
    background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.b-crit { border-color: #ef4444; background: #fef2f2; }
.b-warn { border-color: #f59e0b; background: #fffbeb; }
.b-ok { border-color: #10b981; background: #f0fdf4; }

/* Typography */
h1, h2, h3 { letter-spacing: -0.02em; color: #0f172a; }
.section-header {
    font-size: 0.95rem; font-weight: 700; color: #334155; 
    text-transform: uppercase; letter-spacing: 0.05em;
    border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; margin: 32px 0 16px 0;
}
</style>
"""

# ==========================================
# 3. PHYSIOLOGY ENGINE
# ==========================================

@dataclass
class PatientConfig:
    shock_type: str = "Septic"
    hb: float = 12.0       
    weight: float = 70.0   
    bsa: float = 1.8       
    map_target: float = 65.0
    compliance: float = 1.0 
    metabolic_rate: float = 1.0 
    hr_variability: float = 0.02 

class PhysiologyEngine:
    """
    0-D Cardiovascular Model.
    """
    def __init__(self, config: PatientConfig, preload=12.0, contractility=1.0, afterload=1.0):
        self.cfg = config
        self.preload_base = float(preload)
        self.contractility_base = float(contractility)
        self.afterload_base = float(afterload)

    def copy(self):
        return PhysiologyEngine(self.cfg, self.preload_base, self.contractility_base, self.afterload_base)

    def step(self, disease_severity=0.0, noise_val=0.0):
        sev = np.clip(disease_severity, 0.0, 1.0)
        
        # --- SHOCK PHENOTYPE LOGIC ---
        mod_preload, mod_contract, mod_afterload = 1.0, 1.0, 1.0
        
        if self.cfg.shock_type == "Septic":
            mod_afterload = max(0.25, 1.0 - 0.7 * sev)
            mod_preload = max(0.6, 1.0 - 0.3 * sev)
        elif self.cfg.shock_type == "Cardiogenic":
            mod_contract = max(0.2, 1.0 - 0.8 * sev)
            mod_afterload = 1.0 + (0.5 * sev)
        elif self.cfg.shock_type == "Hypovolemic":
            mod_preload = max(0.15, 1.0 - 0.9 * sev)
            mod_afterload = 1.0 + (0.6 * sev)
            mod_contract = 1.0 + (0.2 * sev)
        elif self.cfg.shock_type == "Obstructive":
            mod_preload = max(0.1, 1.0 - 0.9 * sev)
            mod_afterload = 1.0 + (0.4 * sev)

        # Apply Modifiers
        eff_preload = self.preload_base * mod_preload
        eff_afterload = self.afterload_base * mod_afterload
        eff_contract = self.contractility_base * mod_contract

        # --- MECHANICS ---
        sv_max = 100.0 * eff_contract
        k_const = 8.0
        noisy_preload = eff_preload * (1 + noise_val * 0.5)
        sv = sv_max * (noisy_preload**2 / (noisy_preload**2 + k_const**2))
        
        # --- BAROREFLEX ---
        map_est = (sv * 70.0 * eff_afterload * 0.05) + 5.0
        max_hr = 130.0 if self.cfg.shock_type == "Cardiogenic" else 170.0
        hr_drive = float(np.clip((85.0 - map_est) * 2.0, 0.0, max_hr - 60.0))
        hr = 60.0 + hr_drive + (noise_val * 50 * self.cfg.hr_variability)

        # --- HEMODYNAMICS ---
        co = (hr * sv) / 1000.0
        ci = co / self.cfg.bsa
        svr_dyne = 1200.0 * eff_afterload
        map_val = (co * svr_dyne / 80.0) + 5.0
        
        # Pulse Pressure
        pp_val = (sv / 1.5) * (1 / self.cfg.compliance)
        
        # --- OXYGEN ---
        spo2 = 0.98 if sev < 0.5 else max(0.85, 0.98 - (sev-0.5)*0.2)
        do2i = ci * self.cfg.hb * 1.34 * spo2 * 10.0
        
        # VO2
        vo2_demand = 110.0 * self.cfg.metabolic_rate * (1.0 + (0.5 * sev))
        vo2i = min(vo2_demand, do2i * 0.7)
        o2er = vo2i / do2i if do2i > 0 else 1.0
        
        # Lactate
        lac_gen = 0.0
        if o2er > 0.4: lac_gen += (o2er - 0.4) * 0.2
        if self.cfg.shock_type == "Septic": lac_gen += (sev * 0.03) 
        
        return {
            "HR": hr, "SV": sv, "CO": co, "CI": ci, "MAP": map_val, "SVR": svr_dyne, 
            "DO2I": do2i, "VO2I": vo2i, "O2ER": o2er, "PP": pp_val,
            "Lac_Gen": lac_gen, "Preload_Status": eff_preload
        }

def simulate_data(config: PatientConfig, mins=720, seed=42):
    rng = np.random.default_rng(seed)
    engine = PhysiologyEngine(config)
    
    x = np.linspace(-6, 6, mins)
    progression = 1 / (1 + np.exp(-x)) * 0.9
    noise_vector = rng.normal(0, config.hr_variability, mins)
    
    history = []
    curr_lac = 1.0
    
    for t in range(mins):
        state = engine.step(disease_severity=progression[t], noise_val=noise_vector[t])
        
        clearance = curr_lac * 0.005 
        curr_lac = curr_lac - clearance + state["Lac_Gen"]
        state["Lactate"] = max(0.5, curr_lac)
        
        map_dist = state["MAP"] - config.map_target
        uo = 1.0 / (1.0 + np.exp(-0.3 * map_dist)) * 1.5
        state["Urine"] = max(0.0, uo + rng.normal(0, 0.05))
        
        history.append(state)
        
    df = pd.DataFrame(history)
    df["SVRI"] = df["SVR"] * config.bsa
    return df

def predict_response(base_engine, current_sev, current_lac, horizon=60):
    times = np.arange(horizon)
    futures = {"Natural": {"mean": [], "upper": [], "lower": []}, 
               "Fluid": {"mean": [], "upper": [], "lower": []},
               "Pressor": {"mean": [], "upper": [], "lower": []}}
    
    # Note: Keys here must match keys used in visualization ("Natural", "Fluid", "Pressor")
    scenarios = [("Natural", base_engine.copy()), ("Fluid", base_engine.copy()), ("Pressor", base_engine.copy())]
    
    # Interventions
    scenarios[1][1].preload_base *= 1.4
    scenarios[2][1].afterload_base *= 1.5
    
    for name, eng in scenarios:
        vals = []
        for t in times:
            s = eng.step(current_sev + 0.001*t, noise_val=0.0)
            vals.append(s["MAP"])
        
        vals = np.array(vals)
        sigma = np.linspace(2, 8, horizon)
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

def _get_layout(height=250, title=None):
    layout = go.Layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=30 if title else 10, b=10),
        font=dict(family="Inter, sans-serif", color=THEME["primary"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor=THEME["neutral"], zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=THEME["neutral"], zeroline=False)
    )
    if title: layout.title = dict(text=f"<b>{title}</b>", font=dict(size=14))
    return layout

def plot_spark(df, col, color, thresh_l, thresh_h):
    data = df[col].iloc[-60:]
    fig = go.Figure()
    fig.add_shape(type="rect", x0=data.index[0], x1=data.index[-1], y0=thresh_l, y1=thresh_h,
                  fillcolor=THEME["neutral"], line_width=0, layer="below")
    fig.add_trace(go.Scatter(
        x=data.index, y=data.values, mode='lines',
        line=dict(color=color, width=2.5),
        fill='tozeroy', fillcolor=hex_to_rgba(color, 0.1),
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[data.index[-1]], y=[data.values[-1]], mode='markers',
        marker=dict(color=color, size=10, line=dict(width=2, color='white'))
    ))
    fig.update_layout(_get_layout(height=50))
    fig.update_xaxes(visible=False)
    y_min = min(data.min(), thresh_l) * 0.95
    y_max = max(data.max(), thresh_h) * 1.05
    fig.update_yaxes(visible=False, range=[y_min, y_max])
    return fig

def plot_horizon(futures, target):
    t = np.arange(len(futures["Natural"]["mean"]))
    fig = go.Figure()
    
    def add_scenario(name, color, data, dash=None):
        fig.add_trace(go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([data["upper"], data["lower"][::-1]]),
            fill='toself', fillcolor=hex_to_rgba(color, 0.1),
            line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=t, y=data["mean"], 
            line=dict(color=color, width=3, dash=dash), 
            name=name, hovertemplate="%{y:.0f} mmHg"
        ))

    add_scenario("Natural Course", THEME["secondary"], futures["Natural"], "dot")
    add_scenario("Fluid (+1L)", THEME["vis_ci"], futures["Fluid"])
    add_scenario("Pressor", THEME["vis_map"], futures["Pressor"])
    
    fig.add_hline(y=target, line_color=THEME["danger"], line_dash="solid", opacity=0.6)
    fig.add_annotation(x=5, y=target+2, text=f"TARGET MAP ({target})", 
                       font=dict(color=THEME["danger"], size=10), showarrow=False, align="left")
    
    fig.update_layout(_get_layout(height=300, title="Projected Response (30 min)"))
    fig.update_layout(legend=dict(orientation="h", y=1.1, x=0))
    fig.update_xaxes(title="Minutes Future", showgrid=False)
    fig.update_yaxes(title="MAP (mmHg)")
    return fig

def plot_compass(df, preds, curr_idx):
    start = max(0, curr_idx - 60)
    data = df.iloc[start:curr_idx]
    cur = df.iloc[curr_idx]
    fig = go.Figure()
    
    zones = [
        (0, 0, 2.5, 65, THEME["danger"], "CRITICAL SHOCK"),
        (2.5, 0, 6.0, 65, THEME["warning"], "VASOPLEGIA"),
        (2.5, 65, 6.0, 110, THEME["success"], "GOAL")
    ]
    for x0, y0, x1, y1, col, txt in zones:
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, 
                      fillcolor=hex_to_rgba(col, 0.1), line_width=0, layer="below")
        fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=txt, 
                           font=dict(color=col, size=10, weight="bold"), showarrow=False)

    fig.add_trace(go.Scatter(x=data["CI"], y=data["MAP"], mode="lines", 
                             line=dict(color=THEME["secondary"], width=2, dash="dot"), name="History"))
    fig.add_trace(go.Scatter(x=[cur["CI"]], y=[cur["MAP"]], mode="markers", 
                             marker=dict(color=THEME["primary"], size=14, symbol="cross", line=dict(width=2, color="white")), 
                             name="Current"))
    
    # Vector Visualization
    # Using heuristics + MAP prediction for vector endpoints
    target_f_map = preds["Fluid"]["mean"][-1]
    target_f_ci = cur["CI"] * 1.25 # Heuristic: Fluids increase CI
    
    target_p_map = preds["Pressor"]["mean"][-1]
    target_p_ci = cur["CI"] * 1.0 # Heuristic: Pressors maintain CI (mostly)
    
    fig.add_annotation(x=target_f_ci, y=target_f_map, ax=cur["CI"], ay=cur["MAP"],
                       xref="x", yref="y", axref="x", ayref="y", arrowhead=2, arrowcolor=THEME["vis_ci"], 
                       text="FLUID", font=dict(color=THEME["vis_ci"], weight="bold"))
    
    fig.add_annotation(x=target_p_ci, y=target_p_map, ax=cur["CI"], ay=cur["MAP"],
                       xref="x", yref="y", axref="x", ayref="y", arrowhead=2, arrowcolor=THEME["vis_map"], 
                       text="PRESSOR", font=dict(color=THEME["vis_map"], weight="bold"))

    fig.update_layout(_get_layout(height=300, title="Hemodynamic Compass"))
    fig.update_xaxes(title="Cardiac Index (L/min/m²)", range=[1.0, 6.0])
    fig.update_yaxes(title="MAP (mmHg)", range=[30, 110])
    fig.update_layout(showlegend=False)
    return fig

def plot_starling(df, curr_idx):
    data = df.iloc[max(0, curr_idx-30):curr_idx]
    fig = go.Figure()
    x = np.linspace(0, 20, 100); y = np.log(x+1)*30
    fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=THEME["vis_ref"], dash='dot'), name='Ideal'))
    fig.add_trace(go.Scatter(x=data['Preload_Status'], y=data['SV'], mode='lines', 
                             line=dict(color=THEME["vis_ci"], width=3)))
    fig.add_trace(go.Scatter(x=[data['Preload_Status'].iloc[-1]], y=[data['SV'].iloc[-1]], 
                             mode='markers', marker=dict(color=THEME["vis_ci"], size=10)))
    
    d_sv = data['SV'].iloc[-1] - data['SV'].iloc[0]
    slope = d_sv / (data['Preload_Status'].iloc[-1] - data['Preload_Status'].iloc[0] + 0.01)
    status = "RESPONSIVE" if slope > 2.0 else "NON-RESPONDER"
    col = THEME["success"] if slope > 2.0 else THEME["danger"]
    fig.add_annotation(x=10, y=10, text=status, font=dict(color=col, weight="bold"), showarrow=False)

    fig.update_layout(_get_layout(height=250, title="Fluid Responsiveness"))
    fig.update_xaxes(title="Preload (CVP)")
    fig.update_yaxes(title="Stroke Volume (mL)")
    fig.update_layout(showlegend=False)
    return fig

def plot_oxygen(df, curr_idx):
    start = max(0, curr_idx - 180)
    data = df.iloc[start:curr_idx]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=data.index, y=data['DO2I'], fill='tozeroy', 
                             fillcolor=hex_to_rgba(THEME["vis_do2"], 0.2),
                             line=dict(color=THEME["vis_do2"]), name='Delivery'), secondary_y=False)
    fig.add_trace(go.Scatter(x=data.index, y=data['Lactate'], 
                             line=dict(color=THEME["danger"], width=2), name='Lactate'), secondary_y=True)
    fig.update_layout(_get_layout(height=250, title="Oxygen Kinetics"))
    fig.update_yaxes(title="DO2I (mL/m²)", secondary_y=False)
    fig.update_yaxes(title="Lactate", secondary_y=True, showgrid=False)
    fig.update_layout(showlegend=False)
    return fig

# ==========================================
# 5. MAIN APP EXECUTION
# ==========================================

st.markdown(STYLING, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("TITAN | Control")
    curr_time = st.slider("Timeline", 60, 720, 720)
    
    st.markdown("### ⚙️ Physiology")
    shock = st.selectbox("Shock Type", ["Septic", "Cardiogenic", "Hypovolemic", "Obstructive"])
    hb = st.slider("Hemoglobin", 7.0, 15.0, 12.0, 0.5, help="Affects DO2I calculation")
    map_target = st.number_input("Target MAP", 55, 85, 65)
    
    config = PatientConfig(shock_type=shock, hb=hb, map_target=map_target)

# --- SIMULATION ---
df = simulate_data(config, mins=720, seed=42)
idx = curr_time - 1
row = df.iloc[idx]
prev = df.iloc[idx-15]

# --- 1. HEADER ---
status = "STABLE"
action = "Routine Monitoring"
style = "b-ok"

if row["Lactate"] > 2.0:
    status = "HYPOPERFUSION DETECTED"
    action = "Assess Volume Status & DO2"
    style = "b-warn"

if row["MAP"] < map_target:
    status = f"{shock.upper()} SHOCK"
    style = "b-crit"
    if shock == "Septic": action = "PROTOCOL: Vasopressor (Norepinephrine)"
    elif shock == "Cardiogenic": action = "PROTOCOL: Inotrope (Dobutamine)"
    elif shock == "Hypovolemic": action = "PROTOCOL: Rapid Volume Expansion"
    else: action = "PROTOCOL: Relieve Obstruction"

st.markdown(f"""
<div class="banner {style}">
    <div>
        <div style="font-weight:800; font-size:1.25rem;">{status}</div>
        <div style="color:#64748b; font-weight:500;">Target: MAP > {map_target} • Lactate < 2.0</div>
    </div>
    <div style="text-align:right;">
        <div style="font-size:0.75rem; font-weight:700; color:#64748b;">RECOMMENDATION</div>
        <div style="font-weight:700; font-size:1.1rem;">{action}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 2. KPI GRID ---
cols = st.columns(6)

def kpi(col, label, val, unit, color, df_col, l, h, key):
    d = val - prev[df_col]
    color_trend = THEME["danger"] if (val < l or val > h) else THEME["success"]
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="border-top:3px solid {color}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val:.1f} <span class="kpi-unit">{unit}</span></div>
            <div class="kpi-trend" style="color:{color_trend}">
                {d:+.1f} <span style="color:#94a3b8; font-weight:400; font-size:0.7rem; margin-left:4px">15m</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(plot_spark(df.iloc[:idx+1], df_col, color, l, h), 
                        use_container_width=True, config={'displayModeBar': False}, key=key)

kpi(cols[0], "MAP", row["MAP"], "mmHg", THEME["vis_map"], "MAP", map_target, 110, "k1")
kpi(cols[1], "Cardiac Index", row["CI"], "L/min", THEME["vis_ci"], "CI", 2.5, 4.2, "k2")
kpi(cols[2], "SVR Index", row["SVRI"], "dyn", THEME["vis_svr"], "SVRI", 800, 1200, "k3")
kpi(cols[3], "Stroke Vol", row["PP"], "mL", THEME["vis_hr"], "PP", 35, 60, "k4")
kpi(cols[4], "DO2 Index", row["DO2I"], "mL/m²", THEME["vis_do2"], "DO2I", 400, 600, "k5")
kpi(cols[5], "Lactate", row["Lactate"], "mmol", THEME["danger"], "Lactate", 0, 2.0, "k6")

# --- 3. PREDICTIVE LAYER ---
st.markdown('<div class="section-header">🔮 Predictive Decision Support</div>', unsafe_allow_html=True)
c1, c2 = st.columns([1, 1])

engine = PhysiologyEngine(config)
futures = predict_response(engine, 0.9 * (idx/720), row["Lactate"])

with c1:
    st.plotly_chart(plot_compass(df, futures, idx), use_container_width=True, config={'displayModeBar': False})
with c2:
    st.plotly_chart(plot_horizon(futures, map_target), use_container_width=True, config={'displayModeBar': False})

# --- 4. ORGAN LAYER ---
st.markdown('<div class="section-header">🫀 Physiological Mechanics</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)

with c3:
    st.plotly_chart(plot_oxygen(df, idx), use_container_width=True, config={'displayModeBar': False})
with c4:
    st.plotly_chart(plot_starling(df, idx), use_container_width=True, config={'displayModeBar': False})

# Disclaimer
st.markdown("""
<div style="background-color:#fef3c7; border:1px solid #fcd34d; color:#92400e; padding:12px; border-radius:6px; font-size:0.8rem; margin-top:30px;">
    <strong>⚠️ CAUTION: SIMULATION ONLY</strong><br>
    This dashboard is a demonstration of Clinical Decision Support (CDS) capabilities using a 0-D cardiovascular model. 
    It is not a medical device and should not be used for patient care.
</div>
""", unsafe_allow_html=True)
