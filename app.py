import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from scipy.signal import welch

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
st.set_page_config(
    page_title="TITAN | Precision CDS", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Clinical Thresholds (Constants)
THRESHOLDS = {
    "MAP_TARGET": 65.0,
    "MAP_CRIT": 55.0,
    "LACTATE_HIGH": 2.0,
    "CI_LOW": 2.2,
    "SI_WARN": 0.9,
    "URINE_LOW": 0.5
}

# Theme Definition
THEME = {
    "bg": "#f8fafc",
    "primary": "#0f172a",
    "grid": "#e2e8f0",
    "map": "#be185d", 
    "ci": "#059669", 
    "do2": "#7c3aed", 
    "svr": "#d97706", 
    "hr": "#0284c7", 
    "text": "#334155",
    "crit": "#ef4444", 
    "warn": "#f59e0b", 
    "ok": "#10b981",
    "neutral": "#f1f5f9"
}

# Corrected Variable Name
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@500&display=swap');

:root { --bg: #f8fafc; --card: #ffffff; }
.stApp { background-color: var(--bg); font-family: 'Inter', sans-serif; color: #0f172a; }

/* KPI Card */
.kpi-card {
    background: var(--card); border: 1px solid #cbd5e1; border-radius: 8px;
    padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 8px;
    transition: all 0.2s ease;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 4px 6px rgba(0,0,0,0.05); }

.kpi-lbl { 
    font-size: 0.7rem; font-weight: 700; color: #64748b; 
    text-transform: uppercase; letter-spacing: 0.05em; 
}
.kpi-val { 
    font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; 
    font-weight: 700; color: #0f172a; line-height: 1.1; 
}
.kpi-unit { font-size: 0.8rem; color: #94a3b8; margin-left: 2px; }

/* Alert Banner */
.banner {
    padding: 16px; border-radius: 8px; border-left: 6px solid;
    background: white; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 24px;
}
.b-crit { border-color: #ef4444; background: #fef2f2; }
.b-warn { border-color: #f59e0b; background: #fffbeb; }
.b-ok { border-color: #10b981; background: #f0fdf4; }

.section-head {
    font-size: 1.0rem; font-weight: 800; color: #1e293b;
    border-bottom: 2px solid #e2e8f0; margin: 30px 0 15px 0; padding-bottom: 6px;
    text-transform: uppercase; letter-spacing: 0.05em;
}
</style>
"""

SAFETY_DISCLAIMER = """
<div style="background-color:#fff3cd; border:1px solid #ffeeba; color:#856404; padding:12px; border-radius:6px; font-size:0.85rem; margin-top:20px;">
    <strong>⚠️ INVESTIGATIONAL DEVICE PRECAUTION</strong><br>
    This dashboard is a simulation-based Clinical Decision Support (CDS) prototype. 
    Predictive confidence intervals are based on stochastic modeling and may not reflect individual patient idiosyncrasies.
    <strong>Standard clinical judgment must take precedence.</strong>
</div>
"""

# ==========================================
# 2. DATA MODELS
# ==========================================

@dataclass
class PatientConfig:
    """
    Parameter container for the physiology engine.
    """
    shock_type: str = "Septic"
    hb: float = 12.0       
    weight: float = 70.0   
    bsa: float = 1.8       
    map_target: float = 65.0
    compliance: float = 1.0 
    metabolic_rate: float = 1.0 
    hr_variability: float = 0.02

    def __post_init__(self):
        if self.hb < 3.0 or self.hb > 20.0: raise ValueError("Hemoglobin out of physiological range")
        if self.map_target < 40 or self.map_target > 100: raise ValueError("Target MAP unsafe")

# ==========================================
# 3. PHYSIOLOGY ENGINE (Physics)
# ==========================================

class PhysiologyEngine:
    """
    Modular 0-D Cardiovascular Model.
    """
    def __init__(self, config: PatientConfig, preload=12.0, contractility=1.0, afterload=1.0):
        self.cfg = config
        self.preload = float(preload)
        self.contractility = float(contractility)
        self.afterload = float(afterload)
        
        # Internal Constants
        self.MAX_HR = 180.0
        self.MIN_HR = 40.0
        self.SV_MAX_BASE = 100.0
        self.STARLING_K = 8.0

    def copy(self):
        return PhysiologyEngine(self.cfg, self.preload, self.contractility, self.afterload)

    def _apply_shock_modifiers(self, sev: float) -> Tuple[float, float, float]:
        mod_pre, mod_con, mod_aft = 1.0, 1.0, 1.0
        if self.cfg.shock_type == "Septic":
            mod_aft = max(0.25, 1.0 - 0.7 * sev)
            mod_pre = max(0.6, 1.0 - 0.3 * sev)
        elif self.cfg.shock_type == "Cardiogenic":
            mod_con = max(0.2, 1.0 - 0.8 * sev)
            mod_aft = 1.0 + (0.5 * sev)
        elif self.cfg.shock_type == "Hypovolemic":
            mod_pre = max(0.15, 1.0 - 0.9 * sev)
            mod_con = 1.0 + (0.2 * sev)
            mod_aft = 1.0 + (0.4 * sev)
        elif self.cfg.shock_type == "Obstructive":
            mod_pre = max(0.1, 1.0 - 0.9 * sev)
            mod_aft = 1.0 + (0.4 * sev)
        return (self.preload * mod_pre, self.contractility * mod_con, self.afterload * mod_aft)

    def _calc_mechanics(self, eff_preload: float, eff_contract: float, noise: float) -> float:
        sv_max = self.SV_MAX_BASE * eff_contract
        noisy_preload = eff_preload * (1 + noise * 0.5)
        sv = sv_max * (noisy_preload**2 / (noisy_preload**2 + self.STARLING_K**2))
        return sv

    def _calc_baroreflex(self, map_est: float, noise: float) -> float:
        deficit = 85.0 - map_est
        hr_drive = float(np.clip(deficit * 2.0, 0.0, self.MAX_HR - 60.0))
        hrv_component = noise * 50.0 * self.cfg.hr_variability
        hr = 60.0 + hr_drive + hrv_component
        return max(self.MIN_HR, min(self.MAX_HR, hr))

    def _calc_metabolic(self, ci: float, sev: float) -> Tuple[float, float, float, float]:
        spo2 = 0.98 if sev < 0.5 else max(0.85, 0.98 - (sev-0.5)*0.2)
        do2i = ci * self.cfg.hb * 1.34 * spo2 * 10.0
        vo2_demand = 110.0 * self.cfg.metabolic_rate * (1.0 + (0.5 * sev))
        vo2i = min(vo2_demand, do2i * 0.7)
        o2er = vo2i / do2i if do2i > 0 else 1.0
        lac_gen = 0.0
        if o2er > 0.35: lac_gen += (o2er - 0.35) * 0.3
        if self.cfg.shock_type == "Septic": lac_gen += (sev * 0.03)
        return do2i, vo2i, o2er, lac_gen

    def step(self, disease_severity=0.0, noise_val=0.0) -> Dict[str, float]:
        sev = np.clip(disease_severity, 0.0, 1.0)
        eff_pre, eff_con, eff_aft = self._apply_shock_modifiers(sev)
        sv = self._calc_mechanics(eff_pre, eff_con, noise_val)
        svr_dyne = 1200.0 * eff_aft
        map_est = (sv * 70.0 * eff_aft * 0.05) + 5.0
        hr = self._calc_baroreflex(map_est, noise_val)
        co = (hr * sv) / 1000.0
        ci = co / self.cfg.bsa
        map_val = (co * svr_dyne / 80.0) + 5.0
        pp_val = (sv / 1.5) * (1.0 / self.cfg.compliance)
        do2i, vo2i, o2er, lac_gen = self._calc_metabolic(ci, sev)
        return {
            "HR": hr, "SV": sv, "CO": co, "CI": ci, "MAP": map_val, "SVR": svr_dyne, 
            "DO2I": do2i, "VO2I": vo2i, "O2ER": o2er, "PP": pp_val,
            "Lac_Gen": lac_gen, "Preload_Status": eff_pre
        }

# ==========================================
# 4. SIMULATION ENGINE (Orchestrator)
# ==========================================

class SimulationEngine:
    @staticmethod
    def generate_noise(mins: int, magnitude: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        white = rng.normal(0, 1, mins)
        return np.convolve(white, np.ones(10)/10, mode='same') * magnitude

    @staticmethod
    def run_trajectory(config: PatientConfig, mins=720, seed=42) -> pd.DataFrame:
        engine = PhysiologyEngine(config)
        noise_vec = SimulationEngine.generate_noise(mins, 1.0, seed)
        x = np.linspace(-6, 6, mins)
        progression = 1 / (1 + np.exp(-x)) * 0.9
        history = []
        curr_lac = 1.0
        for t in range(mins):
            state = engine.step(progression[t], noise_vec[t])
            clearance = curr_lac * 0.005 
            curr_lac = curr_lac - clearance + state["Lac_Gen"]
            state["Lactate"] = max(0.5, curr_lac)
            map_dist = state["MAP"] - config.map_target
            uo = 1.0 / (1.0 + np.exp(-0.3 * map_dist)) * 1.5
            state["Urine"] = max(0.0, uo + np.random.normal(0, 0.05))
            history.append(state)
        df = pd.DataFrame(history)
        df["SVRI"] = df["SVR"] * config.bsa
        df["SOFA"] = 0
        df.loc[df["MAP"] < 70, "SOFA"] += 1
        df.loc[df["Urine"] < 0.5, "SOFA"] += 1
        df.loc[df["Lactate"] > 2.0, "SOFA"] += 1
        return df

    @staticmethod
    def predict_multiverse(base_engine: PhysiologyEngine, current_sev: float, 
                          current_lac: float, horizon: int = 30, iterations: int = 20) -> Dict[str, Any]:
        times = np.arange(horizon)
        futures = {}
        scenarios = {
            "Natural": lambda e: None,
            "Fluid": lambda e: setattr(e, 'preload', e.preload * 1.4),
            "Pressor": lambda e: setattr(e, 'afterload', e.afterload * 1.5)
        }
        for name, mod_func in scenarios.items():
            mc_runs = []
            for _ in range(iterations):
                eng = base_engine.copy()
                mod_func(eng)
                noise = SimulationEngine.generate_noise(horizon, 1.0, seed=None)
                run_vals = []
                for t in times:
                    s = eng.step(current_sev + 0.001*t, noise[t])
                    run_vals.append(s["MAP"])
                mc_runs.append(run_vals)
            mc_runs = np.array(mc_runs)
            mean_traj = np.mean(mc_runs, axis=0)
            std_traj = np.std(mc_runs, axis=0)
            futures[name] = {"mean": mean_traj, "upper": mean_traj + (1.96 * std_traj), "lower": mean_traj - (1.96 * std_traj)}
        return futures

# ==========================================
# 5. VISUALIZATION ENGINE
# ==========================================

class Visuals:
    @staticmethod
    def _hex_to_rgba(h, alpha):
        h = h.lstrip('#')
        return f"rgba({int(h[0:2],16)}, {int(h[2:4],16)}, {int(h[4:6],16)}, {alpha})"

    @staticmethod
    def _clean_layout(height=200, title=None):
        layout = go.Layout(
            template="plotly_white", margin=dict(l=10, r=10, t=30 if title else 10, b=10),
            height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color=THEME["text"])
        )
        if title: layout.title = dict(text=f"<b>{title}</b>", font=dict(size=14))
        return layout

    @staticmethod
    def plot_spark(df: pd.DataFrame, col: str, color: str, limits: Tuple[float, float]) -> go.Figure:
        data = df[col].iloc[-60:]
        fig = go.Figure()
        fig.add_shape(type="rect", x0=data.index[0], x1=data.index[-1], y0=limits[0], y1=limits[1],
                      fillcolor="rgba(0,0,0,0.04)", line_width=0, layer="below")
        rgba = Visuals._hex_to_rgba(color, 0.1)
        fig.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines', 
                                 line=dict(color=color, width=2.5), fill='tozeroy', fillcolor=rgba, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=[data.index[-1]], y=[data.values[-1]], mode='markers',
                                 marker=dict(color=color, size=8, line=dict(color='white', width=1))))
        fig.update_layout(Visuals._clean_layout(height=50))
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        y_min = min(data.min(), limits[0]) * 0.95
        y_max = max(data.max(), limits[1]) * 1.05
        fig.update_yaxes(range=[y_min, y_max])
        return fig

    @staticmethod
    def plot_horizon(futures: Dict, target: float) -> go.Figure:
        t = np.arange(len(futures["Natural"]["mean"]))
        fig = go.Figure()
        def add_scenario(name, color, data, dash=None):
            rgba = Visuals._hex_to_rgba(color, 0.15)
            fig.add_trace(go.Scatter(x=np.concatenate([t, t[::-1]]), y=np.concatenate([data["upper"], data["lower"][::-1]]),
                                     fill='toself', fillcolor=rgba, line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=t, y=data["mean"], line=dict(color=color, width=3, dash=dash), name=name))
        add_scenario("Natural Course", "#94a3b8", futures["Natural"], "dot")
        add_scenario("Fluid (+1L)", THEME["ci"], futures["Fluid"])
        add_scenario("Pressor", THEME["map"], futures["Pressor"])
        fig.add_hline(y=target, line_color=THEME["crit"], line_dash="solid")
        fig.update_layout(Visuals._clean_layout(height=280, title="Therapeutic Horizon (30 min)"))
        fig.update_layout(legend=dict(orientation="h", y=1.1, x=0))
        fig.update_xaxes(title="Minutes Future", showgrid=False)
        fig.update_yaxes(title="Projected MAP (mmHg)", gridcolor=THEME["grid"])
        return fig

    @staticmethod
    def plot_compass(df: pd.DataFrame, preds: Dict, curr_idx: int) -> go.Figure:
        start = max(0, curr_idx - 60)
        data = df.iloc[start:curr_idx]
        cur = df.iloc[curr_idx]
        fig = go.Figure()
        zones = [(0, 0, 2.5, 65, THEME["crit"], "CRITICAL SHOCK"),
                 (2.5, 0, 6.0, 65, THEME["warn"], "VASOPLEGIA"),
                 (2.5, 65, 6.0, 110, THEME["ok"], "GOAL")]
        for x0, y0, x1, y1, col, txt in zones:
            rgba = Visuals._hex_to_rgba(col, 0.1)
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, fillcolor=rgba, line_width=0, layer="below")
            fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=txt, font=dict(color=col, size=10, weight="bold"), showarrow=False)
        fig.add_trace(go.Scatter(x=data["CI"], y=data["MAP"], mode="lines", line=dict(color="#94a3b8", dash="dot")))
        fig.add_trace(go.Scatter(x=[cur["CI"]], y=[cur["MAP"]], mode="markers", 
                                 marker=dict(color="#0f172a", size=14, symbol="cross", line=dict(width=2, color="white"))))
        target_f_map = preds["Fluid"]["mean"][-1]
        target_f_ci = cur["CI"] * 1.25
        target_p_map = preds["Pressor"]["mean"][-1]
        target_p_ci = cur["CI"] * 1.0
        fig.add_annotation(x=target_f_ci, y=target_f_map, ax=cur["CI"], ay=cur["MAP"], xref="x", yref="y", axref="x", ayref="y", arrowhead=2, arrowcolor=THEME["ci"], text="FLUID")
        fig.add_annotation(x=target_p_ci, y=target_p_map, ax=cur["CI"], ay=cur["MAP"], xref="x", yref="y", axref="x", ayref="y", arrowhead=2, arrowcolor=THEME["map"], text="PRESSOR")
        fig.update_layout(Visuals._clean_layout(height=300, title="Hemodynamic Compass"))
        fig.update_xaxes(title="Cardiac Index (L/min/m²)", range=[1.0, 6.0], gridcolor=THEME["grid"])
        fig.update_yaxes(title="MAP (mmHg)", range=[30, 110], gridcolor=THEME["grid"])
        fig.update_layout(showlegend=False)
        return fig

    @staticmethod
    def plot_starling(df: pd.DataFrame, curr_idx: int) -> go.Figure:
        data = df.iloc[max(0, curr_idx-30):curr_idx]
        fig = go.Figure()
        x = np.linspace(0, 20, 100); y = np.log(x+1)*30
        fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=THEME["neutral"], dash='dot'), name='Ideal'))
        fig.add_trace(go.Scatter(x=data['Preload_Status'], y=data['SV'], mode='lines', line=dict(color=THEME["ci"], width=3)))
        fig.add_trace(go.Scatter(x=[data['Preload_Status'].iloc[-1]], y=[data['SV'].iloc[-1]], mode='markers', marker=dict(color=THEME["ci"], size=10)))
        d_sv = data['SV'].iloc[-1] - data['SV'].iloc[0]
        slope = d_sv / (data['Preload_Status'].iloc[-1] - data['Preload_Status'].iloc[0] + 0.01)
        status = "RESPONSIVE" if slope > 2.0 else "NON-RESPONDER"
        col = THEME["ok"] if slope > 2.0 else THEME["crit"]
        fig.add_annotation(x=10, y=10, text=status, font=dict(color=col, weight="bold"), showarrow=False)
        fig.update_layout(Visuals._clean_layout(height=250, title="Fluid Responsiveness"))
        fig.update_xaxes(title="Preload (CVP)", gridcolor=THEME["grid"])
        fig.update_yaxes(title="Stroke Volume (mL)", gridcolor=THEME["grid"])
        fig.update_layout(showlegend=False)
        return fig

    @staticmethod
    def plot_oxygen(df: pd.DataFrame, curr_idx: int) -> go.Figure:
        start = max(0, curr_idx - 180)
        data = df.iloc[start:curr_idx]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=data.index, y=data['DO2I'], fill='tozeroy', 
                                 fillcolor=Visuals._hex_to_rgba(THEME["do2"], 0.2),
                                 line=dict(color=THEME["do2"]), name='Delivery'), secondary_y=False)
        fig.add_trace(go.Scatter(x=data.index, y=data['Lactate'], 
                                 line=dict(color=THEME["crit"], width=2), name='Lactate'), secondary_y=True)
        fig.update_layout(Visuals._clean_layout(height=250, title="Oxygen Kinetics"))
        fig.update_yaxes(title="DO2I (mL/m²)", secondary_y=False, gridcolor=THEME["grid"])
        fig.update_yaxes(title="Lactate", secondary_y=True, showgrid=False)
        fig.update_layout(showlegend=False)
        return fig

# ==========================================
# 6. KPI COMPONENT
# ==========================================

class KPIComponent:
    @staticmethod
    def render(col, label, val, unit, color, df_col, t_low, t_high, key_id, df, idx, prev_row):
        delta = val - prev_row[df_col]
        color_trend = THEME["crit"] if (val < t_low or val > t_high) else THEME["ok"]
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-top:3px solid {color}">
                <div class="kpi-lbl">{label}</div>
                <div class="kpi-val" style="color:{color}">{val:.1f} <span class="kpi-unit">{unit}</span></div>
                <div class="kpi-trend" style="color:{color_trend}">
                    {delta:+.1f} <span style="color:#94a3b8; font-weight:400; font-size:0.7rem; margin-left:4px">15m</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(Visuals.plot_spark(df.iloc[:idx+1], df_col, color, (t_low, t_high)), 
                            use_container_width=True, config={'displayModeBar': False}, key=key_id)

# ==========================================
# 7. MAIN APP EXECUTION
# ==========================================

st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown(SAFETY_DISCLAIMER, unsafe_allow_html=True)

# --- SIDEBAR CONTROL ---
with st.sidebar:
    st.header("TITAN | Control")
    st.markdown("### ⚙️ Physiology")
    shock = st.selectbox("Shock Type", ["Septic", "Cardiogenic", "Hypovolemic", "Obstructive"])
    hb = st.slider("Hemoglobin", 7.0, 15.0, 12.0, 0.5)
    map_target = st.number_input("Target MAP", 55, 85, 65)
    curr_time = st.slider("Timeline (min)", 60, 720, 720)
    
    if st.button("Refresh Simulation"):
        st.cache_data.clear()
        st.rerun()
    
    config = PatientConfig(shock_type=shock, hb=hb, map_target=map_target)

# --- SIMULATION ---
@st.cache_data
def get_simulation(cfg):
    return SimulationEngine.run_trajectory(cfg, mins=720, seed=42)

df = get_simulation(config)
idx = curr_time - 1
row = df.iloc[idx]
prev = df.iloc[idx-15]

# --- 1. HEADER (TRIAGE) ---
status_msg = "STABLE"
action_msg = "Routine Monitoring"
banner_style = "b-ok"

if row["Lactate"] > 2.0:
    status_msg = "HYPOPERFUSION DETECTED"
    action_msg = "Evaluate DO2 (Fluids/Blood)"
    banner_style = "b-warn"

if row["MAP"] < map_target:
    status_msg = f"{shock.upper()} SHOCK"
    banner_style = "b-crit"
    if shock == "Septic": action_msg = "PROTOCOL: Vasopressor (Norepinephrine)"
    elif shock == "Cardiogenic": action_msg = "PROTOCOL: Inotrope (Dobutamine)"
    elif shock == "Hypovolemic": action_msg = "PROTOCOL: Rapid Volume Expansion"
    else: action_msg = "PROTOCOL: Relieve Obstruction"

st.markdown(f"""
<div class="banner {banner_style}">
    <div>
        <div style="font-weight:800; font-size:1.25rem;">{status_msg}</div>
        <div style="color:#64748b; font-weight:500;">Target: MAP > {map_target} • Lactate < 2.0</div>
    </div>
    <div style="text-align:right;">
        <div style="font-size:0.75rem; font-weight:700; color:#64748b;">RECOMMENDATION</div>
        <div style="font-weight:700; font-size:1.1rem;">{action_msg}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 2. KPI STRIP ---
cols = st.columns(6)
KPIComponent.render(cols[0], "MAP", row["MAP"], "mmHg", THEME["map"], "MAP", map_target, 110, "k1", df, idx, prev)
KPIComponent.render(cols[1], "Cardiac Index", row["CI"], "L/min", THEME["ci"], "CI", 2.5, 4.2, "k2", df, idx, prev)
KPIComponent.render(cols[2], "SVR Index", row["SVRI"], "dyn", THEME["svr"], "SVRI", 800, 1200, "k3", df, idx, prev)
KPIComponent.render(cols[3], "Stroke Vol", row["PP"], "mL", THEME["hr"], "PP", 35, 60, "k4", df, idx, prev)
KPIComponent.render(cols[4], "DO2 Index", row["DO2I"], "mL/m²", THEME["do2"], "DO2I", 400, 600, "k5", df, idx, prev)
KPIComponent.render(cols[5], "Lactate", row["Lactate"], "mmol", THEME["crit"], "Lactate", 0, 2.0, "k6", df, idx, prev)

# --- 3. PREDICTIVE LAYER ---
st.markdown('<div class="section-header">🔮 Predictive Decision Support</div>', unsafe_allow_html=True)
c1, c2 = st.columns([1, 1])

# Generate predictions on the fly (lightweight Monte Carlo)
base_engine = PhysiologyEngine(config, row["Preload_Status"], 1.0, 1.0) # Approx state
futures = SimulationEngine.predict_multiverse(base_engine, 0.9 * (idx/720), row["Lactate"])

with c1:
    st.plotly_chart(Visuals.plot_compass(df, futures, idx), use_container_width=True, config={'displayModeBar': False})
with c2:
    st.plotly_chart(Visuals.plot_horizon(futures, map_target), use_container_width=True, config={'displayModeBar': False})

# --- 4. ORGAN LAYER ---
st.markdown('<div class="section-header">🫀 Physiological Mechanics</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)

with c3:
    st.plotly_chart(Visuals.plot_oxygen(df, idx), use_container_width=True, config={'displayModeBar': False})
with c4:
    st.plotly_chart(Visuals.plot_starling(df, idx), use_container_width=True, config={'displayModeBar': False})
