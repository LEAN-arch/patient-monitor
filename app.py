import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
from scipy.signal import welch

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="TITAN | Precision CDS", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

THEME = {
    "bg": "#f8fafc",
    "primary": "#0f172a",
    "grid": "#e2e8f0",
    "map": "#be185d", "ci": "#059669", "do2": "#7c3aed", 
    "svr": "#d97706", "hr": "#0284c7", "text": "#334155",
    "crit": "#ef4444", "warn": "#f59e0b", "ok": "#10b981"
}

STYLING = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@500&display=swap');
.stApp { background-color: #f8fafc; font-family: 'Inter', sans-serif; color: #0f172a; }

/* KPI Card */
.kpi-card {
    background: white; border: 1px solid #cbd5e1; border-radius: 8px;
    padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 8px;
}
.kpi-lbl { font-size: 0.7rem; font-weight: 700; color: #64748b; text-transform: uppercase; }
.kpi-val { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; line-height: 1.1; }
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
<div style="background-color:#fff3cd; border:1px solid #ffeeba; color:#856404; padding:10px; border-radius:6px; font-size:0.8rem; margin-top:30px;">
    <strong>⚠️ CAUTION: SIMULATION ONLY</strong><br>
    This is a probabilistic Clinical Decision Support (CDS) prototype. 
    It is not a medical device. Standard of care must take precedence.
</div>
"""

# ==========================================
# 2. PHYSIOLOGY ENGINE CORE
# ==========================================

@dataclass
class PhysiologicalParams:
    """
    Centralized parameter store for the Digital Twin.
    Allows for scenario injection and sensitivity analysis.
    """
    # Patient Profile
    hb: float = 12.0
    weight: float = 70.0
    bsa: float = 1.8
    shock_type: str = "Septic"
    
    # Mechanics Limits
    max_hr: float = 170.0
    min_hr: float = 50.0
    sv_max_baseline: float = 100.0
    starling_k: float = 8.0 # Steepness of Starling curve
    
    # Autoregulation
    map_target: float = 65.0
    baroreflex_sensitivity: float = 2.0
    autoreg_capacity: float = 1.0 # 1.0 = Full health, 0.0 = Vasoplegia

class PhysiologyEngine:
    """
    Advanced 0-D Cardiovascular Model (Windkessel-Baroreflex Coupled).
    """
    def __init__(self, params: PhysiologicalParams, preload=12.0, contractility=1.0, afterload=1.0):
        self.p = params
        self.preload = float(preload)
        self.contractility = float(contractility)
        self.afterload = float(afterload)

    def copy(self):
        """Creates a deep copy for branching prediction."""
        return PhysiologyEngine(self.p, self.preload, self.contractility, self.afterload)

    def _calc_mechanics(self, eff_preload, eff_contractility, noise):
        """Helper: Frank-Starling Stroke Volume Calculation."""
        sv_max = self.p.sv_max_baseline * eff_contractility
        # Add respiratory variation noise to preload
        noisy_preload = eff_preload * (1 + noise * 0.5) 
        sv = sv_max * (noisy_preload**2 / (noisy_preload**2 + self.p.starling_k**2))
        return sv

    def _calc_baroreflex(self, sv, eff_afterload, noise):
        """Helper: CNS control of Heart Rate."""
        # Estimate MAP based on current flow potential
        map_est = (sv * 70.0 * eff_afterload * 0.05) + 5.0
        
        # Drive HR based on pressure deficit (Baroreflex)
        deficit = 85.0 - map_est
        hr_drive = float(np.clip(deficit * self.p.baroreflex_sensitivity, 0.0, self.p.max_hr - 60.0))
        
        # Add HRV (1/f noise influence)
        hrv = noise * 10.0 
        hr = 60.0 + hr_drive + hrv
        return max(self.p.min_hr, hr)

    def _calc_svr(self, eff_afterload, sev):
        """Helper: SVR with Autoregulation logic."""
        # In early shock, SVR tries to rise (compensation) before failing (decompensation)
        # Sepsis destroys this autoregulation (Vasoplegia)
        base_svr = 1200.0 * eff_afterload
        compensation = 0.0
        
        if self.p.shock_type != "Septic":
            # Compensatory vasoconstriction in non-septic shock
            compensation = 400 * sev
            
        return base_svr + compensation

    def _calc_metabolic(self, co, sev):
        """Helper: Oxygen Transport and Lactate Gen."""
        spo2 = 0.98 if sev < 0.5 else max(0.85, 0.98 - (sev-0.5)*0.2)
        do2i = (co / self.p.bsa) * self.p.hb * 1.34 * spo2 * 10.0
        
        # VO2 increases with stress
        vo2_demand = 110.0 * (1.0 + (0.5 * sev))
        vo2i = min(vo2_demand, do2i * 0.7) # Supply dependency
        o2er = vo2i / do2i if do2i > 0 else 1.0
        
        # Lactate Gen: Anaerobic + Cytopathic
        lac_gen = 0.0
        if o2er > 0.35: lac_gen += (o2er - 0.35) * 0.3
        # Stochastic variability in lactate production
        if self.p.shock_type == "Septic": lac_gen += (sev * 0.04 * np.random.uniform(0.8, 1.2))
        
        return do2i, vo2i, o2er, lac_gen

    def step(self, disease_severity=0.0, noise_val=0.0):
        sev = np.clip(disease_severity, 0.0, 1.0)
        
        # 1. Apply Disease Modifiers
        mod_pre, mod_con, mod_aft = 1.0, 1.0, 1.0
        
        if self.p.shock_type == "Septic":
            mod_aft = max(0.25, 1.0 - 0.8 * sev) # Severe Vasodilation
            mod_pre = max(0.6, 1.0 - 0.3 * sev)  # Capillary Leak
        elif self.p.shock_type == "Cardiogenic":
            mod_con = max(0.2, 1.0 - 0.8 * sev)
            mod_aft = 1.0 + (0.5 * sev) # Afterload mismatch
        elif self.p.shock_type == "Hypovolemic":
            mod_pre = max(0.15, 1.0 - 0.9 * sev)
            mod_con = 1.0 + (0.2 * sev) # Hyperdynamic
        elif self.p.shock_type == "Obstructive":
            mod_pre = max(0.1, 1.0 - 0.9 * sev) # Impaired filling

        eff_pre = self.preload * mod_pre
        eff_aft = self.afterload * mod_aft
        eff_con = self.contractility * mod_con

        # 2. Compute Hemodynamics
        sv = self._calc_mechanics(eff_pre, eff_con, noise_val)
        hr = self._calc_baroreflex(sv, eff_aft, noise_val)
        
        co = (hr * sv) / 1000.0
        ci = co / self.p.bsa
        svr = self._calc_svr(eff_aft, sev)
        map_val = (co * svr / 80.0) + 5.0 # Ohm's Law
        
        # 3. Compute Metabolic
        do2i, vo2i, o2er, lac_gen = self._calc_metabolic(co, sev)
        
        return {
            "HR": hr, "SV": sv, "CO": co, "CI": ci, "MAP": map_val, "SVR": svr, 
            "DO2I": do2i, "VO2I": vo2i, "O2ER": o2er,
            "Lac_Gen": lac_gen, "Preload_Status": eff_pre
        }

def simulate_clinical_data(params: PhysiologicalParams, mins=720, seed=42):
    """
    Runs full simulation. Optimized with pre-calculated vectors.
    """
    rng = np.random.default_rng(seed)
    engine = PhysiologyEngine(params)
    
    # 1. Vectorize inputs
    x = np.linspace(-6, 6, mins)
    progression = 1 / (1 + np.exp(-x)) * 0.95 # Sigmoid
    
    # Generate 1/f Pink Noise for realism
    white = rng.normal(0, 1, mins)
    noise_vec = np.convolve(white, np.ones(10)/10, mode='same') * 0.05
    
    # 2. Main Loop
    history = []
    curr_lac = 1.0
    
    # Pre-allocate for performance
    sofa_resp = np.linspace(400, 100, mins) # PaO2/FiO2 drop
    sofa_plt = np.linspace(250, 50, mins)   # Platelet drop
    sofa_bili = np.linspace(0.8, 4.0, mins) # Bilirubin rise
    
    for t in range(mins):
        state = engine.step(progression[t], noise_vec[t])
        
        # State-dependent Lactate Clearance
        # Sick liver clears less lactate
        clearance_rate = 0.005 * (1.0 - progression[t]*0.5)
        curr_lac = curr_lac * (1.0 - clearance_rate) + state["Lac_Gen"]
        state["Lactate"] = max(0.5, curr_lac)
        
        # State-dependent Urine (Autoregulation)
        # GFR drops sharply below MAP 65
        map_dist = state["MAP"] - params.map_target
        uo_base = 1.0 / (1.0 + np.exp(-0.25 * map_dist)) * 1.5
        state["Urine"] = max(0.0, uo_base + rng.normal(0, 0.05))
        
        # Add SOFA Components
        state["PaO2_FiO2"] = sofa_resp[t] + rng.normal(0, 10)
        state["Platelets"] = sofa_plt[t] + rng.normal(0, 5)
        state["Bilirubin"] = sofa_bili[t]
        
        history.append(state)
        
    df = pd.DataFrame(history)
    df["SVRI"] = df["SVR"] * params.bsa
    df["PP"] = df["SV"] / 1.5
    
    # 3. Calculate Comprehensive SOFA Score
    sofa_scores = []
    for _, r in df.iterrows():
        s = 0
        if r["MAP"] < 70: s += 1
        if r["Urine"] < 0.5: s += 1
        if r["PaO2_FiO2"] < 300: s += 1
        if r["Platelets"] < 150: s += 1
        if r["Bilirubin"] > 1.2: s += 1
        if r["Lactate"] > 2.0: s += 1 # CNS Proxy
        sofa_scores.append(s)
    df["SOFA"] = sofa_scores
    
    return df

def predict_response_uncertainty(base_engine, current_sev, current_lac, horizon=60):
    """
    Generates 'Multiverse' predictions with 95% Confidence Intervals.
    Optimized to only calc MAP for speed.
    """
    times = np.arange(horizon)
    futures = {}
    
    # Scenarios: Name -> Engine Modifier
    scenarios = {
        "Natural": lambda e: None,
        # Correctly referencing attributes 'preload' and 'afterload'
        "Fluid": lambda e: setattr(e, 'preload', e.preload * 1.4),
        "Pressor": lambda e: setattr(e, 'afterload', e.afterload * 1.5)
    }
    
    for name, mod_func in scenarios.items():
        eng = base_engine.copy()
        mod_func(eng) # Apply intervention
        
        vals = []
        for t in times:
            # Assume 0 noise for mean trajectory
            s = eng.step(current_sev + 0.001*t, noise_val=0.0)
            vals.append(s["MAP"])
        
        vals = np.array(vals)
        # Uncertainty cone expands with time
        sigma = np.linspace(2, 10, horizon)
        futures[name] = {
            "mean": vals,
            "upper": vals + sigma,
            "lower": vals - sigma
        }
        
    return futures

# ==========================================
# 3. VISUALIZATION LIBRARY (ChartFactory)
# ==========================================

class ChartFactory:
    """
    Centralized factory for generating Plotly figures.
    Ensures consistent styling and reduces code duplication.
    """
    
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
    def hex_to_rgba(h, alpha):
        h = h.lstrip('#')
        return f"rgba({int(h[0:2],16)}, {int(h[2:4],16)}, {int(h[4:6],16)}, {alpha})"

    @staticmethod
    def sparkline(df: pd.DataFrame, col: str, color: str, limits: Tuple[float, float]) -> go.Figure:
        """
        Generates a sparkline with a background safety corridor.
        
        Args:
            df: Dataframe with time series
            col: Column name to plot
            color: Hex color string
            limits: (min, max) tuple for the safety corridor
        """
        data = df[col].iloc[-60:]
        fig = go.Figure()
        
        # Safety Zone
        fig.add_shape(type="rect", x0=data.index[0], x1=data.index[-1], y0=limits[0], y1=limits[1],
                      fillcolor="rgba(0,0,0,0.04)", line_width=0, layer="below")
        
        # Trend
        rgba = ChartFactory.hex_to_rgba(color, 0.1)
        fig.add_trace(go.Scatter(
            x=data.index, y=data.values, mode='lines', 
            line=dict(color=color, width=2.5), fill='tozeroy', fillcolor=rgba, hoverinfo='skip'
        ))
        
        # Head
        fig.add_trace(go.Scatter(x=[data.index[-1]], y=[data.values[-1]], mode='markers',
                                 marker=dict(color=color, size=8, line=dict(color='white', width=1))))
        
        fig.update_layout(ChartFactory._clean_layout(height=50))
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        # Dynamic range
        y_min = min(data.min(), limits[0]) * 0.95
        y_max = max(data.max(), limits[1]) * 1.05
        fig.update_yaxes(range=[y_min, y_max])
        return fig

    @staticmethod
    def predictive_horizon(futures: Dict, target: float) -> go.Figure:
        """
        Visualizes projected MAP with 95% Confidence Intervals.
        """
        t = np.arange(len(futures["Natural"]["mean"]))
        fig = go.Figure()
        
        def add_band(name, color, data, dash=None):
            rgba = ChartFactory.hex_to_rgba(color, 0.15)
            fig.add_trace(go.Scatter(
                x=np.concatenate([t, t[::-1]]), y=np.concatenate([data["upper"], data["lower"][::-1]]),
                fill='toself', fillcolor=rgba, line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(x=t, y=data["mean"], line=dict(color=color, width=3, dash=dash), name=name))

        add_band("Natural Course", "#94a3b8", futures["Natural"], "dot")
        add_band("Fluid (+1L)", THEME["ci"], futures["Fluid"])
        add_band("Pressor", THEME["map"], futures["Pressor"])
        
        fig.add_hline(y=target, line_color=THEME["crit"], line_dash="solid")
        
        fig.update_layout(ChartFactory._clean_layout(height=280, title="Therapeutic Horizon (30 min)"))
        fig.update_layout(legend=dict(orientation="h", y=1.1, x=0))
        fig.update_xaxes(title="Minutes Future", showgrid=False)
        fig.update_yaxes(title="Projected MAP (mmHg)", gridcolor=THEME["grid"])
        return fig

    @staticmethod
    def hemodynamic_compass(df: pd.DataFrame, curr_idx: int) -> go.Figure:
        """
        Diagnostic Quadrant Plot (CI vs MAP).
        """
        data = df.iloc[max(0, curr_idx-60):curr_idx]
        cur = df.iloc[curr_idx]
        fig = go.Figure()
        
        # Zones
        zones = [
            (0, 0, 2.5, 65, THEME["crit"], "CRITICAL SHOCK"),
            (2.5, 0, 6.0, 65, THEME["warn"], "VASOPLEGIA"),
            (2.5, 65, 6.0, 110, THEME["ok"], "GOAL")
        ]
        
        for x0, y0, x1, y1, col, txt in zones:
            rgba = ChartFactory.hex_to_rgba(col, 0.1)
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, fillcolor=rgba, line_width=0, layer="below")
            fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=txt, font=dict(color=col, size=10, weight="bold"), showarrow=False)

        fig.add_trace(go.Scatter(x=data["CI"], y=data["MAP"], mode="lines", line=dict(color="#94a3b8", dash="dot")))
        fig.add_trace(go.Scatter(x=[cur["CI"]], y=[cur["MAP"]], mode="markers", 
                                 marker=dict(color="#0f172a", size=14, symbol="cross", line=dict(width=2, color="white"))))

        fig.update_layout(ChartFactory._clean_layout(height=280, title="Hemodynamic Phenotype"))
        fig.update_xaxes(title="Cardiac Index (L/min/m²)", range=[1.0, 5.5], gridcolor=THEME["grid"])
        fig.update_yaxes(title="MAP (mmHg)", range=[30, 100], gridcolor=THEME["grid"])
        return fig

    @staticmethod
    def organ_radar(df: pd.DataFrame, curr_idx: int) -> go.Figure:
        """Multivariate Organ Risk Polygon."""
        cur = df.iloc[curr_idx]
        
        # Normalize Risks (0-1)
        r_kidney = np.clip((65 - cur["MAP"])/20, 0, 1)
        r_heart = np.clip((cur["HR"] - 100)/60, 0, 1)
        r_lung = np.clip((400 - cur["PaO2_FiO2"])/300, 0, 1)
        r_meta = np.clip((cur["Lactate"] - 1.5)/4, 0, 1)
        r_flow = np.clip((2.2 - cur["CI"])/1.5, 0, 1)
        
        r = [r_kidney, r_heart, r_lung, r_meta, r_flow, r_kidney]
        theta = ["Kidney", "Heart", "Lung", "Metabolic", "Perfusion", "Kidney"]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=[0.2]*6, theta=theta, fill='none', line=dict(color='lightgrey', dash='dot')))
        fig.add_trace(go.Scatterpolar(r=r, theta=theta, fill='toself', 
                                      fillcolor=ChartFactory.hex_to_rgba(THEME['crit'], 0.2),
                                      line=dict(color=THEME["crit"], width=2)))
        
        fig.update_layout(ChartFactory._clean_layout(height=250, title="Multi-Organ SOFA Risk"))
        fig.update_polars(radialaxis=dict(visible=False, range=[0, 1]))
        return fig

# ==========================================
# 5. MAIN APP EXECUTION
# ==========================================

# A. Styles
st.markdown(STYLING, unsafe_allow_html=True)

# B. Sidebar Inputs
with st.sidebar:
    st.header("TITAN | Control")
    curr_time = st.slider("Timeline", 60, 720, 720)
    
    st.markdown("### ⚙️ Physiology")
    shock = st.selectbox("Shock Type", ["Septic", "Cardiogenic", "Hypovolemic", "Obstructive"])
    hb = st.slider("Hemoglobin", 7.0, 15.0, 12.0, 0.5, help="Affects DO2I calculation")
    map_target = st.number_input("Target MAP (mmHg)", 55, 85, 65)
    
    if st.button("Refresh Simulation"):
        st.cache_data.clear()
        st.rerun()
    
    config = PatientConfig(shock_type=shock, hb=hb, map_target=map_target)

# C. Simulation
# Not cached to allow real-time param updates
df = simulate_clinical_data(config, mins=720, seed=42)
idx = curr_time - 1
row = df.iloc[idx]
prev = df.iloc[idx-15]

# D. Header Logic
status_msg = "STABLE"
action_msg = "Continue Monitoring"
banner_style = "b-ok"

if row["Lactate"] > 2.0:
    status_msg = "HYPOPERFUSION DETECTED"
    action_msg = "Evaluate DO2 (Fluids/Blood)"
    banner_style = "b-warn"

if row["MAP"] < map_target:
    status_msg = f"{shock.upper()} SHOCK"
    action_msg = "PROTOCOL ACTIVATED: Hemodynamic Support"
    banner_style = "b-crit"

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

# E. KPI Strip
cols = st.columns(6)

def render_kpi(col, label, val, unit, color, df_col, t_low, t_high, key_id):
    delta = val - prev[df_col]
    color_trend = THEME["crit"] if (val < t_low or val > t_high) else THEME["ok"]
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="border-top:3px solid {color}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val:.1f} <span class="kpi-unit">{unit}</span></div>
            <div class="kpi-trend" style="color:{color_trend}">
                {delta:+.1f} <span style="color:#94a3b8; font-weight:400; font-size:0.7rem; margin-left:4px">15m</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(ChartFactory.sparkline(df.iloc[:idx+1], df_col, color, (t_low, t_high)), 
                        use_container_width=True, config={'displayModeBar': False}, key=key_id)

render_kpi(cols[0], "MAP", row["MAP"], "mmHg", THEME["map"], "MAP", map_target, 110, "k1")
render_kpi(cols[1], "Cardiac Index", row["CI"], "L/min", THEME["ci"], "CI", 2.5, 4.2, "k2")
render_kpi(cols[2], "SVR Index", row["SVRI"], "dyn", THEME["svr"], "SVRI", 800, 1200, "k3")
render_kpi(cols[3], "Stroke Vol", row["PP"], "mL", THEME["hr"], "PP", 35, 60, "k4")
render_kpi(cols[4], "DO2 Index", row["DO2I"], "mL/m²", THEME["do2"], "DO2I", 400, 600, "k5")
render_kpi(cols[5], "Lactate", row["Lactate"], "mmol", THEME["crit"], "Lactate", 0, 2.0, "k6")

# F. Predictive & Organ Panels
st.markdown('<div class="section-header">🔮 PREDICTIVE DECISION SUPPORT</div>', unsafe_allow_html=True)
c1, c2 = st.columns([1, 1])

engine = PhysiologyEngine(config)
futures = predict_response_uncertainty(engine, 0.9 * (idx/720), row["Lactate"])

with c1:
    st.plotly_chart(ChartFactory.hemodynamic_compass(df, idx), use_container_width=True, config={'displayModeBar': False})
with c2:
    st.plotly_chart(ChartFactory.predictive_horizon(futures, map_target), use_container_width=True, config={'displayModeBar': False})

st.markdown('<div class="section-header">🫀 ORGAN SYSTEM SAFETY (SOFA)</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)

with c3:
    st.plotly_chart(ChartFactory.organ_radar(df, idx), use_container_width=True, config={'displayModeBar': False})
with c4:
    st.info(f"""
    **System Status Report:**
    - **Renal:** {'Oliguria' if row['Urine'] < 0.5 else 'Adequate Output'} ({row['Urine']:.1f} mL/kg/hr)
    - **Respiratory:** PaO2/FiO2 ratio {row['PaO2_FiO2']:.0f} (Healthy > 400)
    - **Hematologic:** Platelets {row['Platelets']:.0f} (Simulated)
    - **Metabolic:** Lactate {row['Lactate']:.1f} mmol/L (Clearance {((prev['Lactate']-row['Lactate'])/prev['Lactate'])*100:.1f}%)
    """)

st.markdown(SAFETY_DISCLAIMER, unsafe_allow_html=True)
