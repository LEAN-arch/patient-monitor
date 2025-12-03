import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import datetime

# ==========================================
# 1. PREMIUM UX CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="TITAN | Helios", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Color Palette (Medical Grade)
THEME = {
    "bg": "#f8fafc",
    "surface": "#ffffff",
    "text_main": "#0f172a",
    "text_sub": "#64748b",
    "accent": "#0ea5e9",    
    "ok": "#10b981",        
    "warn": "#f59e0b",      
    "crit": "#ef4444",      
    "grid": "#e2e8f0",
    # Advanced Signals
    "cpo": "#8b5cf6",       # Violet (Power)
    "dsi": "#ec4899",       # Pink (Diastolic)
    "coupling": "#14b8a6",  # Teal (Efficiency)
    "fluid": "#3b82f6"      # Blue
}

STYLING = """
<style>
:root { --primary: #0f172a; --surface: #ffffff; }
.stApp { background-color: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #0f172a; }

/* Patient Header */
.pt-banner {
    background: #1e293b; color: white; padding: 12px 24px; border-radius: 8px;
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 15px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
}
.pt-id { font-family: 'Courier New', monospace; font-weight: 700; font-size: 1.1rem; color: #38bdf8; }

/* Novel KPI Tile */
.kpi-tile {
    background: white; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 12px; height: 100%;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05); transition: transform 0.1s;
}
.kpi-head { font-size: 0.7rem; font-weight: 700; color: #64748b; letter-spacing: 0.05em; text-transform: uppercase; }
.kpi-val { font-size: 1.6rem; font-weight: 800; color: #0f172a; line-height: 1.1; }
.kpi-sub { font-size: 0.75rem; font-weight: 600; margin-top: 4px; }

/* Clinical Alert */
.alert-box {
    padding: 12px 16px; border-radius: 6px; border-left: 6px solid;
    background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center;
}
.a-crit { border-color: #ef4444; background: #fef2f2; color: #991b1b; }
.a-warn { border-color: #f59e0b; background: #fffbeb; color: #92400e; }
.a-ok { border-color: #10b981; background: #f0fdf4; color: #166534; }

.section-title {
    font-size: 0.95rem; font-weight: 800; color: #334155;
    border-bottom: 2px solid #e2e8f0; margin: 25px 0 10px 0; padding-bottom: 4px;
    text-transform: uppercase; letter-spacing: 0.05em;
}
</style>
"""

DISCLAIMER = """
<div style="background-color:#f1f5f9; border:1px solid #cbd5e1; color:#475569; padding:8px; border-radius:6px; font-size:0.7rem; margin-top:30px; text-align:center;">
    <strong>CLINICAL SAFETY WARNING</strong><br>
    Titan Helios is an Investigational Clinical Decision Support tool. 
    Parameters like CPO and Eadyn are derived proxies. 
    Do not use as sole basis for intervention.
</div>
"""

# ==========================================
# 2. PATIENT CONTEXT & PARAMETERS
# ==========================================

@dataclass
class PatientProfile:
    mrn: str = "MRN-892401"
    age: int = 64
    weight: float = 78.5 # kg
    bsa: float = 1.95 # m2
    
    # Lab Values
    hb: float = 10.5 # g/dL
    target_map: float = 65.0

class DigitalTwinEngine:
    def __init__(self, profile: PatientProfile):
        self.p = profile
        self.estimated_contractility = 1.0 
        self.estimated_svr_tone = 0.8
        self.estimated_compliance = 1.2
        self.preload_reserve = 12.0 # CVP proxy
        self.lung_water = 5.0 # ELWI Proxy (Normal < 7)

    def step_physics(self, intervention_preload=1.0, intervention_afterload=1.0, intervention_inotropy=1.0):
        # 1. Effective Parameters
        eff_pre = self.preload_reserve * intervention_preload
        eff_aft = self.estimated_svr_tone * intervention_afterload
        eff_con = self.estimated_contractility * intervention_inotropy
        
        # 2. Mechanics
        sv_max = 100.0 * eff_con
        # Starling Curve
        sv = sv_max * (eff_pre**2 / (eff_pre**2 + 64.0))
        
        # 3. Baroreflex (Vasoplegic Shock response)
        # In sepsis, DBP drops faster than SBP
        map_est = (sv * 70.0 * eff_aft * 0.05) + 5.0
        hr_drive = np.clip((85.0 - map_est) * 2.0, 0, 110)
        hr = 60.0 + hr_drive
        
        # 4. Hemodynamics
        co = (hr * sv) / 1000.0
        ci = co / self.p.bsa
        svr = 1200.0 * eff_aft
        map_val = (co * svr / 80.0) + 5.0
        
        # 5. Advanced Derived Params
        # CPO: Cardiac Power Output (Watts)
        cpo = (map_val * co) / 451
        
        # DBP Estimate (Widened PP in sepsis)
        pp = sv / self.estimated_compliance
        dbp = map_val - (pp/3)
        sbp = map_val + (2*pp/3)
        
        # DSI: Diastolic Shock Index
        dsi = hr / dbp if dbp > 0 else 0
        
        # Eadyn (Dynamic Arterial Elastance) Proxy
        # Ratio of PPV/SVV. High = Pressure Responsive. Low = Vasoplegic.
        # Modeled here as function of Tone
        eadyn = 0.5 + (eff_aft * 1.0)
        
        # Therapeutic Conflict (Lung Water)
        # Fluid bolus increases Lung Water if contractility is low
        lung_stress = (eff_pre / eff_con) * 0.5
        
        return {
            "MAP": map_val, "CI": ci, "SVR": svr, "SV": sv, "HR": hr, 
            "CPO": cpo, "DSI": dsi, "Eadyn": eadyn, "DBP": dbp, "SBP": sbp,
            "LungWater": self.lung_water + lung_stress
        }

# ==========================================
# 3. DATA STREAM SIMULATION
# ==========================================

def get_live_data_stream(profile: PatientProfile, hours=6):
    mins = hours * 60
    t = pd.date_range(end=pd.Timestamp.now(), periods=mins, freq="T")
    
    # Clinical Scenario: Progressive Septic Cardiomyopathy
    trend = np.linspace(0, 1, mins)
    
    # Vitals Generation
    map_trace = 80 - (trend * 35) + np.random.normal(0, 2, mins) # 80 -> 45
    hr_trace = 85 + (trend * 45) + np.random.normal(0, 2, mins)  # 85 -> 130
    
    df = pd.DataFrame({"Timestamp": t, "MAP": map_trace, "HR": hr_trace})
    
    # Advanced Calculations
    # DBP falls faster in sepsis
    df["DBP"] = df["MAP"] - 15 - (trend * 10)
    df["DSI"] = df["HR"] / df["DBP"]
    
    # Cardiac Power (Failing)
    # CO maintains early (Compensation) then crashes
    co_trend = 5.0 + (trend * 1.0) - (trend**3 * 3.0) 
    df["CO"] = co_trend + np.random.normal(0, 0.1, mins)
    df["CPO"] = (df["MAP"] * df["CO"]) / 451
    
    # Elastance (Vasodilation)
    df["Eadyn"] = 1.4 - (trend * 0.9) # Becomes pressure-unresponsive
    
    # Lung Water (Leak)
    df["ELWI"] = 6.0 + (trend * 8.0) # Pulmonary Edema developing
    
    return df

def predict_scenarios(engine: DigitalTwinEngine):
    """
    Project 30 mins into future with interventions.
    """
    timeline = np.arange(30)
    # Interventions: Fluid (Preload++), Levo (Afterload++), Dobuta (Contractility++)
    scenarios = {
        "Natural": {"pre": 1.0, "aft": 1.0, "con": 1.0},
        "Fluid":   {"pre": 1.4, "aft": 1.0, "con": 1.0}, 
        "Pressor": {"pre": 1.0, "aft": 1.5, "con": 1.0},
        "Inotrope": {"pre": 1.0, "aft": 1.0, "con": 1.3}
    }
    
    results = {}
    for name, mods in scenarios.items():
        state = engine.step_physics(mods["pre"], mods["aft"], mods["con"])
        results[name] = state
    
    return results

# ==========================================
# 4. VISUALIZATION COMPONENT
# ==========================================

class Visuals:
    
    @staticmethod
    def _layout(height=200, title=None):
        layout = go.Layout(
            template="plotly_white", margin=dict(l=10, r=10, t=30 if title else 10, b=10),
            height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="sans-serif", color=THEME["text_main"]),
            xaxis=dict(showgrid=True, gridcolor=THEME["grid"], zeroline=False),
            yaxis=dict(showgrid=True, gridcolor=THEME["grid"], zeroline=False)
        )
        if title: layout.title = dict(text=f"<b>{title}</b>", font=dict(size=13))
        return layout

    @staticmethod
    def sparkline(df, col, color, target=None):
        data = df.tail(45)
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data["Timestamp"], y=data[col], mode='lines',
            line=dict(color=color, width=2.5), 
            fill='tozeroy', fillcolor=f"rgba{color[1:-1]},0.1)",
            hoverinfo='y'
        ))
        
        if target:
            fig.add_hline(y=target, line_dash="dot", line_color="gray", opacity=0.5)
            
        fig.update_layout(Visuals._layout(height=50))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, range=[data[col].min()*0.9, data[col].max()*1.1])
        return fig

    @staticmethod
    def plot_cpo_gauge(val):
        """Cardiac Power Output Gauge - The Mortality Predictor"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = val,
            number = {'suffix': " W", 'font': {'size': 24}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Cardiac Power (CPO)", 'font': {'size': 12, 'color': '#64748b'}},
            gauge = {
                'axis': {'range': [0, 1.5], 'tickwidth': 1},
                'bar': {'color': THEME["cpo"]},
                'steps': [
                    {'range': [0, 0.6], 'color': "rgba(239, 68, 68, 0.2)"}, # Mortality Zone
                    {'range': [0.6, 1.5], 'color': "rgba(16, 185, 129, 0.1)"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.6}
            }
        ))
        fig.update_layout(height=140, margin=dict(l=20,r=20,t=20,b=20))
        return fig

    @staticmethod
    def plot_therapeutic_conflict(engine_state):
        """
        Visualizes Fluid Responsiveness vs Fluid Tolerance.
        Y-Axis: Stroke Volume Gain (Benefit)
        X-Axis: Lung Water Index (Harm)
        """
        fig = go.Figure()
        
        # Zones
        fig.add_shape(type="rect", x0=0, x1=10, y0=10, y1=20, fillcolor="rgba(16, 185, 129, 0.1)", line_width=0)
        fig.add_annotation(x=5, y=15, text="SAFE TO FILL", font=dict(color="green", size=10), showarrow=False)
        
        fig.add_shape(type="rect", x0=10, x1=20, y0=0, y1=20, fillcolor="rgba(239, 68, 68, 0.1)", line_width=0)
        fig.add_annotation(x=15, y=10, text="DROWNING RISK<br>(Stop Fluids)", font=dict(color="red", size=10), showarrow=False)

        # Simulation Points
        # Current
        curr_elwi = engine_state['Natural']['LungWater']
        curr_sv = engine_state['Natural']['SV']
        
        # With Fluid
        fluid_elwi = engine_state['Fluid']['LungWater']
        fluid_sv = engine_state['Fluid']['SV']
        
        # Vector
        fig.add_annotation(x=fluid_elwi, y=fluid_sv, ax=curr_elwi, ay=curr_sv,
                           xref="x", yref="y", axref="x", ayref="y",
                           arrowhead=2, arrowwidth=2, arrowcolor=THEME["accent"], text="FLUID EFFECT")
        
        fig.add_trace(go.Scatter(x=[curr_elwi], y=[curr_sv], mode='markers', marker=dict(color='black', size=10), name="Now"))

        fig.update_layout(Visuals._layout(height=250, title="Therapeutic Conflict (Lung vs Heart)"))
        fig.update_xaxes(title="Lung Water (ELWI) - Harm", range=[0, 20], gridcolor=THEME["grid"])
        fig.update_yaxes(title="Stroke Volume - Benefit", range=[20, 100], gridcolor=THEME["grid"])
        return fig

    @staticmethod
    def plot_coupling(engine_state):
        """V-A Coupling Efficiency (Ea/Ees)."""
        fig = go.Figure()
        
        # Iso-efficiency lines
        x = np.linspace(0, 150, 100)
        fig.add_trace(go.Scatter(x=x, y=x*1.0, line=dict(color='gray', dash='dot'), name="Optimal (1.0)"))
        
        # Current State (ESP vs SV) - Simplified Ees proxy
        pes = engine_state['Natural']['MAP'] * 0.9
        sv = engine_state['Natural']['SV']
        
        fig.add_trace(go.Scatter(x=[sv], y=[pes], mode='markers', marker=dict(size=15, color=THEME["coupling"]), name="Coupling"))
        
        fig.update_layout(Visuals._layout(height=250, title="V-A Coupling Efficiency"))
        fig.update_xaxes(title="Stroke Volume", gridcolor=THEME["grid"])
        fig.update_yaxes(title="End-Systolic Pressure", gridcolor=THEME["grid"])
        return fig

# ==========================================
# 5. APP LOGIC
# ==========================================

st.markdown(STYLING, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("TITAN | Helios")
    st.markdown("### 🏥 Patient Context")
    p_id = st.text_input("Patient ID", "MRN-892401")
    
    st.markdown("### 🎛️ Digital Twin Params")
    p_contract = st.slider("Contractility %", 20, 100, 40)
    p_svr = st.slider("Vasoplegia %", 0, 100, 70)
    
    profile = PatientProfile(mrn=p_id)
    twin = DigitalTwinEngine(profile)
    # Apply sliders to twin
    twin.estimated_contractility = p_contract / 100.0
    twin.estimated_svr_tone = (100 - p_svr) / 100.0

# --- DATA INGESTION ---
df_stream = get_live_data_stream(profile)
cur = df_stream.iloc[-1]
prv = df_stream.iloc[-15]

# --- PREDICTIONS ---
futures = predict_scenarios(twin)

# --- HEADER ---
st.markdown(f"""
<div class="pt-banner">
    <div>
        <div class="pt-id">{profile.mrn}</div>
        <div style="font-size:0.9rem; opacity:0.8;">Septic Shock • 64M • 78kg</div>
    </div>
    <div style="text-align:right;">
        <div style="font-weight:700; color:#4ade80;">● LIVE CONNECTION</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- ALERTS ---
alert_status = "STABLE"
alert_col = "a-ok"
alert_msg = "Hemodynamics Compensated"

if cur["CPO"] < 0.6:
    alert_status = "CRITICAL: POWER FAILURE"
    alert_col = "a-crit"
    alert_msg = "Cardiac Power < 0.6W. High Mortality Risk. Consider Mechanical Support."
elif cur["DSI"] > 2.0:
    alert_status = "VASOPLEGIC COLLAPSE"
    alert_col = "a-warn"
    alert_msg = "Diastolic Shock Index > 2.0. Severe Vasodilation."

st.markdown(f"""
<div class="alert-box {alert_col}">
    <div>
        <div style="font-size:1.2rem; font-weight:800;">{alert_status}</div>
        <div style="font-weight:500;">{alert_msg}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 1. ESSENTIAL METRICS GRID ---
c1, c2, c3, c4, c5, c6 = st.columns(6)

def render_kpi(col, label, val, unit, color, df_col):
    delta = val - prv[df_col]
    color_txt = THEME["crit"] if delta < 0 and df_col in ["MAP", "CPO"] else THEME["text_main"]
    
    with col:
        st.markdown(f"""
        <div class="kpi-tile" style="border-top: 3px solid {color}">
            <div class="kpi-head">{label}</div>
            <div class="kpi-val" style="color:{color_txt}">{val:.2f}</div>
            <div class="kpi-sub" style="color:{THEME['text_sub']}">{unit} • {delta:+.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(Visuals.sparkline(df_stream, df_col, color), use_container_width=True, config={'displayModeBar': False})

render_kpi(c1, "Cardiac Power", cur["CPO"], "Watts", THEME["cpo"], "CPO")
render_kpi(c2, "Diastolic SI", cur["DSI"], "idx", THEME["dsi"], "DSI")
render_kpi(c3, "Ea_dyn", cur["Eadyn"], "Pressure Resp", THEME["accent"], "Eadyn")
render_kpi(c4, "MAP", cur["MAP"], "mmHg", THEME["map"], "MAP")
render_kpi(c5, "Cardiac Index", cur["CI"], "L/min", THEME["ci"], "CI")
render_kpi(c6, "ELWI (Lung H2O)", cur["ELWI"], "mL/kg", THEME["warn"], "ELWI")

# --- 2. ADVANCED HEMODYNAMICS ---
st.markdown('<div class="section-title">🔮 PREDICTIVE & MECHANICS</div>', unsafe_allow_html=True)
col_a, col_b, col_c = st.columns([1, 2, 1])

with col_a:
    # CPO Gauge
    st.plotly_chart(Visuals.plot_cpo_gauge(cur["CPO"]), use_container_width=True, config={'displayModeBar': False})
    st.info("**Why CPO?** Integration of Flow and Pressure. < 0.6W predicts mortality with 80% specificity.")

with col_b:
    # Therapeutic Conflict Matrix
    st.plotly_chart(Visuals.plot_therapeutic_conflict(futures), use_container_width=True, config={'displayModeBar': False})

with col_c:
    # V-A Coupling
    st.plotly_chart(Visuals.plot_coupling(futures), use_container_width=True, config={'displayModeBar': False})
    st.info("**Why Ea_dyn?** > 1.0 means Pressors will raise MAP. < 0.8 means Vasoplegia requires Vasopressin/Angiotensin.")

st.markdown(DISCLAIMER, unsafe_allow_html=True)
