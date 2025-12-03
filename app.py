import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ==========================================
# 1. CONFIGURATION & MEDICAL THEME
# ==========================================
st.set_page_config(
    page_title="TITAN | Advanced Critical Care",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🫀"
)

# High-Contrast Medical Palette
THEME = {
    "bg": "#f8fafc",        
    "card_bg": "#ffffff",   
    "text_main": "#0f172a", 
    "text_muted": "#64748b",
    "border": "#e2e8f0",    
    "crit": "#dc2626",      # Clinical Red
    "warn": "#d97706",      # Warning Amber
    "ok": "#059669",        # Stable Green
    "info": "#2563eb",      # Info Blue
    "hemo": "#0891b2",      # Hemodynamic Cyan
    "meta": "#7c3aed",      # Metabolic Violet
    "ai": "#be185d",        # AI Pink
}

STYLING = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Roboto+Mono:wght@500;700&display=swap');
    
    .stApp {{ background-color: {THEME['bg']}; color: {THEME['text_main']}; font-family: 'Inter', sans-serif; }}
    
    /* Compact Metric Cards */
    div[data-testid="stMetric"] {{
        background-color: {THEME['card_bg']};
        padding: 8px 12px;
        border-radius: 6px;
        border: 1px solid {THEME['border']};
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }}
    div[data-testid="stMetric"] label {{ font-size: 0.65rem; font-weight: 700; color: {THEME['text_muted']}; text-transform: uppercase; }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ font-family: 'Roboto Mono'; font-size: 1.5rem; font-weight: 800; }}
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {{ font-size: 0.8rem; }}
    
    /* Headers */
    .zone-header {{
        font-size: 0.85rem;
        font-weight: 900;
        color: {THEME['text_muted']};
        text-transform: uppercase;
        border-bottom: 2px solid {THEME['border']};
        margin: 24px 0 12px 0;
        padding-bottom: 4px;
        letter-spacing: 0.05em;
    }}
    
    /* Banner */
    .status-banner {{
        padding: 12px 20px;
        border-radius: 8px;
        display: flex; justify-content: space-between; align-items: center;
        background: white; border-left: 6px solid;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }}
    .b-crit {{ border-color: {THEME['crit']}; background: #fef2f2; color: #991b1b; }}
    .b-warn {{ border-color: {THEME['warn']}; background: #fffbeb; color: #92400e; }}
    .b-ok   {{ border-color: {THEME['ok']};   background: #f0fdf4; color: #166534; }}
</style>
"""

# ==========================================
# 2. PHYSIOLOGY ENGINE (BROWNIAN MOTION)
# ==========================================
class PhysiologyEngine:
    @staticmethod
    def brownian_bridge(n, start, end, volatility=1.0):
        """Generates a random walk from start to end with specific volatility."""
        t = np.linspace(0, 1, n)
        # Standard Brownian motion
        dW = np.random.normal(0, np.sqrt(1/n), n)
        W = np.cumsum(dW)
        # Brownian Bridge formula
        B = start + W - t * (W[-1] - (end - start))
        # Add high-frequency pink noise
        pink = np.convolve(np.random.normal(0, 0.5, n), np.ones(5)/5, mode='same')
        return B + (pink * volatility)

    @staticmethod
    def calc_svr(map_val, co, cvp=8):
        # SVR = (MAP - CVP) / CO * 80
        # Protect against divide by zero
        co = max(co, 0.1)
        return ((map_val - cvp) / co) * 80

# ==========================================
# 3. PATIENT SIMULATOR
# ==========================================
class PatientSimulator:
    def __init__(self, mins=360):
        self.mins = mins
        self.t = np.arange(mins)

    def get_data(self, profile):
        # 1. Define Boundary Conditions based on Scenario
        if profile == "Healthy":
            p = {'hr': (65, 70), 'map': (85, 88), 'co': (5.0, 5.2), 'vol': 1.5}
        elif profile == "Compensated Sepsis":
            p = {'hr': (85, 105), 'map': (82, 78), 'co': (6.0, 7.5), 'vol': 2.0}
        elif profile == "Vasoplegic Shock":
            p = {'hr': (100, 135), 'map': (75, 48), 'co': (7.0, 8.5), 'vol': 1.0}
        elif profile == "Cardiogenic Shock":
            p = {'hr': (90, 100), 'map': (75, 60), 'co': (3.5, 2.2), 'vol': 3.0}

        # 2. Generate Random Walk Vitals
        hr = PhysiologyEngine.brownian_bridge(self.mins, p['hr'][0], p['hr'][1], volatility=p['vol'])
        map_val = PhysiologyEngine.brownian_bridge(self.mins, p['map'][0], p['map'][1], volatility=p['vol']*0.8)
        co = PhysiologyEngine.brownian_bridge(self.mins, p['co'][0], p['co'][1], volatility=0.2)
        
        # 3. Derived Physiology
        pp_factor = 25 if "Cardio" in profile else (60 if "Vasoplegic" in profile else 40)
        pp_noise = np.random.normal(0, 2, self.mins)
        sbp = map_val + (pp_factor/3)*2 + pp_noise
        dbp = map_val - (pp_factor/3) + pp_noise
        
        # 4. Metabolic coupling (Lactate)
        perf_index = (map_val * co) 
        lactate = np.zeros(self.mins)
        lac_current = 0.8 if profile == "Healthy" else 1.5
        
        for i in range(self.mins):
            production = 0.05 if perf_index[i] < 300 else 0.01 
            clearance = 0.02 * co[i]
            if "Shock" in profile: production *= 1.5
            lac_current = max(0.5, lac_current + production - clearance)
            lactate[i] = lac_current

        # 6. SVR Calculation
        svr = [PhysiologyEngine.calc_svr(m, c) for m, c in zip(map_val, co)]

        # 7. Interventions (Simulated)
        ne_dose = np.zeros(self.mins)
        if map_val[-1] < 60:
            start_shock = int(self.mins * 0.5)
            ne_dose[start_shock:] = np.linspace(0, 0.2, self.mins - start_shock)

        df = pd.DataFrame({
            "Time": self.t, "HR": hr, "MAP": map_val, "DBP": dbp, "CO": co,
            "Lactate": lactate, "Platelets": np.full(self.mins, 200), 
            "Creatinine": np.full(self.mins, 1.0),
            "Norepi_Dose": ne_dose, "SVR": svr
        })
        
        # Derived
        df['CPO'] = (df['MAP'] * df['CO']) / 451
        df['SI'] = df['HR'] / sbp
        
        return df

# ==========================================
# 4. VISUALIZATION ENGINE (ADVANCED)
# ==========================================
def plot_chaos_attractor(df):
    """
    Advanced Poincaré Plot.
    Visualizes Heart Rate Variability (HRV) Complexity.
    """
    # Convert HR to RR intervals (ms)
    rr_series = 60000 / df['HR'].iloc[-120:] # Last 2 hours
    rr_n = rr_series.iloc[:-1].values
    rr_n1 = rr_series.iloc[1:].values
    
    fig = go.Figure()
    
    # Identity Line
    min_rr, max_rr = min(rr_series)-50, max(rr_series)+50
    fig.add_trace(go.Scatter(x=[min_rr, max_rr], y=[min_rr, max_rr], mode='lines', 
                             line=dict(color=THEME['border'], dash='dot', width=1), showlegend=False))
    
    # The Attractor (Color by Time Index to show trajectory)
    fig.add_trace(go.Scatter(
        x=rr_n, y=rr_n1, mode='markers',
        marker=dict(
            color=np.arange(len(rr_n)), 
            colorscale='Teal', 
            size=6, 
            opacity=0.8,
            line=dict(width=0.5, color='white')
        ),
        name='RR Interval'
    ))
    
    fig.update_layout(
        title=dict(text="<b>Autonomic Complexity (Chaos)</b>", font=dict(size=14, color=THEME['text_main'])),
        xaxis=dict(title="RR(n) [ms]", showgrid=True, gridcolor='#f1f5f9', zeroline=False),
        yaxis=dict(title="RR(n+1) [ms]", showgrid=True, gridcolor='#f1f5f9', zeroline=False),
        height=280,
        margin=dict(l=40, r=20, t=40, b=20),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_phase_space(df):
    """
    Hemo-Metabolic Coupling with Clinical Risk Zones.
    X: Cardiac Power Output (Pump Strength)
    Y: Lactate (Metabolic Debt)
    """
    recent = df.iloc[-60:].copy() # Last hour
    
    fig = go.Figure()
    
    # 1. RISK ZONES (Background Shapes)
    # Critical Zone (Low Power, High Lactate)
    fig.add_shape(type="rect", x0=0, x1=0.6, y0=2, y1=15, 
                  fillcolor="rgba(220, 38, 38, 0.1)", line_width=0, layer="below")
    fig.add_annotation(x=0.3, y=10, text="CRITICAL<br>UNCOUPLING", showarrow=False, font=dict(color=THEME['crit'], size=10))
    
    # Safe Zone
    fig.add_shape(type="rect", x0=0.6, x1=2.0, y0=0, y1=2.0, 
                  fillcolor="rgba(16, 185, 129, 0.1)", line_width=0, layer="below")
    fig.add_annotation(x=1.3, y=1, text="COMPENSATED", showarrow=False, font=dict(color=THEME['ok'], size=10))

    # 2. TRAJECTORY (Snail Trail)
    fig.add_trace(go.Scatter(
        x=recent['CPO'], y=recent['Lactate'], 
        mode='lines+markers',
        marker=dict(size=recent.index/recent.index.max()*8, color=recent.index, colorscale='Bluered', showscale=False),
        line=dict(color='rgba(148, 163, 184, 0.4)', width=1),
        name="Trajectory"
    ))
    
    # 3. Current State
    fig.add_trace(go.Scatter(x=[recent.iloc[-1]['CPO']], y=[recent.iloc[-1]['Lactate']], 
                             mode='markers', marker=dict(color=THEME['text_main'], size=14, line=dict(color='white', width=2)),
                             name="Current"))
    
    fig.update_layout(
        title=dict(text="<b>Hemo-Metabolic Phase Space</b>", font=dict(size=14, color=THEME['text_main'])),
        xaxis=dict(title="Cardiac Power Output [Watts]", range=[0, 1.8], gridcolor='#f1f5f9'),
        yaxis=dict(title="Lactate [mmol/L]", range=[0, 12], gridcolor='#f1f5f9'),
        height=280, margin=dict(l=40,r=20,t=40,b=20),
        template="plotly_white", showlegend=False
    )
    return fig

def plot_hemodynamic_profile(df):
    """
    Pump vs Pipes (Forrester-style proxy).
    X: Cardiac Output (Flow)
    Y: SVR (Resistance)
    """
    recent = df.iloc[-60:].copy()
    
    fig = go.Figure()
    
    # Quadrants
    # Low Flow / High Res = Cardiogenic
    fig.add_annotation(x=3.0, y=1400, text="CARDIOGENIC", showarrow=False, font=dict(size=10, color=THEME['text_muted']))
    # High Flow / Low Res = Vasoplegic
    fig.add_annotation(x=8.0, y=600, text="VASOPLEGIC", showarrow=False, font=dict(size=10, color=THEME['text_muted']))
    
    # Crosshairs (Normal values: CO=5, SVR=1000)
    fig.add_hline(y=1000, line_dash="dot", line_color=THEME['border'])
    fig.add_vline(x=5.0, line_dash="dot", line_color=THEME['border'])
    
    fig.add_trace(go.Scatter(
        x=recent['CO'], y=recent['SVR'],
        mode='markers',
        marker=dict(color=recent.index, colorscale='Viridis', size=6, showscale=False),
        name="Hemo Profile"
    ))
     # Current State
    fig.add_trace(go.Scatter(x=[recent.iloc[-1]['CO']], y=[recent.iloc[-1]['SVR']], 
                             mode='markers', marker=dict(color=THEME['text_main'], size=12, symbol="diamond", line=dict(color='white', width=1)),
                             name="Current"))

    fig.update_layout(
        title=dict(text="<b>Hemodynamic Profile (Pump vs Pipes)</b>", font=dict(size=14, color=THEME['text_main'])),
        xaxis=dict(title="Cardiac Output [L/min]", range=[1, 10], gridcolor='#f1f5f9'),
        yaxis=dict(title="SVR [dyn·s/cm⁵]", range=[200, 1800], gridcolor='#f1f5f9'),
        height=280, margin=dict(l=50,r=20,t=40,b=20),
        template="plotly_white", showlegend=False
    )
    return fig

def plot_telemetry(df):
    """
    Full Stack Telemetry with Reference Ranges and Horizon Filling.
    """
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, 
        subplot_titles=("<b>Hemodynamics</b> (HR & MAP)", "<b>Perfusion Adequacy</b> (CPO & Lactate)", "<b>Afterload</b> (SVR)")
    )
    
    # --- ROW 1: HEMODYNAMICS ---
    # Normal MAP band
    fig.add_hrect(y0=65, y1=100, row=1, col=1, fillcolor="green", opacity=0.05, line_width=0)
    
    fig.add_trace(go.Scatter(x=df['Time'], y=df['HR'], name="HR [bpm]", 
                             line=dict(color=THEME['crit'], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Time'], y=df['MAP'], name="MAP [mmHg]", 
                             line=dict(color=THEME['hemo'], width=2)), row=1, col=1)
    
    # --- ROW 2: PERFUSION ---
    # CPO Threshold
    fig.add_hline(y=0.6, line_dash="dash", line_color=THEME['crit'], annotation_text="Shock Threshold", row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df['Time'], y=df['CPO'], name="CPO [W]", 
                             fill='tozeroy', fillcolor='rgba(8, 145, 178, 0.1)', 
                             line=dict(color=THEME['info'], width=2)), row=2, col=1)
    
    # Secondary Y for Lactate? No, keep single axis for clarity, scale fits OK usually, or normalize.
    # Actually, let's put Lactate on secondary y for better visibility
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Lactate'], name="Lactate [mmol/L]", 
                             line=dict(color=THEME['meta'], dash='dot', width=2)), row=2, col=1)
    
    # --- ROW 3: RESISTANCE ---
    fig.add_trace(go.Scatter(x=df['Time'], y=df['SVR'], name="SVR [dyn·s/cm⁵]", 
                             line=dict(color=THEME['warn'], width=2)), row=3, col=1)
    
    # Update Axes
    fig.update_yaxes(title_text="HR / MAP", row=1, col=1, gridcolor='#f1f5f9')
    fig.update_yaxes(title_text="CPO / Lac", row=2, col=1, gridcolor='#f1f5f9')
    fig.update_yaxes(title_text="SVR", row=3, col=1, gridcolor='#f1f5f9')
    fig.update_xaxes(title_text="Time [min]", row=3, col=1, gridcolor='#f1f5f9')

    fig.update_layout(
        height=500, 
        template="plotly_white", 
        margin=dict(l=20,r=20,t=40,b=20), 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ==========================================
# 5. EXECUTION
# ==========================================
st.markdown(STYLING, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("TITAN | CDS")
    st.markdown("<b>CLINICAL COMMAND CENTER</b>", unsafe_allow_html=True)
    scenario = st.selectbox("Simulation Scenario", ["Healthy", "Compensated Sepsis", "Vasoplegic Shock", "Cardiogenic Shock"])
    
    st.divider()
    st.markdown("<b>INTERVENTIONS</b>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: in_fi = st.number_input("FiO2", 0.21, 1.0, 0.40, 0.1)
    with c2: in_ne = st.number_input("Norepi", 0.0, 2.0, 0.0, 0.05)
    st.caption("Brownian Motion Engine Active")

# --- DATA ---
sim = PatientSimulator(mins=360)
df = sim.get_data(scenario)

# Apply Overrides
idx = df.index[-1]
if in_ne > 0: df.loc[idx, 'Norepi_Dose'] = in_ne
cur = df.iloc[-1]
prev_1h = df.iloc[-60]

# Logic
sofa = 0
if cur['MAP'] < 70: sofa += 1
if cur['Platelets'] < 150: sofa += 1
if cur['Creatinine'] > 1.2: sofa += 1
ai_msg = "HEMODYNAMICALLY STABLE"
ai_cls = "b-ok"
if cur['CPO'] < 0.6: 
    ai_msg = "CRITICAL HYPOPERFUSION" 
    ai_cls = "b-crit"
elif cur['Lactate'] > 2.0:
    ai_msg = "METABOLIC STRESS / SEPSIS"
    ai_cls = "b-warn"

# --- LAYOUT ---

# 1. BANNER
st.markdown(f"""
<div class="status-banner {ai_cls}">
    <div>
        <div style="font-size:0.7rem; color:{THEME['ai']}; font-weight:800; letter-spacing:1px;">AI PHENOTYPE CLASSIFICATION</div>
        <div style="font-size:1.8rem; font-weight:800; color:{THEME['text_main']}">{ai_msg}</div>
    </div>
    <div style="text-align:right">
        <div style="font-size:0.7rem; font-weight:700;">SOFA SCORE</div>
        <div style="font-size:2.5rem; font-weight:800; line-height:1;">{sofa}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. ZONE A: VITALS (6-PACK)
st.markdown('<div class="zone-header">ZONE A: IMMEDIATE HEMODYNAMICS</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5, c6 = st.columns(6)
def m(c, l, v, u, d, inv=False):
    c.metric(l, f"{v:.0f} {u}" if v > 10 else f"{v:.2f} {u}", f"{d:+.1f}", delta_color="inverse" if inv else "normal")

m(c1, "MAP", cur['MAP'], "mmHg", cur['MAP']-prev_1h['MAP'])
m(c2, "Heart Rate", cur['HR'], "bpm", cur['HR']-prev_1h['HR'], True)
m(c3, "CPO", cur['CPO'], "W", cur['CPO']-prev_1h['CPO'])
m(c4, "Lactate", cur['Lactate'], "mmol/L", cur['Lactate']-prev_1h['Lactate'], True)
m(c5, "SVR", cur['SVR'], "dyn·s/cm⁵", cur['SVR']-prev_1h['SVR'])
m(c6, "Shock Index", cur['SI'], "bpm/mmHg", cur['SI']-prev_1h['SI'], True)

# 3. ZONE B: ADVANCED DIAGNOSTICS (THE "WHY")
st.markdown('<div class="zone-header">ZONE B: ADVANCED PHYSIOLOGIC PROFILING</div>', unsafe_allow_html=True)
b1, b2, b3 = st.columns(3)

with b1:
    # THE CHAOS PLOT
    st.plotly_chart(plot_chaos_attractor(df), use_container_width=True, config={'displayModeBar': False})
with b2:
    # PHASE SPACE
    st.plotly_chart(plot_phase_space(df), use_container_width=True, config={'displayModeBar': False})
with b3:
    # HEMO PROFILE
    st.plotly_chart(plot_hemodynamic_profile(df), use_container_width=True, config={'displayModeBar': False})

# 4. ZONE C: HISTORY
st.markdown('<div class="zone-header">ZONE C: CONTINUOUS TELEMETRY STACK</div>', unsafe_allow_html=True)
st.plotly_chart(plot_telemetry(df), use_container_width=True, config={'displayModeBar': False})
