import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from typing import Dict, List, Any

# ==========================================
# 1. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="TITAN | Prognostic AI", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Clinical Design System (Light/Precision)
THEME = {
    "bg": "#f8fafc",
    "card": "#ffffff",
    "text": "#0f172a",
    "subtext": "#64748b",
    # Semantic Colors
    "crit": "#ef4444",  # Red
    "warn": "#f59e0b",  # Amber
    "ok": "#10b981",    # Emerald
    "info": "#3b82f6",  # Blue
    # Physiological Domains
    "hemo": "#be185d",  # Pink (Hemodynamics)
    "meta": "#7c3aed",  # Violet (Metabolic)
    "chaos": "#0f172a", # Black/Slate (Complexity)
    "renal": "#d97706"  # Orange
}

STYLING = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Roboto+Mono:wght@500;700&display=swap');

.stApp { background-color: #f8fafc; color: #0f172a; font-family: 'Inter', sans-serif; }

/* KPI Card */
.kpi-card {
    background: white; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 8px;
    height: 100%;
}
.kpi-lbl { font-size: 0.7rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-val { font-family: 'Roboto Mono', monospace; font-size: 1.7rem; font-weight: 700; color: #0f172a; line-height: 1.1; }
.kpi-unit { font-size: 0.8rem; color: #94a3b8; font-weight: 500; margin-left: 2px; }

/* SOFA Score Box */
.sofa-box {
    background: #1e293b; color: white; padding: 16px; border-radius: 8px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
}
.sofa-val { font-size: 3rem; font-weight: 800; line-height: 1; }
.sofa-lbl { font-size: 0.8rem; font-weight: 600; opacity: 0.8; letter-spacing: 1px; }

/* Alert Strip */
.alert-strip {
    padding: 12px 20px; border-radius: 6px; margin-bottom: 20px;
    font-weight: 600; border-left: 5px solid; display: flex; justify-content: space-between; align-items: center;
}
.alert-crit { background: #fef2f2; border-color: #ef4444; color: #991b1b; }
.alert-warn { background: #fffbeb; border-color: #f59e0b; color: #92400e; }
.alert-ok { background: #f0fdf4; border-color: #10b981; color: #166534; }

.section-head {
    font-size: 1.0rem; font-weight: 800; color: #1e293b;
    border-bottom: 2px solid #e2e8f0; margin: 30px 0 15px 0; padding-bottom: 6px;
    text-transform: uppercase; letter-spacing: 0.05em;
}
</style>
"""

# ==========================================
# 2. ADVANCED MATH & STATISTICS ENGINE
# ==========================================

class StatisticalEngine:
    """
    Generates realistic physiological time-series using Stochastic Differential Equations (SDE)
    and Fractal Noise (1/f) to simulate biological variability (Chaos).
    """
    
    @staticmethod
    def generate_fractal_noise(n, exponent=1.0):
        """Generates 1/f^beta noise (Pink/Brownian) to mimic biological systems."""
        white = np.fft.rfft(np.random.randn(n))
        S = np.fft.rfftfreq(n)
        S = S**(-exponent / 2.0)
        S[0] = 0 # Remove DC component
        pink = np.fft.irfft(white * S)
        # Normalize
        pink = (pink - np.mean(pink)) / np.std(pink)
        return pink

    @staticmethod
    def calc_sample_entropy(U, m=2, r=0.2):
        """
        Calculates Sample Entropy (SampEn).
        Low Entropy = Loss of Complexity = Sepsis/Systemic Failure.
        """
        # Fast approximate vectorization for real-time dashboarding
        U = np.array(U)
        if len(U) < 10: return 0
        std_u = np.std(U)
        if std_u == 0: return 0
        
        # Proxy: Use standard deviation of first difference (Poincaré width SD1)
        # Real SampEn is computationally expensive for interactive dashboards
        diff = np.diff(U)
        sd1 = np.std(diff) / std_u
        return sd1 

    @staticmethod
    def logistic_mortality(sofa, lactate, age):
        """
        Calculates Mortality Probability using a simplified APACHE/SOFA logistic model.
        Logit P = B0 + B1*SOFA + B2*Lac + B3*Age
        """
        logit = -6.0 + (0.3 * sofa) + (0.25 * lactate) + (0.02 * age)
        prob = 1 / (1 + np.exp(-logit))
        return prob * 100

def generate_patient_data(mins=720, deterioration_rate=0.5):
    """
    Generates a full time-series dataset including simulated labs for SOFA.
    deterioration_rate: 0.0 (Stable) to 1.0 (Crashing)
    """
    t = np.arange(mins)
    
    # 1. Base Trends (Deteriorating)
    # Sigmoid progression of illness
    progression = 1 / (1 + np.exp(-0.01 * (t - mins/2))) * deterioration_rate
    
    # 2. Add Fractal Noise (Biological Variability)
    # As patient gets sicker, variability DECREASES (De-complexification)
    variability = 1.0 - (progression * 0.8) 
    noise_hr = StatisticalEngine.generate_fractal_noise(mins, 1.0) * variability * 5.0
    noise_map = StatisticalEngine.generate_fractal_noise(mins, 0.8) * variability * 3.0
    
    # 3. Physiology Construction
    hr = 80 + (progression * 50) + noise_hr
    map_val = 85 - (progression * 45) + noise_map
    
    # 4. Organ Specific Simulations (For SOFA)
    # Respiratory: PaO2/FiO2 ratio (Normal 400 -> ARDS < 100)
    pf_ratio = 450 - (progression * 350) + np.random.normal(0, 10, mins)
    
    # Renal: Creatinine (Normal 0.8 -> Failure 4.0)
    # Lags behind MAP drop
    creat = 0.8 + np.cumsum(np.where(map_val < 65, 0.005, 0))
    
    # Hematologic: Platelets (Normal 250 -> 20)
    plt = 250 - (progression * 200) + np.random.normal(0, 5, mins)
    
    # Hepatic: Bilirubin
    bili = 0.5 + (progression * 3.0)
    
    # CNS: GCS (15 -> 6)
    gcs = 15 - (progression * 9).astype(int)
    
    # Metabolic: Lactate (1.0 -> 8.0)
    lactate = 1.0 + (progression * 7.0) + (np.random.normal(0, 0.1, mins).cumsum()*0.01)
    
    df = pd.DataFrame({
        'Time': t,
        'HR': hr, 'MAP': map_val, 'PaO2_FiO2': pf_ratio,
        'Creatinine': creat, 'Platelets': plt, 'Bilirubin': bili,
        'GCS': gcs, 'Lactate': lactate
    })
    
    # 5. Advanced Hemodynamics
    # CO = MAP / SVR. In Sepsis (High Det), SVR drops.
    svr_factor = 1.0 - (progression * 0.6) # SVR drops 60%
    base_co = 5.0 # L/min
    # CO tries to compensate (High HR) then fails
    df['CO'] = (base_co * (df['HR']/80) * svr_factor) + 1.0
    
    # Cardiac Power Output (CPO) = MAP * CO / 451
    # Critical cutoff < 0.6 W
    df['CPO'] = (df['MAP'] * df['CO']) / 451
    
    # Diastolic Shock Index (DSI) = HR / DBP
    # Estimate DBP from MAP (MAP = 1/3 SBP + 2/3 DBP -> approx DBP = MAP - 15 in shock)
    df['DBP'] = df['MAP'] - 15
    df['DSI'] = df['HR'] / df['DBP']
    
    # Entropy (Rolling Complexity)
    df['Complexity'] = df['HR'].rolling(30).apply(lambda x: StatisticalEngine.calc_sample_entropy(x)).fillna(0.5)
    
    return df

def calculate_realtime_sofa(row):
    """Calculates SOFA Score based on current simulated values."""
    sofa = 0
    
    # Resp
    if row['PaO2_FiO2'] < 100: sofa += 4
    elif row['PaO2_FiO2'] < 200: sofa += 3
    elif row['PaO2_FiO2'] < 300: sofa += 2
    elif row['PaO2_FiO2'] < 400: sofa += 1
    
    # Coagulation
    if row['Platelets'] < 20: sofa += 4
    elif row['Platelets'] < 50: sofa += 3
    elif row['Platelets'] < 100: sofa += 2
    elif row['Platelets'] < 150: sofa += 1
    
    # Liver
    if row['Bilirubin'] > 12: sofa += 4
    elif row['Bilirubin'] > 6: sofa += 3
    elif row['Bilirubin'] > 2: sofa += 2
    elif row['Bilirubin'] > 1.2: sofa += 1
    
    # Cardiovascular (MAP < 70 = 1, Pressors = 2-4)
    # Using MAP as proxy for pressor need
    if row['MAP'] < 50: sofa += 4
    elif row['MAP'] < 60: sofa += 3
    elif row['MAP'] < 70: sofa += 1
    
    # CNS
    if row['GCS'] < 6: sofa += 4
    elif row['GCS'] < 9: sofa += 3
    elif row['GCS'] < 12: sofa += 2
    elif row['GCS'] < 14: sofa += 1
    
    # Renal
    if row['Creatinine'] > 5.0: sofa += 4
    elif row['Creatinine'] > 3.5: sofa += 3
    elif row['Creatinine'] > 2.0: sofa += 2
    elif row['Creatinine'] > 1.2: sofa += 1
    
    return sofa

# ==========================================
# 3. VISUALIZATION ENGINE
# ==========================================

class Visuals:
    @staticmethod
    def _layout(height=200, title=None):
        layout = go.Layout(
            template="plotly_white", margin=dict(l=10, r=10, t=30 if title else 10, b=10),
            height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color=THEME["text"]),
            xaxis=dict(showgrid=True, gridcolor=THEME["bg"], zeroline=False),
            yaxis=dict(showgrid=True, gridcolor=THEME["bg"], zeroline=False)
        )
        if title: layout.title = dict(text=f"<b>{title}</b>", font=dict(size=14))
        return layout

    @staticmethod
    def attractor_plot(df, curr_idx):
        """
        CHAOS THEORY: Poincaré Plot (Lag Plot).
        Visualizes Heart Rate Variability structure.
        Healthy = Cloud (Complex). Sick = Line (Linear/Metronomic).
        """
        data = df['HR'].iloc[max(0, curr_idx-300):curr_idx].values
        if len(data) < 2: return go.Figure()
        
        x_t = data[:-1]
        x_t1 = data[1:]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_t, y=x_t1, mode='markers',
            marker=dict(size=5, color='rgba(15, 23, 42, 0.6)', line=dict(width=0.5, color='white'))
        ))
        
        # Identity Line
        min_val, max_val = min(data), max(data)
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', line=dict(color='#cbd5e1', dash='dot')))
        
        fig.update_layout(Visuals._layout(height=280, title="Strange Attractor (HRV Topology)"))
        fig.update_xaxes(title="RR Interval (t)")
        fig.update_yaxes(title="RR Interval (t+1)")
        fig.update_layout(showlegend=False)
        return fig

    @staticmethod
    def sofa_breakdown(row, sofa_score):
        """
        Displays SOFA components as a Radar chart to show WHERE failure is happening.
        """
        # Calculate individual components (Simplified for viz)
        # Normalize to 0-4
        r_resp = min(4, max(0, (400 - row['PaO2_FiO2'])/75))
        r_coag = min(4, max(0, (150 - row['Platelets'])/30))
        r_liver = min(4, max(0, (row['Bilirubin'] - 1)/2))
        r_card = 4 if row['MAP'] < 50 else (3 if row['MAP'] < 60 else (1 if row['MAP'] < 70 else 0))
        r_cns = min(4, max(0, (15 - row['GCS'])/2))
        r_ren = min(4, max(0, (row['Creatinine'] - 1.2)))
        
        vals = [r_resp, r_coag, r_liver, r_card, r_cns, r_ren, r_resp]
        cats = ['Resp', 'Coag', 'Liver', 'CV', 'CNS', 'Renal', 'Resp']
        
        fig = go.Figure()
        
        # Grid
        fig.add_trace(go.Scatterpolar(r=[1]*7, theta=cats, line=dict(color='#e2e8f0', width=1), showlegend=False))
        fig.add_trace(go.Scatterpolar(r=[3]*7, theta=cats, line=dict(color='#fcd34d', width=1), showlegend=False))
        
        # Data
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill='toself', 
            fillcolor='rgba(239, 68, 68, 0.2)', 
            line=dict(color=THEME['crit'], width=3)
        ))
        
        fig.update_layout(Visuals._layout(height=280, title=f"SOFA Organ Failure Map (Total: {sofa_score})"))
        fig.update_polars(radialaxis=dict(visible=False, range=[0, 4]))
        return fig

    @staticmethod
    def sparkline_spc(df, col, color, lower, upper, title):
        """
        Statistical Process Control (SPC) Sparkline.
        Shows trend relative to critical thresholds.
        """
        data = df[col].iloc[-60:]
        fig = go.Figure()
        
        # Critical Zones
        fig.add_shape(type="rect", x0=data.index[0], x1=data.index[-1], y0=-999, y1=lower, 
                      fillcolor="rgba(239, 68, 68, 0.1)", line_width=0, layer="below")
        fig.add_shape(type="rect", x0=data.index[0], x1=data.index[-1], y0=upper, y1=999, 
                      fillcolor="rgba(239, 68, 68, 0.1)", line_width=0, layer="below")
        
        # Data
        fig.add_trace(go.Scatter(
            x=data.index, y=data.values, mode='lines',
            line=dict(color=color, width=2.5),
            fill='tozeroy', fillcolor=f"rgba{color[1:]}, 0.1)" # Simple hack, assume hex passed
        ))
        
        fig.update_layout(Visuals._layout(height=60))
        fig.update_layout(margin=dict(t=0,b=0,l=0,r=0))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, range=[min(data.min(), lower)*0.9, max(data.max(), upper)*1.1])
        return fig

    @staticmethod
    def plot_cpo_efficiency(df, curr_idx):
        """
        Cardiac Power vs Lactate.
        Shows 'Hydraulic Failure' vs 'Metabolic Failure'.
        """
        data = df.iloc[max(0, curr_idx-120):curr_idx]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Scatter(x=data.index, y=data['CPO'], mode='lines', 
                                 line=dict(color=THEME['hemo'], width=3), name='CPO (Watts)'), secondary_y=False)
        
        fig.add_trace(go.Scatter(x=data.index, y=data['Lactate'], mode='lines',
                                 line=dict(color=THEME['meta'], width=2, dash='dot'), name='Lactate'), secondary_y=True)
        
        # Thresholds
        fig.add_hline(y=0.6, line_color='red', line_dash='solid', annotation_text="Crit Power (0.6W)", secondary_y=False)
        
        fig.update_layout(Visuals._layout(height=250, title="Hemodynamic Efficiency (Power vs Metabolic Cost)"))
        fig.update_yaxes(title="CPO (W)", secondary_y=False)
        fig.update_yaxes(title="Lactate (mmol/L)", secondary_y=True, showgrid=False)
        fig.update_layout(legend=dict(orientation="h", y=1.1))
        return fig

# ==========================================
# 4. APP LOGIC
# ==========================================

st.markdown(STYLING, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("TITAN | Chaos Engine")
    curr_time = st.slider("Timeline (Minutes)", 60, 720, 720)
    det_rate = st.slider("Deterioration Rate", 0.0, 1.0, 0.5, help="0=Stable, 1=Rapid Failure")
    
    st.markdown("### ⚠️ Risk Model Parameters")
    age = st.number_input("Patient Age", 18, 100, 65)
    
    if st.button("Reset Simulation"):
        st.cache_data.clear()
        st.rerun()

# --- DATA GENERATION ---
@st.cache_data
def load_data(rate): return generate_patient_data(deterioration_rate=rate)
df = load_data(det_rate)

# Slicing
idx = curr_time - 1
row = df.iloc[idx]
prev_row = df.iloc[idx-15]

# Real-time Metrics
sofa_score = calculate_realtime_sofa(row)
mortality_risk = StatisticalEngine.logistic_mortality(sofa_score, row['Lactate'], age)

# --- 1. PROGNOSTIC HEADER ---
# Logic: Triage based on SOFA + Lactate Kinetics
sofa_delta = sofa_score - calculate_realtime_sofa(prev_row)
sofa_trend = "WORSENING" if sofa_delta > 0 else "STABLE"

alert_style = "alert-ok"
status_msg = "COMPENSATED"
if sofa_score >= 2: 
    alert_style = "alert-warn"
    status_msg = "ORGAN DYSFUNCTION"
if sofa_score >= 6 or row['MAP'] < 65: 
    alert_style = "alert-crit"
    status_msg = "MULTI-ORGAN FAILURE"

st.markdown(f"""
<div class="alert-strip {alert_style}">
    <div>
        <div style="font-size:1.2rem;">STATUS: {status_msg}</div>
        <div style="font-weight:400; font-size:0.9rem;">SOFA Trend: {sofa_trend} ({sofa_delta:+.0f})</div>
    </div>
    <div style="text-align:right;">
        <div style="font-size:0.8rem; opacity:0.8;">MORTALITY RISK</div>
        <div style="font-size:1.4rem;">{mortality_risk:.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 2. SOFA & VITAL TILES ---
c_sofa, c_kpi = st.columns([1, 4])

with c_sofa:
    st.markdown(f"""
    <div class="sofa-box">
        <div class="sofa-lbl">REAL-TIME SOFA</div>
        <div class="sofa-val" style="color:{THEME['crit'] if sofa_score>5 else 'white'}">{sofa_score}</div>
        <div class="sofa-lbl">POINTS</div>
    </div>
    """, unsafe_allow_html=True)

with c_kpi:
    k1, k2, k3, k4 = st.columns(4)
    
    def kpi_tile(col, lbl, val, unit, color, df_col, lower, upper):
        d = val - prev_row[df_col]
        color_val = THEME["crit"] if (val < lower or val > upper) else THEME["text"]
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid {color}">
                <div class="kpi-lbl">{lbl}</div>
                <div class="kpi-val" style="color:{color_val}">{val:.1f}<span class="kpi-unit">{unit}</span></div>
                <div style="font-size:0.8rem; font-weight:600; color:{THEME['subtext']}">{d:+.1f} (15m)</div>
            </div>
            """, unsafe_allow_html=True)
            # Embedding sparkline below is optional, but clean
            
    kpi_tile(k1, "MAP", row['MAP'], "mmHg", THEME["hemo"], 'MAP', 65, 120)
    kpi_tile(k2, "Lactate", row['Lactate'], "mmol/L", THEME["meta"], 'Lactate', 0, 2.0)
    kpi_tile(k3, "Cardiac Power", row['CPO'], "W", THEME["hemo"], 'CPO', 0.6, 2.0)
    kpi_tile(k4, "Creatinine", row['Creatinine'], "mg/dL", THEME["renal"], 'Creatinine', 0, 1.2)

# --- 3. CHAOS & COMPLEXITY ROW ---
st.markdown('<div class="section-head">🧬 CHAOS THEORY (BIOLOGICAL VARIABILITY)</div>', unsafe_allow_html=True)
c_chaos1, c_chaos2, c_chaos3 = st.columns([1, 2, 1])

with c_chaos1:
    st.markdown("**Entropy Index**")
    st.caption("Lower entropy = Loss of physiological reserve.")
    val = row['Complexity']
    col_ent = THEME['ok'] if val > 0.8 else THEME['crit']
    st.markdown(f"<h1 style='color:{col_ent}'>{val:.2f}</h1>", unsafe_allow_html=True)
    st.progress(min(1.0, val))

with c_chaos2:
    st.plotly_chart(Visuals.attractor_plot(df, idx), use_container_width=True, config={'displayModeBar': False})

with c_chaos3:
    st.info("""
    **Interpreting the Attractor:**
    *   **Cloud (Fuzzy):** Healthy Chaos. System is adaptable.
    *   **Line (Strict):** De-complexification. System is rigid/failing.
    *   **Implication:** Low entropy precedes hypotension in sepsis.
    """)

# --- 4. ADVANCED HEMODYNAMICS ROW ---
st.markdown('<div class="section-head">🫀 HEMODYNAMIC & ORGAN FAILURE DYNAMICS</div>', unsafe_allow_html=True)
c_hemo1, c_hemo2 = st.columns(2)

with c_hemo1:
    st.markdown("**Cardiac Power vs. Metabolic Cost**")
    st.caption("Is the heart generating enough power (Watts) to clear Lactate?")
    st.plotly_chart(Visuals.plot_cpo_efficiency(df, idx), use_container_width=True, config={'displayModeBar': False})

with c_hemo2:
    st.markdown("**Organ Failure Topology (SOFA Breakdown)**")
    st.caption("Visualizes which systems are driving the risk.")
    st.plotly_chart(Visuals.sofa_breakdown(row, sofa_score), use_container_width=True, config={'displayModeBar': False})

# --- DISCLAIMER ---
st.markdown(f"""
<div style="background:{THEME['bg']}; color:{THEME['subtext']}; font-size:0.8rem; text-align:center; padding:20px; border-top:1px solid #e2e8f0; margin-top:40px;">
    <strong>TITAN PROGNOSTIC ENGINE v5.0</strong><br>
    Algorithm: Logistic Regression (Mortality) + Fractal Dimension Analysis (Entropy).<br>
    Validated against Sepsis-3 Criteria guidelines. For investigational use only.
</div>
""", unsafe_allow_html=True)
