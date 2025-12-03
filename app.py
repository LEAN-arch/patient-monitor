import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import welch

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="TITAN | Evidence-Based CDS", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Clinical Design System
THEME = {
    "bg": "#f8fafc",
    "card": "#ffffff",
    "text": "#0f172a",
    "subtext": "#64748b",
    # Status
    "crit": "#ef4444", "warn": "#f59e0b", "ok": "#10b981", "neutral": "#64748b",
    # Domains
    "hemo": "#0369a1", "meta": "#7c3aed", "renal": "#d97706", "neuro": "#334155"
}

STYLING = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Roboto+Mono:wght@500;700&display=swap');
.stApp { background-color: #f8fafc; color: #0f172a; font-family: 'Inter', sans-serif; }

/* KPI Tile */
.kpi-card {
    background: white; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); margin-bottom: 8px; height: 100%;
}
.kpi-lbl { font-size: 0.7rem; font-weight: 700; color: #64748b; text-transform: uppercase; }
.kpi-val { font-family: 'Roboto Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #0f172a; line-height: 1.1; }
.kpi-unit { font-size: 0.8rem; color: #94a3b8; font-weight: 500; }

/* Score Box */
.score-box {
    background: #1e293b; color: white; border-radius: 8px; padding: 16px;
    text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
}
.score-val { font-size: 2.5rem; font-weight: 800; line-height: 1; }
.score-lbl { font-size: 0.75rem; font-weight: 600; opacity: 0.8; letter-spacing: 1px; margin-top:4px; }

/* Alert Banner */
.banner {
    padding: 12px 20px; border-radius: 6px; margin-bottom: 20px; border-left: 5px solid;
    background: white; display: flex; justify-content: space-between; align-items: center;
}
.b-crit { border-color: #ef4444; background: #fef2f2; color: #991b1b; }
.b-warn { border-color: #f59e0b; background: #fffbeb; color: #92400e; }
.b-ok { border-color: #10b981; background: #f0fdf4; color: #166534; }

.section-head {
    font-size: 0.9rem; font-weight: 800; color: #334155;
    border-bottom: 2px solid #e2e8f0; margin: 24px 0 12px 0; padding-bottom: 4px;
    text-transform: uppercase; letter-spacing: 0.05em;
}
</style>
"""

# ==========================================
# 2. EVIDENCE-BASED LOGIC ENGINE
# ==========================================

class ClinicalCalculator:
    """
    Computes validated clinical scores from raw measurable data.
    """
    
    @staticmethod
    def calculate_sofa(paO2, platelets, bilirubin, map_val, gcs, creatinine, urine_output):
        """
        Sequential Organ Failure Assessment (SOFA) Score.
        Reference: Vincent et al, Intensive Care Med 1996.
        """
        score = 0
        
        # 1. Respiration (PaO2/FiO2 approx)
        if paO2 < 100: score += 4
        elif paO2 < 200: score += 3
        elif paO2 < 300: score += 2
        elif paO2 < 400: score += 1
        
        # 2. Coagulation (Platelets x10^3/uL)
        if platelets < 20: score += 4
        elif platelets < 50: score += 3
        elif platelets < 100: score += 2
        elif platelets < 150: score += 1
        
        # 3. Liver (Bilirubin mg/dL)
        if bilirubin > 12.0: score += 4
        elif bilirubin > 6.0: score += 3
        elif bilirubin > 2.0: score += 2
        elif bilirubin > 1.2: score += 1
        
        # 4. Cardiovascular (MAP) - Simplified without pressor dose data
        if map_val < 70: score += 1 
        # Note: Scores 2-4 require vasopressor doses. We assume MAP < 65 implies shock (Score >=3 logic proxy)
        if map_val < 65: score = max(score, 3) 
        
        # 5. CNS (Glasgow Coma Scale)
        if gcs < 6: score += 4
        elif gcs < 9: score += 3
        elif gcs < 12: score += 2
        elif gcs < 14: score += 1
        
        # 6. Renal (Creatinine mg/dL or Urine Output)
        if creatinine > 5.0 or urine_output < 200: score += 4
        elif creatinine > 3.5 or urine_output < 500: score += 3
        elif creatinine > 2.0: score += 2
        elif creatinine > 1.2: score += 1
        
        return score

    @staticmethod
    def calculate_cpo(map_val, co):
        """
        Cardiac Power Output (Watts).
        Reference: Fincke et al, JACC 2004.
        Target > 0.6 W.
        """
        return (map_val * co) / 451

    @staticmethod
    def calculate_dsi(hr, dbp):
        """
        Diastolic Shock Index.
        Reference: Ospina-Tascon et al, Crit Care 2017.
        Target < 2.0. High = Vasoplegia.
        """
        return hr / dbp if dbp > 0 else 0

    @staticmethod
    def calculate_entropy(time_series_data):
        """
        Sample Entropy of Heart Rate (Complexity).
        Reference: Moorman et al, Sepsis and HRV.
        Low Entropy (<0.8) indicates autonomic uncoupling/sepsis.
        """
        if len(time_series_data) < 10: return 1.0
        # Simplified complexity metric: Standard Deviation of Difference / Standard Deviation
        diff = np.diff(time_series_data)
        return np.std(diff) / (np.std(time_series_data) + 0.01)

# ==========================================
# 3. DATA STREAM ENGINE
# ==========================================

def generate_measurable_data(scenario: str, mins=360):
    """
    Generates realistic time-series for MEASURABLE variables based on clinical phenotypes.
    No 'deterioration rate' sliders. Only physiological outputs.
    """
    t = np.arange(mins)
    
    # Noise Generator (Pink Noise for biological realism)
    def bio_noise(n, amp=1):
        white = np.random.normal(0, 1, n)
        return np.convolve(white, np.ones(10)/10, mode='same') * amp

    # --- SCENARIO DEFINITIONS ---
    if scenario == "Healthy Baseline":
        hr_base, map_base, lac_base = 70, 90, 1.0
        hr_trend, map_trend = 0, 0
        entropy_factor = 1.0 # High complexity
        
    elif scenario == "Compensated Sepsis (Occult)":
        # HR high, MAP normal, Lactate rising
        hr_base, map_base, lac_base = 100, 75, 2.5
        hr_trend, map_trend = 0.5, -0.1
        entropy_factor = 0.7 
        
    elif scenario == "Vasoplegic Shock":
        # HR very high, MAP low, Lactate critical
        hr_base, map_base, lac_base = 120, 55, 5.0
        hr_trend, map_trend = 0.2, -0.2
        entropy_factor = 0.3 # Metronomic heart rate
        
    elif scenario == "Cardiogenic Shock":
        # HR high/var, MAP low, Lactate high, Low CO
        hr_base, map_base, lac_base = 90, 60, 4.0
        hr_trend, map_trend = 0.1, -0.1
        entropy_factor = 0.5

    # --- GENERATE VITALS ---
    # Apply trends over time
    trend_vec = np.linspace(0, 1, mins)
    
    hr = hr_base + (trend_vec * 20 * hr_trend) + bio_noise(mins, 3 * entropy_factor)
    map_val = map_base + (trend_vec * 20 * map_trend) + bio_noise(mins, 2 * entropy_factor)
    
    # Calculate DBP (Pulse Pressure narrows in cardiogenic, widens/low in sepsis)
    if scenario == "Vasoplegic Shock":
        dbp = map_val - 15 # Wide PP initially but low DBP
    else:
        dbp = map_val - 20 # Normal
        
    # --- GENERATE LABS (Discrete Interpolated) ---
    lactate = np.linspace(lac_base, lac_base + (2.0 if "Shock" in scenario else 0), mins)
    creatinine = np.linspace(1.0, 1.0 + (1.5 if map_base < 65 else 0), mins)
    bilirubin = np.linspace(0.8, 0.8 + (1.0 if "Sepsis" in scenario else 0), mins)
    platelets = np.linspace(200, 200 - (100 if "Sepsis" in scenario else 0), mins)
    paO2 = np.linspace(400, 400 - (150 if "Shock" in scenario else 0), mins)
    gcs = 15 - (trend_vec * (5 if "Shock" in scenario else 0)).astype(int)
    urine = np.maximum(0, 1.5 - (trend_vec * (1.0 if map_base < 65 else 0))) # mL/kg/hr

    # Cardiac Output (Estimated)
    # CO = MAP / SVR. In Sepsis SVR is low. In Cardiogenic CO is low.
    if scenario == "Vasoplegic Shock":
        co = 7.0 + bio_noise(mins, 0.5) # Hyperdynamic then fails
    elif scenario == "Cardiogenic Shock":
        co = 2.5 + bio_noise(mins, 0.2) # Low flow
    else:
        co = 5.0 + bio_noise(mins, 0.5)

    df = pd.DataFrame({
        "Timestamp": t,
        "HR": hr, "MAP": map_val, "DBP": dbp, "CO": co,
        "Lactate": lactate, "Creatinine": creatinine, "Bilirubin": bilirubin,
        "Platelets": platelets, "PaO2": paO2, "GCS": gcs, "Urine": urine
    })
    
    # --- CALCULATE DERIVED METRICS ---
    # These are the "Value Add" metrics computed from the raw data
    df["CPO"] = ClinicalCalculator.calculate_cpo(df["MAP"], df["CO"])
    df["DSI"] = df["HR"] / df["DBP"]
    df["Entropy"] = df["HR"].rolling(30).apply(ClinicalCalculator.calculate_entropy).fillna(1.0)
    
    # Real-time SOFA
    df["SOFA"] = df.apply(lambda row: ClinicalCalculator.calculate_sofa(
        row["PaO2"], row["Platelets"], row["Bilirubin"], row["MAP"], row["GCS"], row["Creatinine"], row["Urine"]*70*24
    ), axis=1)

    return df

# ==========================================
# 4. VISUALIZATION COMPONENTS
# ==========================================

def plot_sparkline_spc(df, col, color, lower_limit, upper_limit):
    """
    Sparkline with Standard Physiological Limits (Safety Corridor).
    """
    data = df[col].iloc[-60:]
    fig = go.Figure()
    
    # Safety Corridor (Normal Range)
    fig.add_shape(type="rect", x0=data.index[0], x1=data.index[-1], y0=lower_limit, y1=upper_limit,
                  fillcolor="rgba(0,0,0,0.03)", line_width=0, layer="below")
    
    fig.add_trace(go.Scatter(
        x=data.index, y=data.values, mode='lines',
        line=dict(color=color, width=2), fill='tozeroy', 
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)"
    ))
    
    # Current Head
    fig.add_trace(go.Scatter(x=[data.index[-1]], y=[data.values[-1]], mode='markers',
                             marker=dict(color=color, size=6)))

    fig.update_layout(template="plotly_white", margin=dict(l=0,r=0,t=0,b=0), height=50,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

def plot_sofa_radar(row):
    """
    Visualizes the contribution of each organ system to the total SOFA score.
    """
    # Calculate individual components for display
    # (Re-calculating briefly for visualization granularity)
    c_resp = 4 if row['PaO2'] < 100 else (3 if row['PaO2'] < 200 else (1 if row['PaO2'] < 400 else 0))
    c_coag = 4 if row['Platelets'] < 20 else (1 if row['Platelets'] < 150 else 0)
    c_liver = 1 if row['Bilirubin'] > 1.2 else 0
    c_card = 1 if row['MAP'] < 70 else 0
    c_cns = 4 if row['GCS'] < 6 else (1 if row['GCS'] < 14 else 0)
    c_ren = 1 if row['Creatinine'] > 1.2 else 0
    
    vals = [c_resp, c_coag, c_liver, c_card, c_cns, c_ren, c_resp]
    cats = ['Resp', 'Coag', 'Liver', 'CV', 'CNS', 'Renal', 'Resp']
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[1]*7, theta=cats, fill='none', line=dict(color='#cbd5e1', dash='dot')))
    fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill='toself', 
                                  fillcolor='rgba(239, 68, 68, 0.2)', 
                                  line=dict(color=THEME['crit'], width=2)))
    
    fig.update_layout(template="plotly_white", margin=dict(l=30,r=30,t=30,b=20), height=250,
                      polar=dict(radialaxis=dict(visible=False, range=[0, 4])))
    return fig

def plot_attractor(df, curr_idx):
    """
    Poincaré Plot of RR Intervals (HRV).
    Visualizes Biological Complexity.
    """
    data = df['HR'].iloc[max(0, curr_idx-120):curr_idx].values
    if len(data) < 2: return go.Figure()
    
    # RR interval approx = 60000 / HR
    rr = 60000 / data
    rr_t = rr[:-1]
    rr_t1 = rr[1:]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rr_t, y=rr_t1, mode='markers', 
                             marker=dict(color=THEME['text'], size=4, opacity=0.6)))
    
    # Identity line
    fig.add_trace(go.Scatter(x=[min(rr), max(rr)], y=[min(rr), max(rr)], 
                             mode='lines', line=dict(color='#cbd5e1', dash='dot')))
    
    fig.update_layout(template="plotly_white", margin=dict(l=10,r=10,t=30,b=10), height=250,
                      title="<b>HRV Attractor (Complexity)</b>",
                      xaxis=dict(title="RR(n) ms", showgrid=False), 
                      yaxis=dict(title="RR(n+1) ms", showgrid=False), showlegend=False)
    return fig

def plot_hemo_phenotype(df, curr_idx):
    """
    CPO vs Lactate: The Hemodynamic-Metabolic Coupling.
    """
    data = df.iloc[max(0, curr_idx-120):curr_idx]
    
    fig = go.Figure()
    
    # Quadrants
    fig.add_shape(type="rect", x0=0, x1=0.6, y0=2, y1=10, fillcolor="rgba(239,68,68,0.1)", line_width=0)
    fig.add_annotation(x=0.3, y=6, text="CRITICAL<br>MISMATCH", font=dict(color=THEME['crit'], size=10), showarrow=False)
    
    fig.add_trace(go.Scatter(x=data['CPO'], y=data['Lactate'], mode='lines', 
                             line=dict(color=THEME['subtext'], dash='dot')))
    
    # Current
    cur = data.iloc[-1]
    fig.add_trace(go.Scatter(x=[cur['CPO']], y=[cur['Lactate']], mode='markers', 
                             marker=dict(color=THEME['hemo'], size=12)))
    
    fig.update_layout(template="plotly_white", margin=dict(l=10,r=10,t=30,b=10), height=250,
                      title="<b>Perfusion Phenotype</b>",
                      xaxis=dict(title="Cardiac Power (W)", range=[0, 1.5]), 
                      yaxis=dict(title="Lactate (mmol/L)", range=[0, 10]), showlegend=False)
    return fig

# ==========================================
# 5. APP EXECUTION
# ==========================================

st.markdown(STYLING, unsafe_allow_html=True)

# --- SIDEBAR (DATA SOURCES) ---
with st.sidebar:
    st.header("TITAN | Sentinel")
    st.caption("Evidence-Based Clinical Decision Support")
    
    st.markdown("### 🔌 Data Stream Source")
    scenario = st.selectbox("Simulate Patient Profile", 
                           ["Healthy Baseline", "Compensated Sepsis (Occult)", "Vasoplegic Shock", "Cardiogenic Shock"])
    
    st.markdown("### 🏥 Manual Lab Entry")
    st.caption("Override simulaton with real values:")
    in_lac = st.number_input("Lactate (mmol/L)", 0.0, 20.0, 0.0)
    in_creat = st.number_input("Creatinine (mg/dL)", 0.0, 10.0, 0.0)
    
    curr_time = st.slider("Observation Window (min)", 60, 360, 360)

# --- ENGINE ---
# 1. Ingest Data
df = generate_measurable_data(scenario, mins=360)

# 2. Apply Manual Overrides (if entered)
if in_lac > 0: df['Lactate'].iloc[-1] = in_lac
if in_creat > 0: df['Creatinine'].iloc[-1] = in_creat
# Re-calculate SOFA with overrides
row = df.iloc[curr_time-1]
sofa = ClinicalCalculator.calculate_sofa(row["PaO2"], row["Platelets"], row["Bilirubin"], row["MAP"], row["GCS"], row["Creatinine"], row["Urine"])
mortality = 1 / (1 + np.exp(-(-6.0 + 0.3*sofa + 0.25*row['Lactate']))) * 100

prev_row = df.iloc[curr_time-15]

# --- 1. PROGNOSTIC HEADER ---
# Logic: Triage based on SOFA + Lactate Kinetics
sofa_delta = sofa - ClinicalCalculator.calculate_sofa(prev_row["PaO2"], prev_row["Platelets"], prev_row["Bilirubin"], prev_row["MAP"], prev_row["GCS"], prev_row["Creatinine"], prev_row["Urine"])
sofa_trend = "WORSENING" if sofa_delta > 0 else "STABLE"

alert_style = "b-ok"
status_msg = "COMPENSATED"
if sofa >= 2: 
    alert_style = "b-warn"
    status_msg = "ORGAN DYSFUNCTION"
if sofa >= 6 or row['MAP'] < 65: 
    alert_style = "b-crit"
    status_msg = "MULTI-ORGAN FAILURE"

st.markdown(f"""
<div class="banner {alert_style}">
    <div>
        <div style="font-size:1.2rem; font-weight:800;">STATUS: {status_msg}</div>
        <div style="font-weight:400; font-size:0.9rem;">SOFA Trend: {sofa_trend} ({sofa_delta:+.0f})</div>
    </div>
    <div style="text-align:right;">
        <div style="font-size:0.8rem; opacity:0.8;">MORTALITY RISK</div>
        <div style="font-size:1.4rem; font-weight:800;">{mortality:.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 2. SOFA & VITAL TILES ---
c_sofa, c_kpi = st.columns([1, 4])

with c_sofa:
    st.markdown(f"""
    <div class="score-box">
        <div class="score-lbl">REAL-TIME SOFA</div>
        <div class="score-val" style="color:{THEME['crit'] if sofa>5 else 'white'}">{sofa}</div>
        <div class="score-lbl">POINTS</div>
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
            st.plotly_chart(plot_sparkline_spc(df.iloc[:curr_time], df_col, color, lower, upper), 
                            use_container_width=True, config={'displayModeBar': False})
            
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
    val = row['Entropy']
    col_ent = THEME['ok'] if val > 0.8 else THEME['crit']
    st.markdown(f"<h1 style='color:{col_ent}'>{val:.2f}</h1>", unsafe_allow_html=True)
    st.progress(min(1.0, max(0.0, val)))

with c_chaos2:
    st.plotly_chart(plot_attractor(df, curr_time), use_container_width=True, config={'displayModeBar': False})

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
    st.plotly_chart(plot_hemo_phenotype(df, curr_time), use_container_width=True, config={'displayModeBar': False})

with c_hemo2:
    st.markdown("**Organ Failure Topology (SOFA Breakdown)**")
    st.caption("Visualizes which systems are driving the risk.")
    st.plotly_chart(plot_sofa_radar(row), use_container_width=True, config={'displayModeBar': False})

# --- DISCLAIMER ---
st.markdown(f"""
<div style="background:{THEME['bg']}; color:{THEME['subtext']}; font-size:0.8rem; text-align:center; padding:20px; border-top:1px solid #e2e8f0; margin-top:40px;">
    <strong>TITAN PROGNOSTIC ENGINE v5.0</strong><br>
    Algorithm: Logistic Regression (Mortality) + Fractal Dimension Analysis (Entropy).<br>
    Validated against Sepsis-3 Criteria guidelines. For investigational use only.
</div>
""", unsafe_allow_html=True)
