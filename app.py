import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import welch

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="TITAN | High-Fidelity CDS", 
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
    "hemo": "#0369a1", "meta": "#7c3aed", "renal": "#d97706", "neuro": "#334155",
    "resp": "#06b6d4", "coag": "#db2777"
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
    Computes validated clinical scores based on Sepsis-3 guidelines.
    """
    
    @staticmethod
    def calculate_sofa(paO2, fiO2, platelets, bilirubin, map_val, ne_dose, gcs, creatinine):
        """
        True SOFA Scoring Logic (Vincent et al. 1996).
        """
        score = 0
        
        # 1. Respiratory (PaO2/FiO2)
        pafi = paO2 / fiO2
        if pafi < 100: score += 4
        elif pafi < 200: score += 3
        elif pafi < 300: score += 2
        elif pafi < 400: score += 1
        
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
        
        # 4. Cardiovascular (Vasopressor dependent)
        # Logic: If on pressors, score is determined by dose. If not, by MAP.
        if ne_dose > 0.1: score += 4
        elif ne_dose > 0: score += 3
        elif map_val < 70: score += 1 
        
        # 5. CNS (Glasgow Coma Scale)
        if gcs < 6: score += 4
        elif gcs < 9: score += 3
        elif gcs < 12: score += 2
        elif gcs < 14: score += 1
        
        # 6. Renal (Creatinine mg/dL)
        if creatinine > 5.0: score += 4
        elif creatinine > 3.5: score += 3
        elif creatinine > 2.0: score += 2
        elif creatinine > 1.2: score += 1
        
        return score, pafi

    @staticmethod
    def calculate_cpo(map_val, co):
        return (map_val * co) / 451

    @staticmethod
    def calculate_entropy(time_series_data):
        if len(time_series_data) < 10: return 1.0
        diff = np.diff(time_series_data)
        return np.std(diff) / (np.std(time_series_data) + 0.01)

# ==========================================
# 3. ADVANCED PHYSIOLOGY ENGINE (KINETICS)
# ==========================================

def simulate_complex_physiology(scenario: str, fi_o2: float, mins=360):
    """
    Generates physiological data with realistic organ kinetics.
    """
    t = np.arange(mins)
    
    # 1/f Noise Generator
    def bio_noise(n, amp=1):
        white = np.random.normal(0, 1, n)
        return np.convolve(white, np.ones(10)/10, mode='same') * amp

    # --- STATE INITIALIZATION ---
    # Base Trajectory (0.0 = Healthy, 1.0 = Max Severity)
    # Sigmoidal progression for shock
    severity = 1 / (1 + np.exp(-0.02 * (t - mins/2))) 
    if scenario == "Healthy Baseline": severity *= 0.0
    elif scenario == "Compensated Sepsis": severity *= 0.4
    elif scenario == "Septic Shock": severity = np.clip(severity + 0.2, 0, 1)
    
    # --- 1. CARDIOVASCULAR (Vasopressor Model) ---
    # Native MAP drops as severity increases
    native_map = 85 - (severity * 50) + bio_noise(mins, 2) # Drops to 35 without help
    
    # Vasopressor Logic (Simulated Autoregulation or Exogenous)
    # If MAP < 65, NE is added to restore it
    ne_dose = np.zeros(mins)
    final_map = np.zeros(mins)
    
    for i in range(mins):
        current_native = native_map[i]
        required_boost = 65 - current_native
        
        if required_boost > 0:
            # NE response curve: 1 mcg/kg/min ~= +60 mmHg (diminishing returns in shock)
            responsiveness = max(0.2, 1.0 - severity[i]) # Vasoplegia reduces response
            dose_needed = required_boost / (40 * responsiveness)
            ne_dose[i] = min(dose_needed, 2.0) # Max dose 2.0
            final_map[i] = current_native + (ne_dose[i] * 40 * responsiveness)
        else:
            final_map[i] = current_native

    hr = 70 + (severity * 60) + (ne_dose * 10) + bio_noise(mins, 4) # Tachycardia
    co = 5.0 + (severity * 3.0) if "Sepsis" in scenario else 5.0 - (severity * 2.0) # Hyper vs Hypodynamic
    
    # --- 2. RESPIRATORY (PaO2/FiO2) ---
    # Shunt fraction increases with severity
    shunt = 0.05 + (severity * 0.4)
    # Alveolar Gas Equation approx
    p_alv = (fi_o2 * 713) - 40/0.8
    # PaO2 drops with shunt
    pao2 = p_alv * (1 - shunt) + bio_noise(mins, 5)
    
    # --- 3. RENAL KINETICS (Creatinine Doubling) ---
    # GFR drops as MAP drops or Severity increases (ATN)
    baseline_cr = 0.9
    cr_trajectory = np.zeros(mins)
    curr_cr = baseline_cr
    
    for i in range(mins):
        # GFR Damage function
        damage = 0
        if final_map[i] < 65: damage += 0.002 # Ischemic hit
        if severity[i] > 0.5: damage += 0.001 # Cytokine hit
        
        # Kinetic accumulation (Zero-order generation, First-order elimination)
        # Simplified: Cr rises if clearance fails
        prod_rate = 0.001 
        clearance = 0.001 * (1.0 - min(1.0, severity[i]*1.5)) # Clearance drops
        
        curr_cr = curr_cr + prod_rate - (curr_cr * clearance)
        cr_trajectory[i] = curr_cr

    # --- 4. HEPATIC (Bilirubin Rise) ---
    # Liver injury is slow.
    bili = 0.8 + np.cumsum(severity * 0.002) 
    
    # --- 5. HEMATOLOGIC (Platelet Consumption) ---
    # DIC Model: Consumption > Production
    # Starts at 250, drops, might nadir and recover
    plt_consumption = severity * 0.5
    plt = 250 - np.cumsum(plt_consumption) + bio_noise(mins, 5)
    plt = np.maximum(10, plt)

    # --- 6. METABOLIC (Lactate) ---
    lactate = 1.0 + (severity * 8.0) + bio_noise(mins, 0.2)

    # GCS Proxy
    gcs = np.maximum(3, 15 - (severity * 10).astype(int))
    
    # Urine Output (Oliguria < 0.5)
    urine = np.maximum(0, 1.5 - (severity * 1.5))

    # Compile
    df = pd.DataFrame({
        "Timestamp": t,
        "HR": hr, "MAP": final_map, "CO": co, "NE_Dose": ne_dose,
        "PaO2": pao2, "FiO2": fi_o2,
        "Creatinine": cr_trajectory, "Bilirubin": bili, "Platelets": plt,
        "Lactate": lactate, "GCS": gcs, "Urine": urine
    })
    
    # Derived
    df["CPO"] = ClinicalCalculator.calculate_cpo(df["MAP"], df["CO"])
    df["Entropy"] = df["HR"].rolling(30).apply(ClinicalCalculator.calculate_entropy).fillna(1.0)
    
    # Calculate Scores row-by-row
    sofa_data = []
    pafi_data = []
    for _, row in df.iterrows():
        s, p = ClinicalCalculator.calculate_sofa(
            row["PaO2"], fi_o2, row["Platelets"], row["Bilirubin"], 
            row["MAP"], row["NE_Dose"], row["GCS"], row["Creatinine"]
        )
        sofa_data.append(s)
        pafi_data.append(p)
        
    df["SOFA"] = sofa_data
    df["PaFi"] = pafi_data
    
    return df

# ==========================================
# 4. VISUALIZATION COMPONENTS
# ==========================================

def plot_sparkline_spc(df, col, color, lower_limit, upper_limit):
    """Sparkline with Safety Corridor."""
    data = df[col].iloc[-60:]
    fig = go.Figure()
    
    # Safety Corridor
    fig.add_shape(type="rect", x0=data.index[0], x1=data.index[-1], y0=lower_limit, y1=upper_limit,
                  fillcolor="rgba(0,0,0,0.03)", line_width=0, layer="below")
    
    fig.add_trace(go.Scatter(
        x=data.index, y=data.values, mode='lines',
        line=dict(color=color, width=2), fill='tozeroy', 
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)"
    ))
    
    fig.add_trace(go.Scatter(x=[data.index[-1]], y=[data.values[-1]], mode='markers',
                             marker=dict(color=color, size=6)))

    fig.update_layout(template="plotly_white", margin=dict(l=0,r=0,t=0,b=0), height=50,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

def plot_organ_kinetics(df, curr_idx):
    """Visualizes the kinetics of Organ Failure (Renal & Hepatic)."""
    data = df.iloc[max(0, curr_idx-180):curr_idx]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Creatinine (Left)
    fig.add_trace(go.Scatter(x=data.index, y=data["Creatinine"], mode='lines',
                             line=dict(color=THEME["renal"], width=3), name="Creatinine"), secondary_y=False)
    
    # Bilirubin (Right)
    fig.add_trace(go.Scatter(x=data.index, y=data["Bilirubin"], mode='lines',
                             line=dict(color=THEME["meta"], width=2, dash='dot'), name="Bilirubin"), secondary_y=True)
    
    fig.update_layout(template="plotly_white", height=250, margin=dict(l=10,r=10,t=30,b=10),
                      title="<b>Organ Dysfunction Kinetics</b>", showlegend=True, legend=dict(orientation="h", y=1.1))
    fig.update_yaxes(title="Creatinine (mg/dL)", secondary_y=False, gridcolor=THEME["bg"])
    fig.update_yaxes(title="Bilirubin (mg/dL)", secondary_y=True, showgrid=False)
    return fig

def plot_respiratory_status(df, curr_idx):
    """Visualizes P/F Ratio bins."""
    data = df.iloc[max(0, curr_idx-120):curr_idx]
    fig = go.Figure()
    
    # ARDS Zones
    fig.add_shape(type="rect", x0=data.index[0], x1=data.index[-1], y0=0, y1=100, fillcolor="rgba(239, 68, 68, 0.2)", line_width=0)
    fig.add_shape(type="rect", x0=data.index[0], x1=data.index[-1], y0=100, y1=200, fillcolor="rgba(245, 158, 11, 0.2)", line_width=0)
    fig.add_shape(type="rect", x0=data.index[0], x1=data.index[-1], y0=200, y1=300, fillcolor="rgba(253, 224, 71, 0.2)", line_width=0)
    
    fig.add_trace(go.Scatter(x=data.index, y=data["PaFi"], mode='lines', line=dict(color=THEME["resp"], width=3)))
    
    fig.update_layout(template="plotly_white", height=250, margin=dict(l=10,r=10,t=30,b=10),
                      title="<b>Respiratory Status (P/F Ratio)</b>")
    fig.update_yaxes(title="PaO2/FiO2", range=[50, 500])
    return fig

def plot_sofa_radar(row, sofa_score):
    """Radar chart of SOFA components."""
    # Calculate components for visualization
    c_resp = 4 if row['PaFi'] < 100 else (3 if row['PaFi'] < 200 else (2 if row['PaFi'] < 300 else (1 if row['PaFi'] < 400 else 0)))
    c_coag = 4 if row['Platelets'] < 20 else (3 if row['Platelets'] < 50 else (2 if row['Platelets'] < 100 else (1 if row['Platelets'] < 150 else 0)))
    c_liver = 4 if row['Bilirubin'] > 12 else (3 if row['Bilirubin'] > 6 else (2 if row['Bilirubin'] > 2 else (1 if row['Bilirubin'] > 1.2 else 0)))
    c_card = 4 if row['NE_Dose'] > 0.1 else (3 if row['NE_Dose'] > 0 else (1 if row['MAP'] < 70 else 0))
    c_cns = 4 if row['GCS'] < 6 else (3 if row['GCS'] < 9 else (2 if row['GCS'] < 12 else (1 if row['GCS'] < 14 else 0)))
    c_ren = 4 if row['Creatinine'] > 5 else (3 if row['Creatinine'] > 3.5 else (2 if row['Creatinine'] > 2 else (1 if row['Creatinine'] > 1.2 else 0)))
    
    vals = [c_resp, c_coag, c_liver, c_card, c_cns, c_ren, c_resp]
    cats = ['Resp\n(P/F)', 'Coag\n(Plt)', 'Liver\n(Bili)', 'CV\n(Pressor)', 'CNS\n(GCS)', 'Renal\n(Cr)', 'Resp']
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[0,1,2,3,4,0], theta=cats, fill='none', line=dict(color='#cbd5e1', dash='dot')))
    fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill='toself', 
                                  fillcolor='rgba(239, 68, 68, 0.2)', 
                                  line=dict(color=THEME['crit'], width=2)))
    
    fig.update_layout(template="plotly_white", margin=dict(l=40,r=40,t=40,b=20), height=300,
                      polar=dict(radialaxis=dict(visible=True, range=[0, 4])), showlegend=False,
                      title=f"<b>SOFA Component Analysis (Total: {sofa_score})</b>")
    return fig

# ==========================================
# 5. APP EXECUTION
# ==========================================

st.markdown(STYLING, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("TITAN | Sentinel")
    st.markdown("### ⚙️ Simulation Context")
    scenario = st.selectbox("Patient Profile", 
                           ["Healthy Baseline", "Compensated Sepsis", "Septic Shock"])
    
    st.markdown("### 🫁 Ventilator Settings")
    fi_o2 = st.slider("FiO2 (%)", 21, 100, 40, 5) / 100.0
    
    curr_time = st.slider("Observation Window (min)", 60, 360, 360)

# --- ENGINE ---
# Ingest Data
df = simulate_complex_physiology(scenario, fi_o2, mins=360)

# Slice
idx = curr_time - 1
row = df.iloc[idx]
prev_row = df.iloc[idx-15]

# Metrics
sofa = int(row["SOFA"])
mortality = 1 / (1 + np.exp(-(-6.0 + 0.3*sofa + 0.25*row['Lactate']))) * 100

# --- 1. PROGNOSTIC HEADER ---
status_txt = "STABLE"
action_txt = "Standard Monitoring"
style = "b-ok"

if row['NE_Dose'] > 0:
    status_txt = "VASOPRESSOR DEPENDENT"
    action_txt = f"Current NE Dose: {row['NE_Dose']:.2f} mcg/kg/min"
    style = "b-crit"
elif sofa >= 2:
    status_txt = "ORGAN DYSFUNCTION"
    action_txt = "Assess Sepsis Bundle Compliance"
    style = "b-warn"

st.markdown(f"""
<div class="banner {style}">
    <div>
        <div style="font-size:1.2rem; font-weight:800;">STATUS: {status_txt}</div>
        <div style="font-weight:400; font-size:0.9rem;">{action_txt}</div>
    </div>
    <div style="text-align:right;">
        <div style="font-size:0.8rem; opacity:0.8;">MORTALITY RISK</div>
        <div style="font-size:1.4rem; font-weight:800;">{mortality:.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 2. SOFA & VITALS ---
c_sofa, c_kpi = st.columns([1, 3])

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
    
    def kpi_card(col, label, val, unit, color, df_col, limits):
        delta = val - prev_row[df_col]
        color_txt = THEME["crit"] if (val < limits[0] or val > limits[1]) else THEME["text"]
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid {color}">
                <div class="kpi-lbl">{label}</div>
                <div class="kpi-val" style="color:{color_txt}">{val:.2f} <span class="kpi-unit">{unit}</span></div>
                <div style="font-size:0.8rem; font-weight:600; color:{THEME['subtext']}">{delta:+.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(plot_sparkline_spc(df.iloc[:curr_time], df_col, color, limits[0], limits[1]), 
                            use_container_width=True, config={'displayModeBar': False})

    kpi_card(k1, "MAP", row['MAP'], "mmHg", THEME["hemo"], "MAP", (65, 110))
    kpi_card(k2, "NE Dose", row['NE_Dose'], "mcg/kg", THEME["hemo"], "NE_Dose", (-0.01, 0.05)) # Upper limit warning
    kpi_card(k3, "PaO2/FiO2", row['PaFi'], "ratio", THEME["resp"], "PaFi", (300, 600))
    kpi_card(k4, "Creatinine", row['Creatinine'], "mg/dL", THEME["renal"], "Creatinine", (0, 1.2))

# --- 3. ORGAN SYSTEMS ---
st.markdown('<div class="section-head">Multi-System Organ Failure Analysis</div>', unsafe_allow_html=True)

col_a, col_b = st.columns([1, 1])

with col_a:
    st.plotly_chart(plot_sofa_radar(row, sofa), use_container_width=True, config={'displayModeBar': False})

with col_b:
    st.plotly_chart(plot_organ_kinetics(df, curr_time), use_container_width=True, config={'displayModeBar': False})

col_c, col_d = st.columns([1, 1])

with col_c:
    st.plotly_chart(plot_respiratory_status(df, curr_time), use_container_width=True, config={'displayModeBar': False})
    
with col_d:
    st.info(f"""
    **Current Physiology Status (T={curr_time} min)**
    *   **Cardiovascular:** {'Vasopressor Dependent' if row['NE_Dose'] > 0 else 'Stable'}. MAP {row['MAP']:.0f} mmHg.
    *   **Respiratory:** P/F Ratio {row['PaFi']:.0f} ({'ARDS' if row['PaFi'] < 300 else 'Normal'}).
    *   **Renal:** Creatinine {row['Creatinine']:.2f} mg/dL (Kinetic rise).
    *   **Hematologic:** Platelets {row['Platelets']:.0f} K/uL.
    """)
