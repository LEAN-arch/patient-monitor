import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import welch

# ==========================================
# 1. SYSTEM CONFIGURATION & STYLING (LIGHT THEME)
# ==========================================
st.set_page_config(
    page_title="TITAN | AI-Enhanced CDS",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

# Light Theme Palette (Medical/Clean)
THEME = {
    "bg": "#f8fafc",        # Slate-50
    "card_bg": "#ffffff",   # White
    "text_main": "#0f172a", # Slate-900
    "text_muted": "#64748b",# Slate-500
    "border": "#e2e8f0",    # Slate-200
    "crit": "#ef4444",      # Red-500
    "warn": "#f59e0b",      # Amber-500
    "ok": "#10b981",        # Emerald-500
    "info": "#3b82f6",      # Blue-500
    "hemo": "#0284c7",      # Sky-600
    "meta": "#7c3aed",      # Violet-600
    "ai": "#db2777",        # Pink-600
}

STYLING = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Roboto+Mono:wght@500;700&display=swap');
    
    /* Global Reset */
    .stApp {{
        background-color: {THEME['bg']};
        color: {THEME['text_main']};
        font-family: 'Inter', sans-serif;
    }}
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {{
        background-color: {THEME['card_bg']};
        padding: 15px;
        border-radius: 8px;
        border: 1px solid {THEME['border']};
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }}
    
    div[data-testid="stMetric"] label {{
        color: {THEME['text_muted']};
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }}
    
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        font-family: 'Roboto Mono', monospace;
        font-size: 1.8rem;
        color: {THEME['text_main']};
        font-weight: 700;
    }}

    /* Custom Classes */
    .section-header {{
        font-size: 0.95rem;
        font-weight: 800;
        color: {THEME['text_muted']};
        margin-top: 2rem;
        margin-bottom: 0.75rem;
        border-bottom: 2px solid {THEME['border']};
        padding-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .status-banner {{
        padding: 1.25rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-weight: bold;
        background: white;
        border-left-width: 6px;
        border-left-style: solid;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }}
    
    /* Alert Styles */
    .b-crit {{ border-left-color: {THEME['crit']}; background: #fef2f2; color: #991b1b; }}
    .b-warn {{ border-left-color: {THEME['warn']}; background: #fffbeb; color: #92400e; }}
    .b-ok   {{ border-left-color: {THEME['ok']};   background: #f0fdf4; color: #166534; }}
    .b-ai   {{ border-left-color: {THEME['ai']};   background: #fdf2f8; color: #831843; }}
</style>
"""

# ==========================================
# 2. CLINICAL ENGINE (LOGIC LAYER)
# ==========================================
class ClinicalEngine:
    """Evidence-based calculations for critical care metrics."""
    
    @staticmethod
    def calculate_sofa_components(row):
        """
        True PaO2/FiO2 and Pressor-based CV Scoring.
        """
        # --- 1. Respiration (PaO2/FiO2) ---
        fi = max(0.21, row['FiO2'])
        pafi = row['PaO2'] / fi
        
        resp = 0
        if pafi < 100: resp = 4
        elif pafi < 200: resp = 3
        elif pafi < 300: resp = 2
        elif pafi < 400: resp = 1
        
        # --- 2. Coagulation (Platelets) ---
        coag = 0
        if row['Platelets'] < 20: coag = 4
        elif row['Platelets'] < 50: coag = 3
        elif row['Platelets'] < 100: coag = 2
        elif row['Platelets'] < 150: coag = 1
        
        # --- 3. Liver (Bilirubin) ---
        liver = 0
        if row['Bilirubin'] > 12.0: liver = 4
        elif row['Bilirubin'] > 6.0: liver = 3
        elif row['Bilirubin'] > 2.0: liver = 2
        elif row['Bilirubin'] > 1.2: liver = 1
        
        # --- 4. Cardiovascular (Vasopressor Model) ---
        ne_dose = row['Norepi_Dose']
        map_val = row['MAP']
        
        cv = 0
        if ne_dose > 0.1: cv = 4
        elif ne_dose > 0: cv = 3
        elif map_val < 70: cv = 1
        
        # --- 5. CNS (GCS) ---
        cns = 0
        if row['GCS'] < 6: cns = 4
        elif row['GCS'] < 9: cns = 3
        elif row['GCS'] < 12: cns = 2
        elif row['GCS'] < 14: cns = 1
        
        # --- 6. Renal (Creatinine) ---
        renal = 0
        if row['Creatinine'] > 5.0: renal = 4
        elif row['Creatinine'] > 3.5: renal = 3
        elif row['Creatinine'] > 2.0: renal = 2
        elif row['Creatinine'] > 1.2: renal = 1
        
        return {
            "Resp": resp, "Coag": coag, "Liver": liver, 
            "CV": cv, "CNS": cns, "Renal": renal,
            "Total": resp + coag + liver + cv + cns + renal,
            "PaFi": pafi
        }

    @staticmethod
    def calculate_hemodynamics(row):
        """Advanced derived hemodynamic variables."""
        # Cardiac Power Output (Watts)
        cpo = (row['MAP'] * row['CO']) / 451
        
        # Diastolic Shock Index (DSI)
        dsi = row['HR'] / row['DBP'] if row['DBP'] > 0 else 0
        
        # Shock Index = HR / SBP
        sbp = (3 * row['MAP']) - (2 * row['DBP'])
        si = row['HR'] / sbp if sbp > 0 else 0
        
        # SVR (Dynes)
        svr = ((row['MAP'] - 10) / row['CO']) * 80
        
        return cpo, si, dsi, svr

    @staticmethod
    def complexity_index(time_series):
        """HRV / Entropy Calculation."""
        if len(time_series) < 20: return 1.0
        ts = np.array(time_series)
        diff = np.diff(ts)
        return np.std(diff) / (np.std(ts) + 0.001)

    @staticmethod
    def ai_phenotype_classifier(row, entropy, lactate_slope):
        """Mock AI Classifier (Deterministic Decision Tree)."""
        cpo = row['CPO']
        svr = row['SVR']
        dsi = row['DSI']
        map_val = row['MAP']
        ne_dose = row['Norepi_Dose']
        
        # Decision Logic
        if map_val < 65 or ne_dose > 0:
            if cpo < 0.6:
                if svr > 1200:
                    return "Cardiogenic Shock", "High SVR + Low CPO", "b-crit"
                else:
                    return "Mixed/Metabolic Shock", "Low SVR + Low CPO", "b-crit"
            else:
                if dsi > 2.0 or svr < 800:
                    return "Vasoplegic Shock", "High DSI + Low SVR", "b-crit"
                elif lactate_slope > 0.1:
                    return "Cryptic Sepsis", "Rising Lactate + Preserved CPO", "b-warn"
                else:
                    return "Obstructive/Other", "Atypical Hemodynamics", "b-warn"
        else:
            if lactate_slope > 0.05 and entropy < 0.8:
                return "Compensated Sepsis", "Loss of Complexity + Lactate Rise", "b-warn"
            elif entropy < 0.5:
                return "Pre-Shock State", "Autonomic Uncoupling", "b-ai"
            else:
                return "Hemodynamically Stable", "Normal Parameters", "b-ok"

# ==========================================
# 3. PATIENT SIMULATOR (KINETIC ENHANCED)
# ==========================================
class PatientSimulator:
    def __init__(self, mins=360):
        self.mins = mins
        self.t = np.arange(mins)

    def _generate_noise(self, n, amp=1):
        white = np.random.normal(0, 1, n)
        b = np.ones(10)/10
        pink = np.convolve(white, b, mode='same')
        rsa = np.sin(np.linspace(0, n/4, n)) * 0.3
        return (pink + rsa) * amp

    def get_data(self, profile):
        trend = np.linspace(0, 1, self.mins)
        
        # --- SCENARIO PARAMETERS ---
        if profile == "Healthy":
            hr_base, hr_end = 70, 75
            map_base, map_end = 90, 85
            co_base = 5.0
            fi_base = 0.21
            ne_base = 0.0
            complexity = 1.0
            creat_func = lambda t: 0.8
            bili_func = lambda t: 0.6
            plt_func = lambda t: 250
            
        elif profile == "Vasoplegic Shock":
            hr_base, hr_end = 90, 130
            map_base, map_end = 75, 50
            co_base = 7.0 
            fi_base = 0.4
            ne_base = 0.05
            complexity = 0.4 
            creat_func = lambda t: 0.8 * (2 ** (t / (60*12))) 
            plt_func = lambda t: 100 + (150 * np.exp(-t/200))
            bili_func = lambda t: 0.8 + (t/100)
            
        elif profile == "Cardiogenic Shock":
            hr_base, hr_end = 90, 110
            map_base, map_end = 80, 60
            co_base = 2.5
            fi_base = 0.6
            ne_base = 0.1
            complexity = 0.6
            creat_func = lambda t: 1.0 + (t/150)
            bili_func = lambda t: 1.2 + (t/200)
            plt_func = lambda t: 200
            
        elif profile == "Compensated Sepsis":
            hr_base, hr_end = 80, 105
            map_base, map_end = 85, 75
            co_base = 6.0
            fi_base = 0.3
            ne_base = 0.0
            complexity = 0.7
            creat_func = lambda t: 0.9
            bili_func = lambda t: 0.7
            plt_func = lambda t: 220
            
        # --- GENERATE VITALS ---
        hr = np.linspace(hr_base, hr_end, self.mins) + self._generate_noise(self.mins, 3 * complexity)
        map_val = np.linspace(map_base, map_end, self.mins) + self._generate_noise(self.mins, 2 * complexity)
        co = np.linspace(co_base, co_base * (0.8 if "Cardio" in profile else 1.1), self.mins) + self._generate_noise(self.mins, 0.2)
        
        # --- GENERATE INTERVENTIONS & LABS ---
        ne_dose = np.zeros(self.mins)
        if "Shock" in profile:
            ne_dose = ne_base + (0.3 / (1 + np.exp(-0.05 * (self.t - 180))))
            
        fio2 = np.linspace(fi_base, fi_base + 0.2, self.mins)
        
        # FIX: Ensure variable name matches definition
        pao2 = np.linspace(400, 400 - (200 if "Shock" in profile else 0), self.mins)
        
        creat = np.array([creat_func(x) for x in self.t])
        bili = np.array([bili_func(x) for x in self.t])
        plt = np.array([plt_func(x) for x in self.t])
        
        if "Shock" in profile:
            lactate = 1.0 + 8.0 * (trend ** 2)
        elif "Compensated" in profile:
            lactate = 1.0 + 2.0 * trend
        else:
            lactate = np.full(self.mins, 0.9)
            
        gcs = np.full(self.mins, 15)
        if "Shock" in profile:
            gcs[180:] = 13
            gcs[300:] = 10

        pp = 25 if profile == "Cardiogenic Shock" else 50
        dbp = map_val - (pp/3)

        # FIX: The dictionary key "PaO2" now maps to the defined variable 'pao2'
        df = pd.DataFrame({
            "Time": self.t,
            "HR": hr, "MAP": map_val, "DBP": dbp, "CO": co,
            "Lactate": lactate, "Creatinine": creat, 
            "Bilirubin": bili, "Platelets": plt, 
            "PaO2": pao2, "FiO2": fio2, "GCS": gcs,
            "Norepi_Dose": ne_dose
        })
        
        df[['CPO', 'SI', 'DSI', 'SVR']] = df.apply(
            lambda x: ClinicalEngine.calculate_hemodynamics(x), axis=1, result_type='expand'
        )
        
        return df

# ==========================================
# 4. VISUALIZATION COMPONENTS (LIGHT THEME)
# ==========================================
def plot_full_telemetry(df):
    """Multi-axis Plotly Subplots for Full Telemetry."""
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=("Heart Rate & MAP", "Cardiac Power Output (CPO)", "Lactate Kinetics", "Diastolic Shock Index")
    )
    
    # Row 1: HR and MAP
    fig.add_trace(go.Scatter(x=df['Time'], y=df['HR'], name="HR", line=dict(color=THEME['crit'])), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Time'], y=df['MAP'], name="MAP", line=dict(color=THEME['hemo'])), row=1, col=1)
    fig.add_hline(y=65, line_dash="dot", line_color=THEME['text_muted'], row=1, col=1)
    
    # Row 2: CPO
    fig.add_trace(go.Scatter(x=df['Time'], y=df['CPO'], name="CPO", fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)', line=dict(color=THEME['info'])), row=2, col=1)
    fig.add_hline(y=0.6, line_dash="dash", line_color=THEME['crit'], row=2, col=1)
    
    # Row 3: Lactate
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Lactate'], name="Lactate", line=dict(color=THEME['meta'])), row=3, col=1)
    
    # Row 4: DSI
    fig.add_trace(go.Scatter(x=df['Time'], y=df['DSI'], name="DSI", line=dict(color=THEME['warn'])), row=4, col=1)
    
    fig.update_layout(height=800, template="plotly_white", margin=dict(l=20,r=20,t=40,b=20))
    return fig

def plot_shock_dashboard(df):
    """Specific Dashboard for Shock Indices."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Shock Index (HR/SBP)", "Variability Index (Entropy)", "Pressor-Lactate Relation"])
    
    # Col 1: SI
    fig.add_trace(go.Scatter(x=df['Time'], y=df['SI'], name="SI", line=dict(color=THEME['crit'])), row=1, col=1)
    fig.add_hline(y=0.9, line_dash="dot", line_color=THEME['warn'], row=1, col=1)
    
    # Col 2: Rolling Entropy
    entropy_roll = df['HR'].rolling(30).apply(ClinicalEngine.complexity_index)
    fig.add_trace(go.Scatter(x=df['Time'], y=entropy_roll, name="Entropy", line=dict(color=THEME['ok'])), row=1, col=2)
    
    # Col 3: Pressor vs Lactate (Scatter)
    fig.add_trace(go.Scatter(x=df['Norepi_Dose'], y=df['Lactate'], mode='markers', 
                             marker=dict(color=df['Time'], colorscale='Viridis', showscale=False), name="Dose-Response"), row=1, col=3)
    
    fig.update_layout(height=250, template="plotly_white", margin=dict(l=10,r=10,t=40,b=10))
    return fig

def plot_coupling_trajectory(df):
    """Hemodynamic-Metabolic Coupling Phase Space."""
    recent = df.iloc[-120:].copy()
    fig = go.Figure()
    
    # Zones
    fig.add_shape(type="rect", x0=0, x1=0.6, y0=4, y1=15, fillcolor="rgba(239, 68, 68, 0.1)", line_width=0, layer="below")
    
    # Trajectory
    fig.add_trace(go.Scatter(
        x=recent['CPO'], y=recent['Lactate'], mode='lines+markers',
        marker=dict(size=recent.index/recent.index.max() * 6, color=recent.index, colorscale='Bluered', showscale=False),
        line=dict(color='rgba(148, 163, 184, 0.4)', width=1),
        name='Trajectory'
    ))
    
    # Current
    curr = recent.iloc[-1]
    fig.add_trace(go.Scatter(x=[curr['CPO']], y=[curr['Lactate']], mode='markers', marker=dict(color=THEME['text_main'], size=15)))

    fig.update_layout(
        title="<b>Perfusion Phenotype Trajectory</b>",
        xaxis=dict(title="Cardiac Power Output (W)", range=[0, 1.8]),
        yaxis=dict(title="Lactate (mmol/L)", range=[0, 12]),
        template="plotly_white",
        height=300, showlegend=False,
        margin=dict(l=20,r=20,t=40,b=20)
    )
    return fig

def plot_organ_radar(sofa_dict):
    """Radar chart for Organ Failure Distribution."""
    categories = ['Resp', 'Coag', 'Liver', 'CV', 'CNS', 'Renal']
    values = [sofa_dict[c] for c in categories]
    values += [values[0]]
    categories += [categories[0]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        fillcolor='rgba(6, 182, 212, 0.2)', line=dict(color=THEME['hemo'], width=2)
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 4], color=THEME['text_muted'])),
        template="plotly_white",
        height=200, margin=dict(l=40, r=40, t=20, b=20), showlegend=False
    )
    return fig

# ==========================================
# 5. MAIN APP EXECUTION
# ==========================================
st.markdown(STYLING, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("TITAN | CDS")
    st.caption("AI-Enhanced Hemodynamic Intelligence")
    
    st.markdown("### 🧬 Scenario Injection")
    scenario = st.selectbox("Patient Profile", ["Healthy", "Compensated Sepsis", "Vasoplegic Shock", "Cardiogenic Shock"])
    
    st.markdown("### ⚙️ Live Titration")
    in_fi = st.slider("FiO2 Override", 0.21, 1.0, 0.21, 0.05)
    in_ne = st.slider("Norepi Dose (mcg/kg/min)", 0.0, 1.0, 0.0, 0.05)
    
    st.divider()
    st.info("**TITAN v3.1**\n\nModules Active:\n- Kinetic Labs\n- True SOFA CV/Resp\n- AI Phenotyping\n- Telemetry Stack")

# --- DATA ENGINE ---
sim = PatientSimulator(mins=360)
df = sim.get_data(scenario)

# Apply Overrides to latest window
idx_end = df.index[-1]
if in_fi > 0.21: df.loc[idx_end, 'FiO2'] = in_fi
if in_ne > 0.0: df.loc[idx_end, 'Norepi_Dose'] = in_ne

# Recalculate Latest State
cur = df.iloc[-1]
prev_1h = df.iloc[-60]

# --- AI & CALCULATOR ---
# 1. SOFA
sofa_comp = ClinicalEngine.calculate_sofa_components(cur)
sofa_total = sofa_comp['Total']

# 2. Complexity & Slope
complexity = ClinicalEngine.complexity_index(df['HR'].iloc[-30:])
lac_slope = (cur['Lactate'] - prev_1h['Lactate'])

# 3. AI Classification
ai_class, ai_reason, ai_style = ClinicalEngine.ai_phenotype_classifier(cur, complexity, lac_slope)

# --- HEADER: AI STATUS ---
st.markdown(f"""
<div class="status-banner {ai_style}">
    <div>
        <div style="font-size:0.75rem; color:{THEME['ai']}; font-weight:800; letter-spacing:1px; margin-bottom:4px;">AI PHENOTYPE CLASSIFIER</div>
        <div style="font-size:1.6rem; color:{THEME['text_main']}">{ai_class}</div>
        <div style="font-size:0.9rem; font-weight:500; color:{THEME['text_muted']}">{ai_reason}</div>
    </div>
    <div style="text-align:right">
        <div style="font-size:0.75rem; opacity:0.8; font-weight:700;">REAL-TIME SOFA</div>
        <div style="font-size:2.2rem; line-height:1; color:{THEME['crit'] if sofa_total > 5 else THEME['ok']}">{sofa_total}</div>
        <div style="font-size:0.7rem; color:{THEME['text_muted']}; margin-top:4px;">PaO2/FiO2: {sofa_comp['PaFi']:.0f} | NE: {cur['Norepi_Dose']:.2f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- TABS FOR EXPANDED VIEWS ---
tab1, tab2, tab3 = st.tabs(["📊 Clinical Overview", "📈 Full Telemetry", "⚡ Shock Dashboard"])

with tab1:
    # --- ROW 1: THE MONITOR ---
    c1, c2, c3, c4, c5 = st.columns(5)
    def d_met(col, lbl, val, unit, delta):
        col.metric(lbl, f"{val:.1f} {unit}", f"{delta:+.1f}")
        
    d_met(c1, "MAP", cur['MAP'], "mmHg", cur['MAP']-prev_1h['MAP'])
    d_met(c2, "Heart Rate", cur['HR'], "bpm", cur['HR']-prev_1h['HR'])
    d_met(c3, "Lactate", cur['Lactate'], "mmol/L", cur['Lactate']-prev_1h['Lactate'])
    d_met(c4, "CPO", cur['CPO'], "W", cur['CPO']-prev_1h['CPO'])
    d_met(c5, "SVR", cur['SVR'], "dyn", cur['SVR']-prev_1h['SVR'])

    # --- ROW 2: ADVANCED PLOTS ---
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.plotly_chart(plot_coupling_trajectory(df), use_container_width=True)
    with col_b:
        st.markdown('<div class="section-header">Organ Dysfunction Topology</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_organ_radar(sofa_comp), use_container_width=True)

    # --- ROW 3: KINETIC LABS ---
    st.markdown('<div class="section-header">Kinetic Lab Progression (Modeled)</div>', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    def plot_mini_kinetic(y_col, color, title):
        fig = px.line(df.iloc[-120:], x="Time", y=y_col)
        fig.update_traces(line_color=color)
        fig.update_layout(
            title=dict(text=title, font=dict(size=12, color=THEME['text_muted'])),
            height=150, margin=dict(l=0,r=0,t=30,b=0), 
            template="plotly_white",
            xaxis=dict(showticklabels=False, title=None),
            yaxis=dict(showticklabels=True)
        )
        return fig
    
    with k1: st.plotly_chart(plot_mini_kinetic("Creatinine", THEME['warn'], "Creatinine Kinetics"), use_container_width=True)
    with k2: st.plotly_chart(plot_mini_kinetic("Bilirubin", THEME['meta'], "Bilirubin Kinetics"), use_container_width=True)
    with k3: st.plotly_chart(plot_mini_kinetic("Platelets", THEME['crit'], "Platelet Consumption"), use_container_width=True)

with tab2:
    st.markdown("### 📡 Multi-Axis Telemetry Stack")
    st.plotly_chart(plot_full_telemetry(df), use_container_width=True)

with tab3:
    st.markdown("### ⚡ Shock Index & Variability Dashboard")
    # Key Shock Metrics
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Shock Index (SI)", f"{cur['SI']:.2f}", help="Normal: 0.5-0.7. >0.9 suggests shock.")
    s2.metric("Diastolic SI (DSI)", f"{cur['DSI']:.2f}", help=">2.0 suggests Vasoplegia.")
    s3.metric("Entropy (HRV)", f"{complexity:.2f}", help="<0.6 suggests autonomic failure.")
    s4.metric("PaO2/FiO2", f"{sofa_comp['PaFi']:.0f}", help="<300 = ARDS Mild.")
    
    st.plotly_chart(plot_shock_dashboard(df), use_container_width=True)

# --- DISCLAIMER ---
st.caption("TITAN Clinical Intelligence Engine | Algorithm: Linear Kinetic Modeling + Deterministic Decision Tree | For Educational Use Only.")
