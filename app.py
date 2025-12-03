import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import welch

# ==========================================
# 1. SYSTEM CONFIGURATION & STYLING (MEDICAL GRADE)
# ==========================================
st.set_page_config(
    page_title="TITAN | Command Center",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚡"
)

# Light Theme Palette (High Contrast Medical)
THEME = {
    "bg": "#f8fafc",        # Slate-50
    "card_bg": "#ffffff",   # White
    "text_main": "#0f172a", # Slate-900
    "text_muted": "#64748b",# Slate-500
    "border": "#e2e8f0",    # Slate-200
    "crit": "#dc2626",      # Red-600
    "warn": "#d97706",      # Amber-600
    "ok": "#059669",        # Emerald-600
    "info": "#2563eb",      # Blue-600
    "hemo": "#0891b2",      # Cyan-600
    "meta": "#7c3aed",      # Violet-600
    "ai": "#be185d",        # Pink-700
}

STYLING = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Roboto+Mono:wght@500;700&display=swap');
    
    .stApp {{ background-color: {THEME['bg']}; color: {THEME['text_main']}; font-family: 'Inter', sans-serif; }}
    
    /* Compact Metric Cards */
    div[data-testid="stMetric"] {{
        background-color: {THEME['card_bg']};
        padding: 10px 15px;
        border-radius: 6px;
        border: 1px solid {THEME['border']};
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }}
    div[data-testid="stMetric"] label {{ font-size: 0.7rem; font-weight: 700; color: {THEME['text_muted']}; }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ font-family: 'Roboto Mono'; font-size: 1.6rem; font-weight: 800; }}
    
    /* Headers */
    .zone-header {{
        font-size: 0.85rem;
        font-weight: 800;
        color: {THEME['text_muted']};
        text-transform: uppercase;
        border-bottom: 2px solid {THEME['border']};
        margin: 25px 0 10px 0;
        padding-bottom: 5px;
        letter-spacing: 0.05em;
    }}
    
    /* AI Banner */
    .status-banner {{
        padding: 15px 20px;
        border-radius: 8px;
        display: flex; justify-content: space-between; align-items: center;
        background: white; border-left: 6px solid;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }}
    .b-crit {{ border-color: {THEME['crit']}; background: #fef2f2; color: #991b1b; }}
    .b-warn {{ border-color: {THEME['warn']}; background: #fffbeb; color: #92400e; }}
    .b-ok   {{ border-color: {THEME['ok']};   background: #f0fdf4; color: #166534; }}
    .b-ai   {{ border-color: {THEME['ai']};   background: #fdf2f8; color: #831843; }}
</style>
"""

# ==========================================
# 2. LOGIC ENGINE (UNCHANGED)
# ==========================================
class ClinicalEngine:
    @staticmethod
    def calculate_sofa_components(row):
        fi = max(0.21, row['FiO2'])
        pafi = row['PaO2'] / fi
        
        resp = 0
        if pafi < 100: resp = 4
        elif pafi < 200: resp = 3
        elif pafi < 300: resp = 2
        elif pafi < 400: resp = 1
        
        coag = 0
        if row['Platelets'] < 20: coag = 4
        elif row['Platelets'] < 50: coag = 3
        elif row['Platelets'] < 100: coag = 2
        elif row['Platelets'] < 150: coag = 1
        
        liver = 0
        if row['Bilirubin'] > 12.0: liver = 4
        elif row['Bilirubin'] > 6.0: liver = 3
        elif row['Bilirubin'] > 2.0: liver = 2
        elif row['Bilirubin'] > 1.2: liver = 1
        
        ne_dose = row['Norepi_Dose']
        map_val = row['MAP']
        cv = 0
        if ne_dose > 0.1: cv = 4
        elif ne_dose > 0: cv = 3
        elif map_val < 70: cv = 1
        
        cns = 0
        if row['GCS'] < 6: cns = 4
        elif row['GCS'] < 9: cns = 3
        elif row['GCS'] < 12: cns = 2
        elif row['GCS'] < 14: cns = 1
        
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
        cpo = (row['MAP'] * row['CO']) / 451
        dsi = row['HR'] / row['DBP'] if row['DBP'] > 0 else 0
        sbp = (3 * row['MAP']) - (2 * row['DBP'])
        si = row['HR'] / sbp if sbp > 0 else 0
        svr = ((row['MAP'] - 10) / row['CO']) * 80
        return cpo, si, dsi, svr

    @staticmethod
    def complexity_index(time_series):
        if len(time_series) < 20: return 1.0
        ts = np.array(time_series)
        diff = np.diff(ts)
        return np.std(diff) / (np.std(ts) + 0.001)

    @staticmethod
    def ai_phenotype_classifier(row, entropy, lactate_slope):
        cpo = row['CPO']
        svr = row['SVR']
        dsi = row['DSI']
        map_val = row['MAP']
        ne_dose = row['Norepi_Dose']
        
        if map_val < 65 or ne_dose > 0:
            if cpo < 0.6:
                if svr > 1200: return "Cardiogenic Shock", "High SVR + Low CPO", "b-crit"
                else: return "Mixed/Metabolic Shock", "Low SVR + Low CPO", "b-crit"
            else:
                if dsi > 2.0 or svr < 800: return "Vasoplegic Shock", "High DSI + Low SVR", "b-crit"
                elif lactate_slope > 0.1: return "Cryptic Sepsis", "Rising Lactate + Preserved CPO", "b-warn"
                else: return "Obstructive/Other", "Atypical Hemodynamics", "b-warn"
        else:
            if lactate_slope > 0.05 and entropy < 0.8: return "Compensated Sepsis", "Loss of Complexity + Lactate Rise", "b-warn"
            elif entropy < 0.5: return "Pre-Shock State", "Autonomic Uncoupling", "b-ai"
            else: return "Hemodynamically Stable", "Normal Parameters", "b-ok"

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
        
        if profile == "Healthy":
            hr_base, hr_end = 70, 75
            map_base, map_end = 90, 85
            co_base = 5.0
            fi_base = 0.21
            ne_base = 0.0
            complexity = 1.0
            creat_func, bili_func, plt_func = lambda t: 0.8, lambda t: 0.6, lambda t: 250
            
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
            creat_func, bili_func, plt_func = lambda t: 0.9, lambda t: 0.7, lambda t: 220
            
        hr = np.linspace(hr_base, hr_end, self.mins) + self._generate_noise(self.mins, 3 * complexity)
        map_val = np.linspace(map_base, map_end, self.mins) + self._generate_noise(self.mins, 2 * complexity)
        co = np.linspace(co_base, co_base * (0.8 if "Cardio" in profile else 1.1), self.mins) + self._generate_noise(self.mins, 0.2)
        ne_dose = np.zeros(self.mins)
        if "Shock" in profile:
            ne_dose = ne_base + (0.3 / (1 + np.exp(-0.05 * (self.t - 180))))
        fio2 = np.linspace(fi_base, fi_base + 0.2, self.mins)
        pao2 = np.linspace(400, 400 - (200 if "Shock" in profile else 0), self.mins)
        creat = np.array([creat_func(x) for x in self.t])
        bili = np.array([bili_func(x) for x in self.t])
        plt = np.array([plt_func(x) for x in self.t])
        if "Shock" in profile: lactate = 1.0 + 8.0 * (trend ** 2)
        elif "Compensated" in profile: lactate = 1.0 + 2.0 * trend
        else: lactate = np.full(self.mins, 0.9)
        gcs = np.full(self.mins, 15)
        if "Shock" in profile: gcs[180:] = 13
        pp = 25 if profile == "Cardiogenic Shock" else 50
        dbp = map_val - (pp/3)

        df = pd.DataFrame({
            "Time": self.t, "HR": hr, "MAP": map_val, "DBP": dbp, "CO": co,
            "Lactate": lactate, "Creatinine": creat, "Bilirubin": bili, "Platelets": plt, 
            "PaO2": pao2, "FiO2": fio2, "GCS": gcs, "Norepi_Dose": ne_dose
        })
        df[['CPO', 'SI', 'DSI', 'SVR']] = df.apply(lambda x: ClinicalEngine.calculate_hemodynamics(x), axis=1, result_type='expand')
        return df

# ==========================================
# 3. VISUALIZATION FUNCTIONS (COMPACT)
# ==========================================
def plot_full_telemetry(df):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=("HR & MAP", "Cardiac Power", "Lactate", "Diastolic SI"))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['HR'], name="HR", line=dict(color=THEME['crit'])), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Time'], y=df['MAP'], name="MAP", line=dict(color=THEME['hemo'])), row=1, col=1)
    fig.add_hline(y=65, line_dash="dot", line_color=THEME['text_muted'], row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Time'], y=df['CPO'], name="CPO", fill='tozeroy', fillcolor='rgba(8, 145, 178, 0.1)', line=dict(color=THEME['hemo'])), row=2, col=1)
    fig.add_hline(y=0.6, line_dash="dash", line_color=THEME['crit'], row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Lactate'], name="Lactate", line=dict(color=THEME['meta'])), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['Time'], y=df['DSI'], name="DSI", line=dict(color=THEME['warn'])), row=4, col=1)
    fig.update_layout(height=600, template="plotly_white", margin=dict(l=20,r=20,t=20,b=20), showlegend=False)
    return fig

def plot_shock_dashboard_row(df):
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Shock Index", "Entropy (HRV)", "Pressor/Lactate Curve"])
    # SI
    fig.add_trace(go.Scatter(x=df['Time'], y=df['SI'], name="SI", line=dict(color=THEME['crit'])), row=1, col=1)
    fig.add_hline(y=0.9, line_dash="dot", line_color=THEME['warn'], row=1, col=1)
    # Entropy
    entropy_roll = df['HR'].rolling(30).apply(ClinicalEngine.complexity_index)
    fig.add_trace(go.Scatter(x=df['Time'], y=entropy_roll, name="Entropy", line=dict(color=THEME['ok'])), row=1, col=2)
    # Dose Response
    fig.add_trace(go.Scatter(x=df['Norepi_Dose'], y=df['Lactate'], mode='markers', 
                             marker=dict(color=df['Time'], colorscale='Viridis', size=6), name="Dose-Response"), row=1, col=3)
    fig.update_layout(height=200, template="plotly_white", margin=dict(l=10,r=10,t=30,b=10), showlegend=False)
    return fig

def plot_coupling_trajectory(df):
    recent = df.iloc[-120:].copy()
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=0.6, y0=4, y1=15, fillcolor="rgba(220, 38, 38, 0.1)", line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=recent['CPO'], y=recent['Lactate'], mode='lines+markers',
        marker=dict(size=recent.index/recent.index.max()*8, color=recent.index, colorscale='Bluered', showscale=False),
        line=dict(color='rgba(100, 116, 139, 0.4)', width=1)))
    fig.add_trace(go.Scatter(x=[recent.iloc[-1]['CPO']], y=[recent.iloc[-1]['Lactate']], mode='markers', marker=dict(color=THEME['text_main'], size=12)))
    fig.update_layout(
        title="<b>Perfusion Phenotype</b>", xaxis=dict(title="Cardiac Power (W)", range=[0, 1.8]), yaxis=dict(title="Lactate", range=[0, 12]),
        template="plotly_white", height=280, margin=dict(l=10,r=10,t=40,b=10)
    )
    return fig

def plot_organ_radar(sofa_dict):
    cats = ['Resp', 'Coag', 'Liver', 'CV', 'CNS', 'Renal', 'Resp']
    vals = [sofa_dict[c] for c in ['Resp', 'Coag', 'Liver', 'CV', 'CNS', 'Renal']] + [sofa_dict['Resp']]
    fig = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill='toself', fillcolor='rgba(8, 145, 178, 0.2)', line=dict(color=THEME['hemo'], width=2)))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 4], color=THEME['text_muted'])),
        template="plotly_white", height=200, margin=dict(l=30,r=30,t=20,b=20), showlegend=False)
    return fig

def plot_mini_kinetic(df, col, color, title):
    fig = px.line(df.iloc[-120:], x="Time", y=col)
    fig.update_traces(line_color=color, line_width=2)
    fig.update_layout(
        title=dict(text=title, font=dict(size=11, color=THEME['text_muted'])),
        height=100, margin=dict(l=0,r=0,t=25,b=0), template="plotly_white",
        xaxis=dict(showticklabels=False, title=None), yaxis=dict(showticklabels=True, title=None)
    )
    return fig

# ==========================================
# 4. APP EXECUTION
# ==========================================
st.markdown(STYLING, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("TITAN | CDS")
    st.markdown("<b>CLINICAL COMMAND CENTER</b>", unsafe_allow_html=True)
    scenario = st.selectbox("Simulation Scenario", ["Healthy", "Compensated Sepsis", "Vasoplegic Shock", "Cardiogenic Shock"])
    st.divider()
    st.markdown("<b>INTERVENTION</b>", unsafe_allow_html=True)
    in_fi = st.slider("FiO2", 0.21, 1.0, 0.21, 0.05)
    in_ne = st.slider("Norepinephrine", 0.0, 1.0, 0.0, 0.05)
    st.caption(f"v3.2 Single Pane • {scenario}")

# --- DATA GENERATION ---
sim = PatientSimulator(mins=360)
df = sim.get_data(scenario)
# Overrides
idx_end = df.index[-1]
if in_fi > 0.21: df.loc[idx_end, 'FiO2'] = in_fi
if in_ne > 0.0: df.loc[idx_end, 'Norepi_Dose'] = in_ne
# Recalc
cur = df.iloc[-1]
prev_1h = df.iloc[-60]
sofa_comp = ClinicalEngine.calculate_sofa_components(cur)
complexity = ClinicalEngine.complexity_index(df['HR'].iloc[-30:])
ai_class, ai_reason, ai_style = ClinicalEngine.ai_phenotype_classifier(cur, complexity, cur['Lactate'] - prev_1h['Lactate'])

# ==========================================
# SINGLE PAGE LAYOUT
# ==========================================

# --- 1. AI TRIAGE BANNER ---
st.markdown(f"""
<div class="status-banner {ai_style}">
    <div>
        <div style="font-size:0.8rem; color:{THEME['ai']}; font-weight:800; letter-spacing:1px; margin-bottom:4px;">AI PHENOTYPE CLASSIFIER</div>
        <div style="font-size:1.8rem; line-height:1.1; color:{THEME['text_main']}">{ai_class}</div>
        <div style="font-size:1rem; color:{THEME['text_muted']}">{ai_reason}</div>
    </div>
    <div style="text-align:right">
        <div style="font-size:0.8rem; opacity:0.8; font-weight:700;">SOFA SCORE</div>
        <div style="font-size:2.5rem; line-height:1; font-weight:800; color:{THEME['crit'] if sofa_comp['Total'] > 5 else THEME['text_main']}">{sofa_comp['Total']}</div>
        <div style="font-size:0.8rem; color:{THEME['text_muted']};">PaO2/FiO2: {sofa_comp['PaFi']:.0f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 2. ZONE A: HEMODYNAMICS & PERFUSION (THE "NOW") ---
st.markdown('<div class="zone-header">ZONE A: HEMODYNAMICS & PERFUSION</div>', unsafe_allow_html=True)
col_top_L, col_top_R = st.columns([2, 1])

with col_top_L:
    # 2x3 Grid of Metrics
    r1c1, r1c2, r1c3 = st.columns(3)
    r2c1, r2c2, r2c3 = st.columns(3)
    
    def m(c, l, v, u, d, inv=False):
        c.metric(l, f"{v:.1f} {u}", f"{d:+.1f}", delta_color="inverse" if inv else "normal")

    m(r1c1, "MAP", cur['MAP'], "mmHg", cur['MAP']-prev_1h['MAP'])
    m(r1c2, "Heart Rate", cur['HR'], "bpm", cur['HR']-prev_1h['HR'], True)
    m(r1c3, "CPO (Power)", cur['CPO'], "W", cur['CPO']-prev_1h['CPO'])
    
    m(r2c1, "Lactate", cur['Lactate'], "mM", cur['Lactate']-prev_1h['Lactate'], True)
    m(r2c2, "SVR", cur['SVR'], "dyn", cur['SVR']-prev_1h['SVR'])
    m(r2c3, "Shock Index", cur['SI'], "", cur['SI']-prev_1h['SI'], True)

with col_top_R:
    st.plotly_chart(plot_coupling_trajectory(df), use_container_width=True, config={'displayModeBar': False})

# --- 3. ZONE B: ETIOLOGY & ORGAN FAILURE (THE "WHY") ---
st.markdown('<div class="zone-header">ZONE B: ETIOLOGY, ORGANS & KINETICS</div>', unsafe_allow_html=True)
c_mid1, c_mid2, c_mid3 = st.columns(3)

with c_mid1:
    st.markdown("**Shock Dynamics**")
    st.caption("Index | HRV | Dose-Response")
    st.plotly_chart(plot_shock_dashboard_row(df), use_container_width=True, config={'displayModeBar': False})

with c_mid2:
    st.markdown("**Organ Topology**")
    st.caption("SOFA Breakdown")
    st.plotly_chart(plot_organ_radar(sofa_comp), use_container_width=True, config={'displayModeBar': False})

with c_mid3:
    st.markdown("**Kinetic Labs**")
    st.caption("Modeled Progression (Last 2h)")
    st.plotly_chart(plot_mini_kinetic(df, "Creatinine", THEME['warn'], "Creatinine"), use_container_width=True, config={'displayModeBar': False})
    st.plotly_chart(plot_mini_kinetic(df, "Bilirubin", THEME['meta'], "Bilirubin"), use_container_width=True, config={'displayModeBar': False})
    st.plotly_chart(plot_mini_kinetic(df, "Platelets", THEME['crit'], "Platelets"), use_container_width=True, config={'displayModeBar': False})

# --- 4. ZONE C: TELEMETRY TRENDS (THE "HISTORY") ---
st.markdown('<div class="zone-header">ZONE C: CONTINUOUS TELEMETRY STACK</div>', unsafe_allow_html=True)
st.plotly_chart(plot_full_telemetry(df), use_container_width=True, config={'displayModeBar': False})
