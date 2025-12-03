import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import welch
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import time

# ==========================================
# 1. CONFIGURATION & MEDICAL THEME
# ==========================================
st.set_page_config(
    page_title="TITAN | LEVEL 5 CDS",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧬"
)

THEME = {
    "bg": "#f8fafc", "card_bg": "#ffffff", "text_main": "#0f172a", "text_muted": "#64748b",
    "border": "#e2e8f0", "crit": "#dc2626", "warn": "#d97706", "ok": "#059669",
    "info": "#2563eb", "hemo": "#0891b2", "resp": "#7c3aed", "ai": "#be185d", "drug": "#4f46e5"
}

STYLING = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Roboto+Mono:wght@500;700&display=swap');
    .stApp {{ background-color: {THEME['bg']}; color: {THEME['text_main']}; font-family: 'Inter', sans-serif; }}
    @keyframes flash-crit {{ 0% {{ box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.7); }} 70% {{ box-shadow: 0 0 0 10px rgba(220, 38, 38, 0); }} 100% {{ box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }} }}
    .crit-pulse {{ animation: flash-crit 2s infinite; border: 1px solid {THEME['crit']} !important; }}
    div[data-testid="stMetric"] {{ background-color: {THEME['card_bg']}; padding: 5px 10px; border-radius: 6px; border: 1px solid {THEME['border']}; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); }}
    div[data-testid="stMetric"] label {{ font-size: 0.6rem; font-weight: 700; color: {THEME['text_muted']}; text-transform: uppercase; }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ font-family: 'Roboto Mono'; font-size: 1.2rem; font-weight: 800; }}
    .zone-header {{ font-size: 0.8rem; font-weight: 900; color: {THEME['text_muted']}; text-transform: uppercase; border-bottom: 2px solid {THEME['border']}; margin: 15px 0 10px 0; padding-bottom: 4px; letter-spacing: 0.05em; }}
    .status-banner {{ padding: 12px 20px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; background: white; border-left: 6px solid; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 10px; }}
</style>
"""

class CONSTANTS:
    # Hemodynamic Multipliers (Standardized)
    DRUG_MULTS = {
        'norepi': {'svr': 900, 'map': 28, 'co': 0.6},
        'vaso':   {'svr': 700, 'map': 18, 'co': 0.0},
        'dobu':   {'svr': -350, 'map': 2, 'co': 2.8, 'hr': 18},
        'bb':     {'svr': 50, 'map': -8, 'co': -1.2, 'hr': -22}
    }
    ATM_PRESSURE = 760
    H2O_PRESSURE = 47
    R_QUOTIENT = 0.8
    MAX_PAO2 = 600
    LAC_PROD_THRESH = 380 # mL/min/m2 DO2I
    LAC_CLEAR_RATE = 0.04

# ==========================================
# 2. VECTORIZED PHYSICS ENGINE
# ==========================================
class VectorizedPharmaEngine:
    @staticmethod
    def apply_drugs_vectorized(map_base, ci_base, hr_base, svri_base, drugs, mins):
        # Calculate effect deltas
        svr_eff = (drugs['norepi'] * CONSTANTS.DRUG_MULTS['norepi']['svr'] + 
                   drugs['vaso'] * CONSTANTS.DRUG_MULTS['vaso']['svr'] + 
                   drugs['dobu'] * CONSTANTS.DRUG_MULTS['dobu']['svr'])
        
        map_eff = (drugs['norepi'] * CONSTANTS.DRUG_MULTS['norepi']['map'] + 
                   drugs['vaso'] * CONSTANTS.DRUG_MULTS['vaso']['map'] + 
                   drugs['bb'] * CONSTANTS.DRUG_MULTS['bb']['map'])
        
        co_eff  = (drugs['norepi'] * CONSTANTS.DRUG_MULTS['norepi']['co'] + 
                   drugs['dobu'] * CONSTANTS.DRUG_MULTS['dobu']['co'] + 
                   drugs['bb'] * CONSTANTS.DRUG_MULTS['bb']['co'])
        
        hr_eff  = (drugs['dobu'] * CONSTANTS.DRUG_MULTS['dobu']['hr'] + 
                   drugs['bb'] * CONSTANTS.DRUG_MULTS['bb']['hr'])
        
        return (map_base + map_eff), (ci_base + co_eff), (hr_base + hr_eff), (svri_base + svr_eff)

class AdvancedPhysiologyEngine:
    @staticmethod
    def brownian_bridge_autonomic(n, start, end, base_volatility=1.0, seed=None):
        """
        Simulates Autonomic Tone.
        If system is crashing (end < start), volatility increases (loss of baroreflex).
        """
        if seed: np.random.seed(seed)
        t = np.linspace(0, 1, n)
        
        # Stress-induced volatility: As physiology degrades, variance increases
        stress_factor = 1.0 + (max(0, start - end) / 20.0) 
        
        dW = np.random.normal(0, np.sqrt(1/n), n)
        W = np.cumsum(dW)
        B = start + W - t * (W[-1] - (end - start))
        
        # Pink noise overlay (Autonomic variability)
        pink = np.convolve(np.random.normal(0, 0.5, n), np.ones(8)/8, mode='same')
        
        return B + (pink * base_volatility * stress_factor)

    @staticmethod
    def alveolar_gas_equation(fio2, paco2):
        p_ideal = (fio2 * (CONSTANTS.ATM_PRESSURE - CONSTANTS.H2O_PRESSURE)) - (paco2 / CONSTANTS.R_QUOTIENT)
        return np.clip(p_ideal, 0, CONSTANTS.MAX_PAO2)

    @staticmethod
    def calculate_shunt_effect(p_ideal, shunt_fraction, peep):
        # PEEP Recruitment: Higher PEEP reduces shunt
        recruit_factor = np.exp(-0.08 * peep) 
        effective_shunt = shunt_fraction * recruit_factor
        
        pao2 = p_ideal * (1 - effective_shunt)
        pao2_safe = np.maximum(pao2, 0.1)
        # Severinghaus
        spo2 = 100 / (1 + (23400 / (pao2_safe**3 + 150*pao2_safe)))
        return pao2, spo2

    @staticmethod
    def lactate_kinetics_vectorized(do2i, mins, liver_function_pct=1.0):
        lactate = np.zeros(mins)
        lac_curr = 1.2 # Baseline
        
        # Production driven by anaerobic threshold (DO2I critical)
        prod = np.where(do2i < CONSTANTS.LAC_PROD_THRESH, 0.15, 0.0)
        
        # Clearance driven by liver perfusion and function
        # Washout requires adequate flow
        clear_base = CONSTANTS.LAC_CLEAR_RATE * liver_function_pct
        clear = np.where(do2i > 450, clear_base, clear_base * 0.2)
        
        net = prod - clear
        
        # Autoregressive accumulation
        for i in range(mins):
            lac_curr = max(0.5, lac_curr + net[i])
            lactate[i] = lac_curr
        return lactate

# ==========================================
# 3. PATIENT SIMULATOR (CLINICAL PROFILE BASED)
# ==========================================
class PatientSimulator:
    def __init__(self, mins=360):
        self.mins = mins
        self.t = np.arange(mins)

    def get_data(self, case_id, drugs, fluids, bsa, peep):
        # --- 1. CLINICAL CASE PROFILING ---
        # Instead of user "Drift", the drift is intrinsic to the pathology
        
        cases = {
            "65M Post-CABG": {
                'ci': (2.2, 1.8), 'map': (75, 55), 'svri': (1800, 1400), 
                'hr': (85, 95), 'shunt': 0.10, 'drift_severity': 1.5, 'liver': 0.9,
                'desc': "Cardiogenic/Vasoplegic Mix"
            },
            "24F Septic Shock": {
                'ci': (4.5, 5.5), 'map': (65, 45), 'svri': (1000, 600), 
                'hr': (110, 140), 'shunt': 0.15, 'drift_severity': 2.0, 'liver': 0.8,
                'desc': "Hyperdynamic Distributive"
            },
            "82M HFpEF Sepsis": {
                'ci': (2.0, 1.9), 'map': (85, 60), 'svri': (2200, 1600), 
                'hr': (90, 110), 'shunt': 0.20, 'drift_severity': 1.2, 'liver': 0.6,
                'desc': "Mixed Etiology (Restricted Reserve)"
            },
            "50M Trauma (Hemorrhage)": {
                'ci': (3.0, 1.5), 'map': (70, 40), 'svri': (2500, 3000), 
                'hr': (100, 150), 'shunt': 0.05, 'drift_severity': 2.5, 'liver': 1.0,
                'desc': "Hypovolemic (High SVR)"
            }
        }
        
        p = cases[case_id]
        
        # Use a consistent internal seed based on case name length + mins (Pseudo-random but stable for visual)
        internal_seed = len(case_id) + 42
        
        # --- 2. GENERATE PHYSIOLOGY ---
        # Generate base trajectories
        hr = AdvancedPhysiologyEngine.brownian_bridge_autonomic(self.mins, p['hr'][0], p['hr'][1], 1.5, internal_seed)
        map_raw = AdvancedPhysiologyEngine.brownian_bridge_autonomic(self.mins, p['map'][0], p['map'][1], 1.2, internal_seed+1)
        ci_raw = AdvancedPhysiologyEngine.brownian_bridge_autonomic(self.mins, p['ci'][0], p['ci'][1], 0.2, internal_seed+2)
        svri_raw = AdvancedPhysiologyEngine.brownian_bridge_autonomic(self.mins, p['svri'][0], p['svri'][1], 100, internal_seed+3)
        rr = AdvancedPhysiologyEngine.brownian_bridge_autonomic(self.mins, 16, 28, 2.0, internal_seed+4)
        
        # --- 3. FLUID PHYSICS (Indexed) ---
        # Pulse Pressure Variation
        ppv_base = 20 if "Hemorrhage" in case_id else (5 if "HFpEF" in case_id else 12)
        ppv = ppv_base + (np.sin(self.t/8) * 4) + np.random.normal(0, 1, self.mins)
        
        # Starling Curve (Indexed)
        # Fluid responsiveness depends on PPV
        ci_gain_per_500 = 0.4 if np.mean(ppv) > 13 else 0.05
        ci_fluid = (fluids / 500) * ci_gain_per_500
        
        # --- 4. PHARMACODYNAMICS ---
        map_f, ci_f, hr_f, svri_f = VectorizedPharmaEngine.apply_drugs_vectorized(
            map_raw, ci_raw + ci_fluid, hr, svri_raw, drugs, self.mins
        )
        
        # --- 5. RESPIRATORY ---
        paco2 = 40 + (18 - rr) * 1.5
        p_ideal = AdvancedPhysiologyEngine.alveolar_gas_equation(drugs['fio2'], paco2)
        pao2, spo2 = AdvancedPhysiologyEngine.calculate_shunt_effect(p_ideal, p['shunt'], peep)
        
        # --- 6. METABOLIC ---
        # DO2I = CI * Hb * 1.34 * SpO2
        hb = 8.0 if "Hemorrhage" in case_id else 12.0
        do2i = ci_f * hb * 1.34 * (spo2/100) * 10
        lactate = AdvancedPhysiologyEngine.lactate_kinetics_vectorized(do2i, self.mins, p['liver'])
        
        # --- 7. DATA PACKAGING ---
        # Convert Indexed back to Absolute for some raw displays if needed, but modern ICU uses Index
        co_abs = ci_f * bsa
        svr_abs = svri_f / bsa
        
        df = pd.DataFrame({
            "Time": self.t, "HR": hr_f, "MAP": map_f, "CI": ci_f, "SVRI": svri_f,
            "CO": co_abs, "SVR": svr_abs,
            "Lactate": lactate, "SpO2": spo2, "PaO2": pao2, "PaCO2": paco2, "RR": rr,
            "PPV": ppv, "DO2I": do2i,
            "Creatinine": np.linspace(0.9, 1.4 if map_f[-1] < 60 else 1.0, self.mins)
        })
        
        # Advanced Derived Metrics
        df['CPO'] = (df['MAP'] * df['CO']) / 451
        df['CPO_Index'] = (df['MAP'] * df['CI']) / 451 # CPI
        df['SI'] = df['HR'] / df['MAP']
        
        return df, p['desc']

# ==========================================
# 4. DECISION SUPPORT & ANALYTICS
# ==========================================
class DecisionSupport:
    @staticmethod
    def decision_tree_classifier(row):
        # Professional Clinical Classifications
        if row['MAP'] < 65:
            if row['CI'] < 2.2:
                if row['SVRI'] > 2000: return "Cardiogenic Shock (Cold/Wet)", "b-crit"
                else: return "Mixed/Vasoplegic Failure", "b-crit"
            else:
                if row['SVRI'] < 1200: return "Distributive Shock (Warm/Dry)", "b-crit"
                else: return "Uncompensated Hypotension", "b-warn"
        else:
            if row['Lactate'] > 4.0: return "Occult Hypoperfusion", "b-crit"
            elif row['Lactate'] > 2.0: return "Stress / Early Sepsis", "b-warn"
            else: return "Hemodynamically Stable", "b-ok"

    @staticmethod
    def monte_carlo_forecast(df, target='MAP', horizons=[10, 20, 30], n_sims=50):
        current_val = df[target].iloc[-1]
        # Volatility is inherent to the recent history (autonomic instability)
        volatility = np.std(df[target].iloc[-30:])
        paths = []
        for _ in range(n_sims):
            noise = np.random.normal(0, volatility, max(horizons))
            # Drift assumption: Current trend continues with decay
            trend = np.mean(np.diff(df[target].iloc[-15:]))
            paths.append(current_val + np.cumsum(noise + trend))
        paths = np.array(paths)
        return np.percentile(paths, 10, axis=0), np.percentile(paths, 50, axis=0), np.percentile(paths, 90, axis=0)

    @staticmethod
    def inverse_transform_centroids(df):
        # K-Means on Indexed values
        features = ['CI', 'SVRI', 'Lactate']
        X = df[features].iloc[-60:]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
        real_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        return [f"C{i+1}: CI={c[0]:.1f}, SVRI={c[1]:.0f}, Lac={c[2]:.1f}" for i, c in enumerate(real_centers)]

# ==========================================
# 5. VISUALIZATION SUITE (CLINICAL GRADE)
# ==========================================
def plot_monte_carlo(df, target, p10, p50, p90):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(30), y=df[target].iloc[-30:], name="History", line=dict(color=THEME['text_main'])))
    future_x = np.arange(30, 30 + len(p50))
    fig.add_trace(go.Scatter(x=np.concatenate([future_x, future_x[::-1]]), y=np.concatenate([p90, p10[::-1]]), fill='toself', fillcolor='rgba(59, 130, 246, 0.2)', line=dict(color='rgba(255,255,255,0)'), name='80% CI'))
    fig.add_trace(go.Scatter(x=future_x, y=p50, name="Median", line=dict(color=THEME['info'], dash='dot')))
    fig.update_layout(height=200, title=f"Stochastic Forecast: {target}", margin=dict(l=20,r=20,t=30,b=20))
    return fig

def plot_chaos_attractor(df):
    rr_series = 60000 / df['HR'].iloc[-120:]
    rr_n = rr_series.iloc[:-1].values
    rr_n1 = rr_series.iloc[1:].values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rr_n, y=rr_n1, mode='markers', marker=dict(color=np.arange(len(rr_n)), colorscale='Teal', size=4, opacity=0.7)))
    fig.update_layout(title="Autonomic Integrity (Poincaré)", xaxis_title="RR(n)", yaxis_title="RR(n+1)", height=200, margin=dict(l=20,r=20,t=30,b=20))
    return fig

def plot_hemodynamic_profile(df):
    """Forrester-style Pump vs Pipes."""
    recent = df.iloc[-60:]
    fig = go.Figure()
    # Reference lines for shock types
    fig.add_hline(y=2000, line_dash="dot", line_color="gray", annotation_text="Vasoconstriction")
    fig.add_vline(x=2.2, line_dash="dot", line_color="gray", annotation_text="Low Flow")
    fig.add_trace(go.Scatter(x=recent['CI'], y=recent['SVRI'], mode='markers', marker=dict(color=recent.index, colorscale='Viridis', size=6)))
    fig.update_layout(title="Hemodynamic Profile (CI vs SVRI)", xaxis_title="Cardiac Index (L/min/m²)", yaxis_title="SVRI (dyn·s/cm⁵/m²)", height=200, margin=dict(l=20,r=20,t=30,b=20))
    return fig

def plot_phase_space_advanced(df):
    recent = df.iloc[-60:]
    fig = go.Figure()
    # Kill zone
    fig.add_shape(type="rect", x0=0, x1=0.6, y0=4, y1=15, fillcolor="rgba(220, 38, 38, 0.1)", line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=recent['CPO'], y=recent['Lactate'], mode='lines+markers', marker=dict(color=recent.index, colorscale='Bluered', size=5)))
    fig.update_layout(title="Coupling (CPO vs Lactate)", xaxis_title="Cardiac Power (W)", yaxis_title="Lactate (mmol/L)", height=200, margin=dict(l=20,r=20,t=30,b=20))
    return fig

def plot_3d_attractor(df):
    recent = df.iloc[-60:]
    fig = go.Figure(data=[go.Scatter3d(x=recent['CPO'], y=recent['SVRI'], z=recent['Lactate'], mode='lines+markers', marker=dict(size=3, color=recent.index, colorscale='Viridis'), line=dict(width=2))])
    fig.update_layout(scene=dict(xaxis_title='Power', yaxis_title='SVRI', zaxis_title='Lac'), margin=dict(l=0, r=0, b=0, t=0), height=250, title="3D Attractor State")
    return fig

def plot_spectral_analysis(df):
    data = df['HR'].iloc[-120:].to_numpy()
    f, Pxx = welch(data, fs=1/60) 
    fig = px.line(x=f, y=Pxx, title="HRV Spectral Density")
    fig.add_vline(x=0.04, line_dash="dot", annotation_text="LF"); fig.add_vline(x=0.15, line_dash="dot", annotation_text="HF")
    fig.update_layout(height=200, margin=dict(l=20,r=20,t=30,b=20))
    return fig

def plot_vq_scatter(df):
    fig = px.scatter(df.iloc[-60:], x="PaO2", y="SpO2", color="PaCO2", title="Oxygenation (Shunt Proxy)", color_continuous_scale="Bluered")
    fig.update_layout(height=200, margin=dict(l=20,r=20,t=30,b=20))
    return fig

def plot_counterfactual(df, drugs, case_id, bsa, peep):
    # Simulate "What if no drugs were given"
    sim = PatientSimulator(mins=60)
    base_drugs = {'norepi':0, 'vaso':0, 'dobu':0, 'bb':0, 'fio2':0.21} 
    df_base, _ = sim.get_data(case_id, base_drugs, 0, bsa, peep)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df['MAP'].iloc[-60:], name="Current", line=dict(color=THEME['ok'])))
    fig.add_trace(go.Scatter(y=df_base['MAP'].iloc[-60:], name="Untreated Proj.", line=dict(dash='dot', color=THEME['crit'])))
    fig.update_layout(title="Intervention Efficacy (MAP)", height=200, margin=dict(l=20,r=20,t=30,b=20))
    return fig

# ==========================================
# 6. MAIN APP EXECUTION
# ==========================================
st.markdown(STYLING, unsafe_allow_html=True)
if 'events' not in st.session_state: st.session_state['events'] = []
if 'fluids_given' not in st.session_state: st.session_state['fluids_given'] = 0

# --- CLINICAL SIDEBAR ---
with st.sidebar:
    st.title("TITAN | L5 CDS")
    st.markdown("### 🏥 Admission & Profile")
    case_id = st.selectbox("Select Case", ["65M Post-CABG", "24F Septic Shock", "82M HFpEF Sepsis", "50M Trauma (Hemorrhage)"])
    
    col_bio1, col_bio2 = st.columns(2)
    with col_bio1: height = st.number_input("Height (cm)", 150, 200, 175)
    with col_bio2: weight = st.number_input("Weight (kg)", 50, 150, 80)
    # Mosteller Formula for BSA
    bsa = np.sqrt((height * weight) / 3600)
    st.caption(f"Calculated BSA: {bsa:.2f} m²")

    st.markdown("### 💉 Infusion Pumps")
    c1, c2 = st.columns(2)
    with c1: 
        norepi = st.number_input("Norepinephrine", 0.0, 2.0, 0.0, 0.05, format="%.2f", help="mcg/kg/min")
        vaso = st.number_input("Vasopressin", 0.0, 0.06, 0.0, 0.01, format="%.2f", help="U/min")
    with c2: 
        dobu = st.number_input("Dobutamine", 0.0, 10.0, 0.0, 0.5, format="%.1f", help="mcg/kg/min")
        bb = st.number_input("Esmolol (BB)", 0.0, 1.0, 0.0, 0.1, format="%.1f")
    
    st.markdown("### 🫁 Ventilator & Fluids")
    c3, c4 = st.columns(2)
    with c3: fio2 = st.slider("FiO2", 0.21, 1.0, 0.40)
    with c4: peep = st.slider("PEEP", 0, 20, 5)
    
    if st.button("💧 500mL Crystalloid"): 
        st.session_state['fluids_given'] += 500
        st.session_state['events'].append({"time": 360, "event": "Fluid Bolus"})
    
    live_mode = st.checkbox("🔴 LIVE MONITORING")

drug_dict = {'norepi': norepi, 'vaso': vaso, 'dobu': dobu, 'bb': bb, 'fio2': fio2}

# --- SIMULATION ---
sim = PatientSimulator(mins=360)
df_full, pathology_desc = sim.get_data(case_id, drug_dict, st.session_state['fluids_given'], bsa, peep)

# Compute Analytics
cur = df_full.iloc[-1]
prev = df_full.iloc[-60]
dt_class, dt_style = DecisionSupport.decision_tree_classifier(cur)
p10, p50, p90 = DecisionSupport.monte_carlo_forecast(df_full, 'MAP')
centroids = DecisionSupport.inverse_transform_centroids(df_full)

# --- 1. CLINICAL HEADER ---
st.markdown(f"""
<div class="status-banner" style="border-left-color: {THEME['ai']};">
    <div>
        <div style="font-size:0.8rem; color:{THEME['ai']}; font-weight:800;">DECISION SUPPORT CLASSIFICATION</div>
        <div style="font-size:1.8rem; font-weight:800; color:{THEME['text_main']}">{dt_class}</div>
        <div style="font-size:0.9rem; color:{THEME['text_muted']};">Pathology: {pathology_desc} | BSA: {bsa:.2f} m²</div>
    </div>
    <div style="text-align:right">
        <div style="font-size:0.8rem; font-weight:700;">PHYSIOLOGIC STATE CLUSTERS</div>
        <div style="font-size:0.7rem;">{centroids[0]}</div>
        <div style="font-size:0.7rem;">{centroids[1]}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 2. MAIN DASHBOARD RENDERER ---
live_container = st.empty()

def render_dashboard(df, is_live=False):
    c_df = df.iloc[-1]
    p_df = df.iloc[-60]
    
    with live_container.container():
        # ZONE A: PREDICTIVE & INTERVENTION
        st.markdown('<div class="zone-header">ZONE A: PREDICTION & INTERVENTION EFFICACY</div>', unsafe_allow_html=True)
        z1, z2, z3 = st.columns(3)
        with z1: st.plotly_chart(plot_monte_carlo(df, 'MAP', p10, p50, p90), use_container_width=True)
        with z2: st.plotly_chart(plot_counterfactual(df, drug_dict, case_id, bsa, peep), use_container_width=True)
        with z3: st.plotly_chart(plot_vq_scatter(df), use_container_width=True)

        # ZONE B: ADVANCED PHASE SPACE
        st.markdown('<div class="zone-header">ZONE B: ADVANCED HEMODYNAMIC PHYSICS</div>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        with b1: st.plotly_chart(plot_3d_attractor(df), use_container_width=True)
        with b2: st.plotly_chart(plot_phase_space_advanced(df), use_container_width=True)
        with b3: st.plotly_chart(plot_hemodynamic_profile(df), use_container_width=True)

        # ZONE C: COMPLEXITY
        st.markdown('<div class="zone-header">ZONE C: AUTONOMIC COMPLEXITY</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_chaos_attractor(df), use_container_width=True)
        with c2: st.plotly_chart(plot_spectral_analysis(df), use_container_width=True)

        # ZONE D: VITALS MATRIX (INDEXED)
        st.markdown('<div class="zone-header">ZONE D: REAL-TIME INDICES (INDEXED TO BSA)</div>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        def arrow(val): return "↑" if val > 0 else "↓" if val < 0 else "→"
        def metric_card(col, label, val, unit, delta):
            col.metric(label, f"{val:.1f} {unit}", f"{arrow(delta)} {abs(delta):.1f}", delta_color="inverse")
        
        # Displaying CI instead of CO, SVRI instead of SVR
        metric_card(m1, "MAP", c_df['MAP'], "mmHg", c_df['MAP']-p_df['MAP'])
        metric_card(m2, "HR", c_df['HR'], "bpm", c_df['HR']-p_df['HR'])
        metric_card(m3, "CI (Index)", c_df['CI'], "L/min/m²", c_df['CI']-p_df['CI'])
        metric_card(m4, "Lactate", c_df['Lactate'], "mmol/L", c_df['Lactate']-p_df['Lactate'])
        metric_card(m5, "DO2I", c_df['DO2I'], "mL/m²", c_df['DO2I']-p_df['DO2I'])
        metric_card(m6, "SVRI", c_df['SVRI'], "dyn·s", c_df['SVRI']-p_df['SVRI'])

        # ZONE E: TELEMETRY
        st.markdown('<div class="zone-header">ZONE E: TELEMETRY TRENDS</div>', unsafe_allow_html=True)
        fig_tele = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=("Hemodynamics (MAP)", "Perfusion (CPO)", "Respiratory (SpO2)"))
        fig_tele.add_trace(go.Scatter(x=df['Time'], y=df['MAP'], name="MAP", line=dict(color=THEME['hemo'])), row=1, col=1)
        fig_tele.add_trace(go.Scatter(x=df['Time'], y=df['CPO'], name="CPO", fill='tozeroy', line=dict(color=THEME['info'])), row=2, col=1)
        fig_tele.add_trace(go.Scatter(x=df['Time'], y=df['SpO2'], name="SpO2", line=dict(color=THEME['resp'])), row=3, col=1)
        for e in st.session_state['events']: fig_tele.add_vline(x=e['time'], line_dash="dash", line_color="green")
        fig_tele.update_layout(height=400, margin=dict(l=0,r=0,t=20,b=20), template="plotly_white")
        st.plotly_chart(fig_tele, use_container_width=True)

if live_mode:
    start_idx = 300
    for i in range(start_idx, 360):
        render_dashboard(df_full.iloc[:i], is_live=True)
        time.sleep(0.5)
else:
    render_dashboard(df_full, is_live=False)

st.caption("TITAN L5 CDS | Modules: Vectorized Physics, Autonomic Baroreflex Modeling, Monte-Carlo Uncertainty, 3D Attractors.")
