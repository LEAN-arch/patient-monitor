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
# 1. CONFIGURATION & CONSTANTS
# ==========================================
st.set_page_config(
    page_title="TITAN | ULTIMATE CDS",
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
    DRUG_MULTS = {
        'norepi': {'svr': 800, 'map': 25, 'co': 0.5},
        'vaso':   {'svr': 600, 'map': 15, 'co': 0.0},
        'dobu':   {'svr': -300, 'map': 0, 'co': 2.5, 'hr': 15},
        'bb':     {'svr': 0, 'map': -5, 'co': -0.8, 'hr': -20}
    }
    ATM_PRESSURE = 760
    H2O_PRESSURE = 47
    R_QUOTIENT = 0.8
    MAX_PAO2 = 600
    LAC_PROD_THRESH = 400
    LAC_CLEAR_RATE = 0.05

# ==========================================
# 2. VECTORIZED PHYSICS ENGINE (L5)
# ==========================================
class VectorizedPharmaEngine:
    @staticmethod
    def apply_drugs_vectorized(map_base, co_base, hr_base, svr_base, drugs, mins):
        svr_eff = drugs['norepi'] * CONSTANTS.DRUG_MULTS['norepi']['svr'] + drugs['vaso'] * CONSTANTS.DRUG_MULTS['vaso']['svr'] + drugs['dobu'] * CONSTANTS.DRUG_MULTS['dobu']['svr']
        map_eff = drugs['norepi'] * CONSTANTS.DRUG_MULTS['norepi']['map'] + drugs['vaso'] * CONSTANTS.DRUG_MULTS['vaso']['map'] + drugs['bb'] * CONSTANTS.DRUG_MULTS['bb']['map']
        co_eff  = drugs['norepi'] * CONSTANTS.DRUG_MULTS['norepi']['co'] + drugs['dobu'] * CONSTANTS.DRUG_MULTS['dobu']['co'] + drugs['bb'] * CONSTANTS.DRUG_MULTS['bb']['co']
        hr_eff  = drugs['dobu'] * CONSTANTS.DRUG_MULTS['dobu']['hr'] + drugs['bb'] * CONSTANTS.DRUG_MULTS['bb']['hr']
        return (map_base + map_eff), (co_base + co_eff), (hr_base + hr_eff), (svr_base + svr_eff)

class AdvancedPhysiologyEngine:
    @staticmethod
    def brownian_bridge_vector(n, start, end, volatility=1.0, seed=42):
        np.random.seed(seed)
        t = np.linspace(0, 1, n)
        dW = np.random.normal(0, np.sqrt(1/n), n)
        W = np.cumsum(dW)
        B = start + W - t * (W[-1] - (end - start))
        pink = np.convolve(np.random.normal(0, 0.5, n), np.ones(5)/5, mode='same')
        return B + (pink * volatility)

    @staticmethod
    def alveolar_gas_equation(fio2, paco2, peep):
        p_ideal = (fio2 * (CONSTANTS.ATM_PRESSURE - CONSTANTS.H2O_PRESSURE)) - (paco2 / CONSTANTS.R_QUOTIENT)
        return np.clip(p_ideal, 0, CONSTANTS.MAX_PAO2)

    @staticmethod
    def calculate_shunt_effect(p_ideal, shunt_fraction, peep):
        effective_shunt = shunt_fraction * np.exp(-0.05 * peep)
        pao2 = p_ideal * (1 - effective_shunt)
        pao2_safe = np.maximum(pao2, 0.1)
        spo2 = 100 / (1 + (23400 / (pao2_safe**3 + 150*pao2_safe)))
        return pao2, spo2

    @staticmethod
    def lactate_kinetics_vectorized(do2, mins):
        lactate = np.zeros(mins)
        lac_curr = 1.0
        prod = np.where(do2 < CONSTANTS.LAC_PROD_THRESH, 0.1, 0.0)
        clear = np.where(do2 > 500, CONSTANTS.LAC_CLEAR_RATE, 0.01)
        net = prod - clear
        for i in range(mins):
            lac_curr = max(0.5, lac_curr + net[i])
            lactate[i] = lac_curr
        return lactate

# ==========================================
# 3. PATIENT SIMULATOR
# ==========================================
class PatientSimulator:
    def __init__(self, mins=360):
        self.mins = mins
        self.t = np.arange(mins)

    def get_data(self, profile, drugs, fluids, age, chronic, drift, seed, peep):
        age_svr = 1.2 if age > 70 else 1.0
        hf_mod = 0.7 if "Heart Failure" in chronic else 1.0
        copd_mod = 1.2 if "COPD" in chronic else 1.0
        
        scenarios = {
            "Healthy": {'hr': (65, 75), 'map': (85, 85), 'co': (5.0, 5.0), 'vol': 1.0, 'shunt': 0.05},
            "Compensated Sepsis": {'hr': (85, 105), 'map': (80, 75), 'co': (6.0, 7.0), 'vol': 1.5, 'shunt': 0.10},
            "Vasoplegic Shock": {'hr': (110, 135), 'map': (60, 48), 'co': (7.0, 8.5), 'vol': 1.2, 'shunt': 0.15},
            "Cardiogenic Shock": {'hr': (90, 105), 'map': (70, 58), 'co': (3.5, 2.2), 'vol': 2.0, 'shunt': 0.20}
        }
        p = scenarios[profile]
        drift_vec = np.linspace(1.0, drift, self.mins)
        
        hr = AdvancedPhysiologyEngine.brownian_bridge_vector(self.mins, p['hr'][0], p['hr'][1], p['vol'], seed) * drift_vec
        map_raw = AdvancedPhysiologyEngine.brownian_bridge_vector(self.mins, p['map'][0], p['map'][1], p['vol']*0.8, seed+1) / drift_vec
        co_raw = AdvancedPhysiologyEngine.brownian_bridge_vector(self.mins, p['co'][0], p['co'][1], 0.2, seed+2) * hf_mod / drift_vec
        rr = AdvancedPhysiologyEngine.brownian_bridge_vector(self.mins, 14, 24 if "Shock" in profile else 16, 2.0, seed+3)
        
        ppv = (15 if "Shock" in profile else 5) + (np.sin(self.t/10) * 3)
        co_fluid = np.where(np.mean(ppv) > 12, (fluids/500)*1.2, (fluids/500)*0.1)
        
        svr_base = ((map_raw - 8) / (co_raw + 0.1)) * 800 * age_svr
        map_f, co_f, hr_f, svr_f = VectorizedPharmaEngine.apply_drugs_vectorized(
            map_raw, co_raw + co_fluid, hr, svr_base, drugs, self.mins
        )
        
        paco2 = 40 + (16 - rr) * 1.5
        p_ideal = AdvancedPhysiologyEngine.alveolar_gas_equation(drugs['fio2'], paco2, peep)
        pao2, spo2 = AdvancedPhysiologyEngine.calculate_shunt_effect(p_ideal, p['shunt'] * copd_mod, peep)
        
        do2 = co_f * 12.0 * 1.34 * (spo2/100) * 10
        lactate = AdvancedPhysiologyEngine.lactate_kinetics_vectorized(do2, self.mins)
        
        df = pd.DataFrame({
            "Time": self.t, "HR": hr_f, "MAP": map_f, "CO": co_f, "SVR": svr_f,
            "Lactate": lactate, "SpO2": spo2, "PaO2": pao2, "PaCO2": paco2, "RR": rr,
            "PPV": ppv, "DO2": do2,
            "Creatinine": np.linspace(0.8, 1.2 if map_f[-1] < 60 else 0.9, self.mins)
        })
        
        df['CPO'] = (df['MAP'] * df['CO']) / 451
        df['SI'] = df['HR'] / df['MAP']
        return df

# ==========================================
# 4. DECISION SUPPORT & ANALYTICS
# ==========================================
class DecisionSupport:
    @staticmethod
    def decision_tree_classifier(row):
        if row['MAP'] < 65:
            if row['Lactate'] > 2.0 or row['CPO'] < 0.6:
                if row['CO'] < 4.0: return "Cardiogenic Shock", "b-crit"
                elif row['SVR'] < 800: return "Distributive Shock", "b-crit"
                else: return "Mixed/Obstructive Shock", "b-crit"
            else: return "Compensated Hypotension", "b-warn"
        else:
            if row['Lactate'] > 2.0: return "Cryptic Sepsis", "b-warn"
            elif row['HR'] > 100: return "Hyperdynamic", "b-info"
            else: return "Stable", "b-ok"

    @staticmethod
    def monte_carlo_forecast(df, target='MAP', horizons=[10, 20, 30], n_sims=50):
        current_val = df[target].iloc[-1]
        volatility = np.std(df[target].iloc[-30:])
        paths = []
        for _ in range(n_sims):
            noise = np.random.normal(0, volatility, max(horizons))
            paths.append(current_val + np.cumsum(noise))
        paths = np.array(paths)
        return np.percentile(paths, 10, axis=0), np.percentile(paths, 50, axis=0), np.percentile(paths, 90, axis=0)

    @staticmethod
    def inverse_transform_centroids(df):
        features = ['CO', 'SVR', 'Lactate']
        X = df[features].iloc[-60:]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
        real_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        return [f"C{i+1}: CO={c[0]:.1f}, SVR={c[1]:.0f}, Lac={c[2]:.1f}" for i, c in enumerate(real_centers)]

# ==========================================
# 5. VISUALIZATION SUITE (ALL RESTORED)
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
    """Restored Poincaré Plot."""
    rr_series = 60000 / df['HR'].iloc[-120:]
    rr_n = rr_series.iloc[:-1].values
    rr_n1 = rr_series.iloc[1:].values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rr_n, y=rr_n1, mode='markers', marker=dict(color=np.arange(len(rr_n)), colorscale='Teal', size=4, opacity=0.7)))
    fig.update_layout(title="Chaos (HRV Poincaré)", xaxis_title="RR(n)", yaxis_title="RR(n+1)", height=200, margin=dict(l=20,r=20,t=30,b=20))
    return fig

def plot_hemodynamic_profile(df):
    """Restored Pump vs Pipes."""
    recent = df.iloc[-60:]
    fig = go.Figure()
    fig.add_hline(y=1000, line_dash="dot", line_color="gray")
    fig.add_vline(x=5.0, line_dash="dot", line_color="gray")
    fig.add_trace(go.Scatter(x=recent['CO'], y=recent['SVR'], mode='markers', marker=dict(color=recent.index, colorscale='Viridis', size=6)))
    fig.update_layout(title="Pump (CO) vs Pipes (SVR)", xaxis_title="CO", yaxis_title="SVR", height=200, margin=dict(l=20,r=20,t=30,b=20))
    return fig

def plot_phase_space_advanced(df):
    """Restored 2D Phase Space with Risk Zones."""
    recent = df.iloc[-60:]
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=0.6, y0=2, y1=15, fillcolor="rgba(220, 38, 38, 0.1)", line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=recent['CPO'], y=recent['Lactate'], mode='lines+markers', marker=dict(color=recent.index, colorscale='Bluered', size=5)))
    fig.update_layout(title="Hemo-Metabolic Coupling", xaxis_title="CPO (W)", yaxis_title="Lactate", height=200, margin=dict(l=20,r=20,t=30,b=20))
    return fig

def plot_3d_attractor(df):
    recent = df.iloc[-60:]
    fig = go.Figure(data=[go.Scatter3d(x=recent['CPO'], y=recent['SVR'], z=recent['Lactate'], mode='lines+markers', marker=dict(size=3, color=recent.index, colorscale='Viridis'), line=dict(width=2))])
    fig.update_layout(scene=dict(xaxis_title='Power', yaxis_title='SVR', zaxis_title='Lac'), margin=dict(l=0, r=0, b=0, t=0), height=250, title="3D Attractor")
    return fig

def plot_spectral_analysis(df):
    data = df['HR'].iloc[-120:].to_numpy()
    f, Pxx = welch(data, fs=1/60) 
    fig = px.line(x=f, y=Pxx, title="Spectral HRV (Autonomics)")
    fig.add_vline(x=0.04, line_dash="dot"); fig.add_vline(x=0.15, line_dash="dot")
    fig.update_layout(height=200, margin=dict(l=20,r=20,t=30,b=20))
    return fig

def plot_vq_scatter(df):
    fig = px.scatter(df.iloc[-60:], x="PaO2", y="SpO2", color="PaCO2", title="V/Q Scatter", color_continuous_scale="Bluered")
    fig.update_layout(height=200, margin=dict(l=20,r=20,t=30,b=20))
    return fig

def plot_counterfactual(df, drugs, profile):
    sim = PatientSimulator(mins=60)
    base_drugs = {'norepi':0, 'vaso':0, 'dobu':0, 'bb':0, 'fio2':0.21} 
    df_base = sim.get_data(profile, base_drugs, 0, 50, [], 1.0, 42, 5)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df['MAP'].iloc[-60:], name="Intervention", line=dict(color=THEME['ok'])))
    fig.add_trace(go.Scatter(y=df_base['MAP'].iloc[-60:], name="Untreated", line=dict(dash='dot', color=THEME['crit'])))
    fig.update_layout(title="Intervention Effect (MAP)", height=200, margin=dict(l=20,r=20,t=30,b=20))
    return fig

# ==========================================
# 6. MAIN APP EXECUTION
# ==========================================
st.markdown(STYLING, unsafe_allow_html=True)
if 'events' not in st.session_state: st.session_state['events'] = []
if 'fluids_given' not in st.session_state: st.session_state['fluids_given'] = 0

with st.sidebar:
    st.title("TITAN | L5 CDS")
    seed = st.number_input("Seed", 1, 1000, 42)
    drift = st.slider("Drift", 1.0, 2.0, 1.0, 0.1)
    age = st.slider("Age", 18, 95, 65)
    chronic = st.multiselect("Comorbidities", ["Heart Failure", "COPD", "CKD"])
    scenario = st.selectbox("Phenotype", ["Healthy", "Compensated Sepsis", "Vasoplegic Shock", "Cardiogenic Shock"])
    c1, c2 = st.columns(2)
    with c1: norepi = st.slider("Norepi", 0.0, 1.0, 0.0, 0.05); vaso = st.slider("Vaso", 0.0, 0.06, 0.0, 0.01)
    with c2: dobu = st.slider("Dobutamine", 0.0, 10.0, 0.0, 1.0); bb = st.slider("B-Block", 0.0, 1.0, 0.0, 0.1)
    fio2 = st.slider("FiO2", 0.21, 1.0, 0.40); peep = st.slider("PEEP", 0, 20, 5)
    if st.button("💧 500mL Bolus"): st.session_state['fluids_given'] += 500; st.session_state['events'].append({"time": 360, "event": "Fluid Bolus"})
    live_mode = st.checkbox("🔴 LIVE FEED")

drug_dict = {'norepi': norepi, 'vaso': vaso, 'dobu': dobu, 'bb': bb, 'fio2': fio2}
sim = PatientSimulator(mins=360)
df_full = sim.get_data(scenario, drug_dict, st.session_state['fluids_given'], age, chronic, drift, seed, peep)

# Compute Analytics
cur = df_full.iloc[-1]
prev = df_full.iloc[-60]
dt_class, dt_style = DecisionSupport.decision_tree_classifier(cur)
p10, p50, p90 = DecisionSupport.monte_carlo_forecast(df_full, 'MAP')
centroids = DecisionSupport.inverse_transform_centroids(df_full)

# --- 1. BANNER ---
st.markdown(f"""
<div class="status-banner" style="border-left-color: {THEME['ai']};">
    <div>
        <div style="font-size:0.8rem; color:{THEME['ai']}; font-weight:800;">CLINICAL DECISION TREE</div>
        <div style="font-size:1.8rem; font-weight:800; color:{THEME['text_main']}">{dt_class}</div>
        <div style="font-size:0.9rem; color:{THEME['text_muted']};">FiO2: {fio2} | PEEP: {peep} | Age: {age}</div>
    </div>
    <div style="text-align:right">
        <div style="font-size:0.8rem; font-weight:700;">CLUSTERS</div>
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
        # ZONE A: PREDICTIVE & INTERVENTION (Row 1)
        st.markdown('<div class="zone-header">ZONE A: PREDICTION & INTERVENTION</div>', unsafe_allow_html=True)
        z1, z2, z3 = st.columns(3)
        with z1: st.plotly_chart(plot_monte_carlo(df, 'MAP', p10, p50, p90), use_container_width=True)
        with z2: st.plotly_chart(plot_counterfactual(df, drug_dict, scenario), use_container_width=True)
        with z3: st.plotly_chart(plot_vq_scatter(df), use_container_width=True)

        # ZONE B: ADVANCED PHASE SPACE PHYSICS (Row 2)
        st.markdown('<div class="zone-header">ZONE B: ADVANCED PHASE SPACE PHYSICS</div>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        with b1: st.plotly_chart(plot_3d_attractor(df), use_container_width=True)
        with b2: st.plotly_chart(plot_phase_space_advanced(df), use_container_width=True)
        with b3: st.plotly_chart(plot_hemodynamic_profile(df), use_container_width=True)

        # ZONE C: COMPLEXITY & AUTONOMICS (Row 3)
        st.markdown('<div class="zone-header">ZONE C: COMPLEXITY & AUTONOMICS</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_chaos_attractor(df), use_container_width=True)
        with c2: st.plotly_chart(plot_spectral_analysis(df), use_container_width=True)

        # ZONE D: VITALS MATRIX (Row 4)
        st.markdown('<div class="zone-header">ZONE D: REAL-TIME METRICS</div>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        def arrow(val): return "↑" if val > 0 else "↓" if val < 0 else "→"
        def metric_card(col, label, val, unit, delta):
            col.metric(label, f"{val:.1f} {unit}", f"{arrow(delta)} {abs(delta):.1f}", delta_color="inverse")
        metric_card(m1, "MAP", c_df['MAP'], "mmHg", c_df['MAP']-p_df['MAP'])
        metric_card(m2, "HR", c_df['HR'], "bpm", c_df['HR']-p_df['HR'])
        metric_card(m3, "CPO", c_df['CPO'], "W", c_df['CPO']-p_df['CPO'])
        metric_card(m4, "Lac", c_df['Lactate'], "mM", c_df['Lactate']-p_df['Lactate'])
        metric_card(m5, "PaO2", c_df['PaO2'], "mmHg", c_df['PaO2']-p_df['PaO2'])
        metric_card(m6, "SVR", c_df['SVR'], "dyn", c_df['SVR']-p_df['SVR'])

        # ZONE E: TELEMETRY (Row 5)
        st.markdown('<div class="zone-header">ZONE E: TELEMETRY</div>', unsafe_allow_html=True)
        fig_tele = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=("Hemodynamics", "Perfusion", "Respiratory"))
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

st.caption("TITAN L5 CDS | Modules: Vectorized Physics, Monte-Carlo Uncertainty, 3D Attractors, Spectral HRV.")
