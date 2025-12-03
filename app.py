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
# 1. CONFIGURATION & CONSTANTS (CENTRALIZED)
# ==========================================
st.set_page_config(
    page_title="TITAN | L5 CDS",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧬"
)

# Advanced Medical Palette
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
    div[data-testid="stMetric"] {{ background-color: {THEME['card_bg']}; padding: 8px 12px; border-radius: 6px; border: 1px solid {THEME['border']}; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); }}
    div[data-testid="stMetric"] label {{ font-size: 0.65rem; font-weight: 700; color: {THEME['text_muted']}; text-transform: uppercase; }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ font-family: 'Roboto Mono'; font-size: 1.4rem; font-weight: 800; }}
    .zone-header {{ font-size: 0.8rem; font-weight: 900; color: {THEME['text_muted']}; text-transform: uppercase; border-bottom: 2px solid {THEME['border']}; margin: 20px 0 10px 0; padding-bottom: 4px; letter-spacing: 0.05em; }}
    .status-banner {{ padding: 12px 20px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; background: white; border-left: 6px solid; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 10px; }}
</style>
"""

class CONSTANTS:
    """Centralized Clinical & Physics Constants."""
    # Drug Multipliers (Effect on Base Vitals)
    DRUG_MULTS = {
        'norepi': {'svr': 800, 'map': 25, 'co': 0.5},  # Alpha-1 strong
        'vaso':   {'svr': 600, 'map': 15, 'co': 0.0},  # V1
        'dobu':   {'svr': -300, 'map': 0, 'co': 2.5, 'hr': 15}, # Beta-1/2
        'bb':     {'svr': 0, 'map': -5, 'co': -0.8, 'hr': -20}  # Beta blockade
    }
    
    # Respiratory Constants
    ATM_PRESSURE = 760  # mmHg
    H2O_PRESSURE = 47   # mmHg
    R_QUOTIENT = 0.8
    MAX_PAO2 = 600      # Ceiling effect
    
    # Lactate Kinetics
    LAC_PROD_THRESH = 400 # DO2 threshold
    LAC_CLEAR_RATE = 0.05

# ==========================================
# 2. VECTORIZED PHYSICS & PHARMA ENGINE
# ==========================================
class VectorizedPharmaEngine:
    """Optimized Array-based Pharmacodynamics."""
    @staticmethod
    def apply_drugs_vectorized(map_base, co_base, hr_base, svr_base, drugs, mins):
        # Create Drug Arrays (Broadcasting)
        # Note: In a full sim, these would be time-varying arrays. 
        # Here we apply the current slider value as a "steady state" infusion simulation
        
        # Norepinephrine
        svr_eff = drugs['norepi'] * CONSTANTS.DRUG_MULTS['norepi']['svr']
        map_eff = drugs['norepi'] * CONSTANTS.DRUG_MULTS['norepi']['map']
        co_eff  = drugs['norepi'] * CONSTANTS.DRUG_MULTS['norepi']['co']
        
        # Vasopressin
        svr_eff += drugs['vaso'] * CONSTANTS.DRUG_MULTS['vaso']['svr']
        map_eff += drugs['vaso'] * CONSTANTS.DRUG_MULTS['vaso']['map']
        
        # Dobutamine
        co_eff  += drugs['dobu'] * CONSTANTS.DRUG_MULTS['dobu']['co']
        hr_eff   = drugs['dobu'] * CONSTANTS.DRUG_MULTS['dobu']['hr']
        svr_eff += drugs['dobu'] * CONSTANTS.DRUG_MULTS['dobu']['svr']
        
        # Beta Blockers
        hr_eff  += drugs['bb'] * CONSTANTS.DRUG_MULTS['bb']['hr']
        co_eff  += drugs['bb'] * CONSTANTS.DRUG_MULTS['bb']['co']
        
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
        """
        Calculates Alveolar Oxygen (PAO2).
        Includes PEEP effect on FRC/Oxygenation indirectly via Shunt logic later, 
        but here strictly calculates available O2.
        """
        # PAO2 = (Patm - PH2O) * FiO2 - (PaCO2 / R)
        p_ideal = (fio2 * (CONSTANTS.ATM_PRESSURE - CONSTANTS.H2O_PRESSURE)) - (paco2 / CONSTANTS.R_QUOTIENT)
        # Ceiling effect (Law of physics)
        return np.clip(p_ideal, 0, CONSTANTS.MAX_PAO2)

    @staticmethod
    def calculate_shunt_effect(p_ideal, shunt_fraction, peep):
        """
        Vectorized Shunt Calculation.
        PEEP reduces shunt fraction (recruitability).
        """
        # PEEP Benefit: 5cmH2O reduces shunt by ~10% relative (simplified model)
        effective_shunt = shunt_fraction * np.exp(-0.05 * peep)
        
        # PaO2 = PAO2 * (1 - Shunt)
        pao2 = p_ideal * (1 - effective_shunt)
        
        # SpO2 Sigmoid (Severinghaus Equation) - Vectorized
        # SpO2 = 100 / (1 + (23400 / (PaO2^3 + 150*PaO2)))
        # Add epsilon to PaO2 to avoid div by zero
        pao2_safe = np.maximum(pao2, 0.1)
        spo2 = 100 / (1 + (23400 / (pao2_safe**3 + 150*pao2_safe)))
        
        return pao2, spo2

    @staticmethod
    def lactate_kinetics_vectorized(do2, mins):
        """
        Vectorized Lactate Accumulation.
        Uses numpy accumulate/scan logic since L(t) depends on L(t-1).
        """
        lactate = np.zeros(mins)
        lac_curr = 1.0
        
        # Production/Clearance Rates
        prod = np.where(do2 < CONSTANTS.LAC_PROD_THRESH, 0.1, 0.0)
        clear = np.where(do2 > 500, CONSTANTS.LAC_CLEAR_RATE, 0.01)
        net = prod - clear
        
        # Accumulate
        # We cannot use pure cumsum because lactate can't go below 0.5 (physiologic floor)
        # We use a fast loop here as it's O(N)
        for i in range(mins):
            lac_curr = max(0.5, lac_curr + net[i])
            lactate[i] = lac_curr
            
        return lactate

# ==========================================
# 3. PATIENT SIMULATOR (REAL-TIME ENABLED)
# ==========================================
class PatientSimulator:
    def __init__(self, mins=360):
        self.mins = mins
        self.t = np.arange(mins)

    def get_data(self, profile, drugs, fluids, age, chronic, drift, seed, peep):
        # 1. Profile Coefficients
        age_svr = 1.2 if age > 70 else 1.0
        age_co = 0.8 if age > 70 else 1.0
        hf_mod = 0.7 if "Heart Failure" in chronic else 1.0
        copd_mod = 1.2 if "COPD" in chronic else 1.0 # Increases shunt
        
        # 2. Scenario Base Params
        scenarios = {
            "Healthy": {'hr': (65, 75), 'map': (85, 85), 'co': (5.0, 5.0), 'vol': 1.0, 'shunt': 0.05},
            "Compensated Sepsis": {'hr': (85, 105), 'map': (80, 75), 'co': (6.0, 7.0), 'vol': 1.5, 'shunt': 0.10},
            "Vasoplegic Shock": {'hr': (110, 135), 'map': (60, 48), 'co': (7.0, 8.5), 'vol': 1.2, 'shunt': 0.15},
            "Cardiogenic Shock": {'hr': (90, 105), 'map': (70, 58), 'co': (3.5, 2.2), 'vol': 2.0, 'shunt': 0.20}
        }
        p = scenarios[profile]

        drift_vec = np.linspace(1.0, drift, self.mins)
        
        # 3. Vectorized Random Walks
        hr = AdvancedPhysiologyEngine.brownian_bridge_vector(self.mins, p['hr'][0], p['hr'][1], p['vol'], seed) * drift_vec
        map_raw = AdvancedPhysiologyEngine.brownian_bridge_vector(self.mins, p['map'][0], p['map'][1], p['vol']*0.8, seed+1) / drift_vec
        co_raw = AdvancedPhysiologyEngine.brownian_bridge_vector(self.mins, p['co'][0], p['co'][1], 0.2, seed+2) * hf_mod / drift_vec
        rr = AdvancedPhysiologyEngine.brownian_bridge_vector(self.mins, 14, 24 if "Shock" in profile else 16, 2.0, seed+3)
        
        # 4. Fluids (Starling)
        ppv = (15 if "Shock" in profile else 5) + (np.sin(self.t/10) * 3)
        # Simplified: Fluid given at t=0 affects whole curve for simulation, 
        # in real-time mode we would step this. Here we assume steady state volume load.
        co_fluid = np.where(np.mean(ppv) > 12, (fluids/500)*1.2, (fluids/500)*0.1)
        
        # 5. Apply Drugs (Vectorized)
        svr_base = ((map_raw - 8) / (co_raw + 0.1)) * 800 * age_svr
        map_f, co_f, hr_f, svr_f = VectorizedPharmaEngine.apply_drugs_vectorized(
            map_raw, co_raw + co_fluid, hr, svr_base, drugs, self.mins
        )
        
        # 6. Respiratory Physics
        paco2 = 40 + (16 - rr) * 1.5
        p_ideal = AdvancedPhysiologyEngine.alveolar_gas_equation(drugs['fio2'], paco2, peep)
        pao2, spo2 = AdvancedPhysiologyEngine.calculate_shunt_effect(p_ideal, p['shunt'] * copd_mod, peep)
        
        # 7. Lactate Kinetics
        hb = 12.0
        do2 = co_f * hb * 1.34 * (spo2/100) * 10
        lactate = AdvancedPhysiologyEngine.lactate_kinetics_vectorized(do2, self.mins)
        
        # 8. Build DataFrame
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
# 4. AI & UNCERTAINTY ENGINE
# ==========================================
class DecisionSupport:
    @staticmethod
    def decision_tree_classifier(row):
        """
        Deterministic Decision Tree based on Literature Cutpoints.
        Replaces 'Black Box' Clustering for explication.
        """
        # Node 1: Hypotension?
        if row['MAP'] < 65:
            # Node 2: Perfusion?
            if row['Lactate'] > 2.0 or row['CPO'] < 0.6:
                # Node 3: Output State?
                if row['CO'] < 4.0:
                    return "Cardiogenic Shock (Cold/Wet)", "b-crit"
                elif row['SVR'] < 800:
                    return "Distributive Shock (Warm/Dry)", "b-crit"
                else:
                    return "Mixed/Obstructive Shock", "b-crit"
            else:
                return "Compensated Hypotension", "b-warn"
        else:
            # Normotensive
            if row['Lactate'] > 2.0:
                return "Cryptic Sepsis (Occult)", "b-warn"
            elif row['HR'] > 100:
                return "Hyperdynamic/Stress", "b-info"
            else:
                return "Hemodynamically Stable", "b-ok"

    @staticmethod
    def monte_carlo_forecast(df, target='MAP', horizons=[10, 20, 30], n_sims=50):
        """
        Stochastic Forecasting with Confidence Intervals (Fan Chart).
        """
        current_val = df[target].iloc[-1]
        volatility = np.std(df[target].iloc[-30:])
        
        paths = []
        for _ in range(n_sims):
            # Random walk projection
            noise = np.random.normal(0, volatility, max(horizons))
            path = current_val + np.cumsum(noise)
            paths.append(path)
            
        paths = np.array(paths)
        
        # Calculate percentiles
        p10 = np.percentile(paths, 10, axis=0)
        p50 = np.percentile(paths, 50, axis=0)
        p90 = np.percentile(paths, 90, axis=0)
        
        return p10, p50, p90

    @staticmethod
    def inverse_transform_centroids(df):
        """
        K-Means with Inverse Transform to show REAL physiological values.
        """
        features = ['CO', 'SVR', 'Lactate']
        X = df[features].iloc[-60:]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
        
        # Inverse transform the centers
        real_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Format for display
        centroids = []
        for i, center in enumerate(real_centers):
            centroids.append(f"Cluster {i+1}: CO={center[0]:.1f}, SVR={center[1]:.0f}, Lac={center[2]:.1f}")
        
        return centroids

# ==========================================
# 5. VISUALIZATION (ENHANCED)
# ==========================================
def plot_monte_carlo(df, target, p10, p50, p90):
    """Uncertainty Fan Chart."""
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(x=np.arange(30), y=df[target].iloc[-30:], name="History", line=dict(color=THEME['text_main'])))
    
    # Forecast Time
    future_x = np.arange(30, 30 + len(p50))
    
    # Fan (Confidence Interval)
    fig.add_trace(go.Scatter(
        x=np.concatenate([future_x, future_x[::-1]]),
        y=np.concatenate([p90, p10[::-1]]),
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='80% CI'
    ))
    
    # Median
    fig.add_trace(go.Scatter(x=future_x, y=p50, name="Median Forecast", line=dict(color=THEME['info'], dash='dot')))
    
    fig.update_layout(height=250, title=f"Stochastic Forecast: {target}", margin=dict(l=20,r=20,t=30,b=20))
    return fig

def plot_vq_scatter(df):
    """V/Q Scatter Plot (Shunt Visualization)."""
    fig = px.scatter(df.iloc[-60:], x="PaO2", y="SpO2", color="PaCO2", 
                     title="Oxygenation Status (V/Q Proxy)",
                     color_continuous_scale="Bluered")
    fig.add_shape(type="line", x0=60, x1=60, y0=0, y1=100, line=dict(dash="dot", color="red"))
    fig.add_annotation(x=55, y=50, text="Hypoxemia", textangle=-90)
    fig.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20))
    return fig

# ==========================================
# 6. MAIN APP EXECUTION
# ==========================================
st.markdown(STYLING, unsafe_allow_html=True)

# --- SESSION STATE & CACHING ---
if 'events' not in st.session_state: st.session_state['events'] = []
if 'fluids_given' not in st.session_state: st.session_state['fluids_given'] = 0
if 'sim_data' not in st.session_state: st.session_state['sim_data'] = None

# --- SIDEBAR ---
with st.sidebar:
    st.title("TITAN | L5 CDS")
    seed = st.number_input("Random Seed", 1, 1000, 42)
    drift = st.slider("Scenario Drift", 1.0, 2.0, 1.0, 0.1)
    
    st.markdown("### 👤 Profile")
    age = st.slider("Age", 18, 95, 65)
    chronic = st.multiselect("Comorbidities", ["Heart Failure", "COPD", "CKD"])
    scenario = st.selectbox("Phenotype", ["Healthy", "Compensated Sepsis", "Vasoplegic Shock", "Cardiogenic Shock"])
    
    st.markdown("### 💉 Drugs & Vent")
    c1, c2 = st.columns(2)
    with c1:
        norepi = st.slider("Norepi", 0.0, 1.0, 0.0, 0.05)
        vaso = st.slider("Vasopressin", 0.0, 0.06, 0.0, 0.01)
    with c2:
        dobu = st.slider("Dobutamine", 0.0, 10.0, 0.0, 1.0)
        bb = st.slider("Beta-Blocker", 0.0, 1.0, 0.0, 0.1)
    
    fio2 = st.slider("FiO2", 0.21, 1.0, 0.40)
    peep = st.slider("PEEP", 0, 20, 5)
    
    if st.button("💧 500mL Bolus"):
        st.session_state['fluids_given'] += 500
        st.session_state['events'].append({"time": 360, "event": "Fluid Bolus"})
    
    live_mode = st.checkbox("🔴 LIVE FEED (1-min updates)")

# --- DATA GENERATION ---
drug_dict = {'norepi': norepi, 'vaso': vaso, 'dobu': dobu, 'bb': bb, 'fio2': fio2}
sim = PatientSimulator(mins=360)
# Generate Full History
df_full = sim.get_data(scenario, drug_dict, st.session_state['fluids_given'], age, chronic, drift, seed, peep)

# --- ANALYTICS ---
cur = df_full.iloc[-1]
prev = df_full.iloc[-60]

# Decision Tree Classification
dt_class, dt_style = DecisionSupport.decision_tree_classifier(cur)

# Monte Carlo
p10, p50, p90 = DecisionSupport.monte_carlo_forecast(df_full, 'MAP')

# Inverse Centroids
centroids = DecisionSupport.inverse_transform_centroids(df_full)

# ==========================================
# UI LAYOUT (SINGLE PANE COMMAND CENTER)
# ==========================================

# --- 1. BANNER ---
st.markdown(f"""
<div class="status-banner" style="border-left-color: {THEME['ai']};">
    <div>
        <div style="font-size:0.8rem; color:{THEME['ai']}; font-weight:800;">CLINICAL DECISION TREE</div>
        <div style="font-size:1.8rem; font-weight:800; color:{THEME['text_main']}">{dt_class}</div>
        <div style="font-size:0.9rem; color:{THEME['text_muted']};">FiO2: {fio2} | PEEP: {peep} | Age: {age}</div>
    </div>
    <div style="text-align:right">
        <div style="font-size:0.8rem; font-weight:700;">PHYSIOLOGIC CLUSTERS (CENTROIDS)</div>
        <div style="font-size:0.8rem; color:{THEME['text_muted']}">{centroids[0]}</div>
        <div style="font-size:0.8rem; color:{THEME['text_muted']}">{centroids[1]}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 2. REAL-TIME CONTAINER ---
# This container holds the live feed if enabled
live_container = st.empty()

def render_dashboard(data_frame, is_live=False):
    """Renders the dashboard components into the container."""
    c_df = data_frame.iloc[-1]
    p_df = data_frame.iloc[-60]
    
    with live_container.container():
        # ZONE A: PREDICTIONS
        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(plot_monte_carlo(data_frame, 'MAP', p10, p50, p90), use_container_width=True)
        with c2:
            st.markdown("**🫁 V/Q Status**")
            st.plotly_chart(plot_vq_scatter(data_frame), use_container_width=True)

        # ZONE B: METRICS
        st.markdown('<div class="zone-header">ZONE B: HEMODYNAMICS & METABOLICS</div>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        def arrow(val): return "↑" if val > 0 else "↓" if val < 0 else "→"
        def metric_card(col, label, val, unit, delta):
            col.metric(label, f"{val:.1f} {unit}", f"{arrow(delta)} {abs(delta):.1f}", delta_color="inverse")

        metric_card(m1, "MAP", c_df['MAP'], "mmHg", c_df['MAP']-p_df['MAP'])
        metric_card(m2, "Heart Rate", c_df['HR'], "bpm", c_df['HR']-p_df['HR'])
        metric_card(m3, "Cardiac Power", c_df['CPO'], "W", c_df['CPO']-p_df['CPO'])
        metric_card(m4, "Lactate", c_df['Lactate'], "mM", c_df['Lactate']-p_df['Lactate'])
        metric_card(m5, "PaO2", c_df['PaO2'], "mmHg", c_df['PaO2']-p_df['PaO2'])
        metric_card(m6, "SpO2", c_df['SpO2'], "%", c_df['SpO2']-p_df['SpO2'])
        
        # ZONE C: TELEMETRY
        st.markdown('<div class="zone-header">ZONE C: REAL-TIME TELEMETRY</div>', unsafe_allow_html=True)
        fig_tele = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=("Hemodynamics", "Perfusion", "Respiratory"))
        fig_tele.add_trace(go.Scatter(x=data_frame['Time'], y=data_frame['MAP'], name="MAP", line=dict(color=THEME['hemo'])), row=1, col=1)
        fig_tele.add_trace(go.Scatter(x=data_frame['Time'], y=data_frame['CPO'], name="CPO", fill='tozeroy', line=dict(color=THEME['info'])), row=2, col=1)
        fig_tele.add_trace(go.Scatter(x=data_frame['Time'], y=data_frame['SpO2'], name="SpO2", line=dict(color=THEME['resp'])), row=3, col=1)
        
        # Add events
        for e in st.session_state['events']: 
            fig_tele.add_vline(x=e['time'], line_dash="dash", line_color="green")
            
        fig_tele.update_layout(height=400, margin=dict(l=0,r=0,t=20,b=20), template="plotly_white")
        st.plotly_chart(fig_tele, use_container_width=True)

# --- 3. EXECUTION LOGIC ---
if live_mode:
    # Real-Time Simulation Loop
    # We display a window that "scrolls" or updates. 
    # For this demo, we iterate through the last 30 minutes of generated data 
    # to simulate a live feed from the history buffer.
    
    start_idx = 300
    for i in range(start_idx, 360):
        # Create a "growing" dataframe view
        live_slice = df_full.iloc[:i]
        render_dashboard(live_slice, is_live=True)
        time.sleep(0.5) # 1-second real-time simulation tick
else:
    # Static Full View
    render_dashboard(df_full, is_live=False)

# Footer
st.caption("TITAN L5 CDS | Modules: Vectorized Physics, Monte-Carlo Uncertainty, Alveolar Gas Eq, Decision Tree Classifier.")
