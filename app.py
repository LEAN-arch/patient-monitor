import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import welch
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import time

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
st.set_page_config(
    page_title="TITAN | LEVEL 6 CDS",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧬"
)

THEME = {
    "bg": "#f8fafc", "card_bg": "#ffffff", "text_main": "#0f172a", "text_muted": "#64748b",
    "border": "#e2e8f0", "crit": "#dc2626", "warn": "#d97706", "ok": "#059669",
    "info": "#2563eb", "hemo": "#0891b2", "resp": "#7c3aed", "ai": "#be185d", 
    "drug": "#4f46e5", "anomaly": "#db2777"
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
    .rational-box {{ font-size: 0.75rem; color: {THEME['text_muted']}; background: #f1f5f9; padding: 8px; border-radius: 4px; margin-top: 5px; border-left: 3px solid {THEME['info']}; }}
</style>
"""

class CONSTANTS:
    # Drug Parameters (Potency, Wash-in Time Constant (min), Tolerance Half-life (min))
    DRUG_PK = {
        'norepi': {'svr': 900, 'map': 28, 'co': 0.6, 'tau': 2.0, 'tol': 1440},
        'vaso':   {'svr': 700, 'map': 18, 'co': 0.0, 'tau': 5.0, 'tol': 2880},
        'dobu':   {'svr': -350, 'map': 2, 'co': 2.8, 'hr': 18, 'tau': 3.0, 'tol': 720}, # Fast tolerance (downregulation)
        'bb':     {'svr': 50, 'map': -8, 'co': -1.2, 'hr': -22, 'tau': 4.0, 'tol': 5000}
    }
    ATM_PRESSURE = 760
    H2O_PRESSURE = 47
    R_QUOTIENT = 0.8
    MAX_PAO2 = 600
    LAC_PROD_THRESH = 380
    LAC_CLEAR_RATE = 0.04
    VCO2_CONST = 200 # mL/min production reference

# ==========================================
# 2. DYNAMICS MODULES (SPLIT ARCHITECTURE)
# ==========================================

class AutonomicDynamics:
    @staticmethod
    def generate_vitals(mins, p, seed):
        """Generates base autonomic output (HR, MAP, CI, SVRI) using Brownian Bridge."""
        def noise(n, start, end, vol, s):
            np.random.seed(s)
            t = np.linspace(0, 1, n)
            # Stress factor: Volatility increases as physiology decompensates (end < start)
            stress = 1.0 + (max(0, start - end) / 20.0)
            dW = np.random.normal(0, np.sqrt(1/n), n)
            W = np.cumsum(dW)
            B = start + W - t * (W[-1] - (end - start))
            pink = np.convolve(np.random.normal(0, 0.5, n), np.ones(8)/8, mode='same')
            return B + (pink * vol * stress)

        hr = noise(mins, p['hr'][0], p['hr'][1], 1.5, seed)
        map_r = noise(mins, p['map'][0], p['map'][1], 1.2, seed+1)
        ci = noise(mins, p['ci'][0], p['ci'][1], 0.2, seed+2)
        svri = noise(mins, p['svri'][0], p['svri'][1], 100, seed+3)
        rr = noise(mins, 16, 28, 2.0, seed+4)
        return hr, map_r, ci, svri, rr

class PharmacokineticsEngine:
    @staticmethod
    def apply_pk_pd(map_base, ci_base, hr_base, svri_base, drugs, mins):
        """Applies Wash-in (Ramp) and Tolerance (Decay) curves."""
        t_vec = np.arange(mins)
        
        def calculate_effect(drug_name, dose, param):
            if dose <= 0: return np.zeros(mins)
            pk = CONSTANTS.DRUG_PK[drug_name]
            if param not in pk: return np.zeros(mins)
            
            max_effect = dose * pk[param]
            # Wash-in: 1 - exp(-t/tau)
            wash_in = 1 - np.exp(-t_vec / pk['tau'])
            # Tolerance: exp(-t/tol)
            tolerance = np.exp(-t_vec / pk['tol'])
            
            return max_effect * wash_in * tolerance

        # Aggregating effects
        d_svr = calculate_effect('norepi', drugs['norepi'], 'svr') + \
                calculate_effect('vaso', drugs['vaso'], 'svr') + \
                calculate_effect('dobu', drugs['dobu'], 'svr')
                
        d_map = calculate_effect('norepi', drugs['norepi'], 'map') + \
                calculate_effect('vaso', drugs['vaso'], 'map') + \
                calculate_effect('bb', drugs['bb'], 'map')
                
        d_co  = calculate_effect('norepi', drugs['norepi'], 'co') + \
                calculate_effect('dobu', drugs['dobu'], 'co') + \
                calculate_effect('bb', drugs['bb'], 'co')
                
        d_hr  = calculate_effect('dobu', drugs['dobu'], 'hr') + \
                calculate_effect('bb', drugs['bb'], 'hr')

        return (map_base + d_map), (ci_base + d_co), (hr_base + d_hr), (svri_base + d_svr)

class RespiratoryDynamics:
    @staticmethod
    def simulate_gas_exchange(fio2, rr, shunt_base, peep, mins, copd_factor):
        """
        Includes Dead Space, V/Q Mismatch, and Alveolar Gas Equation.
        """
        # Dead Space Fraction (Vd/Vt)
        # Normal 0.3. Increases with COPD, Shock (Low perfusion to lung apices).
        vd_vt = 0.3 * copd_factor
        if copd_factor > 1.0: vd_vt += 0.1 # COPD penalty
        
        # Alveolar Ventilation (VA) approx
        # PaCO2 = (VCO2 * 0.863) / VA
        # VA = VE * (1 - Vd/Vt)
        # VE = RR * Vt (assume Vt = 0.5L)
        vt = 0.5
        ve = rr * vt
        va = ve * (1 - vd_vt)
        # Prevent division by zero
        va = np.maximum(va, 2.0)
        
        paco2 = (CONSTANTS.VCO2_CONST * 0.863) / va
        
        # Alveolar Gas Eq: PAO2 = FiO2(Patm-PH2O) - PaCO2/R
        p_ideal = (fio2 * (CONSTANTS.ATM_PRESSURE - CONSTANTS.H2O_PRESSURE)) - (paco2 / CONSTANTS.R_QUOTIENT)
        p_ideal = np.clip(p_ideal, 0, CONSTANTS.MAX_PAO2)
        
        # Shunt & PEEP
        # PEEP Benefit: Decreases Shunt
        shunt_eff = shunt_base * np.exp(-0.06 * peep)
        pao2 = p_ideal * (1 - shunt_eff)
        
        # SpO2 Sigmoid
        pao2_safe = np.maximum(pao2, 0.1)
        spo2 = 100 / (1 + (23400 / (pao2_safe**3 + 150*pao2_safe)))
        
        return pao2, paco2, spo2, vd_vt

class MetabolicDynamics:
    @staticmethod
    def simulate_metabolism(ci, hb, spo2, mins):
        """Calculates Oxygen Delivery, Consumption, Extraction, and Lactate."""
        # DO2I = CI * Hb * 1.34 * SpO2
        do2i = ci * hb * 1.34 * (spo2/100) * 10
        
        # VO2I (Consumption) - Assume ~125 mL/min/m2 resting, changes with stress?
        # Fixed for now, or proportional to CI in supply-dependency
        vo2i = np.full(mins, 125.0) 
        
        # O2ER (Extraction Ratio)
        o2er = vo2i / np.maximum(do2i, 1.0)
        
        # Lactate Kinetics
        lactate = np.zeros(mins)
        lac_curr = 1.0
        
        # Vectorized lookahead not possible for autoregressive, using loop for logic
        # Production increases if DO2I < Threshold OR O2ER > 50%
        prod_stress = np.where((do2i < CONSTANTS.LAC_PROD_THRESH) | (o2er > 0.45), 0.15, 0.0)
        
        for i in range(mins):
            # Clearance depends on liver flow (CI)
            clear = CONSTANTS.LAC_CLEAR_RATE * (ci[i] / 3.0) # Normalized to CI=3
            lac_curr = max(0.5, lac_curr + prod_stress[i] - clear)
            lactate[i] = lac_curr
            
        return do2i, vo2i, o2er, lactate

# ==========================================
# 3. ADVANCED ANALYTICS (AI LAYERS)
# ==========================================

class BayesianClassifier:
    @staticmethod
    def predict_shock_probs(row):
        """
        Calculates posterior probability of shock states using Gaussian Naive Bayes logic.
        P(Shock|Data) ~ Likelihood * Prior
        """
        # Shock Profiles: {Name: {Mean_CI, Mean_SVR, Prior}}
        profiles = {
            "Cardiogenic": {"CI": 1.8, "SVRI": 2400, "Prior": 0.2},
            "Distributive": {"CI": 4.5, "SVRI": 800,  "Prior": 0.3},
            "Hypovolemic": {"CI": 2.0, "SVRI": 2800, "Prior": 0.2},
            "Stable":      {"CI": 3.0, "SVRI": 1800, "Prior": 0.3}
        }
        
        scores = {}
        total_likelihood = 0
        
        for name, p in profiles.items():
            # Likelihood p(Data|State) assuming normal dist
            # Simplified PDF calculation
            p_ci = norm.pdf(row['CI'], loc=p['CI'], scale=0.8)
            p_svr = norm.pdf(row['SVRI'], loc=p['SVRI'], scale=400)
            
            likelihood = p_ci * p_svr * p['Prior']
            scores[name] = likelihood
            total_likelihood += likelihood
            
        # Normalize
        if total_likelihood == 0: return {k:0.25 for k in profiles}
        return {k: (v/total_likelihood)*100 for k, v in scores.items()}

class ReinforcementAgent:
    @staticmethod
    def suggest_dose(row, drug_state):
        """
        Rule-based policy acting as an RL Agent (Actor-Critic proxy).
        Target: MAP > 65, CI > 2.2.
        """
        map_val = row['MAP']
        ci_val = row['CI']
        svr_val = row['SVRI']
        
        suggestion = []
        confidence = 0
        
        if map_val < 65:
            if ci_val < 2.2:
                # Cold/Wet or Cold/Dry -> Needs Flow + Pressure
                suggestion.append("Titrate Dobutamine +2.5 mcg")
                confidence = 85
            else:
                # Warm/Vasoplegic -> Needs Vasopressor
                if svr_val < 1000:
                    suggestion.append("Increase Norepinephrine +0.05")
                    confidence = 92
                else:
                    suggestion.append("Consider Volume / Blood")
                    confidence = 60
        else:
            if drug_state['norepi'] > 0 and map_val > 75:
                 suggestion.append("Wean Norepinephrine -0.05")
                 confidence = 80
            else:
                suggestion.append("Maintain Current Therapy")
                confidence = 99
                
        return suggestion[0], confidence

class AnomalyDetector:
    @staticmethod
    def detect_anomalies(df):
        """
        Isolation Forest to detect physiologic instability/outliers.
        """
        features = ['MAP', 'CI', 'SVRI', 'HR']
        X = df[features].fillna(0)
        
        # Fit model on history
        iso = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly_score'] = iso.fit_predict(X)
        # -1 is anomaly, 1 is normal
        return df

# ==========================================
# 4. PATIENT SIMULATOR
# ==========================================
class PatientSimulator:
    def __init__(self, mins=360):
        self.mins = mins
        self.t = np.arange(mins)

    def get_data(self, case_id, drugs, fluids, bsa, peep):
        # Case Config
        cases = {
            "65M Post-CABG": {'ci':(2.2,1.8), 'map':(75,55), 'svri':(1800,1400), 'hr':(85,95), 'shunt':0.10, 'liver':0.9, 'copd':1.0},
            "24F Septic Shock": {'ci':(4.5,5.5), 'map':(65,45), 'svri':(1000,600), 'hr':(110,140), 'shunt':0.15, 'liver':0.8, 'copd':1.0},
            "82M HFpEF Sepsis": {'ci':(2.0,1.9), 'map':(85,60), 'svri':(2200,1600), 'hr':(90,110), 'shunt':0.20, 'liver':0.6, 'copd':1.5},
            "50M Trauma": {'ci':(3.0,1.5), 'map':(70,40), 'svri':(2500,3000), 'hr':(100,150), 'shunt':0.05, 'liver':1.0, 'copd':1.0}
        }
        p = cases[case_id]
        seed = len(case_id) + 42
        
        # 1. Autonomic
        hr, map_r, ci_r, svri_r, rr = AutonomicDynamics.generate_vitals(self.mins, p, seed)
        
        # 2. Fluid Physics
        ppv_base = 20 if "Trauma" in case_id else 12
        ppv = ppv_base + (np.sin(self.t/8)*4)
        ci_fluid = (fluids/500) * (0.4 if np.mean(ppv)>13 else 0.05)
        
        # 3. Pharmacokinetics
        map_f, ci_f, hr_f, svri_f = PharmacokineticsEngine.apply_pk_pd(map_r, ci_r + ci_fluid, hr, svri_r, drugs, self.mins)
        
        # 4. Respiratory
        pao2, paco2, spo2, vd_vt = RespiratoryDynamics.simulate_gas_exchange(drugs['fio2'], rr, p['shunt'], peep, self.mins, p['copd'])
        
        # 5. Metabolic
        hb = 8.0 if "Trauma" in case_id else 12.0
        do2i, vo2i, o2er, lactate = MetabolicDynamics.simulate_metabolism(ci_f, hb, spo2, self.mins)
        
        # 6. Assembly
        df = pd.DataFrame({
            "Time": self.t, "HR": hr_f, "MAP": map_f, "CI": ci_f, "SVRI": svri_f,
            "CO": ci_f * bsa, "SVR": svri_f / bsa,
            "Lactate": lactate, "SpO2": spo2, "PaO2": pao2, "PaCO2": paco2, "RR": rr,
            "DO2I": do2i, "VO2I": vo2i, "O2ER": o2er, "Vd/Vt": np.full(self.mins, vd_vt),
            "CPO": (map_f * (ci_f * bsa)) / 451
        })
        
        return df

# ==========================================
# 5. VISUALIZATION & UI COMPONENTS
# ==========================================
def plot_sparkline(data, color):
    # Minimalist sparkline for metric cards
    fig = px.line(x=np.arange(len(data)), y=data)
    fig.update_traces(line_color=color, line_width=2)
    fig.update_layout(showlegend=False, xaxis_visible=False, yaxis_visible=False, margin=dict(l=0,r=0,t=0,b=0), height=35)
    return fig

def plot_bayes_bars(probs):
    fig = go.Figure(go.Bar(
        x=list(probs.values()), y=list(probs.keys()), orientation='h',
        marker=dict(color=[THEME['hemo'], THEME['crit'], THEME['warn'], THEME['ok']], opacity=0.8)
    ))
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=120, xaxis=dict(range=[0,100], showticklabels=False), yaxis=dict(tickfont=dict(size=10)))
    return fig

def render_dashboard(df, probs, suggestion, confidence, anomalies):
    c_df = df.iloc[-1]
    
    # --- ZONE A: AI & DIAGNOSTICS ---
    st.markdown('<div class="zone-header">ZONE A: AI DIAGNOSTICS & PHARMACOLOGY</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.5, 1.5, 1])
    
    with c1:
        st.markdown("**Bayesian Shock Classification**")
        st.plotly_chart(plot_bayes_bars(probs), use_container_width=True)
        st.markdown(f"<div class='rational-box'>Confidence: {max(probs.values()):.1f}% in top class. Based on CI/SVRI Gaussian likelihood.</div>", unsafe_allow_html=True)
        
    with c2:
        st.markdown("**Reinforcement Learning Advisor**")
        st.info(f"🤖 **Suggestion:** {suggestion}")
        st.progress(confidence/100, f"Policy Confidence: {confidence}%")
        st.markdown(f"<div class='rational-box'>Target: MAP>65, CI>2.2. Reward function penalties applied for vasoactive load.</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("**Risk / Anomaly**")
        is_anomaly = anomalies.iloc[-1]['anomaly_score'] == -1
        status = "DETECTED" if is_anomaly else "STABLE"
        color = THEME['anomaly'] if is_anomaly else THEME['ok']
        st.markdown(f"<h2 style='color:{color}; margin:0;'>{status}</h2>", unsafe_allow_html=True)
        st.caption("IsolationForest Algorithm")

    # --- ZONE B: METRICS GRID WITH SPARKLINES ---
    st.markdown('<div class="zone-header">ZONE B: PHYSIOLOGIC METRICS (60m TREND)</div>', unsafe_allow_html=True)
    
    metrics = [
        ("MAP", c_df['MAP'], "mmHg", df['MAP'].iloc[-60:], THEME['hemo']),
        ("CI", c_df['CI'], "L/min/m²", df['CI'].iloc[-60:], THEME['info']),
        ("SVRI", c_df['SVRI'], "dyn·s", df['SVRI'].iloc[-60:], THEME['warn']),
        ("Lactate", c_df['Lactate'], "mM", df['Lactate'].iloc[-60:], THEME['crit']),
        ("O2ER", c_df['O2ER']*100, "%", df['O2ER'].iloc[-60:], THEME['resp']),
        ("PaCO2", c_df['PaCO2'], "mmHg", df['PaCO2'].iloc[-60:], THEME['text_muted'])
    ]
    
    cols = st.columns(6)
    for i, (label, val, unit, trend, color) in enumerate(metrics):
        with cols[i]:
            st.metric(label, f"{val:.1f} {unit}")
            st.plotly_chart(plot_sparkline(trend, color), use_container_width=True)

    # --- ZONE C: ADVANCED RESPIRATORY & METABOLICS ---
    st.markdown('<div class="zone-header">ZONE C: RESPIRATORY & METABOLIC DYNAMICS</div>', unsafe_allow_html=True)
    rc1, rc2 = st.columns(2)
    with rc1:
        # V/Q Scatter
        fig = px.scatter(df.iloc[-120:], x="PaO2", y="PaCO2", color="Vd/Vt", 
                         title="Gas Exchange: V/Q & Dead Space", color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)
    with rc2:
        # DO2 vs VO2
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Time'], y=df['DO2I'], name="DO2I", line=dict(color=THEME['info'])))
        fig.add_trace(go.Scatter(x=df['Time'], y=df['VO2I'], name="VO2I", line=dict(color=THEME['resp'], dash='dot')))
        fig.update_layout(title="Supply (DO2) / Demand (VO2) Match", height=300, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
st.markdown(STYLING, unsafe_allow_html=True)
if 'events' not in st.session_state: st.session_state['events'] = []
if 'fluids' not in st.session_state: st.session_state['fluids'] = 0

with st.sidebar:
    st.title("TITAN | L6 CDS")
    res_mins = st.select_slider("Physics Resolution (mins)", options=[60, 180, 360, 720], value=360)
    
    st.markdown("### 🏥 Case")
    case_id = st.selectbox("Profile", ["65M Post-CABG", "24F Septic Shock", "82M HFpEF Sepsis", "50M Trauma"])
    bsa = 1.9 # Simplified default
    
    st.markdown("### 💉 Infusions")
    with st.form("drugs_form"):
        c1, c2 = st.columns(2)
        norepi = c1.number_input("Norepi", 0.0, 2.0, 0.0, 0.05)
        vaso = c1.number_input("Vaso", 0.0, 0.1, 0.0, 0.01)
        dobu = c2.number_input("Dobutamine", 0.0, 10.0, 0.0, 0.5)
        bb = c2.number_input("Esmolol", 0.0, 1.0, 0.0, 0.1)
        fio2 = st.slider("FiO2", 0.21, 1.0, 0.4)
        peep = st.slider("PEEP", 0, 20, 5)
        submit = st.form_submit_button("Update Model")
    
    if st.button("💧 Bolus 500mL"): st.session_state['fluids'] += 500

# Run Simulation
sim = PatientSimulator(mins=res_mins)
drugs = {'norepi':norepi, 'vaso':vaso, 'dobu':dobu, 'bb':bb, 'fio2':fio2}
df = sim.get_data(case_id, drugs, st.session_state['fluids'], bsa, peep)

# Analytics
probs = BayesianClassifier.predict_shock_probs(df.iloc[-1])
sugg, conf = ReinforcementAgent.suggest_dose(df.iloc[-1], drugs)
df_anom = AnomalyDetector.detect_anomalies(df)

render_dashboard(df_anom, probs, sugg, conf, df_anom)

st.caption("TITAN L6 | Bayesian Inference, RL Policy, IsolationForest Anomaly Detection, Pharmacokinetic Time-Constants.")
