import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import welch
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import time

# ==========================================
# 1. CONFIGURATION & MEDICAL THEME
# ==========================================
st.set_page_config(
    page_title="TITAN | ULTIMATE COMMAND CENTER",
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
    # High-Fidelity Pharmacokinetics (Max Effect at typical high dose)
    DRUG_PK = {
        'norepi': {'svr': 2500, 'map': 120, 'co': 0.8, 'tau': 2.0, 'tol': 2880}, 
        'vaso':   {'svr': 4000, 'map': 150, 'co': -0.2, 'tau': 5.0, 'tol': 2880}, 
        'dobu':   {'svr': -600, 'map': 5, 'co': 4.5, 'hr': 25, 'tau': 3.0, 'tol': 720}, 
        'bb':     {'svr': 50, 'map': -15, 'co': -2.0, 'hr': -35, 'tau': 4.0, 'tol': 5000}
    }
    ATM_PRESSURE = 760; H2O_PRESSURE = 47; R_QUOTIENT = 0.8; MAX_PAO2 = 600
    LAC_PROD_THRESH = 330 # Critical DO2I (mL/min/m2)
    LAC_CLEAR_RATE = 0.05; VCO2_CONST = 130 # mL/min/m2

# ==========================================
# 2. DYNAMICS MODULES (PHYSICS ENGINE)
# ==========================================

class AutonomicDynamics:
    @staticmethod
    def generate_vitals(mins, p, seed):
        def noise(n, start, end, vol, s):
            np.random.seed(s)
            t = np.linspace(0, 1, n)
            # Stress factor: Volatility increases as physiology decompensates
            stress = 1.0 + (max(0, start - end) / 20.0)
            dW = np.random.normal(0, np.sqrt(1/n), n)
            pink = np.convolve(np.random.normal(0, 0.5, n), np.ones(8)/8, mode='same')
            B = start + np.cumsum(dW) - t * (np.cumsum(dW)[-1] - (end - start))
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
        t_vec = np.arange(mins)
        def calculate_effect(drug_name, dose, param):
            if dose <= 0: return np.zeros(mins)
            pk = CONSTANTS.DRUG_PK[drug_name]
            if param not in pk: return np.zeros(mins)
            max_eff = dose * pk[param]
            # Wash-in (1-exp) and Tolerance (exp decay)
            return max_eff * (1 - np.exp(-t_vec / pk['tau'])) * np.exp(-t_vec / pk['tol'])

        d_svr = calculate_effect('norepi', drugs['norepi'], 'svr') + calculate_effect('vaso', drugs['vaso'], 'svr') + calculate_effect('dobu', drugs['dobu'], 'svr')
        d_map = calculate_effect('norepi', drugs['norepi'], 'map') + calculate_effect('vaso', drugs['vaso'], 'map') + calculate_effect('bb', drugs['bb'], 'map')
        d_co  = calculate_effect('norepi', drugs['norepi'], 'co') + calculate_effect('dobu', drugs['dobu'], 'co') + calculate_effect('bb', drugs['bb'], 'co')
        d_hr  = calculate_effect('dobu', drugs['dobu'], 'hr') + calculate_effect('bb', drugs['bb'], 'hr')

        return (map_base + d_map), (ci_base + d_co), (hr_base + d_hr), (svri_base + d_svr)

class RespiratoryDynamics:
    @staticmethod
    def simulate_gas_exchange(fio2, rr, shunt_base, peep, mins, copd_factor):
        # Dead Space and Alveolar Ventilation
        vd_vt = 0.3 * copd_factor + (0.1 if copd_factor > 1.0 else 0)
        va = np.maximum((rr * 0.5) * (1 - vd_vt), 2.0)
        
        # PaCO2
        paco2 = (CONSTANTS.VCO2_CONST * 0.863) / va
        
        # Alveolar Gas Equation
        p_ideal = np.clip((fio2 * (CONSTANTS.ATM_PRESSURE - CONSTANTS.H2O_PRESSURE)) - (paco2 / CONSTANTS.R_QUOTIENT), 0, CONSTANTS.MAX_PAO2)
        
        # Shunt with PEEP Recruitment
        shunt_eff = shunt_base * np.exp(-0.06 * peep)
        pao2 = p_ideal * (1 - shunt_eff)
        
        # SpO2 Sigmoid (Severinghaus)
        pao2_safe = np.maximum(pao2, 0.1)
        spo2 = 100 / (1 + (23400 / (pao2_safe**3 + 150*pao2_safe)))
        
        return pao2, paco2, spo2, vd_vt

class MetabolicDynamics:
    @staticmethod
    def simulate_metabolism(ci, hb, spo2, mins, vo2_stress_factor):
        # Oxygen Delivery Index
        do2i = ci * hb * 1.34 * (spo2/100) * 10
        # Oxygen Consumption Index
        vo2i = np.full(mins, 125.0 * vo2_stress_factor)
        # Extraction Ratio
        o2er = vo2i / np.maximum(do2i, 10.0)
        
        lactate = np.zeros(mins)
        lac_curr = 1.0
        
        # Production: Increases if critical DO2I breach or high extraction
        prod = np.where((do2i < CONSTANTS.LAC_PROD_THRESH) | (o2er > 0.50), 0.2, 0.0)
        
        for i in range(mins):
            # Clearance: Proportional to liver perfusion (CI)
            clear = CONSTANTS.LAC_CLEAR_RATE * (ci[i] / 2.5)
            lac_curr = max(0.6, lac_curr + prod[i] - clear)
            lactate[i] = lac_curr
            
        return do2i, vo2i, o2er, lactate

# ==========================================
# 3. AI & ANALYTICS LAYERS
# ==========================================

class AI_Services:
    @staticmethod
    def predict_shock_bayes(row):
        profiles = {
            "Cardiogenic": {"CI": 1.8, "SVRI": 2800, "Prior": 0.2},
            "Distributive": {"CI": 5.0, "SVRI": 800,  "Prior": 0.3},
            "Hypovolemic": {"CI": 2.0, "SVRI": 3000, "Prior": 0.2},
            "Stable":      {"CI": 3.2, "SVRI": 1900, "Prior": 0.3}
        }
        scores = {}
        total = 0
        for name, p in profiles.items():
            like = norm.pdf(row['CI'], p['CI'], 1.0) * norm.pdf(row['SVRI'], p['SVRI'], 500)
            scores[name] = like * p['Prior']
            total += scores[name]
        return {k: (v/total)*100 for k, v in scores.items()} if total > 0 else {k:25 for k in profiles}

    @staticmethod
    def rl_advisor(row, drugs):
        # Actor-Critic Proxy Logic
        if row['MAP'] < 65:
            if row['CI'] < 2.2: 
                if drugs['dobu'] < 5: return "Titrate Dobutamine +2.5", 88
                else: return "Consider Mechanical Support", 75
            else: 
                if row['SVRI'] < 1200:
                    if drugs['norepi'] < 0.5: return "Increase Norepi +0.05", 92
                    else: return "Add Vasopressin 0.03", 85
                else:
                    return "Fluid Bolus 500mL", 80
        else:
            if drugs['norepi'] > 0 and row['MAP'] > 75: return "Wean Norepi -0.05", 85
        return "Maintain Current Therapy", 98

    @staticmethod
    def detect_anomalies(df):
        iso = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly'] = iso.fit_predict(df[['MAP','CI','SVRI']].fillna(0))
        return df

    @staticmethod
    def monte_carlo_forecast(df, target='MAP', n_sims=50):
        curr = df[target].iloc[-1]
        vol = np.std(df[target].iloc[-30:])
        paths = np.array([curr + np.cumsum(np.random.normal(0, vol, 30)) for _ in range(n_sims)])
        return np.percentile(paths, 10, axis=0), np.percentile(paths, 50, axis=0), np.percentile(paths, 90, axis=0)
    
    @staticmethod
    def inverse_centroids(df):
        try:
            scaler = StandardScaler()
            X = scaler.fit_transform(df[['CI','SVRI','Lactate']].iloc[-60:])
            km = KMeans(3, random_state=42).fit(X)
            ctrs = scaler.inverse_transform(km.cluster_centers_)
            return [f"C{i+1}: CI={c[0]:.1f}, SVR={c[1]:.0f}, Lac={c[2]:.1f}" for i,c in enumerate(ctrs)]
        except: return ["Insufficient Data"]

# ==========================================
# 4. SIMULATOR ORCHESTRATOR
# ==========================================
class PatientSimulator:
    def __init__(self, mins=360):
        self.mins = mins
        self.t = np.arange(mins)
        
    def run(self, case_id, drugs, fluids, bsa, peep):
        cases = {
            "65M Post-CABG": {
                'ci':(2.4, 1.8), 'map':(75, 55), 'svri':(2000, 1600), 
                'hr':(85, 95), 'shunt':0.10, 'copd':1.0, 'vo2_stress': 1.0
            },
            "24F Septic Shock": {
                'ci':(3.5, 5.5), 'map':(65, 45), 'svri':(1200, 500), 
                'hr':(110, 140), 'shunt':0.15, 'copd':1.0, 'vo2_stress': 1.4
            },
            "82M HFpEF Sepsis": {
                'ci':(2.2, 1.8), 'map':(85, 55), 'svri':(2400, 1800), 
                'hr':(80, 110), 'shunt':0.20, 'copd':1.5, 'vo2_stress': 1.1
            },
            "50M Trauma": {
                'ci':(3.0, 1.5), 'map':(70, 40), 'svri':(2500, 3200), 
                'hr':(100, 150), 'shunt':0.05, 'copd':1.0, 'vo2_stress': 0.9
            }
        }
        p = cases[case_id]
        seed = len(case_id)+42
        
        hr, map_r, ci_r, svri_r, rr = AutonomicDynamics.generate_vitals(self.mins, p, seed)
        ppv = (20 if "Trauma" in case_id else 12) + (np.sin(self.t/8)*4)
        
        # Starling Curve (Fluid Responsiveness)
        ci_fluid = (fluids/500) * (0.5 if np.mean(ppv)>13 else 0.05)
        
        map_f, ci_f, hr_f, svri_f = PharmacokineticsEngine.apply_pk_pd(map_r, ci_r+ci_fluid, hr, svri_r, drugs, self.mins)
        pao2, paco2, spo2, vd_vt = RespiratoryDynamics.simulate_gas_exchange(drugs['fio2'], rr, p['shunt'], peep, self.mins, p['copd'])
        
        hb = 8.0 if "Trauma" in case_id else 12.0
        do2i, vo2i, o2er, lactate = MetabolicDynamics.simulate_metabolism(ci_f, hb, spo2, self.mins, p['vo2_stress'])
        
        df = pd.DataFrame({
            "Time": self.t, "HR": hr_f, "MAP": map_f, "CI": ci_f, "SVRI": svri_f,
            "CO": ci_f * bsa, "SVR": svri_f / bsa,
            "Lactate": lactate, "SpO2": spo2, "PaO2": pao2, "PaCO2": paco2, "RR": rr,
            "DO2I": do2i, "VO2I": vo2i, "O2ER": o2er, "Vd/Vt": np.full(self.mins, vd_vt),
            "CPO": (map_f * (ci_f * bsa)) / 451
        })
        return df

# ==========================================
# 5. VISUALIZATION FUNCTIONS
# ==========================================
def plot_sparkline(data, color, key):
    fig = px.line(x=np.arange(len(data)), y=data)
    fig.update_traces(line_color=color, line_width=2)
    fig.update_layout(showlegend=False, xaxis_visible=False, yaxis_visible=False, margin=dict(l=0,r=0,t=0,b=0), height=35)
    return fig

def plot_bayes_bars(probs):
    fig = go.Figure(go.Bar(x=list(probs.values()), y=list(probs.keys()), orientation='h', marker=dict(color=[THEME['hemo'], THEME['crit'], THEME['warn'], THEME['ok']])))
    fig.update_layout(
        margin=dict(l=0,r=0,t=0,b=0), height=100, 
        xaxis=dict(showticklabels=True, title="Probability [%]", range=[0, 100]),
        yaxis=dict(title=None)
    )
    return fig

def plot_3d_attractor(df):
    recent = df.iloc[-60:]
    fig = go.Figure(data=[go.Scatter3d(x=recent['CPO'], y=recent['SVRI'], z=recent['Lactate'], mode='lines+markers', marker=dict(size=3, color=recent.index, colorscale='Viridis'), line=dict(width=2))])
    fig.update_layout(
        scene=dict(
            xaxis_title='Power [W]', 
            yaxis_title='SVRI [dyn·s·cm⁻⁵·m²]', 
            zaxis_title='Lactate [mmol/L]'
        ), 
        margin=dict(l=0, r=0, b=0, t=0), height=250, title="3D Attractor Phase Space"
    )
    return fig

def plot_chaos_attractor(df):
    rr = 60000 / df['HR'].iloc[-120:]
    fig = go.Figure(go.Scatter(x=rr.iloc[:-1], y=rr.iloc[1:], mode='markers', marker=dict(color='teal', size=4, opacity=0.6), name="R-R Interval"))
    fig.update_layout(
        title="Chaos (Poincaré Plot)", height=200, margin=dict(l=20,r=20,t=30,b=20), 
        xaxis_title="RR(n) [ms]", yaxis_title="RR(n+1) [ms]", showlegend=False
    )
    return fig

def plot_spectral_analysis(df):
    # Fix: convert to numpy to avoid scipy KeyError
    data = df['HR'].iloc[-120:].to_numpy()
    f, Pxx = welch(data, fs=1/60)
    fig = px.line(x=f, y=Pxx, title="Spectral HRV Analysis")
    fig.add_vline(x=0.04, line_dash="dot", annotation_text="LF"); fig.add_vline(x=0.15, line_dash="dot", annotation_text="HF")
    fig.update_layout(
        height=200, margin=dict(l=20,r=20,t=30,b=20),
        xaxis_title="Frequency [Hz]", yaxis_title="Power [ms²/Hz]"
    )
    return fig

def plot_monte_carlo_fan(df, p10, p50, p90):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(30), y=df['MAP'].iloc[-30:], name="History", line=dict(color=THEME['text_main'])))
    fx = np.arange(30, 60)
    fig.add_trace(go.Scatter(x=np.concatenate([fx, fx[::-1]]), y=np.concatenate([p90, p10[::-1]]), fill='toself', fillcolor='rgba(0,0,255,0.1)', line=dict(width=0), name="80% Conf. Interval"))
    fig.add_trace(go.Scatter(x=fx, y=p50, line=dict(dash='dot', color='blue'), name="Median Forecast"))
    fig.update_layout(
        height=200, title="Monte Carlo Forecast (MAP)", margin=dict(l=20,r=20,t=30,b=20),
        xaxis_title="Time [min]", yaxis_title="MAP [mmHg]", legend=dict(orientation="h", y=1.1)
    )
    return fig

def plot_hemodynamic_profile(df):
    recent = df.iloc[-60:]
    fig = go.Figure()
    fig.add_hline(y=2000, line_dash="dot", annotation_text="Vasoconstriction"); fig.add_vline(x=2.2, line_dash="dot", annotation_text="Low Flow")
    fig.add_trace(go.Scatter(x=recent['CI'], y=recent['SVRI'], mode='markers', marker=dict(color=recent.index, colorscale='Viridis'), name="State"))
    fig.update_layout(
        title="Pump vs Pipes (Forrester Proxy)", height=200, margin=dict(l=20,r=20,t=30,b=20),
        xaxis_title="Cardiac Index [L/min/m²]", yaxis_title="SVRI [dyn·s·cm⁻⁵·m²]"
    )
    return fig

def plot_phase_space(df):
    recent = df.iloc[-60:]
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=0.6, y0=2, y1=15, fillcolor="rgba(255,0,0,0.1)", line_width=0)
    fig.add_trace(go.Scatter(x=recent['CPO'], y=recent['Lactate'], mode='lines+markers', marker=dict(color=recent.index, colorscale='Bluered'), name="Trajectory"))
    fig.update_layout(
        title="Hemo-Metabolic Coupling", height=200, margin=dict(l=20,r=20,t=30,b=20),
        xaxis_title="Cardiac Power [W]", yaxis_title="Lactate [mmol/L]"
    )
    return fig

def plot_counterfactual(df, drugs, case, bsa, peep):
    sim = PatientSimulator(60)
    base = {'norepi':0, 'vaso':0, 'dobu':0, 'bb':0, 'fio2':0.21}
    df_b = sim.run(case, base, 0, bsa, peep)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df['MAP'].iloc[-60:], name="With Intervention", line=dict(color=THEME['ok'])))
    fig.add_trace(go.Scatter(y=df_b['MAP'], name="Natural History", line=dict(dash='dot', color=THEME['crit'])))
    fig.update_layout(
        title="Counterfactual Projection", height=200, margin=dict(l=20,r=20,t=30,b=20),
        xaxis_title="Time [min]", yaxis_title="MAP [mmHg]", legend=dict(orientation="h", y=1.1)
    )
    return fig

# ==========================================
# 6. MAIN APP LOGIC
# ==========================================
st.markdown(STYLING, unsafe_allow_html=True)
if 'events' not in st.session_state: st.session_state['events'] = []
if 'fluids' not in st.session_state: st.session_state['fluids'] = 0

with st.sidebar:
    st.title("TITAN | L6 CDS")
    res_mins = st.select_slider("Resolution", [60, 180, 360, 720], value=360)
    case_id = st.selectbox("Profile", ["65M Post-CABG", "24F Septic Shock", "82M HFpEF Sepsis", "50M Trauma"])
    
    # Calculate BSA dynamically
    col_bio1, col_bio2 = st.columns(2)
    with col_bio1: height = st.number_input("Height (cm)", 150, 200, 175)
    with col_bio2: weight = st.number_input("Weight (kg)", 50, 150, 80)
    bsa = np.sqrt((height * weight) / 3600)
    st.caption(f"Calculated BSA: {bsa:.2f} m²")
    
    with st.form("drugs"):
        st.markdown("**Infusion Pumps**")
        c1, c2 = st.columns(2)
        norepi = c1.number_input("Norepi (mcg/kg/min)", 0.0, 2.0, 0.0, 0.05, format="%.2f")
        vaso = c1.number_input("Vaso (U/min)", 0.0, 0.1, 0.0, 0.01, format="%.2f")
        dobu = c2.number_input("Dobutamine (mcg)", 0.0, 10.0, 0.0, 0.5, format="%.1f")
        bb = c2.number_input("Esmolol (mg/kg)", 0.0, 1.0, 0.0, 0.1, format="%.1f")
        fio2 = st.slider("FiO2", 0.21, 1.0, 0.4); peep = st.slider("PEEP", 0, 20, 5)
        st.form_submit_button("Update Model")
    
    if st.button("💧 Bolus 500mL"): st.session_state['fluids'] += 500
    live_mode = st.checkbox("🔴 LIVE MODE")

sim = PatientSimulator(mins=res_mins)
drugs = {'norepi':norepi, 'vaso':vaso, 'dobu':dobu, 'bb':bb, 'fio2':fio2}
df = sim.run(case_id, drugs, st.session_state['fluids'], bsa, peep)
df = AI_Services.detect_anomalies(df)
probs = AI_Services.predict_shock_bayes(df.iloc[-1])
sugg, conf = AI_Services.rl_advisor(df.iloc[-1], drugs)
p10, p50, p90 = AI_Services.monte_carlo_forecast(df)
centroids = AI_Services.inverse_centroids(df)

# --- DASHBOARD RENDERER ---
container = st.empty()

def render(d_slice):
    ix = len(d_slice) # Unique key for Streamlit loop
    curr = d_slice.iloc[-1]
    anom = "DETECTED" if curr['anomaly'] == -1 else "STABLE"
    
    with container.container():
        # ZONE A: HEADER & AI
        st.markdown(f"""
        <div class="status-banner" style="border-left-color: {THEME['ai']};">
            <div><div style="font-size:0.8rem; font-weight:800; color:{THEME['ai']}">BAYESIAN SHOCK PROBABILITY</div>
            <div style="font-size:1.5rem; font-weight:800;">{max(probs, key=probs.get).upper()} ({max(probs.values()):.0f}%)</div>
            <div style="font-size:0.8rem;">{centroids[0]}</div></div>
            <div style="text-align:right">
                <div style="font-size:0.8rem; font-weight:700;">ANOMALY STATUS</div>
                <div class="{'crit-pulse' if anom=='DETECTED' else ''}" style="font-size:2rem; font-weight:800; color:{THEME['crit'] if anom=='DETECTED' else THEME['ok']}">{anom}</div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1: st.plotly_chart(plot_bayes_bars(probs), use_container_width=True, key=f"bayes_{ix}")
        with c2: 
            st.info(f"🤖 **RL Agent:** {sugg} ({conf}%)")
            st.plotly_chart(plot_monte_carlo_fan(d_slice, p10, p50, p90), use_container_width=True, key=f"fan_{ix}")
        with c3: st.plotly_chart(plot_counterfactual(d_slice, drugs, case_id, bsa, peep), use_container_width=True, key=f"count_{ix}")

        # ZONE B: METRICS
        st.markdown('<div class="zone-header">ZONE B: PHYSIOLOGIC METRICS (60m)</div>', unsafe_allow_html=True)
        cols = st.columns(6)
        metrics = [("MAP", curr['MAP'], THEME['hemo']), ("CI", curr['CI'], THEME['info']), 
                   ("SVRI", curr['SVRI'], THEME['warn']), ("Lactate", curr['Lactate'], THEME['crit']),
                   ("O2ER", curr['O2ER']*100, THEME['resp']), ("CPO", curr['CPO'], THEME['drug'])]
        for i, (l, v, c) in enumerate(metrics):
            with cols[i]:
                st.metric(l, f"{v:.1f}")
                st.plotly_chart(plot_sparkline(d_slice[l].iloc[-60:], c, key=f"spark_{i}_{ix}"), use_container_width=True, key=f"spark_chart_{i}_{ix}")

        # ZONE C: ADVANCED PHYSICS (Restored)
        st.markdown('<div class="zone-header">ZONE C: PHASE SPACE & COMPLEXITY</div>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        with b1: st.plotly_chart(plot_3d_attractor(d_slice), use_container_width=True, key=f"3d_{ix}")
        with b2: st.plotly_chart(plot_hemodynamic_profile(d_slice), use_container_width=True, key=f"hemo_{ix}")
        with b3: st.plotly_chart(plot_phase_space(d_slice), use_container_width=True, key=f"phase_{ix}")
        
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_chaos_attractor(d_slice), use_container_width=True, key=f"chaos_{ix}")
        with c2: st.plotly_chart(plot_spectral_analysis(d_slice), use_container_width=True, key=f"spec_{ix}")

        # ZONE E: TELEMETRY
        st.markdown('<div class="zone-header">ZONE E: TELEMETRY</div>', unsafe_allow_html=True)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03)
        fig.add_trace(go.Scatter(x=d_slice['Time'], y=d_slice['MAP'], name="MAP", line=dict(color=THEME['hemo'])), row=1, col=1)
        fig.add_trace(go.Scatter(x=d_slice['Time'], y=d_slice['CPO'], name="CPO", fill='tozeroy', line=dict(color=THEME['info'])), row=2, col=1)
        fig.add_trace(go.Scatter(x=d_slice['Time'], y=d_slice['SpO2'], name="SpO2", line=dict(color=THEME['resp'])), row=3, col=1)
        fig.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0), template="plotly_white")
        
        # Telemetry Labels
        fig.update_yaxes(title_text="MAP [mmHg]", row=1, col=1)
        fig.update_yaxes(title_text="CPO [W]", row=2, col=1)
        fig.update_yaxes(title_text="SpO2 [%]", row=3, col=1)
        fig.update_xaxes(title_text="Time [min]", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True, key=f"tele_{ix}")

if live_mode:
    for i in range(max(10, res_mins-60), res_mins):
        render(df.iloc[:i])
        time.sleep(0.1)
else:
    render(df)
st.caption("TITAN L6 | Bayesian Inference, RL Policy, IsolationForest Anomaly Detection, Pharmacokinetic Time-Constants.")
