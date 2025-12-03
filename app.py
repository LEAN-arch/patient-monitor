import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import welch
from scipy.stats import norm, multivariate_normal
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
    "drug": "#4f46e5", "anomaly": "#db2777", "external": "#d946ef"
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
    DRUG_PK = {
        'norepi': {'svr': 2500, 'map': 120, 'co': 0.8, 'tau': 2.0, 'tol': 2880}, 
        'vaso':   {'svr': 4000, 'map': 150, 'co': -0.2, 'tau': 5.0, 'tol': 2880}, 
        'dobu':   {'svr': -600, 'map': 5, 'co': 4.5, 'hr': 25, 'tau': 3.0, 'tol': 720}, 
        'bb':     {'svr': 50, 'map': -15, 'co': -2.0, 'hr': -35, 'tau': 4.0, 'tol': 5000}
    }
    ATM_PRESSURE = 760; H2O_PRESSURE = 47; R_QUOTIENT = 0.8; MAX_PAO2 = 600
    LAC_PROD_THRESH = 330; LAC_CLEAR_RATE = 0.05; VCO2_CONST = 130 
    EPSILON = 1e-5

# ==========================================
# 2. DYNAMICS MODULES
# ==========================================

class AutonomicDynamics:
    @staticmethod
    def generate_vitals(mins, p, seed, is_paced=False, vent_mode='Spontaneous'):
        def noise(n, start, end, vol, s, signal_type='bio'):
            np.random.seed(s)
            t = np.linspace(0, 1, n)
            dW = np.random.normal(0, np.sqrt(1/n), n)
            B = start + np.cumsum(dW) - t * (np.cumsum(dW)[-1] - (end - start))
            
            if signal_type == 'paced':
                return B + np.random.normal(0, 0.05, n)
            elif signal_type == 'mechanical_vent':
                resp_wave = np.sin(np.linspace(0, n/4, n)) * 2.0
                return B + resp_wave + np.random.normal(0, 0.2, n)
            else:
                stress = 1.0 + (max(0, start - end) / 20.0)
                pink = np.convolve(np.random.normal(0, 0.5, n), np.ones(8)/8, mode='same')
                return B + (pink * vol * stress)

        if is_paced: hr = noise(mins, p['hr'][0], p['hr'][0], 0.1, seed, 'paced')
        elif vent_mode == 'Control (AC)': hr = noise(mins, p['hr'][0], p['hr'][1], 1.5, seed, 'mechanical_vent')
        else: hr = noise(mins, p['hr'][0], p['hr'][1], 1.5, seed, 'bio')

        map_r = np.maximum(noise(mins, p['map'][0], p['map'][1], 1.2, seed+1, 'bio'), 20.0)
        ci = np.maximum(noise(mins, p['ci'][0], p['ci'][1], 0.2, seed+2, 'bio'), 0.5)
        svri = np.maximum(noise(mins, p['svri'][0], p['svri'][1], 100, seed+3, 'bio'), 100.0)
        rr = np.maximum(noise(mins, 16, 28, 2.0, seed+4, 'bio'), 4.0)
        
        return hr, map_r, ci, svri, rr

class PharmacokineticsEngine:
    @staticmethod
    def apply_pk_pd(map_base, ci_base, hr_base, svri_base, drugs, mins):
        t_vec = np.arange(mins)
        def calculate_effect(drug_name, dose, param):
            dose = max(0.0, dose) 
            if dose <= 0: return np.zeros(mins)
            pk = CONSTANTS.DRUG_PK[drug_name]
            if param not in pk: return np.zeros(mins)
            max_eff = dose * pk[param]
            return max_eff * (1 - np.exp(-t_vec / pk['tau'])) * np.exp(-t_vec / pk['tol'])

        d_svr = calculate_effect('norepi', drugs['norepi'], 'svr') + calculate_effect('vaso', drugs['vaso'], 'svr') + calculate_effect('dobu', drugs['dobu'], 'svr')
        d_map = calculate_effect('norepi', drugs['norepi'], 'map') + calculate_effect('vaso', drugs['vaso'], 'map') + calculate_effect('bb', drugs['bb'], 'map')
        d_co  = calculate_effect('norepi', drugs['norepi'], 'co') + calculate_effect('dobu', drugs['dobu'], 'co') + calculate_effect('bb', drugs['bb'], 'co')
        d_hr  = calculate_effect('dobu', drugs['dobu'], 'hr') + calculate_effect('bb', drugs['bb'], 'hr')

        return (
            np.maximum(map_base + d_map, 10.0), 
            np.maximum(ci_base + d_co, 0.1), 
            np.maximum(hr_base + d_hr, 30.0), 
            np.maximum(svri_base + d_svr, 50.0)
        )

class RespiratoryDynamics:
    @staticmethod
    def simulate_gas_exchange(fio2, rr, shunt_base, peep, mins, copd_factor):
        fio2 = np.clip(fio2, 0.21, 1.0)
        rr = np.maximum(rr, 4.0)
        vd_vt = np.clip(0.3 * copd_factor + (0.1 if copd_factor > 1.0 else 0), 0.1, 0.8)
        va = np.maximum((rr * 0.5) * (1 - vd_vt), 0.5) 
        paco2 = (CONSTANTS.VCO2_CONST * 0.863) / va
        p_ideal = np.clip((fio2 * (CONSTANTS.ATM_PRESSURE - CONSTANTS.H2O_PRESSURE)) - (paco2 / CONSTANTS.R_QUOTIENT), 0, CONSTANTS.MAX_PAO2)
        shunt_eff = shunt_base * np.exp(-0.06 * peep)
        pao2 = p_ideal * (1 - shunt_eff)
        pao2_safe = np.maximum(pao2, 0.1)
        spo2 = 100 / (1 + (23400 / (pao2_safe**3 + 150*pao2_safe)))
        return pao2, paco2, spo2, vd_vt

class MetabolicDynamics:
    @staticmethod
    def simulate_metabolism(ci, hb, spo2, mins, vo2_stress_factor):
        ci_safe = np.maximum(ci, 0.1)
        do2i = ci_safe * hb * 1.34 * (spo2/100) * 10
        vo2i = np.full(mins, 125.0 * vo2_stress_factor)
        o2er = vo2i / np.maximum(do2i, 1.0)
        lactate = np.zeros(mins)
        lac_curr = 1.0
        prod = np.where((do2i < CONSTANTS.LAC_PROD_THRESH) | (o2er > 0.50), 0.2, 0.0)
        for i in range(mins):
            clear = CONSTANTS.LAC_CLEAR_RATE * (ci_safe[i] / 2.5)
            lac_curr = max(0.6, lac_curr + prod[i] - clear)
            lactate[i] = lac_curr
        return do2i, vo2i, o2er, lactate

# ==========================================
# 3. AI, ANALYTICS & SIGNAL FORENSICS
# ==========================================

class SignalForensics:
    @staticmethod
    def analyze_driver_source(time_series, is_paced, drug_delta):
        std_dev = np.std(time_series)
        if is_paced or std_dev < 0.5:
            return "EXTERNAL: PACEMAKER", 99, "Zero Variance (Quartz Precision)"
        grad = np.gradient(time_series)
        if np.max(np.abs(grad)) > 5.0: 
            return "EXTERNAL: INFUSION/BOLUS", 90, "Non-Physiologic Step Change"
        f, Pxx = welch(time_series, fs=1/60)
        pxx_norm = Pxx / np.sum(Pxx)
        spectral_entropy = -np.sum(pxx_norm * np.log2(pxx_norm + 1e-12))
        if spectral_entropy < 1.5: 
            return "EXTERNAL: VENTILATOR ENTRAINMENT", 85, f"Periodic Oscillation"
        return "INTERNAL: AUTONOMIC", 80, "Fractal Complexity (Pink Noise)"

class AI_Services:
    @staticmethod
    def predict_shock_bayes(row):
        means = {"Cardiogenic": [1.8, 2800], "Distributive": [5.0, 800], "Hypovolemic": [2.0, 3000], "Stable": [3.2, 1900]}
        covs = {"Cardiogenic": [[0.5, -100], [-100, 150000]], "Distributive": [[1.0, -200], [-200, 100000]],
                "Hypovolemic": [[0.4, -50], [-50, 200000]], "Stable": [[0.6, -150], [-150, 150000]]}
        priors = {"Cardiogenic": 0.2, "Distributive": 0.3, "Hypovolemic": 0.2, "Stable": 0.3}
        scores, total = {}, 0
        x = [row['CI'], row['SVRI']]
        for name in means:
            try:
                like = multivariate_normal.pdf(x, mean=means[name], cov=covs[name])
                scores[name] = like * priors[name]; total += scores[name]
            except: scores[name] = 0.0
        return {k: (v/total)*100 for k, v in scores.items()} if total > 1e-9 else {k:25.0 for k in means}

    @staticmethod
    def rl_advisor(row, drugs):
        if pd.isna(row['MAP']): return "Data Invalid", 0
        if row['MAP'] < 65:
            if row['CI'] < 2.2: 
                return ("Titrate Dobutamine +2.5", 88) if drugs['dobu'] < 5 else ("Consider Mechanical Support", 75)
            else: 
                return ("Increase Norepi +0.05", 92) if row['SVRI'] < 1200 else ("Fluid Bolus 500mL", 80)
        else:
            return ("Wean Norepi -0.05", 85) if drugs['norepi'] > 0 and row['MAP'] > 75 else ("Maintain Current Therapy", 98)

    @staticmethod
    def detect_anomalies(df):
        X = df[['MAP','CI','SVRI']].fillna(method='bfill').fillna(0)
        df['anomaly'] = IsolationForest(contamination=0.05, random_state=42).fit_predict(X)
        return df

    @staticmethod
    def monte_carlo_forecast(df, target='MAP', n_sims=50):
        curr, hist = df[target].iloc[-1], df[target].iloc[-30:]
        vol = max(np.std(hist) if len(hist) > 1 else 1.0, 0.5)
        paths = np.array([curr + np.cumsum(np.random.normal(0, vol, 30)) for _ in range(n_sims)])
        return np.percentile(paths, 10, 0), np.percentile(paths, 50, 0), np.percentile(paths, 90, 0)
    
    @staticmethod
    def inverse_centroids(df):
        try:
            if len(df)<10: return ["Data Insufficient"]
            sc = StandardScaler()
            X = sc.fit_transform(df[['CI','SVRI','Lactate']].fillna(0))
            ctrs = sc.inverse_transform(KMeans(3, random_state=42, n_init=10).fit(X).cluster_centers_)
            return [f"C{i+1}: CI={c[0]:.1f}, SVR={c[1]:.0f}" for i,c in enumerate(ctrs)]
        except: return ["Calc Error"]

# ==========================================
# 4. SIMULATOR ORCHESTRATOR
# ==========================================
class PatientSimulator:
    def __init__(self, mins=360):
        self.mins = mins
        self.t = np.arange(mins)
        
    def run(self, case_id, drugs, fluids, bsa, peep, is_paced, vent_mode):
        cases = {
            "65M Post-CABG": {'ci':(2.4, 1.8), 'map':(75, 55), 'svri':(2000, 1600), 'hr':(85, 95), 'shunt':0.10, 'copd':1.0, 'vo2_stress': 1.0},
            "24F Septic Shock": {'ci':(3.5, 5.5), 'map':(65, 45), 'svri':(1200, 500), 'hr':(110, 140), 'shunt':0.15, 'copd':1.0, 'vo2_stress': 1.4},
            "82M HFpEF Sepsis": {'ci':(2.2, 1.8), 'map':(85, 55), 'svri':(2400, 1800), 'hr':(80, 110), 'shunt':0.20, 'copd':1.5, 'vo2_stress': 1.1},
            "50M Trauma": {'ci':(3.0, 1.5), 'map':(70, 40), 'svri':(2500, 3200), 'hr':(100, 150), 'shunt':0.05, 'copd':1.0, 'vo2_stress': 0.9}
        }
        p = cases[case_id]
        seed = len(case_id)+42
        
        hr, map_r, ci_r, svri_r, rr = AutonomicDynamics.generate_vitals(self.mins, p, seed, is_paced, vent_mode)
        ppv = (20 if "Trauma" in case_id else 12) + (np.sin(self.t/8)*4)
        ci_fluid = (fluids/500) * (0.4 if np.mean(ppv)>13 else 0.05)
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
        }).fillna(0)
        return df

# ==========================================
# 5. VISUALIZATION FUNCTIONS (COMPLETE)
# ==========================================
def plot_sparkline(data, color, key):
    fig = px.line(x=np.arange(len(data)), y=data)
    fig.update_traces(line_color=color, line_width=2)
    fig.update_layout(showlegend=False, xaxis_visible=False, yaxis_visible=False, margin=dict(l=0,r=0,t=0,b=0), height=35)
    return fig

def plot_bayes_bars(probs, key):
    fig = go.Figure(go.Bar(x=list(probs.values()), y=list(probs.keys()), orientation='h', marker=dict(color=[THEME['hemo'], THEME['crit'], THEME['warn'], THEME['ok']])))
    fig.update_layout(
        margin=dict(l=0,r=0,t=20,b=0), height=120, 
        xaxis=dict(showticklabels=True, title="Probability [%]", range=[0, 100]),
        yaxis=dict(title=None),
        title=dict(text="P(Shock State | CI, SVR)", font=dict(size=12, color=THEME['text_muted']))
    )
    return fig

def plot_3d_attractor(df, key):
    recent = df.iloc[-60:]
    fig = go.Figure(data=[go.Scatter3d(x=recent['CPO'], y=recent['SVRI'], z=recent['Lactate'], 
                                       mode='lines+markers', marker=dict(size=3, color=recent.index, colorscale='Viridis', showscale=False), 
                                       line=dict(width=2, color='rgba(50,50,50,0.5)'), name="State")])
    fig.update_layout(
        scene=dict(xaxis_title='Power [W]', yaxis_title='SVRI', zaxis_title='Lactate'), 
        margin=dict(l=0, r=0, b=0, t=25), height=250, 
        title=dict(text="3D Hemodynamic Phase Space", font=dict(size=12, color=THEME['text_muted']))
    )
    return fig

def plot_chaos_attractor(df, key, source_label):
    hr_safe = np.maximum(df['HR'].iloc[-120:], 1.0)
    rr = 60000 / hr_safe
    color = THEME['external'] if "EXTERNAL" in source_label else 'teal'
    fig = go.Figure(go.Scatter(x=rr.iloc[:-1], y=rr.iloc[1:], mode='markers', marker=dict(color=color, size=4, opacity=0.6), name="R-R"))
    fig.update_layout(
        title=dict(text=f"Chaos: {source_label}", font=dict(size=12, color=THEME['text_muted'])), 
        height=200, margin=dict(l=20,r=20,t=30,b=20), xaxis_title="RR(n)", yaxis_title="RR(n+1)", showlegend=False
    )
    return fig

def plot_spectral_analysis(df, key):
    data = df['HR'].iloc[-120:].to_numpy()
    f, Pxx = welch(data, fs=1/60)
    fig = px.line(x=f, y=Pxx)
    fig.add_vline(x=0.04, line_dash="dot", annotation_text="LF"); fig.add_vline(x=0.15, line_dash="dot", annotation_text="HF")
    fig.update_layout(
        title=dict(text="Spectral HRV Analysis", font=dict(size=12, color=THEME['text_muted'])),
        height=200, margin=dict(l=20,r=20,t=30,b=20), xaxis_title="Hz", yaxis_title="Power"
    )
    return fig

def plot_monte_carlo_fan(df, p10, p50, p90, key):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(30), y=df['MAP'].iloc[-30:], name="History", line=dict(color=THEME['text_main'])))
    fx = np.arange(30, 60)
    fig.add_trace(go.Scatter(x=np.concatenate([fx, fx[::-1]]), y=np.concatenate([p90, p10[::-1]]), fill='toself', fillcolor='rgba(0,0,255,0.1)', line=dict(width=0), name="80% CI"))
    fig.add_trace(go.Scatter(x=fx, y=p50, line=dict(dash='dot', color='blue'), name="Median"))
    fig.update_layout(
        height=200, title=dict(text="Stochastic Forecast (MAP)", font=dict(size=12, color=THEME['text_muted'])), 
        margin=dict(l=20,r=20,t=30,b=20), xaxis_title="Time", yaxis_title="mmHg", legend=dict(orientation="h", y=1.1)
    )
    return fig

def plot_hemodynamic_profile(df, key):
    recent = df.iloc[-60:]
    fig = go.Figure()
    fig.add_hline(y=2000, line_dash="dot", annotation_text="Vasoconstricted"); fig.add_vline(x=2.2, line_dash="dot", annotation_text="Low Flow")
    fig.add_trace(go.Scatter(x=recent['CI'], y=recent['SVRI'], mode='markers', marker=dict(color=recent.index, colorscale='Viridis'), name="State"))
    fig.update_layout(title=dict(text="Pump vs Pipes (Forrester)", font=dict(size=12, color=THEME['text_muted'])), height=200, margin=dict(l=20,r=20,t=30,b=20), xaxis_title="CI", yaxis_title="SVRI")
    return fig

def plot_phase_space(df, key):
    recent = df.iloc[-60:]
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=0.6, y0=2, y1=15, fillcolor="rgba(255,0,0,0.1)", line_width=0)
    fig.add_trace(go.Scatter(x=recent['CPO'], y=recent['Lactate'], mode='lines+markers', marker=dict(color=recent.index, colorscale='Bluered'), name="Trajectory"))
    fig.update_layout(title=dict(text="Coupling (CPO/Lac)", font=dict(size=12, color=THEME['text_muted'])), height=200, margin=dict(l=20,r=20,t=30,b=20), xaxis_title="CPO", yaxis_title="Lactate")
    return fig

def plot_vq_scatter(df, key):
    fig = px.scatter(df.iloc[-60:], x="PaO2", y="SpO2", color="PaCO2", color_continuous_scale="Bluered")
    fig.update_layout(title=dict(text="V/Q Mismatch & Oxygenation", font=dict(size=12, color=THEME['text_muted'])), height=200, margin=dict(l=20,r=20,t=30,b=20))
    return fig

def plot_counterfactual(df, drugs, case, bsa, peep, key):
    sim = PatientSimulator(60)
    base = {'norepi':0, 'vaso':0, 'dobu':0, 'bb':0, 'fio2':0.21}
    df_b = sim.run(case, base, 0, bsa, peep, False, 'Spontaneous')
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df['MAP'].iloc[-60:], name="Intervention", line=dict(color=THEME['ok'])))
    fig.add_trace(go.Scatter(y=df_b['MAP'], name="Natural Hx", line=dict(dash='dot', color=THEME['crit'])))
    fig.update_layout(title=dict(text="Counterfactual (MAP)", font=dict(size=12, color=THEME['text_muted'])), height=200, margin=dict(l=20,r=20,t=30,b=20), xaxis_title="Time", yaxis_title="MAP")
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
    
    col_bio1, col_bio2 = st.columns(2)
    with col_bio1: height = st.number_input("Height (cm)", 150, 200, 175)
    with col_bio2: weight = st.number_input("Weight (kg)", 50, 150, 80)
    bsa = np.sqrt((height * weight) / 3600)
    st.caption(f"BSA: {bsa:.2f} m²")
    
    with st.form("drugs"):
        c1, c2 = st.columns(2)
        norepi = c1.number_input("Norepi", 0.0, 2.0, 0.0, 0.05)
        vaso = c1.number_input("Vaso", 0.0, 0.1, 0.0, 0.01)
        dobu = c2.number_input("Dobutamine", 0.0, 10.0, 0.0, 0.5)
        bb = c2.number_input("Esmolol", 0.0, 1.0, 0.0, 0.1)
        fio2 = st.slider("FiO2", 0.21, 1.0, 0.4); peep = st.slider("PEEP", 0, 20, 5)
        st.form_submit_button("Update")
    
    st.markdown("### 🔌 External Devices")
    is_paced = st.checkbox("Pacemaker (VVI)")
    vent_mode = st.selectbox("Vent Mode", ["Spontaneous", "Control (AC)"])
    
    if st.button("💧 Bolus 500mL"): st.session_state['fluids'] += 500
    live_mode = st.checkbox("🔴 LIVE MODE")

sim = PatientSimulator(mins=res_mins)
drugs = {'norepi':norepi, 'vaso':vaso, 'dobu':dobu, 'bb':bb, 'fio2':fio2}
df = sim.run(case_id, drugs, st.session_state['fluids'], bsa, peep, is_paced, vent_mode)

# Analytics
df = AI_Services.detect_anomalies(df)
probs = AI_Services.predict_shock_bayes(df.iloc[-1])
sugg, conf = AI_Services.rl_advisor(df.iloc[-1], drugs)
p10, p50, p90 = AI_Services.monte_carlo_forecast(df)
centroids = AI_Services.inverse_centroids(df)
source_label, source_conf, source_reason = SignalForensics.analyze_driver_source(df['HR'].iloc[-120:], is_paced, 0)

# --- DASHBOARD RENDERER ---
container = st.empty()

def render(d_slice):
    ix = len(d_slice) 
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
        with c1: st.plotly_chart(plot_bayes_bars(probs, ix), use_container_width=True, key=f"bayes_{ix}")
        with c2: 
            st.info(f"🤖 **RL Agent:** {sugg} ({conf}%)")
            st.plotly_chart(plot_monte_carlo_fan(d_slice, p10, p50, p90, ix), use_container_width=True, key=f"fan_{ix}")
        with c3: st.plotly_chart(plot_counterfactual(d_slice, drugs, case_id, bsa, peep, ix), use_container_width=True, key=f"count_{ix}")

        # ZONE B: METRICS
        st.markdown('<div class="zone-header">ZONE B: PHYSIOLOGIC METRICS (60m)</div>', unsafe_allow_html=True)
        cols = st.columns(6)
        metrics = [("MAP", curr['MAP'], THEME['hemo']), ("CI", curr['CI'], THEME['info']), 
                   ("SVRI", curr['SVRI'], THEME['warn']), ("Lactate", curr['Lactate'], THEME['crit']),
                   ("O2ER", curr['O2ER']*100, THEME['resp']), ("CPO", curr['CPO'], THEME['drug'])]
        for i, (l, v, c) in enumerate(metrics):
            with cols[i]:
                st.metric(l, f"{v:.1f}")
                st.plotly_chart(plot_sparkline(d_slice[l].iloc[-60:], c, ix), use_container_width=True, key=f"spark_{i}_{ix}")

        # ZONE C: FORENSICS & COMPLEXITY
        st.markdown('<div class="zone-header">ZONE C: SIGNAL FORENSICS & COMPLEXITY</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:{THEME['card_bg']}; padding:10px; border-left:4px solid {THEME['external'] if 'EXTERNAL' in source_label else THEME['ok']}; margin-bottom:10px; border-radius:4px;">
            <div style="font-size:0.75rem; font-weight:700; color:{THEME['text_muted']}">PRIMARY DRIVER</div>
            <div style="font-size:1.1rem; font-weight:800;">{source_label}</div>
            <div style="font-size:0.8rem; color:{THEME['text_muted']}">{source_reason} (Conf: {source_conf}%)</div>
        </div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.plotly_chart(plot_chaos_attractor(d_slice, ix, source_label), use_container_width=True, key=f"chaos_{ix}")
        with c2: st.plotly_chart(plot_spectral_analysis(d_slice, ix), use_container_width=True, key=f"spec_{ix}")
        with c3: st.plotly_chart(plot_3d_attractor(d_slice, ix), use_container_width=True, key=f"3d_{ix}")

        # ZONE D: HEMODYNAMICS
        st.markdown('<div class="zone-header">ZONE D: ADVANCED HEMODYNAMICS</div>', unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1: st.plotly_chart(plot_hemodynamic_profile(d_slice, ix), use_container_width=True, key=f"hemo_{ix}")
        with b2: st.plotly_chart(plot_phase_space(d_slice, ix), use_container_width=True, key=f"phase_{ix}")
        
        # ZONE E: RESPIRATORY & METABOLIC (Restored)
        st.markdown('<div class="zone-header">ZONE E: RESPIRATORY & METABOLIC COUPLING</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_vq_scatter(d_slice, ix), use_container_width=True, key=f"vq_{ix}")

        # ZONE F: TELEMETRY
        st.markdown('<div class="zone-header">ZONE F: TELEMETRY</div>', unsafe_allow_html=True)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03)
        fig.add_trace(go.Scatter(x=d_slice['Time'], y=d_slice['MAP'], name="MAP", line=dict(color=THEME['hemo'])), row=1, col=1)
        fig.add_trace(go.Scatter(x=d_slice['Time'], y=d_slice['CPO'], name="CPO", fill='tozeroy', line=dict(color=THEME['info'])), row=2, col=1)
        fig.add_trace(go.Scatter(x=d_slice['Time'], y=d_slice['SpO2'], name="SpO2", line=dict(color=THEME['resp'])), row=3, col=1)
        fig.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0), template="plotly_white")
        fig.update_yaxes(title_text="MAP", row=1, col=1); fig.update_yaxes(title_text="CPO", row=2, col=1); fig.update_yaxes(title_text="SpO2", row=3, col=1)
        st.plotly_chart(fig, use_container_width=True, key=f"tele_{ix}")

if live_mode:
    for i in range(max(10, res_mins-60), res_mins):
        render(df.iloc[:i])
        time.sleep(0.1)
else:
    render(df)
st.caption("TITAN L6 | Bayesian Inference, RL Policy, IsolationForest Anomaly Detection, Pharmacokinetic Time-Constants.")
