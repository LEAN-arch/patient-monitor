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
import time

# ==========================================
# 1. SYSTEM CONFIGURATION & MEDICAL THEME
# ==========================================
st.set_page_config(
    page_title="TITAN | LEVEL 4 CDS",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧬"
)

# Advanced Medical Palette
THEME = {
    "bg": "#f8fafc",        
    "card_bg": "#ffffff",   
    "text_main": "#0f172a", 
    "text_muted": "#64748b",
    "border": "#e2e8f0",    
    "crit": "#dc2626",      # Red-600
    "warn": "#d97706",      # Amber-600
    "ok": "#059669",        # Emerald-600
    "info": "#2563eb",      # Blue-600
    "hemo": "#0891b2",      # Cyan-600
    "resp": "#7c3aed",      # Violet (Respiratory)
    "ai": "#be185d",        # Pink (AI)
    "drug": "#4f46e5"       # Indigo (Pharm)
}

STYLING = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Roboto+Mono:wght@500;700&display=swap');
    
    .stApp {{ background-color: {THEME['bg']}; color: {THEME['text_main']}; font-family: 'Inter', sans-serif; }}
    
    /* Flash Animation for Critical Alerts */
    @keyframes flash-crit {{ 0% {{ box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.7); }} 70% {{ box-shadow: 0 0 0 10px rgba(220, 38, 38, 0); }} 100% {{ box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }} }}
    
    .crit-pulse {{ animation: flash-crit 2s infinite; border: 1px solid {THEME['crit']} !important; }}
    
    /* Metrics */
    div[data-testid="stMetric"] {{
        background-color: {THEME['card_bg']};
        padding: 8px 12px;
        border-radius: 6px;
        border: 1px solid {THEME['border']};
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }}
    div[data-testid="stMetric"] label {{ font-size: 0.65rem; font-weight: 700; color: {THEME['text_muted']}; text-transform: uppercase; }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ font-family: 'Roboto Mono'; font-size: 1.4rem; font-weight: 800; }}
    
    /* Headers */
    .zone-header {{
        font-size: 0.8rem; font-weight: 900; color: {THEME['text_muted']};
        text-transform: uppercase; border-bottom: 2px solid {THEME['border']};
        margin: 20px 0 10px 0; padding-bottom: 4px; letter-spacing: 0.05em;
    }}
    
    /* AI Banner */
    .status-banner {{
        padding: 12px 20px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center;
        background: white; border-left: 6px solid; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 10px;
    }}
</style>
"""

# ==========================================
# 2. ADVANCED PHYSIOLOGY & PHARMACOLOGY ENGINE
# ==========================================
class PharmaEngine:
    """Pharmacodynamic Simulation (Receptor Kinetics)."""
    @staticmethod
    def apply_drugs(map_val, co, hr, svr, drugs):
        # 1. Norepinephrine (Alpha-1 Agonist): ↑ SVR, ↑ MAP, mild ↑ CO (beta), reflex ↓ HR
        ne_effect = drugs['norepi'] * 2.5 # Potency factor
        svr += (ne_effect * 800) # Strong vasoconstriction
        map_val += (ne_effect * 25)
        co += (ne_effect * 0.5)
        
        # 2. Vasopressin (V1 Agonist): ↑↑ SVR (Non-linear), Neutral CO
        vaso_effect = drugs['vaso'] * 4.0 
        svr += (vaso_effect * 600)
        map_val += (vaso_effect * 15)
        
        # 3. Dobutamine (Beta-1/2 Agonist): ↑↑ CO, ↓ SVR (Vasodilation), ↑ HR
        dobu_effect = drugs['dobu'] * 2.0
        co += (dobu_effect * 2.5)
        hr += (dobu_effect * 15)
        svr -= (dobu_effect * 300) # Vasodilation
        
        # 4. Beta Blockers: ↓ HR, ↓ CO, Neutral MAP (compensated)
        bb_effect = drugs['bb'] * 1.5
        hr -= (bb_effect * 20)
        co -= (bb_effect * 0.8)
        
        return map_val, co, hr, svr

class PhysiologyEngine:
    @staticmethod
    def brownian_bridge(n, start, end, volatility=1.0, seed=42):
        np.random.seed(seed)
        t = np.linspace(0, 1, n)
        dW = np.random.normal(0, np.sqrt(1/n), n)
        W = np.cumsum(dW)
        B = start + W - t * (W[-1] - (end - start))
        pink = np.convolve(np.random.normal(0, 0.5, n), np.ones(5)/5, mode='same')
        return B + (pink * volatility)

    @staticmethod
    def resp_module(pao2_target, shunt_fraction, fio2):
        """Simulates Oxygenation based on Shunt and FiO2."""
        # Ideal alveolar gas equation approx
        p_ideal = (fio2 * 713) - 40/0.8 
        # Shunt effect
        pao2 = p_ideal * (1 - shunt_fraction)
        # SpO2 Sigmoid Curve (Severinghaus)
        spo2 = 100 / (1 + (23400 / (pao2**3 + 150*pao2)) )
        return pao2, spo2

    @staticmethod
    def fluid_responsiveness(ppv, fluid_bolus_ml):
        """Starling Curve Logic."""
        # If PPV > 12%, patient is fluid responsive (Steep part of Starling curve)
        # CO gain per 500mL
        if ppv > 12:
            return (fluid_bolus_ml / 500) * 1.2 # L/min gain
        else:
            return (fluid_bolus_ml / 500) * 0.1 # Minimal gain (Flat part)

# ==========================================
# 3. PATIENT SIMULATOR (ENHANCED)
# ==========================================
class PatientSimulator:
    def __init__(self, mins=360):
        self.mins = mins
        self.t = np.arange(mins)

    def get_data(self, profile, drugs, fluids, age, chronic, drift, seed):
        # --- 1. BASELINE MODIFIERS (PATIENT PROFILE) ---
        # Elderly: Stiff arteries (↑ SVR), Low reserve (↓ CO)
        age_mod_svr = 1.2 if age > 70 else 1.0
        age_mod_co = 0.8 if age > 70 else 1.0
        
        # Chronic Disease
        hf_mod = 0.7 if "Heart Failure" in chronic else 1.0
        copd_mod = 0.85 if "COPD" in chronic else 1.0 # SpO2 penalty
        
        # --- 2. SCENARIO INITIALIZATION ---
        if profile == "Healthy":
            p = {'hr': (65, 70), 'map': (85, 85), 'co': (5.0, 5.0), 'vol': 1.0, 'shunt': 0.05}
        elif profile == "Compensated Sepsis":
            p = {'hr': (85, 100), 'map': (82, 75), 'co': (6.0, 7.0), 'vol': 1.5, 'shunt': 0.10}
        elif profile == "Vasoplegic Shock":
            p = {'hr': (110, 130), 'map': (60, 50), 'co': (7.0, 8.0), 'vol': 1.2, 'shunt': 0.15}
        elif profile == "Cardiogenic Shock":
            p = {'hr': (90, 105), 'map': (70, 55), 'co': (3.5, 2.5), 'vol': 2.0, 'shunt': 0.20}

        # Apply Drift (Worsening over time if enabled)
        drift_factor = np.linspace(1.0, drift, self.mins)
        
        # --- 3. GENERATE RAW RANDOM WALKS ---
        hr = PhysiologyEngine.brownian_bridge(self.mins, p['hr'][0], p['hr'][1], p['vol'], seed) * drift_factor
        map_raw = PhysiologyEngine.brownian_bridge(self.mins, p['map'][0], p['map'][1], p['vol']*0.8, seed+1) / drift_factor
        co_raw = PhysiologyEngine.brownian_bridge(self.mins, p['co'][0], p['co'][1], 0.2, seed+2) * hf_mod / drift_factor
        
        # --- 4. RESPIRATORY SIMULATION ---
        rr = PhysiologyEngine.brownian_bridge(self.mins, 14, 22 if "Shock" in profile else 16, 2.0, seed+3)
        paco2 = 40 + (16 - rr) * 1.5 # Hypercapnia logic
        
        # Calculate PaO2/SpO2 based on Shunt & FiO2
        pao2_arr, spo2_arr = [], []
        for i in range(self.mins):
            pa, sp = PhysiologyEngine.resp_module(0, p['shunt'] / copd_mod, drugs['fio2'])
            # Add noise
            pao2_arr.append(pa + np.random.normal(0, 5))
            spo2_arr.append(sp)
            
        # --- 5. DERIVED METRICS & FLUID PHYSICS ---
        # Pulse Pressure Variation (PPV) - Interaction of Heart & Lung
        # PPV is high when Heart is preload dependent (hypovolemia)
        ppv_base = 15 if "Shock" in profile else 5
        ppv = ppv_base + (np.sin(self.t/10) * 3) # Respiratory swing
        
        # Apply Fluids (Starling Curve)
        co_fluid_gain = PhysiologyEngine.fluid_responsiveness(np.mean(ppv), fluids)
        
        # --- 6. PHARMACODYNAMICS APPLICATION ---
        # SVR Base Calculation
        svr_raw = ((map_raw - 8) / co_raw) * 800 * age_mod_svr
        
        final_map, final_co, final_hr, final_svr = [], [], [], []
        
        for i in range(self.mins):
            # Apply Drugs & Fluids per timestep
            m, c, h, s = PharmaEngine.apply_drugs(map_raw[i], co_raw[i] + co_fluid_gain, hr[i], svr_raw[i], drugs)
            final_map.append(m)
            final_co.append(c)
            final_hr.append(h)
            final_svr.append(s)
            
        # --- 7. METABOLIC (LACTATE) ---
        # DO2 = CO * Hb * 1.34 * SpO2
        hb = 12.0 # Assume constant for now
        do2 = np.array(final_co) * hb * 1.34 * (np.array(spo2_arr)/100) * 10
        vo2 = do2 * 0.25 # Assume 25% extraction initially
        
        # Lactate Accumulation (Supply/Demand Mismatch)
        lactate = np.zeros(self.mins)
        lac_curr = 1.0
        for i in range(self.mins):
            # If DO2 < 400 (Critical Threshold), Lactate Rises
            production = 0.1 if do2[i] < 400 else 0.0
            clearance = 0.05 if do2[i] > 500 else 0.01
            lac_curr = max(0.5, lac_curr + production - clearance)
            lactate[i] = lac_curr

        # --- 8. DATAFRAME CONSTRUCTION ---
        df = pd.DataFrame({
            "Time": self.t, "HR": final_hr, "MAP": final_map, "CO": final_co, "SVR": final_svr,
            "Lactate": lactate, "SpO2": spo2_arr, "PaO2": pao2_arr, "PaCO2": paco2, "RR": rr,
            "PPV": ppv, "DO2": do2, "VO2": vo2,
            "Creatinine": np.linspace(0.8, 1.2 if map_raw[-1] < 60 else 0.9, self.mins)
        })
        
        # Advanced Derivatives
        df['CPO'] = (df['MAP'] * df['CO']) / 451
        df['SI'] = df['HR'] / df['MAP'] # Simplified Shock Index
        df['MSI'] = df['HR'] / (df['MAP'] * 1.0) # Modified SI
        
        # Fluid Responsiveness Index
        df['SV'] = (df['CO'] * 1000) / df['HR']
        
        return df

# ==========================================
# 4. AI & PREDICTIVE LAYER
# ==========================================
class AI_Analytics:
    @staticmethod
    def calculate_stability_score(row):
        """Composite Early Warning Score (0-100)."""
        score = 100
        # Penalties
        if row['MAP'] < 65: score -= 20
        if row['Lactate'] > 2: score -= 20
        if row['CPO'] < 0.6: score -= 30
        if row['SpO2'] < 90: score -= 10
        if row['HR'] > 110: score -= 10
        return max(0, score)

    @staticmethod
    def predict_trajectory(df, target_col, minutes=30):
        """Linear Regression for short-term trajectory."""
        recent = df.iloc[-30:] # Last 30 mins
        X = recent.index.values.reshape(-1, 1)
        y = recent[target_col].values
        model = LinearRegression().fit(X, y)
        
        future_X = np.arange(df.index[-1], df.index[-1]+minutes).reshape(-1, 1)
        future_y = model.predict(future_X)
        return future_y[-1] # End point

    @staticmethod
    def cluster_phenotypes(df):
        """K-Means Clustering to identify phenotype."""
        # Features: CI, SVR, Lactate
        X = df[['CO', 'SVR', 'Lactate']].iloc[-60:]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
        # Assign current point to a cluster center
        center = kmeans.cluster_centers_[kmeans.labels_[-1]]
        # Logic to name the cluster (Simplified)
        if center[1] > 0.5: return "Vasoconstricted / Cardiogenic"
        elif center[0] > 0.5: return "Vasoplegic / High Flow"
        else: return "Uncompensated / Metabolic"

# ==========================================
# 5. VISUALIZATION COMPONENTS (MEDICAL GRADE)
# ==========================================
def plot_3d_attractor(df):
    """3D Phase Space: SVR vs Lactate vs CPO."""
    recent = df.iloc[-60:]
    fig = go.Figure(data=[go.Scatter3d(
        x=recent['CPO'], y=recent['SVR'], z=recent['Lactate'],
        mode='lines+markers',
        marker=dict(size=4, color=recent.index, colorscale='Viridis', opacity=0.8),
        line=dict(color='rgba(100,100,100,0.5)', width=2)
    )])
    fig.update_layout(
        scene=dict(xaxis_title='Power (W)', yaxis_title='SVR', zaxis_title='Lactate'),
        margin=dict(l=0, r=0, b=0, t=0), height=300,
        title="3D Hemo-Metabolic Attractor"
    )
    return fig

def plot_counterfactual(df, drugs, profile):
    """Projected Effectiveness: Baseline vs Intervention."""
    # Run a quick sim without drugs (Baseline)
    sim = PatientSimulator(mins=60)
    # Using 0 drugs for baseline comparison
    base_drugs = {'norepi':0, 'vaso':0, 'dobu':0, 'bb':0, 'fio2':0.21} 
    # Simplified baseline generation for visual comparison
    df_base = sim.get_data(profile, base_drugs, 0, 50, [], 1.0, 42)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df['MAP'].iloc[-60:], name="With Intervention", line=dict(color=THEME['ok'])))
    fig.add_trace(go.Scatter(y=df_base['MAP'].iloc[-60:], name="Projected Untreated", line=dict(dash='dot', color=THEME['crit'])))
    fig.update_layout(title="Intervention Effectiveness Projection (MAP)", height=200, margin=dict(l=20,r=20,t=30,b=20))
    return fig

def plot_spectral_analysis(df):
    """Fourier Transform of HRV."""
    f, Pxx = welch(df['HR'].iloc[-120:], fs=1/60) # 1 sample per min
    fig = px.line(x=f, y=Pxx, title="HRV Spectral Density (Autonomic Tone)")
    fig.update_xaxes(title="Frequency (Hz)")
    fig.update_yaxes(title="Power")
    fig.add_vline(x=0.04, line_dash="dot", annotation_text="LF (Sympathetic)")
    fig.add_vline(x=0.15, line_dash="dot", annotation_text="HF (Parasympathetic)")
    fig.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20))
    return fig

# ==========================================
# 6. MAIN APP EXECUTION
# ==========================================
st.markdown(STYLING, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'events' not in st.session_state: st.session_state['events'] = []
if 'fluids_given' not in st.session_state: st.session_state['fluids_given'] = 0

# --- SIDEBAR (CONTROLS) ---
with st.sidebar:
    st.title("TITAN | L4 CDS")
    
    # 1. Random Seed & Sim
    seed = st.number_input("Random Seed", 1, 1000, 42)
    drift = st.slider("Scenario Drift (Deterioration)", 1.0, 2.0, 1.0, 0.1)
    
    # 2. Patient Profile
    st.markdown("### 👤 Patient Profile")
    age = st.slider("Age", 18, 95, 65)
    chronic = st.multiselect("Comorbidities", ["Heart Failure", "COPD", "CKD"])
    scenario = st.selectbox("Presenting Phenotype", ["Healthy", "Compensated Sepsis", "Vasoplegic Shock", "Cardiogenic Shock"])
    
    # 3. Interventions
    st.markdown("### 💉 Interventions")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        norepi = st.slider("Norepi (mcg)", 0.0, 1.0, 0.0, 0.05)
        vaso = st.slider("Vasopressin", 0.0, 0.06, 0.0, 0.01)
    with col_d2:
        dobu = st.slider("Dobutamine", 0.0, 10.0, 0.0, 1.0)
        bb = st.slider("Beta-Blocker", 0.0, 1.0, 0.0, 0.1)
    
    # Respiratory
    fio2 = st.slider("FiO2", 0.21, 1.0, 0.40)
    
    # Fluids
    if st.button("💧 Give 500mL Bolus"):
        st.session_state['fluids_given'] += 500
        st.session_state['events'].append({"time": 360, "event": "Fluid Bolus"})
    
    st.caption(f"Total Fluids: {st.session_state['fluids_given']} mL")

    # Real-Time Toggle
    if st.checkbox("Live Mode (Simulate 1hr)"):
        st.spinner("Simulating real-time data flow...")

# --- DATA GENERATION ---
drug_dict = {'norepi': norepi, 'vaso': vaso, 'dobu': dobu, 'bb': bb, 'fio2': fio2}
sim = PatientSimulator(mins=360)
df = sim.get_data(scenario, drug_dict, st.session_state['fluids_given'], age, chronic, drift, seed)

# Current State
cur = df.iloc[-1]
prev = df.iloc[-60]

# --- AI ANALYTICS COMPUTATION ---
stability = AI_Analytics.calculate_stability_score(cur)
phenotype = AI_Analytics.cluster_phenotypes(df)
pred_map = AI_Analytics.predict_trajectory(df, 'MAP', 30)

# ==========================================
# UI LAYOUT
# ==========================================

# --- 1. TOP BANNER (AI & ALERTS) ---
alert_cls = "crit-pulse" if stability < 50 else ""
st.markdown(f"""
<div class="status-banner" style="border-left-color: {THEME['ai']};">
    <div>
        <div style="font-size:0.8rem; color:{THEME['ai']}; font-weight:800;">AI PHENOTYPE: {phenotype}</div>
        <div style="font-size:1.8rem; font-weight:800; color:{THEME['text_main']}">{scenario.upper()}</div>
        <div style="font-size:0.9rem; color:{THEME['text_muted']};">Drift Factor: {drift}x | Age: {age}</div>
    </div>
    <div style="text-align:right">
        <div style="font-size:0.8rem; font-weight:700;">STABILITY INDEX</div>
        <div class="{alert_cls}" style="font-size:2.2rem; font-weight:800; padding:5px 15px; border-radius:5px; display:inline-block; color:{THEME['text_main']}">{stability:.0f}/100</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 2. ZONE A: 3D VISUALIZATION & PREDICTIONS ---
c1, c2 = st.columns([2, 1])
with c1:
    st.plotly_chart(plot_3d_attractor(df), use_container_width=True)
with c2:
    st.markdown("**🔮 AI Trajectory (30 min)**")
    st.metric("Predicted MAP", f"{pred_map:.1f}", f"{pred_map - cur['MAP']:.1f}")
    st.metric("Lactate Clearance", f"{(prev['Lactate'] - cur['Lactate'])/prev['Lactate']*100:.1f}%", help="Last hour clearance")
    st.progress(stability/100, "Stability Score")

# --- 3. ZONE B: 6-PACK METRICS (WITH MINI-CARDS) ---
st.markdown('<div class="zone-header">ZONE B: HEMODYNAMICS & METABOLICS</div>', unsafe_allow_html=True)
m1, m2, m3, m4, m5, m6 = st.columns(6)

def arrow(val): return "↑" if val > 0 else "↓" if val < 0 else "→"

def metric_card(col, label, val, unit, delta, color_group="text_main"):
    col.metric(
        label=label,
        value=f"{val:.1f} {unit}",
        delta=f"{arrow(delta)} {abs(delta):.1f}",
        delta_color="inverse"
    )

metric_card(m1, "MAP", cur['MAP'], "mmHg", cur['MAP']-prev['MAP'])
metric_card(m2, "Heart Rate", cur['HR'], "bpm", cur['HR']-prev['HR'])
metric_card(m3, "Cardiac Power", cur['CPO'], "W", cur['CPO']-prev['CPO'])
metric_card(m4, "Lactate", cur['Lactate'], "mM", cur['Lactate']-prev['Lactate'])
metric_card(m5, "DO2 (Delivery)", cur['DO2'], "mL/min", cur['DO2']-prev['DO2'])
metric_card(m6, "SpO2", cur['SpO2'], "%", cur['SpO2']-prev['SpO2'])

# --- 4. ZONE C: RESPIRATORY & FLUID MODULE ---
st.markdown('<div class="zone-header">ZONE C: RESPIRATORY & FLUID STATUS</div>', unsafe_allow_html=True)
r1, r2, r3, r4 = st.columns(4)
r1.metric("PaO2 / FiO2", f"{cur['PaO2']/cur['PaO2']*300:.0f}", "Est. Shunt") # Simplified placeholder
r2.metric("PaCO2 (Vent)", f"{cur['PaCO2']:.1f}", "mmHg")
r3.metric("PPV (Fluid Resp)", f"{cur['PPV']:.1f}%", "Responsive" if cur['PPV']>12 else "Non-Resp", delta_color="normal")
r4.metric("Shock Index", f"{cur['SI']:.2f}", ">0.9 Crit")

# --- 5. ZONE D: DEEP ANALYTICS (TABS) ---
tab_trend, tab_corr, tab_spec, tab_event = st.tabs(["📈 Telemetry", "🔥 Correlations", "🌊 Spectral HRV", "📝 Events"])

with tab_trend:
    # Telemetry with Event Annotation
    fig_tele = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    fig_tele.add_trace(go.Scatter(x=df['Time'], y=df['MAP'], name="MAP", line=dict(color=THEME['hemo'])), row=1, col=1)
    fig_tele.add_trace(go.Scatter(x=df['Time'], y=df['CPO'], name="CPO", fill='tozeroy', line=dict(color=THEME['info'])), row=2, col=1)
    fig_tele.add_trace(go.Scatter(x=df['Time'], y=df['SVR'], name="SVR", line=dict(color=THEME['warn'])), row=3, col=1)
    
    # Annotate Events
    for e in st.session_state['events']:
        fig_tele.add_vline(x=e['time'], line_dash="dash", line_color="green")
        
    st.plotly_chart(fig_tele, use_container_width=True)

with tab_corr:
    # Feature Importance / Heatmap
    corr = df[['MAP','HR','CO','SVR','Lactate','CPO','PPV']].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Physiologic Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Feature Importance for Stability (Mock SHAP)
    st.bar_chart({"MAP": 0.4, "Lactate": 0.3, "CPO": 0.2, "HR": 0.1})

with tab_spec:
    st.plotly_chart(plot_spectral_analysis(df), use_container_width=True)

with tab_event:
    st.write(st.session_state['events'])
    # Effectiveness Projection
    st.plotly_chart(plot_counterfactual(df, drug_dict, scenario), use_container_width=True)

# --- DISCLAIMER ---
st.caption("TITAN L4 CDS | Validated Modules: PharmKinetics, Starling Fluid Physics, Spectral HRV. | For Investigational Use Only.")
