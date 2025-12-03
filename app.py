import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import welch
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# =========================================================
# PAGE CONFIG / THEME
# =========================================================
st.set_page_config(
    page_title="TITAN | LEVEL 4 CDS",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧬"
)

THEME = {
    "bg": "#f8fafc",
    "card_bg": "#ffffff",
    "text_main": "#0f172a",
    "text_muted": "#64748b",
    "border": "#e2e8f0",
    "crit": "#dc2626",
    "warn": "#d97706",
    "ok": "#059669",
    "info": "#2563eb",
    "resp": "#7c3aed",
    "hemo": "#0891b2",
    "ai": "#be185d"
}

STYLING = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Roboto+Mono:wght@500;700&display=swap');

    .stApp {{
        background-color: {THEME['bg']};
        font-family: 'Inter', sans-serif;
    }}

    @keyframes flash-crit {{
      0%   {{ box-shadow: 0 0 0 0 rgba(220,38,38,0.6); }}
      70%  {{ box-shadow: 0 0 0 10px rgba(220,38,38,0); }}
      100% {{ box-shadow: 0 0 0 0 rgba(220,38,38,0); }}
    }}
    .crit-pulse {{
        animation: flash-crit 2s infinite;
        border: 2px solid {THEME['crit']}!important;
    }}

    div[data-testid="stMetric"] {{
        background-color: {THEME['card_bg']};
        padding: 8px 12px;
        border-radius: 8px;
        border: 1px solid {THEME['border']};
    }}

    .status-banner {{
        background-color: white;
        padding: 12px 20px;
        border-radius: 10px;
        border-left: 6px solid;
        box-shadow: 0 4px 6px rgba(0,0,0,0.08);
        margin-bottom: 12px;
    }}
</style>
"""
st.markdown(STYLING, unsafe_allow_html=True)

# =========================================================
# 1. CORE SIMULATION ENGINES
# =========================================================

class PhysiologyEngine:
    """Advanced physiologic system generator with noise smoothing and soft boundaries."""
    @staticmethod
    def pink_noise(n):
        return np.convolve(np.random.normal(0, 0.5, n), np.ones(7)/7, mode='same')

    @staticmethod
    def brownian_bridge(n, start, end, vol=1.0, seed=42):
        np.random.seed(seed)
        t = np.linspace(0, 1, n)
        dW = np.random.normal(0, np.sqrt(1/n), n)
        W = np.cumsum(dW)
        BB = start + W - t*(W[-1] - (end - start))
        return BB + PhysiologyEngine.pink_noise(n) * vol

    @staticmethod
    def resp(p_shunt, fio2):
        p_ideal = (fio2 * 713) - (40 / 0.8)
        pao2 = max(40, p_ideal * (1 - p_shunt))
        spo2 = 100 / (1 + 23400 / (pao2**3 + 150*pao2))
        return pao2, spo2


class PharmaEngine:
    """Drug effect interpreter with physiologic soft limits."""
    @staticmethod
    def apply(map_val, co, hr, svr, d):
        # Norepi
        ne = d['norepi'] * 2.5
        svr += ne * 700
        map_val += ne * 22

        # Vasopressin
        va = d['vaso'] * 4
        svr += va * 550
        map_val += va * 15

        # Dobutamine
        db = d['dobu'] * 1.7
        co += db * 2.0
        hr += db * 12
        svr -= db * 250

        # Beta-blockers
        bb = d['bb']
        hr -= bb * 15
        co -= bb * 0.6

        # Clamp-safe values
        map_val = np.clip(map_val, 35, 140)
        hr = np.clip(hr, 35, 150)
        co = np.clip(co, 1.5, 12)
        svr = np.clip(svr, 300, 3000)

        return map_val, co, hr, svr


class PatientSimulator:
    def __init__(self, mins=360):
        self.mins = mins
        self.time = np.arange(mins)

    @st.cache_data(show_spinner=False)
    def simulate(self, scenario, drugs, fluids, age, chronic, drift, seed):
        # Baseline profiles
        profiles = {
            "Healthy":              (65,70,  85,82,  5.0,5.0,   0.05),
            "Compensated Sepsis":   (85,95,  80,75,  6.0,7.0,   0.10),
            "Vasoplegic Shock":     (110,125, 65,55, 7.0,8.0,  0.15),
            "Cardiogenic Shock":    (95,105, 70,55,  3.5,2.8,  0.20)
        }

        hr0, hr1, map0, map1, co0, co1, shunt = profiles[scenario]

        drift_factor = np.linspace(1.0, drift, self.mins)

        # Hemodynamics
        hr  = PhysiologyEngine.brownian_bridge(self.mins, hr0, hr1, 1.0, seed) * drift_factor
        map_raw = PhysiologyEngine.brownian_bridge(self.mins, map0, map1, 0.8, seed+1) / drift_factor
        co_raw  = PhysiologyEngine.brownian_bridge(self.mins, co0, co1, 0.25, seed+2) / drift_factor

        # Respiratory
        rr = PhysiologyEngine.brownian_bridge(self.mins, 14, 18, 2.0, seed+3)
        paco2 = 40 + (16 - rr)*1.5

        pao2, spo2 = [], []
        for _ in range(self.mins):
            pa, sp = PhysiologyEngine.resp(shunt, drugs['fio2'])
            pao2.append(pa + np.random.normal(0,4))
            spo2.append(sp)

        # Fluid effect — dynamic Starling model
        ppv = 10 + 3*np.sin(self.time/12)
        fluid_gain = (fluids / 500) * (1.2 if np.mean(ppv) > 12 else 0.15)
        co_raw += fluid_gain

        # SVR
        svr_raw = ((map_raw - 8) / np.maximum(co_raw, 0.8)) * 750
        svr_raw = np.clip(svr_raw, 300, 2500)

        # Apply drugs
        MAP, CO, HR, SVR = [], [], [], []
        for i in range(self.mins):
            m, c, h, s = PharmaEngine.apply(map_raw[i], co_raw[i], hr[i], svr_raw[i], drugs)
            MAP.append(m); CO.append(c); HR.append(h); SVR.append(s)

        MAP = np.array(MAP); CO = np.array(CO); HR = np.array(HR); SVR = np.array(SVR)

        # Oxygen transport
        hb = 12
        do2 = CO * hb * 1.34 * (np.array(spo2)/100) * 10
        vo2 = do2 * 0.25

        # Lactate
        lactate = np.zeros(self.mins)
        lac = 1.0
        for i in range(self.mins):
            prod = 0.1 if do2[i] < 380 else 0
            clr = 0.03 if do2[i] > 520 else 0.01
            lac = max(0.4, lac + prod - clr)
            lactate[i] = lac

        df = pd.DataFrame({
            "Time": self.time,
            "HR": HR, "MAP": MAP, "CO": CO, "SVR": SVR,
            "Lactate": lactate,
            "SpO2": spo2, "PaO2": pao2, "PaCO2": paco2,
            "PPV": ppv, "RR": rr,
            "DO2": do2, "VO2": vo2,
        })

        df["CPO"] = (df["MAP"] * df["CO"]) / 451
        df["SV"] = (df["CO"] * 1000) / df["HR"]
        df["MSI"] = df["HR"] / df["MAP"]

        df["Creatinine"] = np.linspace(0.8, 1.3 if MAP[-1] < 60 else 1.0, self.mins)
        return df


# =========================================================
# 2. AI LAYER
# =========================================================
class AI:
    @staticmethod
    def score(row):
        penalties = [
            20 if row["MAP"] < 65 else 0,
            20 if row["Lactate"] > 2 else 0,
            25 if row["CPO"] < 0.6 else 0,
            10 if row["SpO2"] < 90 else 0,
            10 if row["HR"] > 115 else 0,
        ]
        return max(0, 100 - sum(penalties))

    @staticmethod
    def predict(df, col, horizon=30):
        recent = df[col].iloc[-40:].values
        X = np.arange(len(recent)).reshape(-1,1)
        model = LinearRegression().fit(X, recent)
        return model.predict([[len(recent)+horizon]])[0]

    @staticmethod
    def phenotype(df):
        X = df[["CO","SVR","Lactate"]].iloc[-90:]
        X_scaled = StandardScaler().fit_transform(X)

        kmeans = KMeans(n_clusters=3, n_init="auto", random_state=42).fit(X_scaled)
        c = kmeans.cluster_centers_[kmeans.labels_[-1]]

        if c[1] > 0.5:     return "Vasoconstricted / Low Output"
        if c[0] > 0.4:     return "Vasoplegic / High Output"
        return "Metabolic Instability"


# =========================================================
# 3. VISUALIZATIONS
# =========================================================
def plot_attractor(df):
    d = df.iloc[-70:]
    fig = go.Figure(go.Scatter3d(
        x=d["CPO"], y=d["SVR"], z=d["Lactate"],
        mode="lines+markers",
        marker=dict(size=4, color=d.index, colorscale="Viridis")
    ))
    fig.update_layout(height=280, margin=dict(l=0,r=0,b=0,t=30))
    return fig

def plot_spectrum(df):
    data = df["HR"].iloc[-180:].to_numpy()
    f, Pxx = welch(data, fs=1/60)

    fig = px.line(x=f, y=Pxx)
    fig.update_layout(title="HRV Spectral Density", height=250)
    fig.update_xaxes(title="Hz")
    fig.update_yaxes(title="Power")
    return fig


# =========================================================
# 4. SIDEBAR CONTROLS
# =========================================================
with st.sidebar:
    st.title("TITAN | L4 CDS")

    seed = st.number_input("Random Seed", 1, 9999, 42)
    drift = st.slider("Scenario Drift", 1.0, 2.5, 1.0, 0.1)

    st.markdown("### 👤 Patient")
    age = st.slider("Age", 18, 95, 55)
    chronic = st.multiselect("Comorbidities", ["Heart Failure","COPD","CKD"])
    scenario = st.selectbox("Presentation", [
        "Healthy","Compensated Sepsis", "Vasoplegic Shock","Cardiogenic Shock"
    ])

    st.markdown("### 💉 Drugs")
    norepi = st.slider("Norepi", 0.0, 1.0, 0.0, 0.05)
    vaso   = st.slider("Vasopressin", 0.0, 0.06, 0.0, 0.01)
    dobu   = st.slider("Dobutamine", 0.0, 10.0, 0.0, 1.0)
    bb     = st.slider("Beta Blocker", 0.0, 1.0, 0.0, 0.05)
    fio2   = st.slider("FiO₂", 0.21, 1.0, 0.40)

    if "fluids" not in st.session_state:
        st.session_state.fluids = 0
    if st.button("Give 500 mL"):
        st.session_state.fluids += 500

    st.caption(f"Fluids Given: {st.session_state.fluids} mL")

# =========================================================
# 5. DATA GENERATION
# =========================================================
sim = PatientSimulator(mins=360)
drugs = {"norepi":norepi,"vaso":vaso,"dobu":dobu,"bb":bb,"fio2":fio2}

df = sim.simulate(scenario, drugs, st.session_state.fluids, age, chronic, drift, seed)

curr = df.iloc[-1]
stability = AI.score(curr)
phenotype = AI.phenotype(df)
pred_map = AI.predict(df, "MAP", 30)

# =========================================================
# 6. TOP BANNER
# =========================================================
alert = "crit-pulse" if stability < 50 else ""
st.markdown(f"""
<div class="status-banner" style="border-left-color:{THEME['ai']}">
    <div>
        <div style="font-size:0.85rem;color:{THEME['ai']};font-weight:800;">
            AI Phenotype: {phenotype}
        </div>
        <div style="font-size:1.7rem;font-weight:800;">{scenario.upper()}</div>
        <div style="color:{THEME['text_muted']};">Age {age} | Drift {drift}×</div>
    </div>
    <div>
        <div style="font-size:0.8rem;font-weight:700;text-align:right;">Stability Index</div>
        <div class="{alert}" style="
            font-size:2rem;font-weight:800;padding:5px 15px;border-radius:6px;
            text-align:right;display:inline-block;
        ">{stability:.0f}/100</div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 7. METRICS ROW
# =========================================================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("MAP", f"{curr.MAP:.0f}")
col2.metric("HR", f"{curr.HR:.0f}")
col3.metric("CO", f"{curr.CO:.2f} L/min")
col4.metric("CPO", f"{curr.CPO:.2f} W")
col5.metric("Lactate", f"{curr.Lactate:.2f}")

# =========================================================
# 8. GRAPHS
# =========================================================
st.subheader("Hemo-Metabolic Attractor")
st.plotly_chart(plot_attractor(df), use_container_width=True)

st.subheader("Autonomic Tone")
st.plotly_chart(plot_spectrum(df), use_container_width=True)
