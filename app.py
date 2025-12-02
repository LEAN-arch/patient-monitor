import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any, Optional, List

# ==========================================
# 1. STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="TITAN | ICU (Clinical)", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. THEMES & CSS (Light Mode)
# ==========================================
THEME_CSS = """
<style>
:root{
  --bg: #f1f5f9;
  --card-bg: #ffffff;
  --muted: #64748b;
  --text: #0f172a;
  --accent: #0284c7;
  --warn: #d97706;
  --danger: #dc2626;
  --success: #16a34a;
}
.stApp { background-color: var(--bg); color: var(--text); }
.titan-card {
  background: var(--card-bg);
  border-radius: 8px;
  padding: 10px;
  border: 1px solid #e2e8f0;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
  margin-bottom: 10px;
}
.kpi-lbl { font-size:0.75rem; color:var(--muted); font-weight:700; text-transform:uppercase; }
.kpi-val { font-size:1.25rem; font-weight:800; font-family: 'Roboto Mono', monospace; color: var(--text); }
.alert-header { 
    background: #ffffff; 
    border-left: 6px solid var(--accent); 
    padding: 15px; 
    margin-bottom: 15px; 
    display:flex; 
    justify-content:space-between; 
    align-items:center; 
    gap:8px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    border-radius: 6px;
}
.section-header { 
    font-size:1.1rem; 
    font-weight:800; 
    color:var(--text); 
    border-bottom:2px solid #e2e8f0; 
    margin-top:25px; 
    margin-bottom:15px; 
    padding-bottom:8px; 
}
.small { color: var(--muted); font-size:0.9rem; }
</style>
"""

# ==========================================
# 3. UTILS & PHYSIOLOGY ENGINE
# ==========================================

# Theme constants for plotting (High Contrast for Light Mode)
PLOT_THEME = {
    "bg": "#ffffff",
    "map": "#be185d",      # Magenta/Rose
    "ci": "#059669",       # Emerald Green
    "do2": "#7c3aed",      # Violet
    "svr": "#d97706",      # Amber
    "ok": "rgba(22, 163, 74, 0.1)",
    "warn": "rgba(217, 119, 6, 0.1)",
    "crit": "rgba(220, 38, 38, 0.1)",
    "text": "#1e293b",
    "grid": "#e2e8f0"
}

# Clinical thresholds
CLIN_THRESH = {
    "MAP_TARGET": 65.0,
    "MAP_CRIT": 55.0,
    "LACTATE_ELEVATED": 2.0,
    "LACTATE_HIGH": 4.0,
    "CI_LOW": 2.0,
    "SI_WARN": 0.9,
    "ALARM_MAP_DURATION": 10
}

class DigitalTwin:
    """
    Compact hemodynamic twin with parameters for preload/contractility/afterload
    and simple lactate kinetics.
    """
    def __init__(self,
                 preload: float = 12.0,
                 contractility: float = 1.0,
                 afterload: float = 1.0,
                 lactate_half_life_min: float = 120.0):
        self.preload = float(preload)
        self.contractility = float(contractility)
        self.afterload = float(afterload)
        self.lactate_half_life_min = float(lactate_half_life_min)

    def copy(self) -> "DigitalTwin":
        return DigitalTwin(self.preload, self.contractility, self.afterload, self.lactate_half_life_min)

    def step(self, sepsis_severity: float = 0.0, dt_min: float = 1.0) -> Dict[str, float]:
        sev = float(np.clip(sepsis_severity, 0.0, 1.0))
        eff_afterload = self.afterload * max(0.01, 1.0 - 0.7 * sev)
        eff_preload = self.preload * max(0.01, 1.0 - 0.4 * sev)

        k_preload = 8.0
        sv_max = 100.0 * self.contractility
        sv = sv_max * (eff_preload ** 2 / (eff_preload ** 2 + k_preload ** 2))

        map_proxy = (sv * 75.0 * eff_afterload * 0.05) + 5.0
        hr_set = 70.0
        hr_drive = float(np.clip((90.0 - map_proxy) * 1.8, 0.0, 160.0))
        hr = hr_set + hr_drive

        co = (hr * sv) / 1000.0
        svr = 1200.0 * eff_afterload
        map_val = (co * svr / 80.0) + 5.0

        do2 = co * 1.34 * 12.0 * 0.98 * 10.0

        lac_gen = max(0.0, (400.0 - do2) * 0.01) + 0.01
        clearance_k = np.log(2) / max(1.0, self.lactate_half_life_min)
        lac_clear_fraction = clearance_k * dt_min

        return {
            "HR": float(hr),
            "SV": float(sv),
            "CO": float(co),
            "MAP": float(map_val),
            "SVR": float(svr),
            "DO2": float(do2),
            "Lac_Gen": float(lac_gen),
            "Lac_Clear_frac": float(lac_clear_fraction),
            "Preload": float(eff_preload)
        }

def predict_horizon(base_twin: DigitalTwin, last_sev: float, horizon: int = 30) -> Dict[str, Any]:
    """Forecast hemodynamic variables under different intervention scenarios."""
    nat, fluid, press, inot = [], [], [], []
    for i in range(horizon):
        sev = float(np.clip(last_sev + 0.001 * i, 0.0, 1.0))

        t_nat = base_twin.copy()
        s_nat = t_nat.step(sev)
        nat.append({"MAP": s_nat["MAP"], "CI": s_nat["CO"] / 1.8, "Lactate": s_nat["Lac_Gen"]})

        t_fl = base_twin.copy(); t_fl.preload *= 1.30
        s_fl = t_fl.step(sev)
        fluid.append({"MAP": s_fl["MAP"], "CI": s_fl["CO"] / 1.8, "Lactate": s_fl["Lac_Gen"]})

        t_pr = base_twin.copy(); t_pr.afterload *= 1.40
        s_pr = t_pr.step(sev)
        press.append({"MAP": s_pr["MAP"], "CI": s_pr["CO"] / 1.8, "Lactate": s_pr["Lac_Gen"]})

        t_in = base_twin.copy(); t_in.contractility *= 1.15
        s_in = t_in.step(sev)
        inot.append({"MAP": s_in["MAP"], "CI": s_in["CO"] / 1.8, "Lactate": s_in["Lac_Gen"]})

    def arr(listdicts, key):
        return np.array([d[key] for d in listdicts], dtype=float)

    return {
        "time": np.arange(horizon),
        "nat": {"MAP": arr(nat, "MAP"), "CI": arr(nat, "CI"), "Lactate": arr(nat, "Lactate")},
        "fluid": {"MAP": arr(fluid, "MAP"), "CI": arr(fluid, "CI"), "Lactate": arr(fluid, "Lactate")},
        "press": {"MAP": arr(press, "MAP"), "CI": arr(press, "CI"), "Lactate": arr(press, "Lactate")},
        "inot": {"MAP": arr(inot, "MAP"), "CI": arr(inot, "CI"), "Lactate": arr(inot, "Lactate")},
    }

def simulate_titan_data(mins: int = 720,
                        twin: Optional[DigitalTwin] = None,
                        seed: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    base = twin.copy() if twin is not None else DigitalTwin()
    sepsis = np.linspace(0.0, 0.85, mins)
    noise = rng.normal(0.0, 0.02, size=mins)

    rows = []
    curr_lac = 1.0
    for i in range(mins):
        sev = float(np.clip(sepsis[i] + noise[i], 0.0, 1.0))
        s = base.step(sev, dt_min=1.0)
        curr_lac = curr_lac * (1.0 - s["Lac_Clear_frac"]) + s["Lac_Gen"]
        s["Lactate"] = float(max(0.4, curr_lac))
        rows.append(s)

    df = pd.DataFrame(rows)
    df.index.name = "minute"

    df["CI"] = df["CO"] / 1.8  # normalize CO to CI assuming BSA ~1.8
    df["SVRI"] = df["SVR"] * 1.8
    df["PP"] = df["SV"] / 1.5
    df["MAP"] = df["MAP"].replace(0, np.nan).fillna(method="ffill").fillna(1.0)
    df["SI"] = df["HR"] / df["MAP"]

    df["HR_diff"] = df["HR"].diff().fillna(0.0)
    df["Entropy"] = df["HR_diff"].rolling(window=60, min_periods=1).std().fillna(0.0)

    # Urine proxy via sigmoid of MAP
    def urine_sigmoid(m):
        center = CLIN_THRESH["MAP_TARGET"]
        slope = 0.15
        norm = 1.0 / (1.0 + np.exp(-slope * (m - center)))
        return (norm * 2.0)

    df["Urine"] = urine_sigmoid(df["MAP"]) + rng.normal(0.0, 0.06, size=len(df))
    df["Urine"] = df["Urine"].clip(lower=0.0)

    # PCA for embedding (return PC1/PC2)
    pca_feats = ["HR", "MAP", "CI", "SVR"]
    pca_input = df[pca_feats].fillna(method="ffill").fillna(0.0)
    coords = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(pca_input))
    df["PC1"], df["PC2"] = coords[:, 0], coords[:, 1]

    # Predictions: build forecast dictionary using helper
    preds = predict_horizon(base, sepsis[-1], horizon=30)

    return df, preds

def compute_alerts(df: pd.DataFrame, lookback_min: int = 15) -> Dict[str, Any]:
    out = {"alerts": []}
    if df is None or df.shape[0] == 0:
        return out

    lookback = min(lookback_min, len(df))
    window = df.iloc[-lookback:]
    curr = window.iloc[-1]

    # Sustained Hypotension
    below = window["MAP"] < CLIN_THRESH["MAP_TARGET"]
    if below.sum() >= CLIN_THRESH["ALARM_MAP_DURATION"]:
        severity = "critical" if window["MAP"].min() < CLIN_THRESH["MAP_CRIT"] else "warning"
        out["alerts"].append({
            "type": "SUSTAINED_HYPOTENSION",
            "severity": severity,
            "message": f"MAP < {CLIN_THRESH['MAP_TARGET']} mmHg for {int(below.sum())} min.",
            "suggested_action": "Consider vasopressors and reassess fluids."
        })

    # Low CI
    if curr.get("CI", 99.0) < CLIN_THRESH["CI_LOW"]:
        out["alerts"].append({
            "type": "LOW_CI",
            "severity": "warning",
            "message": f"Low cardiac index: {curr.get('CI', 0):.2f}",
            "suggested_action": "Consider inotrope or optimize preload."
        })

    # Rising Lactate
    if "Lactate" in window.columns and len(window) >= 5:
        x = np.arange(len(window))
        y = window["Lactate"].values
        A = np.vstack([x, np.ones_like(x)]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        slope_hr = m * 60.0
        if slope_hr > 0.3:
            out["alerts"].append({
                "type": "LACTATE_RISING",
                "severity": "warning",
                "message": f"Lactate rising ~{slope_hr:.2f} mmol/L/hr",
                "suggested_action": "Assess perfusion, consider increasing DO2."
            })

    # Shock Index
    if curr.get("SI", 0.0) >= CLIN_THRESH["SI_WARN"]:
        out["alerts"].append({
            "type": "ELEVATED_SHOCK_INDEX",
            "severity": "warning",
            "message": f"Shock index {curr.get('SI',0):.2f}",
            "suggested_action": "Reassess for occult shock."
        })

    return out

# ==========================================
# 4. CLINICAL LOGIC
# ==========================================
def build_action_plan(df: pd.DataFrame, alerts: Dict[str, Any]) -> List[Dict[str, str]]:
    plan = []
    if not alerts.get("alerts"):
        plan.append({"priority": "low", "action": "Continue monitoring", "rationale": "No active alerts."})
        return plan

    for a in alerts["alerts"]:
        t = a["type"]
        sev = a["severity"]
        if t == "SUSTAINED_HYPOTENSION" and sev == "critical":
            plan.append({"priority": "critical", "action": "Start/Titrate Vasopressor (Norepinephrine)", "rationale": a["message"]})
        elif t == "SUSTAINED_HYPOTENSION":
            plan.append({"priority": "high", "action": "Assess Fluid Responsiveness (PLR/Echo)", "rationale": a["message"]})
        elif t == "LOW_CI":
            plan.append({"priority": "high", "action": "Consider Inotrope (Dobutamine)", "rationale": a["message"]})
        elif t == "LACTATE_RISING":
            plan.append({"priority": "high", "action": "Investigate Perfusion & Source Control", "rationale": a["message"]})
        elif t == "ELEVATED_SHOCK_INDEX":
            plan.append({"priority": "medium", "action": "Rapid Bedside Assessment", "rationale": a["message"]})
        else:
            plan.append({"priority": "low", "action": a.get("suggested_action", "Review"), "rationale": a.get("message", "")})

    seen = set()
    dedup = []
    for p in plan:
        key = (p["action"], p["priority"])
        if key not in seen:
            seen.add(key)
            dedup.append(p)
    return dedup

# ==========================================
# 5. VISUALIZATIONS
# ==========================================
def _base_layout(fig: go.Figure, height=260, title=None) -> go.Figure:
    fig.update_layout(template="plotly_white",
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=8,r=8,t=30 if title else 6,b=6))
    if title:
        fig.update_layout(title=title)
    return fig

def sparkline(df: pd.DataFrame, col: str, color: str, low: float, high: float) -> go.Figure:
    data = df[col].iloc[-60:]
    fig = go.Figure()
    if data.empty:
        return _base_layout(fig, height=48)
    
    # Calculate RGBA color string manually to avoid errors
    c = color.lstrip('#')
    if len(c) == 6:
        r = int(c[0:2], 16)
        g = int(c[2:4], 16)
        b = int(c[4:6], 16)
        fill_color_str = f"rgba({r}, {g}, {b}, 0.1)"
    else:
        fill_color_str = "rgba(100,100,100,0.1)" # Fallback

    # Background SPC zone
    fig.add_shape(type="rect", x0=data.index[0], x1=data.index[-1], y0=low, y1=high, 
                  fillcolor="rgba(0,0,0,0.03)", line_width=0, layer="below")
    
    fig.add_trace(go.Scatter(x=data.index, y=data.values, mode="lines", 
                             line=dict(color=color, width=2), fill="tozeroy", 
                             fillcolor=fill_color_str))
    _base_layout(fig, height=48)
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return fig

def predictive_compass(df: pd.DataFrame, preds: Dict[str, Any], curr_idx: int) -> go.Figure:
    start = max(0, curr_idx - 60)
    window = df.iloc[start:curr_idx]
    cur = window.iloc[-1] if len(window) else df.iloc[-1]
    fig = go.Figure()
    
    # Diagnostic Zones
    fig.add_shape(type="rect", x0=0, y0=0, x1=2.5, y1=CLIN_THRESH["MAP_TARGET"], fillcolor=PLOT_THEME["crit"], layer="below", line_width=0)
    fig.add_shape(type="rect", x0=2.5, y0=CLIN_THRESH["MAP_TARGET"], x1=6.0, y1=180, fillcolor=PLOT_THEME["ok"], layer="below", line_width=0)
    
    if len(window)>0:
        fig.add_trace(go.Scatter(x=window["CI"], y=window["MAP"], mode="lines", line=dict(color="#94a3b8", dash="dot"), name="history"))
    fig.add_trace(go.Scatter(x=[cur["CI"]], y=[cur["MAP"]], mode="markers", marker=dict(color="#0f172a", size=12, symbol="x"), name="current"))
    
    # Predicted endpoints
    for label, key, color in [("fluid","fluid",PLOT_THEME["ci"]), ("press","press",PLOT_THEME["map"]), ("inot","inot",PLOT_THEME["do2"])]:
        val_map = float(preds[key]["MAP"][-1])
        base_ci = float(max(0.2, cur["CI"]))
        dx = 0.2 if label=="press" else (0.45 if label=="inot" else 0.6)
        fig.add_annotation(x=base_ci+dx, y=val_map, ax=cur["CI"], ay=cur["MAP"], showarrow=True, arrowhead=2, arrowcolor=color, text=label.upper(), font=dict(color=color))
    
    _base_layout(fig, height=340, title="<b>Predictive Compass</b>")
    fig.update_xaxes(title_text="CI (L/min/m²)", range=[0.5,6.0], gridcolor=PLOT_THEME["grid"])
    fig.update_yaxes(title_text="MAP (mmHg)", range=[30,160], gridcolor=PLOT_THEME["grid"])
    fig.update_layout(showlegend=False)
    return fig

def multi_scenario_horizon(preds: Dict[str, Any]) -> go.Figure:
    h = len(preds["time"])
    t = np.arange(h)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=preds["nat"]["MAP"], line=dict(color="#94a3b8", dash="dot"), name="natural"))
    fig.add_trace(go.Scatter(x=t, y=preds["fluid"]["MAP"], line=dict(color=PLOT_THEME["ci"]), name="fluid"))
    fig.add_trace(go.Scatter(x=t, y=preds["press"]["MAP"], line=dict(color=PLOT_THEME["map"]), name="pressor"))
    fig.add_trace(go.Scatter(x=t, y=preds["inot"]["MAP"], line=dict(color=PLOT_THEME["do2"]), name="inotrope"))
    fig.add_hline(y=CLIN_THRESH["MAP_TARGET"], line_color=PLOT_THEME["crit"], line_dash="dot")
    _base_layout(fig, height=300, title="<b>Intervention Horizon - MAP</b>")
    fig.update_xaxes(title_text="Minutes ahead", gridcolor=PLOT_THEME["grid"])
    fig.update_yaxes(title_text="MAP (mmHg)", gridcolor=PLOT_THEME["grid"])
    return fig

def organ_radar(df: pd.DataFrame, curr_idx: int) -> go.Figure:
    cur = df.iloc[max(0, curr_idx-1)]
    r_renal = float(np.clip((CLIN_THRESH["MAP_TARGET"] - cur["MAP"]) / 25.0, 0, 1))
    r_card = float(np.clip((cur["HR"] - 100)/60.0, 0, 1))
    r_meta = float(np.clip((cur["Lactate"] - 1.5)/4.0, 0, 1))
    r_perf = float(np.clip((2.2 - cur["CI"]) / 1.5, 0, 1))
    R = [r_renal, r_card, r_meta, r_perf, r_renal]
    theta = ["Renal","Cardiac","Metabolic","Perfusion","Renal"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[0.2]*5, theta=theta, fill="toself", fillcolor=PLOT_THEME["ok"], line=dict(width=0)))
    fig.add_trace(go.Scatterpolar(r=R, theta=theta, fill="toself", fillcolor=PLOT_THEME["crit"], line=dict(color="red", width=2)))
    _base_layout(fig, height=280, title="<b>Organ Risk Topology</b>")
    fig.update_polars(radialaxis=dict(visible=False, range=[0,1]), bgcolor="rgba(0,0,0,0)")
    fig.update_layout(showlegend=False)
    return fig

# ==========================================
# 6. APP MAIN
# ==========================================

# Apply Styling
st.markdown(THEME_CSS, unsafe_allow_html=True)

# Load Data
@st.cache_data
def load(mins=720):
    df, preds = simulate_titan_data(mins=mins, seed=42)
    return df, preds

df, preds = load(720)

# Sidebar
with st.sidebar:
    st.header("TITAN Controls")
    st.write("Inspect timeline and adjust model knobs.")
    minute = st.slider("Minute", 1, len(df), len(df))
    
    st.markdown("**Model Knobs**")
    preload = st.number_input("Preload", value=12.0, step=0.5)
    contractility = st.number_input("Contractility", value=1.0, step=0.05)
    afterload = st.number_input("Afterload", value=1.0, step=0.05)
    
    if st.button("Re-run Sim"):
        twin = DigitalTwin(preload=preload, contractility=contractility, afterload=afterload)
        df, preds = simulate_titan_data(mins=720, twin=twin, seed=42)
        st.experimental_rerun()

# Processing
idx = max(0, minute - 1)
row = df.iloc[idx]
alerts = compute_alerts(df.iloc[:idx+1], lookback_min=15)
plan = build_action_plan(df, alerts)

# Header Logic
def compute_header(row):
    status = "STABLE"
    color = "var(--accent)"
    action = "Monitor"
    rationale = "Vitals in acceptable range."
    if row["MAP"] < CLIN_THRESH["MAP_TARGET"]:
        status = "HYPOTENSION"
        action = "Fluids ± Vasopressor" if row["CI"] >= CLIN_THRESH["CI_LOW"] else "Vasopressor ± Inotrope"
        color = "var(--danger)" if row["MAP"] < CLIN_THRESH["MAP_CRIT"] else "var(--warn)"
        rationale = f"MAP {row['MAP']:.0f} mmHg. CI {row['CI']:.2f}."
    if row["Lactate"] >= CLIN_THRESH["LACTATE_ELEVATED"]:
        status = "PERFUSION RISK" if status=="STABLE" else status
        rationale += f" Lactate {row['Lactate']:.2f} mmol/L."
    return status, action, color, rationale

status, action, color, rationale = compute_header(row)

# Render UI
st.markdown(f"<div class='alert-header' style='border-color:{color}'><div style='font-size:1.1rem;font-weight:800;color:{color}'>{status}</div><div style='font-weight:700;color:var(--text)'>ACTION: {action}</div></div>", unsafe_allow_html=True)
st.markdown(f"<div class='small'>{rationale}</div>", unsafe_allow_html=True)

# KPI Row
k1, k2, k3, k4, k5, k6 = st.columns(6, gap="small")
def kpi_viz(col, label, val, unit, color, dfcol, low, high, viz_key):
    with col:
        st.markdown(f"<div class='titan-card' style='border-top:3px solid {color}'><div class='kpi-lbl'>{label}</div><div class='kpi-val' style='color:{color}'>{val}<span style='font-size:0.72rem;color:var(--muted)'> {unit}</span></div></div>", unsafe_allow_html=True)
        fig = sparkline(df, dfcol, color, low, high)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=viz_key)

kpi_viz(k1, "MAP", f"{row['MAP']:.0f}", "mmHg", PLOT_THEME["map"], "MAP", 55, 110, "k1")
kpi_viz(k2, "CI", f"{row['CI']:.2f}", "L/min/m²", PLOT_THEME["ci"], "CI", 1.0, 4.5, "k2")
kpi_viz(k3, "SVR", f"{row['SVRI']:.0f}", "dyn", PLOT_THEME["svr"], "SVRI", 400, 1400, "k3")
kpi_viz(k4, "SV", f"{row['SV']:.0f}", "mL", PLOT_THEME["ci"], "SV", 30, 120, "k4")
kpi_viz(k5, "Lactate", f"{row['Lactate']:.2f}", "mmol/L", PLOT_THEME["do2"], "Lactate", 0.4, 6.0, "k5")
kpi_viz(k6, "Entropy", f"{row['Entropy']:.2f}", "σ", "#64748b", "Entropy", 0.0, 2.0, "k6")

# Predictive Panels
st.markdown('<div class="section-header">🔮 Predictive Hemodynamics & Forecasts</div>', unsafe_allow_html=True)
left, right = st.columns((1,1), gap="large")
with left:
    st.markdown("<div class='titan-card'>", unsafe_allow_html=True)
    fig_c = predictive_compass(df, preds, idx)
    st.plotly_chart(fig_c, use_container_width=True, config={"displayModeBar": False}, key="p_compass")
    st.markdown("</div>", unsafe_allow_html=True)
with right:
    st.markdown("<div class='titan-card'>", unsafe_allow_html=True)
    fig_m = multi_scenario_horizon(preds)
    st.plotly_chart(fig_m, use_container_width=True, config={"displayModeBar": False}, key="p_horizon")
    st.markdown("</div>", unsafe_allow_html=True)

# Organ Risk Panels
st.markdown('<div class="section-header">🫀 Organ Mechanics & Risk</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4, gap="large")
with c1:
    st.plotly_chart(organ_radar(df, idx), use_container_width=True, config={"displayModeBar": False}, key="o_radar")
with c2:
    st.plotly_chart(sparkline(df, "SV", PLOT_THEME["ci"], 30, 120), use_container_width=True, config={"displayModeBar": False}, key="o_sv")
with c3:
    st.plotly_chart(sparkline(df, "DO2", PLOT_THEME["do2"], 200, 700), use_container_width=True, config={"displayModeBar": False}, key="o_do2")
with c4:
    st.plotly_chart(sparkline(df, "Urine", "#64748b", 0.0, 2.0), use_container_width=True, config={"displayModeBar": False}, key="o_urine")

# Action Plan
st.markdown('<div class="section-header">🩺 Suggested Actions (Prioritized)</div>', unsafe_allow_html=True)
for i, p in enumerate(plan):
    pr = p["priority"].upper()
    color = "var(--danger)" if pr=="CRITICAL" else "var(--warn)" if pr=="HIGH" else "var(--text)"
    st.markdown(f"- <strong style='color:{color}'>{pr}</strong> — {p['action']}  \n  _{p['rationale']}_", unsafe_allow_html=True)

# Debug
with st.expander("Data Inspector"):
    st.write(f"Index: {idx}")
    st.dataframe(df.iloc[max(0, idx-10):idx+1])
