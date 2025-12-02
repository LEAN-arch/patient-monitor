# utils.py
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -------------------------
# THEME / CLINICAL CONSTANTS
# -------------------------
THEME = {
    "bg": "#000000",
    "card": "#111111",
    "grid": "#333333",
    "text": "#e0e0e0",
    "hr": "#00e5ff",
    "map": "#ff2975",
    "ci": "#00ff33",
    "svr": "#ff9900",
    "do2": "#8c1eff",
    "pca": "#ffffff",
    "zone_ok": "rgba(0, 255, 51, 0.08)",
    "zone_warn": "rgba(255, 195, 0, 0.08)",
    "zone_crit": "rgba(255, 41, 117, 0.08)",
}

# Clinical thresholds (easy to adjust)
CLIN_THRESH = {
    "MAP_OK": 65.0,
    "MAP_CRIT": 55.0,
    "CI_LOW": 2.0,
    "LACTATE_RAISE_RATE": 0.3,  # mM per hour considered clinically meaningful
    "ALARM_MAP_DURATION_MIN": 10,  # minutes below MAP_OK to trigger sustained alarm
}


def hex_to_rgba(hex_color: str, opacity: float = 0.1) -> str:
    """Convert '#RRGGBB' to 'rgba(r,g,b,opacity)'"""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"


# -------------------------
# DIGITAL TWIN: physiologic kernel
# -------------------------
class DigitalTwin:
    """
    Parameterized, deterministic digital twin of hemodynamics.

    Units:
    - preload: arbitrary preload units (higher -> higher SV)
    - contractility: multiplier on SV max
    - afterload: multiplier on vascular tone (SVR)
    """

    def __init__(
        self,
        preload: float = 12.0,
        contractility: float = 1.0,
        afterload: float = 1.0,
        lactate_clearance_half_life_min: float = 120.0,
    ):
        self.preload = float(preload)
        self.contractility = float(contractility)
        self.afterload = float(afterload)
        # lactate clearance modeled as first order (half-life in minutes)
        self.lactate_clearance_half_life_min = float(lactate_clearance_half_life_min)

    def step(self, sepsis_severity: float = 0.0, dt_min: float = 1.0) -> Dict[str, float]:
        """
        Return one timestep of physiology.

        Parameters
        ----------
        sepsis_severity : float
            0..1 severity that reduces preload and afterload (vasodilation & leak)
        dt_min : float
            timestep in minutes (for lactate clearance scaling)

        Returns
        -------
        dict of physiologic metrics
        """
        # ensure safe ranges
        sev = float(np.clip(sepsis_severity, 0.0, 1.0))
        eff_afterload = self.afterload * max(0.01, 1.0 - sev * 0.7)
        eff_preload = self.preload * max(0.01, 1.0 - sev * 0.4)

        # Frank-Starling-like saturation (same functional form as before, explicit constants)
        k_preload = 8.0
        sv_max = 100.0 * self.contractility
        sv = sv_max * (eff_preload ** 2 / (eff_preload ** 2 + k_preload ** 2))

        # Baroreflex/HR: MAP proxy used to drive HR; use a smooth relation
        map_proxy = (sv * 75.0 * eff_afterload * 0.05) + 5.0
        hr_set = 70.0
        hr_drive = float(np.clip((90.0 - map_proxy) * 1.8, 0.0, 140.0))
        hr = hr_set + hr_drive

        # CO, SVR, MAP
        co = (hr * sv) / 1000.0  # L/min
        svr_dyne = 1200.0 * eff_afterload
        map_val = (co * svr_dyne / 80.0) + 5.0

        # DO2 (oxygen delivery) rough proxy
        do2 = co * 1.34 * 12.0 * 0.98 * 10.0

        # lactate generation inversely related to DO2; include small baseline production
        lac_gen_rate = max(0.0, (400.0 - do2) * 0.01) + 0.01

        # lactate clearance factor per minute from half-life
        clearance_k = np.log(2) / max(1.0, self.lactate_clearance_half_life_min)  # per minute
        lac_clearance = clearance_k * dt_min

        return {
            "HR": float(hr),
            "SV": float(sv),
            "CO": float(co),
            "MAP": float(map_val),
            "SVR": float(svr_dyne),
            "DO2": float(do2),
            "Lac_Gen": float(lac_gen_rate),
            "Lac_Clear_k": float(lac_clearance),
            "Preload_Status": float(eff_preload),
        }

    def copy(self) -> "DigitalTwin":
        return DigitalTwin(
            preload=self.preload,
            contractility=self.contractility,
            afterload=self.afterload,
            lactate_clearance_half_life_min=self.lactate_clearance_half_life_min,
        )


# -------------------------
# SIMULATOR: runs the twin over time and produces derived metrics + predictions
# -------------------------
def simulate_titan_data(
    mins: int = 720, random_seed: Optional[int] = None, twin: Optional[DigitalTwin] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simulate TITAN time-series physiology. Returns DataFrame and prediction dict.

    Parameters
    ----------
    mins : int
        duration in minutes
    random_seed : Optional[int]
        deterministic RNG if set
    twin : Optional[DigitalTwin]
        base twin instance (copied internally). If None, a default twin is used.

    Returns
    -------
    df : pd.DataFrame
    preds : dict
    """
    rng = np.random.default_rng(random_seed)
    base = twin.copy() if twin is not None else DigitalTwin()
    history: List[Dict[str, float]] = []

    # sepsis severity ramp with physiologic noise
    sepsis = np.linspace(0.0, 0.85, mins)
    noise = rng.normal(0.0, 0.02, size=mins)

    curr_lac = 1.0
    for t in range(mins):
        sev = float(np.clip(sepsis[t] + noise[t], 0.0, 1.0))
        s = base.step(sev, dt_min=1.0)
        # lactate kinetics: production - clearance
        curr_lac = curr_lac * (1.0 - s["Lac_Clear_k"]) + s["Lac_Gen"]
        s["Lactate"] = float(max(0.4, curr_lac))  # allow baseline minimum
        history.append(s)

    df = pd.DataFrame(history)
    df.index.name = "minute"

    # Derived metrics
    # Cardiac Index (CI) = CO / BSA; assume BSA ~ 1.8 m^2 for normalization used historically
    df["CI"] = df["CO"] / 1.8
    df["SVRI"] = df["SVR"] * 1.8
    df["PP"] = df["SV"] / 1.5
    # MAP might have NaNs if something weird happens; ensure fill-forward
    df["MAP"] = df["MAP"].replace(0, np.nan).fillna(method="ffill").fillna(1.0)
    df["SI"] = df["HR"] / df["MAP"]

    # Entropy: rolling std of HR diffs (fast HRV proxy). Use rolling std for speed.
    df["HR_diff"] = df["HR"].diff().fillna(0.0)
    df["Entropy"] = df["HR_diff"].rolling(window=60, min_periods=1).std().fillna(0.0)

    # Urine production: smooth sigmoid of MAP -> urine (0..2 mL/kg/hr style proxy)
    # logistic center at MAP = 65, steeper slope to emphasize drop
    def urine_from_map(map_series: pd.Series) -> pd.Series:
        center = 65.0
        slope = 0.15
        # normalized 0..1
        norm = 1.0 / (1.0 + np.exp(-slope * (map_series - center)))
        # scale to 0..2 (arbitrary clinical-like units)
        return (norm * 2.0) + rng.normal(0.0, 0.08, size=len(map_series))

    df["Urine"] = urine_from_map(df["MAP"]).clip(lower=0.0)

    # PCA on core features (robust pipeline)
    pca_features = ["HR", "MAP", "CI", "SVR"]
    pca_input = df[pca_features].fillna(method="ffill").fillna(0.0)
    scaled = StandardScaler().fit_transform(pca_input)
    coords = PCA(n_components=2).fit_transform(scaled)
    df["PC1"], df["PC2"] = coords[:, 0], coords[:, 1]

    # Prediction horizon: compute three intervention trajectories for MAP & CI & Lactate
    preds = predict_horizon(base, sepsis[-1], horizon=30)

    return df, preds


def predict_horizon(base_twin: DigitalTwin, last_sev: float, horizon: int = 30) -> Dict[str, Any]:
    """
    Predict MAP/CI/Lactate trajectories for several interventions:
    - natural progression
    - fluid bolus (increase preload)
    - vasopressor (increase afterload)
    - inotrope (increase contractility)
    """
    pred_nat, pred_fluid, pred_press, pred_inot = [], [], [], []
    time_axis = np.arange(horizon)
    for i in range(horizon):
        sev = float(np.clip(last_sev + 0.001 * i, 0.0, 1.0))
        # natural
        twin_nat = base_twin.copy()
        s_nat = twin_nat.step(sev)
        pred_nat.append({"MAP": s_nat["MAP"], "CI": s_nat["CO"] / 1.8, "Lactate": s_nat["Lac_Gen"]})

        # fluid: transient preload increase (30%)
        twin_fluid = base_twin.copy()
        twin_fluid.preload *= 1.30
        s_fluid = twin_fluid.step(sev)
        pred_fluid.append({"MAP": s_fluid["MAP"], "CI": s_fluid["CO"] / 1.8, "Lactate": s_fluid["Lac_Gen"]})

        # pressor: increase afterload (40%) to restore tone (but increases SVR)
        twin_press = base_twin.copy()
        twin_press.afterload *= 1.40
        s_press = twin_press.step(sev)
        pred_press.append({"MAP": s_press["MAP"], "CI": s_press["CO"] / 1.8, "Lactate": s_press["Lac_Gen"]})

        # inotrope: increase contractility (15%)
        twin_inot = base_twin.copy()
        twin_inot.contractility *= 1.15
        s_inot = twin_inot.step(sev)
        pred_inot.append({"MAP": s_inot["MAP"], "CI": s_inot["CO"] / 1.8, "Lactate": s_inot["Lac_Gen"]})

    # assemble into arrays for plotting convenience
    def to_series(list_of_dicts: List[Dict[str, float]], key: str) -> np.ndarray:
        return np.array([d[key] for d in list_of_dicts], dtype=float)

    preds = {
        "time": time_axis,
        "nat": {"MAP": to_series(pred_nat, "MAP"), "CI": to_series(pred_nat, "CI"), "Lactate": to_series(pred_nat, "Lactate")},
        "fluid": {"MAP": to_series(pred_fluid, "MAP"), "CI": to_series(pred_fluid, "CI"), "Lactate": to_series(pred_fluid, "Lactate")},
        "press": {"MAP": to_series(pred_press, "MAP"), "CI": to_series(pred_press, "CI"), "Lactate": to_series(pred_press, "Lactate")},
        "inot": {"MAP": to_series(pred_inot, "MAP"), "CI": to_series(pred_inot, "CI"), "Lactate": to_series(pred_inot, "Lactate")},
    }
    return preds


# -------------------------
# ALERT ENGINE (clinical rules)
# -------------------------
def compute_alerts(df: pd.DataFrame, lookback_min: int = 15) -> Dict[str, Any]:
    """
    Compute clinical alerts from the timeseries.
    Returns a dict with alarm flags, durations, and textual rationale.
    """

    result: Dict[str, Any] = {"alerts": []}
    if df is None or len(df) == 0:
        return result

    lookback = min(lookback_min, len(df))
    window = df.iloc[-lookback:]
    curr = window.iloc[-1]

    # 1) MAP sustained low
    map_below = window["MAP"] < CLIN_THRESH["MAP_OK"]
    if map_below.sum() >= CLIN_THRESH["ALARM_MAP_DURATION_MIN"]:
        result["alerts"].append(
            {
                "type": "SUSTAINED_HYPOTENSION",
                "severity": "critical" if (window["MAP"].min() < CLIN_THRESH["MAP_CRIT"]) else "warning",
                "message": f"MAP < {CLIN_THRESH['MAP_OK']} mmHg for {int(map_below.sum())} min (min MAP {window['MAP'].min():.0f}).",
                "suggested_action": "Consider vasopressors and review fluid status.",
            }
        )

    # 2) Low cardiac index
    if curr.get("CI", 0.0) < CLIN_THRESH["CI_LOW"]:
        result["alerts"].append(
            {
                "type": "LOW_CI",
                "severity": "warning",
                "message": f"Cardiac index low: {curr.get('CI', 0.0):.2f} L/min/m².",
                "suggested_action": "Consider inotrope or optimize preload.",
            }
        )

    # 3) Lactate rising trend (estimate slope minutes->hours)
    # compute slope over lookback using linear regression (simple)
    if "Lactate" in window.columns and len(window) >= 5:
        x = np.arange(len(window))
        y = window["Lactate"].values
        # fit slope per minute
        A = np.vstack([x, np.ones_like(x)]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]  # m is mM per minute
        slope_per_hour = m * 60.0
        if slope_per_hour > CLIN_THRESH["LACTATE_RAISE_RATE"]:
            result["alerts"].append(
                {
                    "type": "LACTATE_RISING",
                    "severity": "warning",
                    "message": f"Lactate rising at {slope_per_hour:.2f} mM/hr (recent slope).",
                    "suggested_action": "Evaluate perfusion, consider increasing DO2.",
                }
            )

    # 4) Rapid HR increase (compensatory tachycardia)
    if curr.get("HR", 0.0) > 120:
        result["alerts"].append(
            {
                "type": "TACHYCARDIA",
                "severity": "warning",
                "message": f"HR {curr.get('HR'):.0f} bpm.",
                "suggested_action": "Evaluate arrhythmia, pain, hypovolemia, or sympathetic drive.",
            }
        )

    return result


# -------------------------
# PLOTTING HELPERS (clinically annotated)
# -------------------------
def _base_layout(fig: go.Figure, height: int = 240, title: Optional[str] = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=8, r=8, t=30 if title else 6, b=6),
    )
    if title:
        fig.update_layout(title=title)
    return fig


def plot_spark_spc(df: pd.DataFrame, col: str, color: str, thresh_low: float, thresh_high: float) -> go.Figure:
    """Compact sparkline with threshold band for the last 60 samples (fast rendering)."""
    data = df[col].iloc[-60:].copy()
    fig = go.Figure()
    if len(data) == 0:
        return _base_layout(fig, height=48)
    x0, x1 = data.index[0], data.index[-1]
    fig.add_shape(
        type="rect", x0=x0, x1=x1, y0=thresh_low, y1=thresh_high, fillcolor="rgba(255,255,255,0.03)", line_width=0, layer="below"
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data.values, mode="lines", line=dict(color=color, width=2), fill="tozeroy", fillcolor=hex_to_rgba(color, 0.12))
    )
    _base_layout(fig, height=48)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def plot_predictive_compass(df: pd.DataFrame, preds: Dict[str, Any], curr_time: int) -> go.Figure:
    """
    CI vs MAP scatter/trajectory showing current point and arrows for interventions.
    preds: output of predict_horizon()
    """
    start = max(0, curr_time - 60)
    window = df.iloc[start:curr_time].copy()
    cur = window.iloc[-1] if len(window) > 0 else df.iloc[-1]

    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=2.5, y1=CLIN_THRESH["MAP_OK"], fillcolor=THEME["zone_crit"], line_width=0, layer="below")
    fig.add_shape(type="rect", x0=2.5, y0=CLIN_THRESH["MAP_OK"], x1=6.0, y1=180, fillcolor=THEME["zone_ok"], line_width=0, layer="below")

    if len(window) > 0:
        fig.add_trace(go.Scatter(x=window["CI"], y=window["MAP"], mode="lines", line=dict(color="#555", width=1, dash="dot"), name="history"))

    fig.add_trace(go.Scatter(x=[cur["CI"]], y=[cur["MAP"]], mode="markers", marker=dict(color="white", size=10, symbol="x"), name="current"))

    # Get predicted endpoints (take final horizon value)
    fluid_map = preds["fluid"]["MAP"][-1]
    press_map = preds["press"]["MAP"][-1]
    inot_map = preds["inot"]["MAP"][-1]

    # arrows (scale on CI axis incrementally to avoid backwards arrows)
    base_ci = float(max(0.2, cur["CI"]))
    fig.add_annotation(x=base_ci + 0.4, y=float(fluid_map), ax=cur["CI"], ay=cur["MAP"], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor=THEME["ci"], text="FLUID", font=dict(color=THEME["ci"]))
    fig.add_annotation(x=base_ci + 0.1, y=float(press_map), ax=cur["CI"], ay=cur["MAP"], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor=THEME["map"], text="PRESSOR", font=dict(color=THEME["map"]))
    fig.add_annotation(x=base_ci + 0.6, y=float(inot_map), ax=cur["CI"], ay=cur["MAP"], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor=THEME["do2"], text="INOTROPE", font=dict(color=THEME["do2"]))

    _base_layout(fig, height=320, title="<b>Predictive Compass</b>")
    fig.update_xaxes(title_text="CI (L/min/m²)", range=[0.5, 6.0], gridcolor=THEME["grid"])
    fig.update_yaxes(title_text="MAP (mmHg)", range=[30, 160], gridcolor=THEME["grid"])
    fig.update_layout(showlegend=False)
    return fig


def plot_multiverse(df: pd.DataFrame, preds: Dict[str, Any], curr_time: int) -> go.Figure:
    """Time-series MAP with intervention horizon traces (natural/fluid/press/inot)."""
    hist = df.iloc[max(0, curr_time - 60):curr_time].copy()
    fig = go.Figure()
    if len(hist) > 0:
        fig.add_trace(go.Scatter(x=hist.index, y=hist["MAP"], line=dict(color="white", width=2), name="history"))
    # predicted traces (time axis is relative 0..horizon-1)
    horizon_time = np.arange(len(preds["time"]))
    fig.add_trace(go.Scatter(x=horizon_time, y=preds["nat"]["MAP"], line=dict(color="#555", dash="dot"), name="natural"))
    fig.add_trace(go.Scatter(x=horizon_time, y=preds["fluid"]["MAP"], line=dict(color=THEME["ci"]), name="fluid"))
    fig.add_trace(go.Scatter(x=horizon_time, y=preds["press"]["MAP"], line=dict(color=THEME["map"]), name="pressor"))
    fig.add_trace(go.Scatter(x=horizon_time, y=preds["inot"]["MAP"], line=dict(color=THEME["do2"]), name="inotrope"))
    fig.add_hline(y=CLIN_THRESH["MAP_OK"], line_color="red", line_dash="dot")
    _base_layout(fig, height=320, title="<b>Intervention Horizon (MAP)</b>")
    fig.update_xaxes(title_text="Minutes ahead", gridcolor=THEME["grid"])
    fig.update_yaxes(title_text="MAP (mmHg)", gridcolor=THEME["grid"])
    fig.update_layout(showlegend=False)
    return fig


def plot_organ_radar(df: pd.DataFrame, curr_time: int) -> go.Figure:
    """Organ risk radar with clinically normalized scales."""
    cur = df.iloc[max(0, curr_time - 1)]
    risk_renal = float(np.clip((CLIN_THRESH["MAP_OK"] - cur["MAP"]) / 25.0, 0.0, 1.0))
    risk_cardiac = float(np.clip((cur["HR"] - 100.0) / 60.0, 0.0, 1.0))
    risk_meta = float(np.clip((cur["Lactate"] - 1.5) / 4.0, 0.0, 1.0))
    risk_perf = float(np.clip((2.2 - cur["CI"]) / 1.5, 0.0, 1.0))

    r = [risk_renal, risk_cardiac, risk_meta, risk_perf, risk_renal]
    theta = ["Renal", "Cardiac", "Metabolic", "Perfusion", "Renal"]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[0.2] * 5, theta=theta, fill="toself", fillcolor=THEME["zone_ok"], line=dict(width=0)))
    fig.add_trace(go.Scatterpolar(r=r, theta=theta, fill="toself", fillcolor=THEME["zone_crit"], line=dict(color="red", width=2)))
    _base_layout(fig, height=260, title="<b>Organ Risk Topology</b>")
    fig.update_layout(showlegend=False)
    fig.update_polars(radialaxis=dict(visible=False, range=[0, 1], gridcolor=THEME["grid"]))
    return fig


def plot_starling_vector(df: pd.DataFrame, curr_time: int) -> go.Figure:
    """Starling curve with recent preload -> stroke volume trace."""
    data = df.iloc[max(0, curr_time - 30):curr_time].copy()
    fig = go.Figure()
    x = np.linspace(0, 20, 200)
    y = (x**2 / (x**2 + 8.0**2)) * 100.0  # consistent with twin
    fig.add_trace(go.Scatter(x=x, y=y, line=dict(color="#333", dash="dot"), name="starling"))

    if len(data) > 0:
        fig.add_trace(go.Scatter(x=data["Preload_Status"], y=data["SV"], mode="lines", line=dict(color=THEME["ci"]), name="trace"))
        fig.add_trace(
            go.Scatter(
                x=[data["Preload_Status"].iloc[-1]],
                y=[data["SV"].iloc[-1]],
                mode="markers",
                marker=dict(color="white", size=8, symbol="triangle-up"),
                name="now",
            )
        )
    _base_layout(fig, height=260, title="<b>Starling Vector</b>")
    fig.update_xaxes(title_text="Preload", gridcolor=THEME["grid"])
    fig.update_yaxes(title_text="SV (mL)", gridcolor=THEME["grid"])
    fig.update_layout(showlegend=False)
    return fig


def plot_oxygen_debt(df: pd.DataFrame, curr_time: int) -> go.Figure:
    """Dual-axis plot for DO2 and Lactate; DO2 on primary, Lactate on secondary axis."""
    data = df.iloc[max(0, curr_time - 120):curr_time].copy()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if len(data) > 0:
        fig.add_trace(go.Scatter(x=data.index, y=data["DO2"], fill="tozeroy", fillcolor=hex_to_rgba(THEME["do2"], 0.12), line=dict(color=THEME["do2"]), name="DO2"), secondary_y=False)
        fig.add_trace(go.Scatter(x=data.index, y=data["Lactate"], line=dict(color="red", width=2), name="Lactate"), secondary_y=True)
    _base_layout(fig, height=260, title="<b>O2 Supply/Demand</b>")
    fig.update_layout(showlegend=False)
    return fig


def plot_renal_cliff(df: pd.DataFrame, curr_time: int) -> go.Figure:
    """MAP vs Urine production showing autoregulation region and critical rectangle."""
    data = df.iloc[max(0, curr_time - 120):curr_time].copy()
    fig = go.Figure()
    fig.add_shape(type="rect", x0=30, y0=0.0, x1=65, y1=0.5, fillcolor=THEME["zone_crit"], line_width=0, layer="below")
    if len(data) > 0:
        fig.add_trace(
            go.Scatter(
                x=data["MAP"],
                y=data["Urine"],
                mode="lines+markers",
                marker=dict(color=np.linspace(0, 1, len(data)), colorscale="Reds", size=6),
                line=dict(color="#555"),
                name="urine",
            )
        )
    _base_layout(fig, height=260, title="<b>Renal Autoregulation</b>")
    fig.update_xaxes(title_text="MAP (mmHg)", gridcolor=THEME["grid"])
    fig.update_yaxes(title_text="Urine (proxy)", gridcolor=THEME["grid"])
    fig.update_layout(showlegend=False)
    return fig
