# utils.py
"""
TITAN utils: DigitalTwin physiology kernel, simulation runner, prediction horizon,
and core alert engine.

Clinical thresholds implemented here follow Surviving Sepsis Campaign and Sepsis-3:
MAP >= 65 mmHg target, lactate >= 2 mmol/L relevant threshold, Shock Index ~0.9
(see inline references in top-level README).
"""
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Theme and clinical constants (kept small; UI uses separate visuals module)
THEME = {
    "bg": "#000000",
    "map": "#ff2975",
    "ci": "#00ff33",
    "do2": "#8c1eff",
    "svr": "#ff9900",
    "ok": "rgba(0,255,51,0.08)",
    "warn": "rgba(255,195,0,0.08)",
    "crit": "rgba(255,41,117,0.08)",
}

# Clinical thresholds (evidence-based anchors)
CLIN_THRESH = {
    "MAP_TARGET": 65.0,           # Surviving Sepsis Campaign
    "MAP_CRIT": 55.0,             # critical risk threshold
    "LACTATE_ELEVATED": 2.0,      # Sepsis-3 diagnostic threshold (mmol/L)
    "LACTATE_HIGH": 4.0,          # high risk
    "CI_LOW": 2.0,                # low cardiac index suspicious for cardiogenic compromise
    "SI_WARN": 0.9,               # Shock index threshold
    "ALARM_MAP_DURATION": 10      # minutes sustained hypotension to escalate
}

# -----------------------
# DigitalTwin model
# -----------------------
class DigitalTwin:
    """
    Compact hemodynamic twin with parameters for preload/contractility/afterload
    and simple lactate kinetics. Intended for simulation and scenario forecasting.
    """
    def __init__(self,
                 preload: float = 12.0,
                 contractility: float = 1.0,
                 afterload: float = 1.0,
                 lactate_half_life_min: float = 120.0):
        self.preload = float(preload)
        self.contractility = float(contractility)
        self.afterload = float(afterload)
        # lactate clearance constant (per minute)
        self.lactate_half_life_min = float(lactate_half_life_min)

    def copy(self) -> "DigitalTwin":
        return DigitalTwin(self.preload, self.contractility, self.afterload, self.lactate_half_life_min)

    def step(self, sepsis_severity: float = 0.0, dt_min: float = 1.0) -> Dict[str, float]:
        """
        Run a single timestep and return dictionary of physiologic quantities.
        dt_min: minutes elapsed (used for lactate clearance scaling).
        """
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

        co = (hr * sv) / 1000.0  # L/min
        svr = 1200.0 * eff_afterload
        map_val = (co * svr / 80.0) + 5.0

        # DO2 heuristic
        do2 = co * 1.34 * 12.0 * 0.98 * 10.0

        # lactate generation and clearance
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

# -----------------------
# Simulator
# -----------------------
def simulate_titan_data(mins: int = 720,
                        twin: Optional[DigitalTwin] = None,
                        seed: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simulate a time-series of length `mins` using a baseline twin (copied).
    Returns (df, preds) where preds is a forecast object for interventions.
    """
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

    # Derived metrics
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

# -----------------------
# Prediction horizon
# -----------------------
def predict_horizon(base_twin: DigitalTwin, last_sev: float, horizon: int = 30) -> Dict[str, Any]:
    """
    Forecast MAP/CI/Lactate under scenarios: natural, fluid, pressor, inotrope.
    Returns nested dict: preds['fluid']['MAP'] etc.
    """
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

    preds = {
        "time": np.arange(horizon),
        "nat": {"MAP": arr(nat, "MAP"), "CI": arr(nat, "CI"), "Lactate": arr(nat, "Lactate")},
        "fluid": {"MAP": arr(fluid, "MAP"), "CI": arr(fluid, "CI"), "Lactate": arr(fluid, "Lactate")},
        "press": {"MAP": arr(press, "MAP"), "CI": arr(press, "CI"), "Lactate": arr(press, "Lactate")},
        "inot": {"MAP": arr(inot, "MAP"), "CI": arr(inot, "CI"), "Lactate": arr(inot, "Lactate")},
    }
    return preds

# -----------------------
# Alert engine (clinical rules)
# -----------------------
def compute_alerts(df: pd.DataFrame, lookback_min: int = 15) -> Dict[str, Any]:
    """
    Evaluate clinical alarms using rolling window rules.
    Returns dict containing 'alerts' list; each alert is a dict with type/severity/message/action.
    """
    out = {"alerts": []}
    if df is None or df.shape[0] == 0:
        return out

    lookback = min(lookback_min, len(df))
    window = df.iloc[-lookback:]
    curr = window.iloc[-1]

    # sustained hypotension
    below = window["MAP"] < CLIN_THRESH["MAP_TARGET"]
    if below.sum() >= CLIN_THRESH["ALARM_MAP_DURATION"]:
        severity = "critical" if window["MAP"].min() < CLIN_THRESH["MAP_CRIT"] else "warning"
        out["alerts"].append({
            "type": "SUSTAINED_HYPOTENSION",
            "severity": severity,
            "message": f"MAP < {CLIN_THRESH['MAP_TARGET']} mmHg for {int(below.sum())} min (min {window['MAP'].min():.0f}).",
            "suggested_action": "Consider vasopressors and reassess fluids."
        })

    # low CI
    if curr.get("CI", 99.0) < CLIN_THRESH["CI_LOW"]:
        out["alerts"].append({
            "type": "LOW_CI",
            "severity": "warning",
            "message": f"Low cardiac index: {curr.get('CI', 0):.2f} L/min/m²",
            "suggested_action": "Consider inotrope or optimize preload based on responsiveness."
        })

    # lactate rising slope
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
                "suggested_action": "Assess perfusion, consider increasing DO2 or source control."
            })

    # shock index
    if curr.get("SI", 0.0) >= CLIN_THRESH["SI_WARN"]:
        out["alerts"].append({
            "type": "ELEVATED_SHOCK_INDEX",
            "severity": "warning",
            "message": f"Shock index {curr.get('SI',0):.2f} >= {CLIN_THRESH['SI_WARN']}",
            "suggested_action": "Reassess for occult shock causes (bleeding, hypovolemia, sepsis)."
        })

    return out

