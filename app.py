# app.py
from typing import Tuple
import streamlit as st
import pandas as pd
import utils

# --- CONFIG ---
st.set_page_config(page_title="TITAN | ICU", layout="wide", initial_sidebar_state="collapsed")

# --- STYLES ---
_STYLES = """
<style>
:root{
  --bg      : #000000;
  --card-bg : #111111;
  --muted    : #888;
  --text    : #e0e0e0;
  --card-border: #333;
}
body { background-color: var(--bg); color: var(--text); }
.stApp .titan-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 8px;
}
.kpi-lbl { font-size:0.75rem; color: var(--muted); font-weight:700; text-transform:uppercase; margin-bottom:2px; }
.kpi-val { font-size:1.2rem; font-weight:800; font-family: 'Roboto Mono', monospace; }
.alert-header {
    background: #111; border-left: 6px solid; padding: 10px; margin-bottom: 12px;
    display:flex; justify-content:space-between; align-items:center; gap:8px;
}
.section-header { font-size:1.05rem; font-weight:800; color:#fff; border-bottom:1px solid var(--card-border); margin-top:18px; margin-bottom:10px; padding-bottom:6px; }
.small { font-size:0.85rem; color:#bbb; }
</style>
"""
st.markdown(_STYLES, unsafe_allow_html=True)

# --- LOAD DATA (cached) ---
@st.cache_data
def load_data(mins: int = 720):
    df, preds = utils.simulate_titan_data(mins=mins, random_seed=42)
    return df, preds


df, preds = load_data(mins=720)

# --- SIDEBAR: clinical controls & quick actions ---
with st.sidebar:
    st.header("TITAN Control (ICU)")
    st.write("Inspect patient timeline and simulate interventions.")
    max_time = int(len(df))
    curr_time_ui = st.slider("Time (minute index)", 1, max_time, max_time, help="1-based minute index")
    st.markdown("---")
    st.subheader("Model knobs")
    preload = st.number_input("Preload (base)", value=12.0, step=0.5)
    contractility = st.number_input("Contractility", value=1.0, step=0.05, format="%.2f")
    afterload = st.number_input("Afterload", value=1.0, step=0.05, format="%.2f")
    if st.button("Resimulate with knobs"):
        # resimulate quickly with new twin
        twin = utils.DigitalTwin(preload=preload, contractility=contractility, afterload=afterload)
        df, preds = utils.simulate_titan_data(mins=720, random_seed=42, twin=twin)
        st.experimental_rerun()

# persist slider in session state to make it consistent across reruns
st.session_state.setdefault("curr_time", int(curr_time_ui))
curr_time = st.session_state["curr_time"]

# safe index (0-based)
cur_idx = max(0, min(len(df) - 1, curr_time - 1))
cur = df.iloc[cur_idx]

# Compute alerts using utils alert engine
alerts = utils.compute_alerts(df, lookback_min=15)

# --- ALERT / PROTOCOL HEADER ---
def compute_status_and_action(row: pd.Series):
    """Return status, suggested action, color and textual rationale for display."""
    status = "STABLE"
    action = "MONITOR"
    color = "#00ff33"
    rationale = "Vitals within acceptable range."

    if row["MAP"] < utils.CLIN_THRESH["MAP_OK"]:
        status = "HYPOTENSION"
        if row["MAP"] < utils.CLIN_THRESH["MAP_CRIT"]:
            color = "#ff0a3d"
            action = "Immediate vasopressor"
            rationale = f"MAP critically low ({row['MAP']:.0f} mmHg)."
        else:
            color = "#ff9900"
            action = "Fluids ± vasopressor"
            rationale = f"MAP below target ({row['MAP']:.0f} mmHg)."
    elif row.get("Lactate", 0.0) > 2.0:
        status = "PERFUSION RISK"
        action = "Optimize flow"
        color = "#ff9900"
        rationale = f"Lactate elevated ({row.get('Lactate', 0.0):.2f} mM)."

    # Low CI
    if row.get("CI", 999.0) < utils.CLIN_THRESH["CI_LOW"]:
        status = "LOW CARDIAC OUTPUT"
        action = "Consider inotrope"
        color = "#8c1eff"
        rationale = rationale + " CI low."

    return status, action, color, rationale


status, protocol_action, status_color, rationale = compute_status_and_action(cur)

st.markdown(
    f"""
<div class="alert-header" style="border-color:{status_color}">
  <div style="font-size:1.05rem; font-weight:bold; color:{status_color}">{status}</div>
  <div style="font-weight:700;color:#ddd">PROTOCOL: {protocol_action}</div>
</div>
<p class="small">{rationale}</p>
""",
    unsafe_allow_html=True,
)

# show active alerts (if any)
if alerts.get("alerts"):
    with st.container():
        for a in alerts["alerts"]:
            sev = a["severity"]
            color = "#ff2975" if sev == "critical" else "#ff9900"
            st.markdown(
                f"<div class='titan-card' style='border-left:4px solid {color}; padding:8px; margin-bottom:6px;'><b>{a['type']}</b> — {a['message']}<br><i style='color:#bbb'>{a['suggested_action']}</i></div>",
                unsafe_allow_html=True,
            )

# --- KPI STRIP ---
kpi_cols = st.columns(6, gap="small")


def kpi_card(col, label: str, value: str, unit: str, color: str, df_col: str, low: float, high: float):
    with col:
        st.markdown(
            f"""<div class='titan-card' style='border-top:3px solid {color}'>
                    <div class='kpi-lbl'>{label}</div>
                    <div class='kpi-val' style='color:{color}'>{value}<span style='font-size:0.72rem;color:#999'> {unit}</span></div>
                </div>""",
            unsafe_allow_html=True,
        )
        fig = utils.plot_spark_spc(df, df_col, color, low, high)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"spark_{df_col}")


kpi_card(kpi_cols[0], "MAP", f"{cur['MAP']:.0f}", "mmHg", utils.THEME["map"], "MAP", 55, 100)
kpi_card(kpi_cols[1], "C. INDEX", f"{cur['CI']:.2f}", "L/min/m²", utils.THEME["ci"], "CI", 1.5, 4.0)
kpi_card(kpi_cols[2], "SVR", f"{cur['SVRI']:.0f}", "dyn·s·cm⁻⁵", utils.THEME["svr"], "SVRI", 600, 1400)
kpi_card(kpi_cols[3], "STROKE VOL", f"{cur['SV']:.0f}", "mL", utils.THEME["hr"], "SV", 40, 120)
kpi_card(kpi_cols[4], "LACTATE", f"{cur['Lactate']:.2f}", "mM", utils.THEME["do2"], "Lactate", 0.4, 4.0)
kpi_card(kpi_cols[5], "ENTROPY", f"{cur['Entropy']:.2f}", "σ", "#ffffff", "Entropy", 0.1, 2.0)

# --- PREDICTIVE HEMODYNAMICS ---
st.markdown('<div class="section-header">🔮 PREDICTIVE HEMODYNAMICS</div>', unsafe_allow_html=True)
c_pred1, c_pred2 = st.columns((1, 1), gap="large")

with c_pred1:
    st.markdown("<div class='titan-card'>", unsafe_allow_html=True)
    fig_compass = utils.plot_predictive_compass(df, preds, curr_time)
    st.plotly_chart(fig_compass, use_container_width=True, config={"displayModeBar": False}, key="compass")
    st.markdown("</div>", unsafe_allow_html=True)

with c_pred2:
    st.markdown("<div class='titan-card'>", unsafe_allow_html=True)
    fig_multiverse = utils.plot_multiverse(df, preds, curr_time)
    st.plotly_chart(fig_multiverse, use_container_width=True, config={"displayModeBar": False}, key="multiverse")
    st.markdown("</div>", unsafe_allow_html=True)

# --- ORGAN MECHANICS ---
st.markdown('<div class="section-header">🫀 ORGAN MECHANICS</div>', unsafe_allow_html=True)
c_org1, c_org2, c_org3, c_org4 = st.columns(4, gap="large")

with c_org1:
    st.markdown("<div class='titan-card'>", unsafe_allow_html=True)
    st.plotly_chart(utils.plot_organ_radar(df, curr_time), use_container_width=True, config={"displayModeBar": False}, key="radar")
    st.markdown("</div>", unsafe_allow_html=True)

with c_org2:
    st.markdown("<div class='titan-card'>", unsafe_allow_html=True)
    st.plotly_chart(utils.plot_starling_vector(df, curr_time), use_container_width=True, config={"displayModeBar": False}, key="starling")
    st.markdown("</div>", unsafe_allow_html=True)

with c_org3:
    st.markdown("<div class='titan-card'>", unsafe_allow_html=True)
    st.plotly_chart(utils.plot_oxygen_debt(df, curr_time), use_container_width=True, config={"displayModeBar": False}, key="oxygen")
    st.markdown("</div>", unsafe_allow_html=True)

with c_org4:
    st.markdown("<div class='titan-card'>", unsafe_allow_html=True)
    st.plotly_chart(utils.plot_renal_cliff(df, curr_time), use_container_width=True, config={"displayModeBar": False}, key="renal")
    st.markdown("</div>", unsafe_allow_html=True)

# --- ACTION PANEL (suggested clinical actions summary) ---
st.markdown('<div class="section-header">🩺 Suggested Clinical Actions</div>', unsafe_allow_html=True)
if alerts.get("alerts"):
    for a in alerts["alerts"]:
        st.markdown(f"**{a['type']}** — {a['message']}  \n*Suggested:* {a['suggested_action']}")
else:
    # show recommended next steps derived from protocol_action
    st.markdown("- Patient stable. Continue monitoring and maintain MAP ≥ 65 mmHg.")
    st.markdown("- If lactate rising, check perfusion, check hemoglobin/SpO2, consider fluids or inotrope based on CI.")

# --- DEBUG / DATA SNAPSHOT (expandable) ---
with st.expander("Debug / Data snapshot", expanded=False):
    st.write("Current minute index (1-based):", curr_time)
    st.dataframe(cur.to_frame(name="value"))
    st.write("Recent row (last 10):")
    st.dataframe(df.tail(10))

# --- END ---
