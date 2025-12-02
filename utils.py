import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
from typing import Tuple, Dict

# --- 1. DX: CONFIGURATION OBJECTS ---
@dataclass
class Theme:
    bg: str = "#ffffff"
    grid: str = "#f0f2f6"
    text_main: str = "#111827"
    text_sub: str = "#6b7280"
    # Physiology Colors
    hr: str = "#2563eb"     # Blue (Standard)
    sbp: str = "#be185d"    # Magenta/Red
    dbp: str = "#f472b6"    # Light Pink
    map: str = "#831843"    # Dark Magenta
    pi: str = "#059669"     # Emerald
    # Zones
    zone_ok: str = "#d1fae5"
    zone_warn: str = "#fef3c7"
    zone_crit: str = "#fee2e2"

THEME = Theme()

# --- 2. PHYSIOLOGY ENGINE ---
def generate_pink_noise(n, scale=1.0):
    """Generates 1/f noise (Pink Noise) which mimics biological signals better than Gaussian."""
    white = np.random.normal(0, scale, n)
    # Simple low-pass filter to approximate pink noise
    b = [0.04] * 25
    a = 1
    import scipy.signal as signal
    pink = signal.lfilter(b, a, white)
    return pink

def simulate_hemodynamic_shock(mins=720) -> pd.DataFrame:
    """
    Simulates 'Compensated' vs 'Decompensated' Shock.
    
    Pathology:
    1. Volume Loss (Hidden) -> 2. Vasoconstriction (PI Drop) -> 
    3. Tachycardia (HR Rise) -> 4. Pulse Pressure Narrowing -> 
    5. BP Crash (Decompensation).
    """
    t = np.arange(mins)
    
    # --- BASELINE STATE (Healthy) ---
    # HR: 70 +/- 5
    # BP: 120/80 (MAP ~93)
    # PI: 3.0 +/- 0.5
    
    noise_hr = generate_pink_noise(mins, scale=2.0)
    noise_sbp = generate_pink_noise(mins, scale=3.0)
    noise_pi = generate_pink_noise(mins, scale=0.2)
    
    hr = 70 + noise_hr
    sbp = 120 + noise_sbp
    dbp = 80 + (noise_sbp * 0.5) # DBP tracks SBP partially
    pi = 4.0 + 0.5 * np.sin(t/50) + noise_pi
    
    # --- EVENT: OCCULT HEMORRHAGE (Starts T=250) ---
    event_start = 250
    decomp_start = 550
    
    # Phase 1: Compensation (250 - 550)
    # Body dumps catecholamines.
    # Result: HR Up, SVR Up (PI Down), DBP Up (Vasoconstriction), SBP Flat.
    
    dur_comp = decomp_start - event_start
    drift_hr = np.linspace(0, 45, dur_comp)     # HR 70 -> 115
    drift_pi = np.linspace(0, 3.5, dur_comp)    # PI 4.0 -> 0.5
    drift_dbp = np.linspace(0, 10, dur_comp)    # DBP 80 -> 90 (Narrowing PP)
    
    hr[event_start:decomp_start] += drift_hr
    pi[event_start:decomp_start] = np.maximum(0.2, pi[event_start:decomp_start] - drift_pi)
    dbp[event_start:decomp_start] += drift_dbp
    
    # Phase 2: Decompensation (550+)
    # Reserve exhausted. Frank hypotension.
    
    dur_crash = mins - decomp_start
    crash_sbp = np.linspace(0, 50, dur_crash) ** 1.2 # Fast crash
    crash_dbp = np.linspace(0, 40, dur_crash)
    
    # Keep HR High (or higher)
    hr[decomp_start:] += 45 + np.random.normal(0, 2, dur_crash)
    
    sbp[decomp_start:] -= crash_sbp
    dbp[decomp_start:] -= crash_dbp
    # PI stays floored
    pi[decomp_start:] = 0.2 + np.random.normal(0, 0.05, dur_crash)

    # --- DERIVED METRICS ---
    df = pd.DataFrame({'HR': hr, 'SBP': sbp, 'DBP': dbp, 'PI': pi}, index=t)
    
    # 1. MAP (Mean Arterial Pressure): The perfusion pressure
    df['MAP'] = (df['SBP'] + 2*df['DBP']) / 3
    
    # 2. Pulse Pressure (PP): SBP - DBP (Narrows in shock)
    df['PP'] = df['SBP'] - df['DBP']
    
    # 3. Shock Index (SI): HR / SBP (Clinical indicator > 0.9)
    df['SI'] = df['HR'] / df['SBP']
    
    return df

# --- 3. ANALYTICS: SPC & FORECASTING ---
def run_spc_analysis(series, window=60):
    """Western Electric Rules calculation."""
    roll = series.rolling(window=window)
    mean = roll.mean()
    std = roll.std()
    
    # Returning zones for visualization
    return {
        'mean': mean,
        'u2': mean + 2*std, 
        'l2': mean - 2*std,
        'u3': mean + 3*std,
        'l3': mean - 3*std
    }

def generate_trend_forecast(series, horizon=30):
    """Simple linear trend with expanding volatility cone."""
    y = series.values[-45:] # Last 45 mins momentum
    x = np.arange(len(y)).reshape(-1, 1)
    
    model = LinearRegression().fit(x, y)
    
    future_x = np.arange(len(y), len(y)+horizon).reshape(-1, 1)
    trend = model.predict(future_x)
    
    # Volatility cone
    sigma = np.std(y)
    cone = np.linspace(1.0, 2.0, horizon) * 1.96 * sigma
    
    return trend, trend+cone, trend-cone

# --- 4. VISUALIZATION: THE "HEMODYNAMIC PROFILE" ---
def plot_hemodynamic_profile(df, curr_time):
    """
    A unified chart showing the interaction of HR, BP, and Stroke Volume (implied by PP).
    This creates a 'Visual Signature' of shock.
    """
    window = 180
    start = max(0, curr_time - window)
    data = df.iloc[start:curr_time]
    t_fut = np.arange(curr_time, curr_time+30)
    
    # Forecasts
    hr_pred, hr_up, hr_lo = generate_trend_forecast(df['HR'].iloc[:curr_time])
    
    # Layout: 2 Rows (Heart vs Flow)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.6, 0.4],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    # --- ROW 1: THE SHOCK CROSSOVER (HR vs MAP) ---
    # HR Forecast Fan
    fig.add_trace(go.Scatter(x=t_fut, y=hr_up, line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_fut, y=hr_lo, fill='tonexty', fillcolor='rgba(37, 99, 235, 0.1)', line=dict(width=0), showlegend=False), row=1, col=1)
    
    # HR Actual
    fig.add_trace(go.Scatter(
        x=data.index, y=data['HR'], name="Heart Rate",
        line=dict(color=THEME.hr, width=2.5)
    ), row=1, col=1)
    
    # MAP (Mean Arterial Pressure) - The perfusion driver
    fig.add_trace(go.Scatter(
        x=data.index, y=data['MAP'], name="MAP (mmHg)",
        line=dict(color=THEME.map, width=2.5, dash='solid')
    ), row=1, col=1, secondary_y=True)

    # Shock Index Threshold Background (Implicit)
    # If HR > SBP, we shade the area
    
    # --- ROW 2: PULSE PRESSURE & PERFUSION (The Early Warning) ---
    # Pulse Pressure Area (Band between SBP and DBP)
    # Visualizes "Narrowing" beautifully
    fig.add_trace(go.Scatter(
        x=data.index, y=data['SBP'], line=dict(width=1, color=THEME.sbp), name="Sys BP"
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data['DBP'], line=dict(width=0), fill='tonexty', 
        fillcolor='rgba(190, 24, 93, 0.15)', name="Pulse Pressure (Stroke Vol)"
    ), row=2, col=1)
    
    # PI overlay
    # We scale PI to fit nicely or put it on its own axis? 
    # Better: Put PI on Right Axis of Row 2 to compare Flow vs Pressure
    # But Plotly subplot secondary axis is tricky with fills. 
    # Solution: Normalize PI visually or just keep separate. Let's keep separate for clarity.
    
    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(l=10, r=10, t=20, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0)
    )
    
    # Axes
    fig.update_yaxes(title="HR (bpm)", row=1, col=1, secondary_y=False, showgrid=True, gridcolor=THEME.grid)
    fig.update_yaxes(title="MAP (mmHg)", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title="BP / Pulse Pressure", row=2, col=1, showgrid=True, gridcolor=THEME.grid)
    
    return fig

def plot_perfusion_radar(current_row):
    """
    A 'Target' chart. The closer to center, the more stable.
    We invert metrics so 'High' is always 'Bad'.
    """
    # Normalization Logic (0 = Good, 1 = Critical)
    
    # HR: > 100 is bad
    val_hr = np.clip((current_row['HR'] - 60) / 80, 0, 1)
    
    # SI: > 0.7 is bad
    val_si = np.clip((current_row['SI'] - 0.6) / 0.8, 0, 1)
    
    # PI: < 3.0 is bad (Inverted)
    val_pi = np.clip((3.0 - current_row['PI']) / 3.0, 0, 1)
    
    # PP: < 40 is bad (Inverted)
    val_pp = np.clip((45 - current_row['PP']) / 25, 0, 1)
    
    r = [val_hr, val_si, val_pi, val_pp, val_hr]
    theta = ['Tachycardia', 'Shock Index', 'Vasoconstriction (PI)', 'Low Stroke Vol (PP)', 'Tachycardia']
    
    fig = go.Figure()
    
    # Safe Zone
    fig.add_trace(go.Scatterpolar(
        r=[0.3]*5, theta=theta, fill='toself', fillcolor=THEME.zone_ok, 
        line=dict(color='green', width=0), hoverinfo='skip'
    ))
    
    # Current State
    fig.add_trace(go.Scatterpolar(
        r=r, theta=theta, fill='toself', 
        fillcolor='rgba(220, 38, 38, 0.4)', 
        line=dict(color='red', width=2),
        name='Current Status'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(tickfont=dict(size=10, color=THEME.text_sub))
        ),
        showlegend=False,
        height=250,
        margin=dict(l=30, r=30, t=20, b=20)
    )
    return fig

def plot_causal_spc(df, curr_time):
    """
    Focuses on the 'Early Warning' signal: PI (Perfusion Index).
    """
    spc = run_spc_analysis(df['PI'].iloc[:curr_time])
    data = df['PI'].iloc[max(0, curr_time-120):curr_time]
    
    fig = go.Figure()
    
    # SPC Bands
    fig.add_trace(go.Scatter(x=data.index, y=spc['mean'].loc[data.index], line=dict(color='gray', dash='dot', width=1)))
    fig.add_trace(go.Scatter(x=data.index, y=spc['l2'].loc[data.index], line=dict(color='orange', width=0), name='Warning'))
    fig.add_trace(go.Scatter(x=data.index, y=spc['l3'].loc[data.index], fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)', line=dict(color='red', width=0), name='Critical'))
    
    # Data
    fig.add_trace(go.Scatter(
        x=data.index, y=data, mode='lines', 
        line=dict(color=THEME.pi, width=2), name='Perfusion Index'
    ))
    
    fig.update_layout(
        template="plotly_white",
        height=200,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text="<b>Early Warning: Perfusion Index (SPC)</b>", font=dict(size=12)),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=THEME.grid)
    )
    
    return fig
