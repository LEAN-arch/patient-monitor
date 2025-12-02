import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# --- 1. CLINICAL THEME (Medical Device Standard) ---
THEME = {
    'bg_paper': '#ffffff',
    'text_main': '#111827',
    'grid': '#f3f4f6',
    # Signals
    'hr': '#2563eb',       # Blue
    'map': '#be185d',      # Magenta (Perfusion)
    'pp': '#7c3aed',       # Violet (Stroke Volume Proxy)
    'pi': '#059669',       # Emerald (Micro-circ)
    'entropy': '#d97706',  # Amber (CNS Status)
    # Status Zones
    'zone_crit': 'rgba(239, 68, 68, 0.1)',
    'zone_warn': 'rgba(251, 191, 36, 0.15)',
    'zone_ok': 'rgba(16, 185, 129, 0.1)'
}

# --- 2. ADVANCED PHYSIOLOGY SIMULATION ---
def simulate_shock_progression(mins=720):
    t = np.arange(mins)
    
    # Noise Generator (Pink Noise for realism)
    def bio_noise(n, amp=1.0):
        w = np.random.normal(0, 1, n)
        return np.convolve(w, [0.05]*20, mode='same') * amp

    # --- BASELINE (Healthy) ---
    true_hr = 70 + bio_noise(mins, 3)
    true_sbp = 125 + bio_noise(mins, 3)
    true_dbp = 80 + bio_noise(mins, 2)
    pi = 4.5 + bio_noise(mins, 0.5)
    
    # --- CLINICAL SCENARIO: Progressive "Cold" Shock ---
    # Characterized by: Rising HR, Narrowing Pulse Pressure, Dropping PI.
    
    start = 240
    
    # 1. Stroke Volume Fails (Pulse Pressure Narrows)
    # SBP drops, but DBP is maintained initially by vasoconstriction
    sbp_drift = np.linspace(0, 45, mins-start)
    dbp_drift = np.linspace(0, 10, mins-start)
    true_sbp[start:] -= sbp_drift
    true_dbp[start:] -= dbp_drift
    
    # 2. Compensatory Tachycardia
    hr_drift = np.linspace(0, 50, mins-start)
    true_hr[start:] += hr_drift
    
    # 3. Micro-circulatory Shut Down (Vasoconstriction)
    pi_drift = np.linspace(0, 4.0, mins-start)
    pi[start:] = np.maximum(0.3, pi[start:] - pi_drift)
    
    # 4. Autonomic De-complexification (Entropy Loss - Sepsis/Stress marker)
    # We dampen the biological noise over time
    dampening = np.linspace(1, 0.2, mins-start)
    true_hr[start:] = true_hr[start:] * dampening + (true_hr[start:].mean() * (1-dampening))

    # Add Sensor Noise (for Kalman to clean)
    obs_hr = true_hr + np.random.normal(0, 3, mins)

    df = pd.DataFrame({'Obs_HR': obs_hr, 'True_HR': true_hr, 'SBP': true_sbp, 'DBP': true_dbp, 'PI': pi}, index=t)
    
    # Derived Hemodynamics
    df['MAP'] = (df['SBP'] + 2*df['DBP']) / 3
    df['PP'] = df['SBP'] - df['DBP'] # Pulse Pressure (Stroke Volume Proxy)
    df['SI'] = df['Obs_HR'] / df['SBP'] # Shock Index
    
    # Rolling Entropy (Physiological Complexity)
    # Simplified proxy: standard deviation of beat-to-beat differences
    df['Entropy'] = df['Obs_HR'].rolling(30).apply(lambda x: np.std(np.diff(x))).fillna(1.0)
    
    return df

# --- 3. DIAGNOSTIC VISUALIZATION ENGINES ---

def plot_prognostic_horizon(df, curr_time):
    """
    VISUALIZATION 1: The "Time-to-Crash" Horizon.
    Combines Kalman-filtered history with a Linear Regression projection.
    """
    start = max(0, curr_time - 180)
    data = df.iloc[start:curr_time]
    
    # Prognostic Projection (MAP)
    # Fit regression on last 30 mins
    recent_map = data['MAP'].iloc[-30:].values
    X = np.arange(len(recent_map)).reshape(-1, 1)
    model = LinearRegression().fit(X, recent_map)
    
    # Project 45 mins out
    future_steps = 45
    X_fut = np.arange(len(recent_map), len(recent_map) + future_steps).reshape(-1, 1)
    map_pred = model.predict(X_fut)
    
    # Find "Crash Time" (MAP < 65)
    crash_idx = np.argmax(map_pred < 65)
    crash_time = crash_idx if crash_idx > 0 else None
    
    t_fut = np.arange(curr_time, curr_time + future_steps)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. MAP History
    fig.add_trace(go.Scatter(
        x=data.index, y=data['MAP'], mode='lines', 
        line=dict(color=THEME['map'], width=2.5), name='MAP (Perfusion)'
    ), secondary_y=False)
    
    # 2. MAP Projection (Dotted)
    fig.add_trace(go.Scatter(
        x=t_fut, y=map_pred, mode='lines', 
        line=dict(color=THEME['map'], dash='dot', width=2), name='Projected MAP'
    ), secondary_y=False)
    
    # 3. Critical Threshold
    fig.add_hline(y=65, line_dash="solid", line_color="red", opacity=0.3, secondary_y=False)
    
    # 4. Crash Annotation
    if crash_time:
        fig.add_vline(x=curr_time + crash_time, line_dash="dot", line_color="red", opacity=0.5)
        fig.add_annotation(
            x=curr_time + crash_time, y=65, 
            text=f"CRITICAL<br>T+{crash_time}min", 
            showarrow=True, arrowhead=2, arrowcolor="red", bgcolor="white"
        )
        
    # 5. Heart Rate (Context)
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Obs_HR'], mode='lines',
        line=dict(color=THEME['hr'], width=1), opacity=0.3, name='HR'
    ), secondary_y=True)

    fig.update_layout(
        template="plotly_white", height=300, 
        title="<b>Hemodynamic Horizon (Time-to-Event)</b>",
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified", legend=dict(orientation="h", y=1.05)
    )
    fig.update_yaxes(title="MAP (mmHg)", range=[40, 130], secondary_y=False, gridcolor=THEME['grid'])
    fig.update_yaxes(title="HR (bpm)", secondary_y=True, showgrid=False)
    
    return fig

def plot_shock_phenotype(df, curr_time):
    """
    VISUALIZATION 2: The "Shock Quadrants" (Diagnostic).
    Plots Pulse Pressure (Stroke Volume Proxy) vs Shock Index (Instability).
    """
    start = max(0, curr_time - 120)
    data = df.iloc[start:curr_time]
    
    fig = go.Figure()
    
    # --- DIAGNOSTIC ZONES ---
    # Zone 1: Stable (Normal PP, Low SI)
    fig.add_shape(type="rect", x0=0.4, y0=40, x1=0.8, y1=80, 
                  fillcolor="rgba(16, 185, 129, 0.1)", line_width=0, layer="below")
    fig.add_annotation(x=0.6, y=60, text="STABLE", showarrow=False, font=dict(color="green"))
    
    # Zone 2: Compensated (Narrow PP, Rising SI)
    fig.add_shape(type="rect", x0=0.8, y0=20, x1=1.2, y1=40, 
                  fillcolor="rgba(245, 158, 11, 0.1)", line_width=0, layer="below")
    fig.add_annotation(x=1.0, y=30, text="COMPENSATED\n(Low SV)", showarrow=False, font=dict(color="orange"))
    
    # Zone 3: Critical (High SI, Narrow PP)
    fig.add_shape(type="rect", x0=1.2, y0=0, x1=1.6, y1=30, 
                  fillcolor="rgba(239, 68, 68, 0.1)", line_width=0, layer="below")
    fig.add_annotation(x=1.4, y=15, text="CRITICAL\nDECOMP", showarrow=False, font=dict(color="red"))

    # Trajectory Line
    fig.add_trace(go.Scatter(
        x=data['SI'], y=data['PP'],
        mode='lines', line=dict(color='gray', width=2), opacity=0.5,
        name='Trajectory'
    ))
    
    # Current State Dot
    fig.add_trace(go.Scatter(
        x=[data['SI'].iloc[-1]], y=[data['PP'].iloc[-1]],
        mode='markers', 
        marker=dict(size=14, color=THEME['pp'], symbol='cross'),
        name='CURRENT STATE'
    ))

    fig.update_layout(
        template="plotly_white", height=320,
        title="<b>Shock Phenotyping (Diagnosis)</b>",
        xaxis=dict(title="Shock Index (Instability)", range=[0.4, 1.6], showgrid=True, gridcolor=THEME['grid']),
        yaxis=dict(title="Pulse Pressure (Stroke Volume Proxy)", range=[0, 80], showgrid=True, gridcolor=THEME['grid']),
        margin=dict(l=10, r=10, t=40, b=10), showlegend=False
    )
    return fig

def plot_autonomic_strip(df, curr_time):
    """
    VISUALIZATION 3: Autonomic & Micro-circulatory Reserve.
    Combines Entropy (Nerves) and PI (Vessels).
    """
    start = max(0, curr_time - 180)
    data = df.iloc[start:curr_time]
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    
    # 1. Entropy (CNS)
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Entropy'], 
        line=dict(color=THEME['entropy'], width=2), name='HR Entropy'
    ), row=1, col=1)
    fig.add_hline(y=0.5, line_dash="dot", line_color="red", row=1, col=1, annotation_text="CNS Stress")
    
    # 2. PI (PNS/Vascular)
    fig.add_trace(go.Scatter(
        x=data.index, y=data['PI'], 
        line=dict(color=THEME['pi'], width=2), fill='tozeroy', 
        fillcolor='rgba(5, 150, 105, 0.1)', name='Perfusion'
    ), row=2, col=1)
    fig.add_hline(y=1.0, line_dash="dot", line_color="red", row=2, col=1, annotation_text="Vasoconstriction")
    
    fig.update_layout(
        template="plotly_white", height=250, margin=dict(l=0,r=0,t=30,b=0),
        title="<b>Autonomic & Micro-vascular Reserve</b>",
        showlegend=False
    )
    fig.update_yaxes(title="Entropy", row=1, col=1, gridcolor=THEME['grid'])
    fig.update_yaxes(title="PI %", row=2, col=1, gridcolor=THEME['grid'])
    
    return fig
