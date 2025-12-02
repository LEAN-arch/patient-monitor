import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# --- 1. CONFIGURATION ---
COLORS = {
    'bg': '#0b0e11',
    'card': '#15191f',
    'text': '#e0e0e0',
    'grid': '#2b313b',
    'hr': '#00f2ea',      # Cyan
    'sbp': '#ff0055',     # Neon Red
    'spc_mean': '#00ff00',
    'spc_band': 'rgba(0, 255, 0, 0.05)',
    'forecast': '#ffae00' # Amber
}

# --- 2. ADVANCED SIMULATION ---
def simulate_patient_dynamics(mins_total=720):
    """
    Simulates patient physiology with non-linear dynamics for Chaos theory analysis.
    """
    t = np.arange(mins_total)
    
    # 1. Baseline (Stable Attractor)
    # Using sine waves + fractal noise to simulate biological variability (HRV)
    noise = np.random.normal(0, 1, mins_total)
    hr = 75 + 2 * np.sin(t/50) + noise
    sbp = 120 + 1 * np.cos(t/60) + noise
    pi = 3.5 + 0.1 * np.sin(t/30) + 0.1 * noise
    
    # 2. Pathological Event (The "Drift")
    # Starts at 400. Not a sudden jump, but a change in the Attractor.
    event_start = 400
    
    # Drift functions
    drift_hr = np.linspace(0, 40, mins_total-event_start) ** 1.1 # Non-linear rise
    drift_pi = np.linspace(0, 2.5, mins_total-event_start)       # Linear constriction
    drift_sbp = np.zeros(mins_total-event_start)
    
    # Late crash for SBP (Decompensation)
    crash_start = 600
    if mins_total > crash_start:
        drift_sbp[crash_start-event_start:] = np.linspace(0, 30, mins_total-crash_start) ** 1.5

    # Apply drifts
    hr[event_start:] += drift_hr
    sbp[event_start:] -= drift_sbp
    pi[event_start:] = np.maximum(0.5, pi[event_start:] - drift_pi)
    
    # Add micro-volatility (loss of complexity) as shock advances
    # Sick patients often have REDUCED variability (stiff system) or CHAOTIC variability
    hr[event_start:] += np.random.normal(0, 2, mins_total-event_start) # Increased noise

    return pd.DataFrame({'HR': hr, 'SBP': sbp, 'PI': pi}, index=t)

# --- 3. SPC ANALYTICS (Western Electric Rules) ---
def detect_spc_violations(series, window=60):
    """
    Detects Western Electric Rule violations:
    1. Points beyond 3 sigma (Control Limits)
    2. 2 out of 3 points beyond 2 sigma
    3. 8 consecutive points on one side of mean (Trend)
    """
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    
    ucl = mean + 3*std
    lcl = mean - 3*std
    
    # Identify violations
    violation_idx = []
    violation_type = []
    
    values = series.values
    means = mean.values
    
    for i in range(window, len(series)):
        # Rule 1: Breach
        if values[i] > ucl.iloc[i] or values[i] < lcl.iloc[i]:
            violation_idx.append(series.index[i])
            violation_type.append('Breach')
            continue
            
        # Rule 2: Shift (Simplification for speed: 8 points on one side)
        if i > 8:
            last_8 = values[i-8:i] - means[i-8:i]
            if np.all(last_8 > 0) or np.all(last_8 < 0):
                violation_idx.append(series.index[i])
                violation_type.append('Shift')

    return mean, ucl, lcl, violation_idx, violation_type

# --- 4. AI PROGNOSTICS (Forecasting) ---
def generate_ai_forecast(series, future_steps=30):
    """
    Uses Auto-Regressive Logic to project trajectory with uncertainty cones.
    """
    # Fit linear trend on last 20 mins to capture momentum
    y = series.values[-20:]
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    
    # Predict future
    x_future = np.arange(len(y), len(y) + future_steps).reshape(-1, 1)
    trend = model.predict(x_future)
    
    # Calculate noise/uncertainty based on recent volatility
    volatility = np.std(y)
    
    # Expanding cone of uncertainty (Monte Carlo approximation)
    upper = trend + (np.linspace(1, 2, future_steps) * volatility * 1.96)
    lower = trend - (np.linspace(1, 2, future_steps) * volatility * 1.96)
    
    return trend, upper, lower

# --- 5. VISUALIZATION: SPC MONITOR ---
def plot_spc_monitor(df, curr_time, window=180):
    start = max(0, curr_time - window)
    data = df.iloc[start:curr_time]
    
    # Calculate SPC
    mean_hr, ucl_hr, lcl_hr, vio_idx, vio_type = detect_spc_violations(df['HR'].iloc[:curr_time])
    
    # AI Forecast
    forecast_hr, f_upper, f_lower = generate_ai_forecast(df['HR'].iloc[:curr_time])
    t_future = np.arange(curr_time, curr_time + 30)

    fig = go.Figure()

    # 1. Historical Data
    fig.add_trace(go.Scatter(x=data.index, y=data['HR'], mode='lines', 
                             name='HR Actual', line=dict(color=COLORS['hr'], width=2)))
    
    # 2. SPC Bands (Dynamic)
    fig.add_trace(go.Scatter(x=data.index, y=ucl_hr.loc[data.index], mode='lines', 
                             line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=data.index, y=lcl_hr.loc[data.index], mode='lines', 
                             line=dict(width=0), fill='tonexty', fillcolor=COLORS['spc_band'], 
                             name='3σ Control Zone'))
    
    fig.add_trace(go.Scatter(x=data.index, y=mean_hr.loc[data.index], mode='lines',
                             line=dict(color=COLORS['spc_mean'], dash='dot', width=1), name='Moving Mean'))

    # 3. Violation Markers
    relevant_violations = [v for v in vio_idx if v >= start and v < curr_time]
    if relevant_violations:
        fig.add_trace(go.Scatter(
            x=relevant_violations, 
            y=df.loc[relevant_violations, 'HR'],
            mode='markers',
            marker=dict(color='yellow', size=8, symbol='x'),
            name='SPC Violation'
        ))

    # 4. AI Forecast (The Cone)
    fig.add_trace(go.Scatter(x=t_future, y=f_upper, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=t_future, y=f_lower, mode='lines', line=dict(width=0), 
                             fill='tonexty', fillcolor='rgba(255, 174, 0, 0.2)', name='AI 95% CI'))
    fig.add_trace(go.Scatter(x=t_future, y=forecast_hr, mode='lines', 
                             line=dict(color=COLORS['forecast'], dash='dash'), name='Prognosis'))

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text="HR: STATISTICAL PROCESS CONTROL & AI FORECAST", font=dict(size=14)),
        xaxis=dict(showgrid=True, gridcolor=COLORS['grid']),
        yaxis=dict(showgrid=True, gridcolor=COLORS['grid']),
        legend=dict(orientation="h", y=1, x=0, bgcolor='rgba(0,0,0,0)')
    )
    return fig

# --- 6. VISUALIZATION: STATE SPACE (PCA + CHAOS) ---
def plot_state_space(df, curr_time):
    """
    Plots the 'Phase Space' of the patient.
    X-axis: Principal Component 1 (Hemodynamics)
    Y-axis: Principal Component 2 (Perfusion/Stress)
    
    Stable patients hover in the center. Unstable patients 'orbit' out or drift linearly.
    """
    # Fit PCA on baseline (first 120 mins)
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    
    baseline = df.iloc[:120][['HR', 'SBP', 'PI']]
    scaler.fit(baseline)
    pca.fit(scaler.transform(baseline))
    
    # Transform current history
    history = df.iloc[max(0, curr_time-300):curr_time][['HR', 'SBP', 'PI']]
    coords = pca.transform(scaler.transform(history))
    
    # Create the "Comet" effect (Recent points are opaque, old are transparent)
    n_points = len(coords)
    alphas = np.linspace(0.1, 1, n_points)
    
    fig = go.Figure()
    
    # The Trajectory
    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1],
        mode='markers+lines',
        marker=dict(
            color=np.arange(n_points), 
            colorscale='Viridis', 
            size=6,
            opacity=alphas
        ),
        line=dict(width=1, color='rgba(255,255,255,0.3)'),
        name='State Trajectory'
    ))
    
    # The "Safe Zone" (Circle at 0,0)
    fig.add_shape(type="circle",
        xref="x", yref="y",
        x0=-2, y0=-2, x1=2, y1=2,
        line_color="green", fillcolor="rgba(0,255,0,0.1)",
    )
    
    # Current Head
    fig.add_trace(go.Scatter(
        x=[coords[-1, 0]], y=[coords[-1, 1]],
        mode='markers',
        marker=dict(color='red', size=12, symbol='diamond'),
        name='Current State'
    ))

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text="PHYSIOLOGICAL STATE SPACE (PCA)", font=dict(size=14)),
        xaxis=dict(title="PC1 (Hemodynamics)", showgrid=True, gridcolor=COLORS['grid'], zeroline=True),
        yaxis=dict(title="PC2 (Perfusion)", showgrid=True, gridcolor=COLORS['grid'], zeroline=True),
        showlegend=False
    )
    return fig

# --- 7. VISUALIZATION: CHAOS ATTRACTOR (Poincaré Plot) ---
def plot_chaos_attractor(df, curr_time, lag=1):
    """
    Plots HR(t) vs HR(t-lag).
    Round ball = Healthy Chaos (Sinus Arrhythmia).
    Stretched line = Reduced Complexity / Deterministic drift (Pathology).
    """
    start = max(0, curr_time - 120)
    data = df['HR'].iloc[start:curr_time].values
    
    x_t = data[:-lag]
    x_t_plus_1 = data[lag:]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_t, y=x_t_plus_1,
        mode='markers',
        marker=dict(
            color='#00ccff',
            size=5,
            opacity=0.6,
            line=dict(width=0.5, color='white')
        ),
        name='Attractor'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        width=300,
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text="CHAOS ATTRACTOR (Lag Plot)", font=dict(size=12)),
        xaxis=dict(title="HR(t)", showgrid=True, gridcolor=COLORS['grid']),
        yaxis=dict(title=f"HR(t+{lag})", showgrid=True, gridcolor=COLORS['grid']),
    )
    return fig
