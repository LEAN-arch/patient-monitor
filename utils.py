import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# --- 1. CONFIGURATION (Light Mode / Clinical Report Style) ---
COLORS = {
    'bg': '#ffffff',
    'card': '#f8f9fa',
    'text': '#2c3e50',
    'grid': '#e9ecef',
    'hr': '#0056b3',      # Clinical Blue
    'sbp': '#d63384',     # Magenta
    'forecast': '#fd7e14',# Orange
    'zone_a': 'rgba(220, 53, 69, 0.1)',   # Red Tint (3 Sigma)
    'zone_b': 'rgba(255, 193, 7, 0.15)',  # Yellow Tint (2 Sigma)
    'zone_c': 'rgba(40, 167, 69, 0.05)',  # Green Tint (1 Sigma)
    'marker': '#dc3545'   # Red for violations
}

# --- 2. ADVANCED SIMULATION ---
def simulate_patient_dynamics(mins_total=720):
    """
    Simulates a patient with subtle "Micro-Drifts" before the crash.
    """
    t = np.arange(mins_total)
    
    # 1. Baseline (Stable Attractor)
    noise = np.random.normal(0, 0.8, mins_total) # Tighter noise for sensitivity demo
    hr = 72 + 1.5 * np.sin(t/50) + noise
    sbp = 120 + 1 * np.cos(t/60) + noise
    pi = 3.5 + 0.1 * np.sin(t/30) + 0.05 * noise
    
    # 2. Pathological Drift (Subtle start at min 300)
    event_start = 300
    
    # "The Creep": Very slow linear rise that standard alarms miss, but SPC catches
    drift_hr = np.linspace(0, 25, mins_total-event_start) 
    drift_pi = np.linspace(0, 1.5, mins_total-event_start)
    
    # Apply drifts
    hr[event_start:] += drift_hr
    hr[event_start:] += np.random.normal(0, 1.5, mins_total-event_start) # Increased entropy
    
    # PI Drops
    pi[event_start:] = np.maximum(0.5, pi[event_start:] - drift_pi)

    return pd.DataFrame({'HR': hr, 'SBP': sbp, 'PI': pi}, index=t)

# --- 3. SENSITIVE SPC LOGIC (Western Electric Zones) ---
def detect_sensitive_spc(series, window=60):
    """
    Implements multi-zone sensitivity:
    - Zone A: Beyond 3 sigma (Critical)
    - Zone B: Beyond 2 sigma (Warning)
    - Zone C: Beyond 1 sigma (Early Shift)
    """
    # Rolling Statistics
    roll = series.rolling(window=window)
    mean = roll.mean()
    std = roll.std()
    
    # Define Sigma Bands
    ucl_3 = mean + 3*std
    lcl_3 = mean - 3*std
    ucl_2 = mean + 2*std
    lcl_2 = mean - 2*std
    ucl_1 = mean + 1*std
    lcl_1 = mean - 1*std
    
    # Violation Detection
    violations = []
    
    values = series.values
    
    for i in range(window, len(series)):
        val = values[i]
        
        # Rule 1: Any point outside 3 Sigma
        if val > ucl_3.iloc[i] or val < lcl_3.iloc[i]:
            violations.append((series.index[i], val, "Zone A Violation (3σ)"))
            continue
            
        # Rule 2: 2 out of 3 points beyond 2 Sigma (Sensitivity check)
        if i > 2:
            # Simple heuristic for this demo: Point outside 2 Sigma
            if val > ucl_2.iloc[i] or val < lcl_2.iloc[i]:
                violations.append((series.index[i], val, "Zone B Warning (2σ)"))

    return {
        'mean': mean, 'std': std,
        'u3': ucl_3, 'l3': lcl_3,
        'u2': ucl_2, 'l2': lcl_2,
        'u1': ucl_1, 'l1': lcl_1,
        'violations': violations
    }

# --- 4. AI PROGNOSTICS ---
def generate_ai_forecast(series, future_steps=30):
    # (Same robust logic as before, optimized for visual output)
    y = series.values[-20:]
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    
    x_future = np.arange(len(y), len(y) + future_steps).reshape(-1, 1)
    trend = model.predict(x_future)
    volatility = np.std(y)
    
    upper = trend + (np.linspace(1, 2, future_steps) * volatility * 2.0)
    lower = trend - (np.linspace(1, 2, future_steps) * volatility * 2.0)
    
    return trend, upper, lower

# --- 5. HIGH-SENSITIVITY VISUALIZATION ---
def plot_sensitive_spc(df, curr_time, window=120):
    start = max(0, curr_time - window)
    data = df.iloc[start:curr_time]
    
    # Run Sensitive SPC
    spc = detect_sensitive_spc(df['HR'].iloc[:curr_time])
    
    # AI Forecast
    forecast_hr, f_upper, f_lower = generate_ai_forecast(df['HR'].iloc[:curr_time])
    t_future = np.arange(curr_time, curr_time + 30)

    fig = go.Figure()

    # --- A. The Sigma Zones (Background Bands) ---
    # We plot these first so they sit behind the data
    # Zone B (2-3 Sigma) - Warning Zone
    fig.add_trace(go.Scatter(x=data.index, y=spc['u3'].loc[data.index], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=data.index, y=spc['u2'].loc[data.index], mode='lines', line=dict(width=0), 
                             fill='tonexty', fillcolor=COLORS['zone_b'], name='Warning Zone (2-3σ)'))
    
    fig.add_trace(go.Scatter(x=data.index, y=spc['l2'].loc[data.index], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=data.index, y=spc['l3'].loc[data.index], mode='lines', line=dict(width=0), 
                             fill='tonexty', fillcolor=COLORS['zone_b'], showlegend=False))

    # Zone C (1 Sigma) - Noise Zone
    fig.add_trace(go.Scatter(x=data.index, y=spc['u1'].loc[data.index], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=data.index, y=spc['l1'].loc[data.index], mode='lines', line=dict(width=0), 
                             fill='tonexty', fillcolor=COLORS['zone_c'], name='Stable Zone (1σ)'))

    # Center Line
    fig.add_trace(go.Scatter(x=data.index, y=spc['mean'].loc[data.index], mode='lines',
                             line=dict(color='#adb5bd', dash='dash', width=1), name='Mean'))

    # --- B. The Data Stream ---
    fig.add_trace(go.Scatter(x=data.index, y=data['HR'], mode='lines', 
                             name='HR Actual', line=dict(color=COLORS['hr'], width=2.5)))

    # --- C. Violation Markers (High Sensitivity) ---
    relevant_violations = [v for v in spc['violations'] if v[0] >= start and v[0] < curr_time]
    if relevant_violations:
        vx = [v[0] for v in relevant_violations]
        vy = [v[1] for v in relevant_violations]
        fig.add_trace(go.Scatter(
            x=vx, y=vy, mode='markers',
            marker=dict(color=COLORS['marker'], size=6, symbol='circle-open', line=dict(width=2)),
            name='SPC Trigger'
        ))

    # --- D. AI Forecast Cone ---
    fig.add_trace(go.Scatter(x=t_future, y=f_upper, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=t_future, y=f_lower, mode='lines', line=dict(width=0), 
                             fill='tonexty', fillcolor='rgba(253, 126, 20, 0.2)', name='AI Prediction'))
    fig.add_trace(go.Scatter(x=t_future, y=forecast_hr, mode='lines', 
                             line=dict(color=COLORS['forecast'], dash='dot', width=2)))

    fig.update_layout(
        template="plotly_white", # Light Mode
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text="High-Sensitivity Process Control (Western Electric Rules)", font=dict(color=COLORS['text'])),
        xaxis=dict(showgrid=True, gridcolor=COLORS['grid'], title="Time (min)"),
        yaxis=dict(showgrid=True, gridcolor=COLORS['grid'], title="Heart Rate (BPM)"),
        legend=dict(orientation="h", y=1.02, x=0)
    )
    return fig

def plot_state_space_light(df, curr_time):
    # PCA Logic (same as before)
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    baseline = df.iloc[:120][['HR', 'SBP', 'PI']]
    scaler.fit(baseline)
    pca.fit(scaler.transform(baseline))
    
    history = df.iloc[max(0, curr_time-300):curr_time][['HR', 'SBP', 'PI']]
    coords = pca.transform(scaler.transform(history))
    
    fig = go.Figure()
    
    # Trajectory (Blue/Grey for light mode)
    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1],
        mode='markers+lines',
        marker=dict(
            color=np.arange(len(coords)), 
            colorscale='Blues', 
            size=5,
            opacity=0.8
        ),
        line=dict(width=1, color='rgba(0,0,0,0.2)'),
        name='State'
    ))
    
    # Safe Zone
    fig.add_shape(type="circle", xref="x", yref="y", x0=-2, y0=-2, x1=2, y1=2,
        line_color="#28a745", fillcolor="rgba(40, 167, 69, 0.1)")
    
    # Current Head
    fig.add_trace(go.Scatter(x=[coords[-1, 0]], y=[coords[-1, 1]], mode='markers',
        marker=dict(color='#dc3545', size=10, symbol='diamond'), name='Current'))

    fig.update_layout(
        template="plotly_white",
        height=300,
        title=dict(text="State Space Trajectory (PCA)", font=dict(color=COLORS['text'], size=12)),
        xaxis=dict(title="PC1", showgrid=True, gridcolor=COLORS['grid']),
        yaxis=dict(title="PC2", showgrid=True, gridcolor=COLORS['grid']),
        showlegend=False
    )
    return fig

def plot_chaos_light(df, curr_time):
    start = max(0, curr_time - 120)
    data = df['HR'].iloc[start:curr_time].values
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[:-1], y=data[1:],
        mode='markers',
        marker=dict(color='#6c757d', size=4, opacity=0.5), # Grey dots
        name='Attractor'
    ))
    
    fig.update_layout(
        template="plotly_white",
        height=250, width=250,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Poincaré Plot (Chaos)", font=dict(color=COLORS['text'], size=12)),
        xaxis=dict(showgrid=True, gridcolor=COLORS['grid']),
        yaxis=dict(showgrid=True, gridcolor=COLORS['grid']),
    )
    return fig
