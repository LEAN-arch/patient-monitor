import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# --- 1. CONFIGURATION (Clinical Light Theme) ---
COLORS = {
    'bg': '#ffffff',
    'text': '#2c3e50',
    'grid': '#e9ecef',
    'hr': '#0056b3',       # Strong Blue
    'sbp': '#d63384',      # Magenta
    'si': '#fd7e14',       # Orange (Shock Index)
    'zone_warn': 'rgba(255, 193, 7, 0.15)',
    'zone_safe': 'rgba(40, 167, 69, 0.05)',
    'violation': '#dc3545' # Red
}

# --- 2. ADVANCED SIMULATION ---
def simulate_patient_dynamics(mins_total=720):
    t = np.arange(mins_total)
    
    # Baseline with noise
    noise = np.random.normal(0, 0.8, mins_total)
    hr = 72 + 1.5 * np.sin(t/50) + noise
    sbp = 120 + 1 * np.cos(t/60) + noise
    pi = 3.5 + 0.1 * np.sin(t/30) + 0.05 * noise
    spo2 = np.random.normal(98, 0.5, mins_total)
    rr = np.random.normal(16, 1, mins_total)

    # EVENT: Gradual Sepsis/Shock "Creep" starting at min 300
    event_start = 300
    
    # HR drifts UP (Tachycardia)
    hr[event_start:] += np.linspace(0, 35, mins_total-event_start) + np.random.normal(0, 2, mins_total-event_start)
    
    # SBP drifts DOWN (Hypotension) - slowly at first
    sbp[event_start:] -= np.linspace(0, 25, mins_total-event_start)
    
    # PI crashes EARLY (Vasoconstriction)
    pi[event_start:] = np.maximum(0.2, pi[event_start:] - np.linspace(0, 2.5, mins_total-event_start))
    
    # RR increases (Compensation)
    rr[event_start:] += np.linspace(0, 10, mins_total-event_start)

    df = pd.DataFrame({'HR': hr, 'SBP': sbp, 'PI': pi, 'SpO2': spo2, 'RR': rr}, index=t)
    df['SI'] = df['HR'] / df['SBP'] # Calculate Shock Index
    return df

# --- 3. ANALYTICS ENGINES ---

def detect_western_electric_spc(series, window=60):
    """
    Detects subtle deviations using Zone Rules.
    """
    roll = series.rolling(window=window)
    mean = roll.mean()
    std = roll.std()
    
    # Zones
    ucl_3 = mean + 3*std
    ucl_2 = mean + 2*std
    ucl_1 = mean + 1*std
    lcl_1 = mean - 1*std
    lcl_2 = mean - 2*std
    lcl_3 = mean - 3*std
    
    violations = []
    values = series.values
    
    for i in range(window, len(series)):
        val = values[i]
        # Rule 1: Breach 3 Sigma
        if val > ucl_3.iloc[i] or val < lcl_3.iloc[i]:
            violations.append((series.index[i], val, "Zone A (Critical)"))
            continue
        # Rule 2: Warning 2 Sigma
        if val > ucl_2.iloc[i] or val < lcl_2.iloc[i]:
            violations.append((series.index[i], val, "Zone B (Warning)"))

    return {'mean': mean, 'u3': ucl_3, 'u2': ucl_2, 'u1': ucl_1, 
            'l1': lcl_1, 'l2': lcl_2, 'l3': lcl_3, 'violations': violations}

def run_pca_analysis(df, curr_time):
    """
    Runs PCA and determines WHICH variables are driving the movement.
    """
    # 1. Fit on baseline (first 120 mins)
    baseline = df.iloc[:120][['HR', 'SBP', 'PI', 'RR', 'SpO2']]
    scaler = StandardScaler()
    scaler.fit(baseline)
    scaled_baseline = scaler.transform(baseline)
    
    pca = PCA(n_components=2)
    pca.fit(scaled_baseline)
    
    # 2. Transform current window
    history = df.iloc[max(0, curr_time-300):curr_time][['HR', 'SBP', 'PI', 'RR', 'SpO2']]
    scaled_history = scaler.transform(history)
    coords = pca.transform(scaled_history)
    
    # 3. Analyze Loadings (The "Why")
    # This tells us: "PC1 is 60% HR and 40% SBP"
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=['HR', 'SBP', 'PI', 'RR', 'SpO2'])
    
    return coords, loadings, pca.explained_variance_ratio_

def calculate_risk_score(row, violations):
    """
    Composite Risk Score (0-100)
    """
    score = 0
    # Physiology
    if row['HR'] > 100: score += 20
    if row['SBP'] < 100: score += 25
    if row['PI'] < 1.5: score += 15
    if row['SI'] > 0.9: score += 20 # Shock Index
    
    # SPC History
    recent_vios = [v for v in violations if v[0] > row.name - 15]
    if len(recent_vios) > 0: score += 20
    
    return min(100, score)

# --- 4. VISUALIZATIONS ---

def plot_sensitive_spc_with_forecast(df, curr_time):
    data = df.iloc[max(0, curr_time-180):curr_time]
    spc = detect_western_electric_spc(df['HR'].iloc[:curr_time])
    
    # AI Forecast
    y = df['HR'].iloc[curr_time-30:curr_time].values
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    future_x = np.arange(len(y), len(y)+30).reshape(-1, 1)
    trend = model.predict(future_x)
    
    fig = go.Figure()

    # Zones (The Sensitivity Layer)
    fig.add_trace(go.Scatter(x=data.index, y=spc['u2'].loc[data.index], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=data.index, y=spc['u3'].loc[data.index], mode='lines', line=dict(width=0), 
                             fill='tonexty', fillcolor=COLORS['zone_warn'], name='Warning Zone'))
    
    fig.add_trace(go.Scatter(x=data.index, y=spc['l1'].loc[data.index], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=data.index, y=spc['u1'].loc[data.index], mode='lines', line=dict(width=0), 
                             fill='tonexty', fillcolor=COLORS['zone_safe'], name='Stable Zone'))

    # Data
    fig.add_trace(go.Scatter(x=data.index, y=data['HR'], mode='lines', name='HR', line=dict(color=COLORS['hr'], width=2)))
    
    # Violations
    v_x = [v[0] for v in spc['violations'] if v[0] >= data.index[0]]
    v_y = [v[1] for v in spc['violations'] if v[0] >= data.index[0]]
    if v_x:
        fig.add_trace(go.Scatter(x=v_x, y=v_y, mode='markers', marker=dict(color='red', size=6, symbol='x'), name='SPC Alert'))

    # Forecast
    t_fut = np.arange(curr_time, curr_time+30)
    fig.add_trace(go.Scatter(x=t_fut, y=trend, mode='lines', line=dict(dash='dot', color='#fd7e14'), name='AI Forecast'))

    fig.update_layout(
        template="plotly_white", height=300, margin=dict(l=0,r=0,t=30,b=0),
        title="<b>High-Sensitivity SPC (Heart Rate)</b>",
        xaxis=dict(showgrid=True, gridcolor=COLORS['grid']),
        yaxis=dict(showgrid=True, gridcolor=COLORS['grid'])
    )
    return fig

def plot_heatmap_restored(df, curr_time):
    """
    Shows Z-score deviation for ALL variables.
    """
    # Calculate Z-scores relative to baseline
    baseline = df.iloc[:120].mean()
    std = df.iloc[:120].std()
    
    window = df.iloc[max(0, curr_time-60):curr_time]
    z_scores = (window - baseline) / std
    z_scores = z_scores[['HR','SBP','PI','RR','SpO2']].T
    
    # Clip for color
    z_scores = z_scores.clip(-4, 4)
    
    fig = go.Figure(data=go.Heatmap(
        z=z_scores.values,
        x=z_scores.columns,
        y=z_scores.index,
        colorscale='RdBu_r', # Red = High, Blue = Low
        zmid=0
    ))
    
    fig.update_layout(
        template="plotly_white", height=250, margin=dict(l=0,r=0,t=30,b=0),
        title="<b>Deviation Heatmap (Z-Score)</b>",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    return fig

def plot_radar_snapshot(row):
    cats = ['Tachycardia', 'Hypotension', 'Hypoperfusion', 'Desaturation', 'Tachypnea']
    # Normalize 0-1 (1 = Bad)
    vals = [
        np.clip((row['HR']-60)/80, 0, 1),
        np.clip((130-row['SBP'])/60, 0, 1), # Inverted
        np.clip((4-row['PI'])/4, 0, 1),     # Inverted
        np.clip((100-row['SpO2'])/15, 0, 1),# Inverted
        np.clip((row['RR']-12)/20, 0, 1)
    ]
    vals += [vals[0]]
    cats += [cats[0]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill='toself', line_color='#dc3545'))
    fig.add_trace(go.Scatterpolar(r=[0.3]*6, theta=cats, line_color='green', line_dash='dot', hoverinfo='skip'))
    
    fig.update_layout(
        template="plotly_white", height=250, margin=dict(l=40,r=40,t=30,b=20),
        title="<b>Physiological Footprint</b>",
        polar=dict(radialaxis=dict(visible=False, range=[0,1]))
    )
    return fig

def plot_pca_explained(coords, loadings):
    """
    PCA Plot with Vector Explanations.
    """
    fig = go.Figure()
    
    # 1. Trajectory
    fig.add_trace(go.Scatter(
        x=coords[:,0], y=coords[:,1], 
        mode='lines', line=dict(color='rgba(0,0,0,0.2)', width=1), name='History'
    ))
    # Current Point
    fig.add_trace(go.Scatter(
        x=[coords[-1,0]], y=[coords[-1,1]], 
        mode='markers', marker=dict(color='#dc3545', size=12, symbol='diamond'), name='Current'
    ))
    # Safe Zone
    fig.add_shape(type="circle", x0=-2, y0=-2, x1=2, y1=2, line_color="green", fillcolor="rgba(0,255,0,0.1)")

    # 2. Add Loading Vectors (Arrows showing what drives direction)
    # Scale up for visibility
    scale = 3 
    for feature in loadings.index:
        x_vec = loadings.loc[feature, 'PC1'] * scale
        y_vec = loadings.loc[feature, 'PC2'] * scale
        
        fig.add_annotation(
            x=x_vec, y=y_vec, ax=0, ay=0, xref='x', yref='y', axref='x', ayref='y',
            text=feature, showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#0056b3'
        )

    fig.update_layout(
        template="plotly_white", height=350,
        title="<b>State Space (PCA) & Driver Vectors</b>",
        xaxis=dict(title="PC1 (Global Instability)", showgrid=True),
        yaxis=dict(title="PC2 (Secondary Variance)", showgrid=True),
        showlegend=False
    )
    return fig

def plot_chaos_attractor(df, curr_time):
    data = df['HR'].iloc[max(0, curr_time-120):curr_time].values
    fig = go.Figure(go.Scatter(
        x=data[:-1], y=data[1:], mode='markers',
        marker=dict(color='rgba(0,0,0,0.5)', size=3)
    ))
    fig.update_layout(
        template="plotly_white", height=250, width=250,
        title="<b>Chaos (Poincaré)</b>",
        xaxis=dict(title="HR(t)", showgrid=True),
        yaxis=dict(title="HR(t+1)", showgrid=True),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig
