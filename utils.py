import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# --- 1. DESIGN SYSTEM (Clinical Light Mode) ---
THEME = {
    'bg_paper': '#ffffff',
    'text': '#1e293b',
    'grid': '#f1f5f9',
    # Physiology Colors
    'hr': '#0369a1',       # Clinical Blue
    'map': '#be185d',      # Magenta (Perfusion)
    'pp': '#7c3aed',       # Violet (Stroke Volume)
    'co': '#059669',       # Emerald (Flow)
    'svr': '#d97706',      # Amber (Resistance)
    'entropy': '#64748b',  # Slate (Neuro)
    # Zones
    'zone_safe': 'rgba(16, 185, 129, 0.1)',
    'zone_warn': 'rgba(245, 158, 11, 0.1)',
    'zone_crit': 'rgba(239, 68, 68, 0.1)'
}

# --- 2. ADVANCED MATH CLASSES ---
class KalmanFilter1D:
    """Estimates true state from noisy sensor data."""
    def __init__(self, process_var, measure_var):
        self.process_var = process_var
        self.measure_var = measure_var
        self.posteri_est = 0.0
        self.posteri_error = 1.0

    def update(self, measurement):
        priori_est = self.posteri_est
        priori_error = self.posteri_error + self.process_var
        blending = priori_error / (priori_error + self.measure_var)
        self.posteri_est = priori_est + blending * (measurement - priori_est)
        self.posteri_error = (1 - blending) * priori_error
        return self.posteri_est

# --- 3. COMPREHENSIVE SIMULATION ---
def simulate_patient_physiology(mins=720):
    t = np.arange(mins)
    
    # 1. Pink Noise Generator (Biological Realism)
    def bio_noise(n, amp=1.0):
        w = np.random.normal(0, 1, n)
        return np.convolve(w, [0.05]*20, mode='same') * amp

    # 2. Baseline Vitals
    true_hr = 72 + bio_noise(mins, 4)
    true_sbp = 125 + bio_noise(mins, 3)
    true_dbp = 80 + bio_noise(mins, 2)
    rr = 16 + bio_noise(mins, 1)
    
    # 3. CLINICAL EVENT: "Cold Sepsis" / Cardiogenic Cascade
    # Timeline: T=240 (Onset) -> T=600 (Crash)
    start = 240
    
    # A. Stroke Volume Failure (Preload/Contractility drop)
    # Pulse Pressure narrows significantly
    sbp_drop = np.linspace(0, 45, mins-start)
    dbp_drop = np.linspace(0, 10, mins-start) # DBP maintained by SVR initially
    
    true_sbp[start:] -= sbp_drop
    true_dbp[start:] -= dbp_drop
    
    # B. Compensatory Tachycardia
    hr_rise = np.linspace(0, 50, mins-start)
    true_hr[start:] += hr_rise
    
    # C. Loss of Complexity (Sepsis Signature - Entropy Drop)
    # Signal becomes "smoother"
    dampening = np.linspace(1, 0.15, mins-start)
    true_hr[start:] = true_hr[start:] * dampening + (true_hr[start:].mean() * (1-dampening))

    # D. Sensor Noise (for Kalman)
    obs_hr = true_hr + np.random.normal(0, 3, mins)

    # 4. Build DataFrame
    df = pd.DataFrame({'Obs_HR': obs_hr, 'True_HR': true_hr, 'SBP': true_sbp, 'DBP': true_dbp, 'RR': rr}, index=t)
    
    # 5. Derived Physics
    # Kalman Filter
    kf = KalmanFilter1D(1e-5, 0.5)
    df['Kalman_HR'] = [kf.update(x) for x in df['Obs_HR']]
    
    # Hemodynamics
    df['MAP'] = (df['SBP'] + 2*df['DBP']) / 3
    df['PP'] = df['SBP'] - df['DBP'] # Stroke Volume Proxy
    df['SI'] = df['Kalman_HR'] / df['SBP'] # Shock Index
    
    # Cardiac Output Proxy (L/min)
    df['CO'] = (df['Kalman_HR'] * df['PP']) / 1000 * 2.2
    
    # SVR Proxy (dyne)
    df['SVR'] = df['MAP'] / df['CO'] * 80
    
    # Cardiac Effort Index (Novel Metric)
    df['CEI'] = df['Kalman_HR'] / df['PP']
    
    # Entropy (Complexity)
    df['Entropy'] = df['Obs_HR'].rolling(30).apply(lambda x: np.std(np.diff(x))).fillna(1.0)
    
    return df

# --- 4. VISUALIZATION ENGINES ---

def plot_command_timeline(df, curr_time):
    """
    HUD: Combines Kalman HR (True State), MAP (Perfusion), and AI Forecast.
    """
    start = max(0, curr_time - 180)
    data = df.iloc[start:curr_time]
    
    # Prognostic Linear Regression on MAP
    recent_map = df['MAP'].iloc[curr_time-30:curr_time].values
    model = LinearRegression().fit(np.arange(len(recent_map)).reshape(-1,1), recent_map)
    fut_x = np.arange(len(recent_map), len(recent_map)+45).reshape(-1,1)
    pred_map = model.predict(fut_x)
    t_fut = np.arange(curr_time, curr_time+45)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 1. Kalman HR (Left Axis)
    fig.add_trace(go.Scatter(x=data.index, y=data['Kalman_HR'], mode='lines', 
                             line=dict(color=THEME['hr'], width=2), name='True HR'), secondary_y=False)
    
    # 2. MAP (Right Axis)
    fig.add_trace(go.Scatter(x=data.index, y=data['MAP'], mode='lines', 
                             line=dict(color=THEME['map'], width=2), name='MAP'), secondary_y=True)
    
    # 3. MAP Forecast (Dotted)
    fig.add_trace(go.Scatter(x=t_fut, y=pred_map, mode='lines', 
                             line=dict(color=THEME['map'], dash='dot'), name='MAP Forecast'), secondary_y=True)

    # 4. Critical Line
    fig.add_hline(y=65, line_color=THEME['map'], line_dash="dot", opacity=0.5, secondary_y=True)

    fig.update_layout(template="plotly_white", height=300, margin=dict(l=0,r=0,t=20,b=0), 
                      hovermode="x unified", legend=dict(orientation="h", y=1.1))
    fig.update_yaxes(title="Heart Rate", secondary_y=False, gridcolor=THEME['grid'])
    fig.update_yaxes(title="MAP (mmHg)", secondary_y=True, showgrid=False)
    
    return fig

def plot_starling_curve(df, curr_time):
    """
    Fluid Physics: Preload (DBP) vs Stroke Volume (PP).
    """
    start = max(0, curr_time - 120)
    data = df.iloc[start:curr_time]
    
    fig = go.Figure()
    
    # Reference Curve
    x_ref = np.linspace(40, 100, 100)
    y_ref = np.log(x_ref)*20 - 40
    fig.add_trace(go.Scatter(x=x_ref, y=y_ref, mode='lines', line=dict(color='#cbd5e1', dash='dot'), name='Ideal'))
    
    # Trajectory
    fig.add_trace(go.Scatter(x=data['DBP'], y=data['PP'], mode='lines', 
                             line=dict(color=THEME['pp'], width=3), name='Patient'))
    
    # Current
    fig.add_trace(go.Scatter(x=[data['DBP'].iloc[-1]], y=[data['PP'].iloc[-1]], 
                             mode='markers', marker=dict(color=THEME['hr'], size=12, symbol='cross'), name='Now'))

    fig.update_layout(template="plotly_white", height=280, margin=dict(l=0,r=0,t=40,b=0),
                      title="<b>Starling Curve (Fluid Physics)</b>",
                      xaxis=dict(title="Preload (DBP)", gridcolor=THEME['grid']),
                      yaxis=dict(title="Stroke Vol (PP)", gridcolor=THEME['grid']), showlegend=False)
    return fig

def plot_shock_phenotype(df, curr_time):
    """
    Diagnostic Quadrants: Shock Index vs Pulse Pressure.
    """
    data = df.iloc[max(0, curr_time-60):curr_time]
    
    fig = go.Figure()
    
    # Zones
    fig.add_shape(type="rect", x0=0.4, y0=40, x1=0.9, y1=90, fillcolor=THEME['zone_safe'], line_width=0, layer="below")
    fig.add_annotation(x=0.65, y=65, text="STABLE", font=dict(color="green", size=10), showarrow=False)
    
    fig.add_shape(type="rect", x0=0.9, y0=20, x1=1.5, y1=45, fillcolor=THEME['zone_warn'], line_width=0, layer="below")
    fig.add_annotation(x=1.2, y=32, text="COMPENSATED\n(Low SV)", font=dict(color="orange", size=10), showarrow=False)
    
    fig.add_shape(type="rect", x0=1.2, y0=0, x1=1.8, y1=25, fillcolor=THEME['zone_crit'], line_width=0, layer="below")
    fig.add_annotation(x=1.5, y=12, text="CRITICAL", font=dict(color="red", size=10), showarrow=False)

    # Path
    fig.add_trace(go.Scatter(x=data['SI'], y=data['PP'], mode='lines+markers', 
                             line=dict(color='gray', width=1), marker=dict(size=4)))
    
    # Head
    fig.add_trace(go.Scatter(x=[data['SI'].iloc[-1]], y=[data['PP'].iloc[-1]], 
                             mode='markers', marker=dict(color='black', size=12, symbol='diamond')))

    fig.update_layout(template="plotly_white", height=280, margin=dict(l=0,r=0,t=40,b=0),
                      title="<b>Shock Phenotype (Diagnosis)</b>",
                      xaxis=dict(title="Shock Index", gridcolor=THEME['grid']),
                      yaxis=dict(title="Pulse Pressure", gridcolor=THEME['grid']), showlegend=False)
    return fig

def plot_organ_matrix(df, curr_time):
    """
    4-Panel System View.
    """
    start = max(0, curr_time - 60)
    data = df.iloc[start:curr_time]
    
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=("Cardiac (Flow)", "Vascular (Resistance)", "Neural (Entropy)", "Metabolic (Comp)"))
    
    fig.add_trace(go.Scatter(x=data.index, y=data['CO'], line=dict(color=THEME['co'])), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SVR'], line=dict(color=THEME['svr'])), row=1, col=2)
    fig.add_trace(go.Scatter(x=data.index, y=data['Entropy'], line=dict(color=THEME['entropy'])), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['RR'], line=dict(color=THEME['map'])), row=2, col=2)
    
    fig.update_layout(template="plotly_white", height=250, margin=dict(l=0,r=0,t=30,b=0), showlegend=False)
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)
    return fig

def plot_horizon(df, curr_time):
    """
    Time-To-Event Horizon.
    """
    recent_map = df['MAP'].iloc[curr_time-30:curr_time].values
    model = LinearRegression().fit(np.arange(len(recent_map)).reshape(-1,1), recent_map)
    fut_x = np.arange(len(recent_map), len(recent_map)+60).reshape(-1,1)
    pred_map = model.predict(fut_x)
    
    # Find crash
    crash_idx = np.argmax(pred_map < 65)
    t_crash = crash_idx if crash_idx > 0 else None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=recent_map, x=np.arange(-30, 0), name="History", line=dict(color=THEME['map'])))
    fig.add_trace(go.Scatter(y=pred_map, x=np.arange(0, 60), name="Forecast", line=dict(color=THEME['map'], dash='dot')))
    fig.add_hline(y=65, line_color='red', opacity=0.3)
    
    if t_crash:
         fig.add_vline(x=t_crash, line_dash="dot", line_color="red")
         fig.add_annotation(x=t_crash, y=65, text=f"CRITICAL<br>+{t_crash}min", showarrow=True, arrowhead=2)

    fig.update_layout(template="plotly_white", height=200, margin=dict(l=0,r=0,t=30,b=0),
                      title="<b>Prognostic Horizon (Time-to-Crash)</b>",
                      xaxis=dict(title="Minutes from Now", gridcolor=THEME['grid']),
                      yaxis=dict(title="MAP", gridcolor=THEME['grid']))
    return fig
