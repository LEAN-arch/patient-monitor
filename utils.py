import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression

# --- 1. DESIGN SYSTEM (Clinical Light Mode) ---
THEME = {
    'bg': '#ffffff',
    'text': '#1f2937',
    'grid': '#f3f4f6',
    # Physiology
    'hr': '#0284c7',       # Vivid Blue (Heart Rate)
    'hr_kalman': '#0369a1',# Darker Blue (True State)
    'map': '#be185d',      # Magenta (Perfusion Pressure)
    'pp': '#8b5cf6',       # Violet (Stroke Volume/Pulse Pressure)
    'cei': '#d97706',      # Amber (Effort Index)
    'entropy': '#059669',  # Emerald (Complexity)
    # Zones
    'zone_ok': 'rgba(16, 185, 129, 0.1)',
    'zone_warn': 'rgba(245, 158, 11, 0.1)',
    'zone_crit': 'rgba(239, 68, 68, 0.1)'
}

# --- 2. ADVANCED MATH CLASSES ---

class KalmanFilter1D:
    """
    Separates true physiological signal from sensor noise.
    """
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

def calc_entropy(series, window=30):
    """
    Computes rolling sample entropy (physiological complexity).
    """
    # Simplified fast proxy for visualization: 
    # Inverse of standard deviation of differences (Smoothness = Low Entropy)
    return series.rolling(window).apply(lambda x: np.std(np.diff(x))).fillna(1.0)

# --- 3. PHYSIOLOGICAL SIMULATION (Pink Noise + Pathophysiology) ---

def simulate_comprehensive_patient(mins=720):
    t = np.arange(mins)
    
    # 1/f Pink Noise Generator (Biologically Realistic)
    def pink_noise(n, amp=1.0):
        white = np.random.normal(0, 1, n)
        b = [0.05] * 20 # Low pass filter
        return np.convolve(white, b, mode='same') * amp

    # Base Vitals
    true_hr = 72 + pink_noise(mins, 5)
    true_sbp = 120 + pink_noise(mins, 3)
    true_dbp = 80 + pink_noise(mins, 2)
    pi = 4.0 + pink_noise(mins, 0.5)

    # EVENT: Septic "Uncoupling" (Starts T=250)
    # The heart works harder (HR Up) but pumps less (PP Down).
    start = 250
    
    # 1. Vasodilation/Constriction Flux
    drift_pi = np.linspace(0, 3.5, mins-start)
    pi[start:] = np.maximum(0.2, pi[start:] - drift_pi)
    
    # 2. Pulse Pressure Narrowing (Preload Failure)
    sbp_drop = np.linspace(0, 35, mins-start)
    dbp_drop = np.linspace(0, 5, mins-start) # DBP stays higher due to clamping
    true_sbp[start:] -= sbp_drop
    true_dbp[start:] -= dbp_drop
    
    # 3. Compensatory Tachycardia
    hr_rise = np.linspace(0, 45, mins-start)
    true_hr[start:] += hr_rise

    # 4. Loss of Entropy (Sepsis Signature)
    # Signal becomes "smoother" (metronomic)
    dampening = np.linspace(1, 0.1, mins-start)
    true_hr[start:] = true_hr[start:] * dampening + (true_hr[start:].mean() * (1-dampening))

    # 5. Add Sensor Noise for Kalman Demo
    sensor_noise = np.random.normal(0, 3, mins) # Movement artifact
    observed_hr = true_hr + sensor_noise

    # Build DataFrame
    df = pd.DataFrame({'Obs_HR': observed_hr, 'True_HR': true_hr, 'SBP': true_sbp, 'DBP': true_dbp, 'PI': pi}, index=t)
    
    # --- DERIVED METRICS ---
    
    # Kalman Filtering
    kf = KalmanFilter1D(1e-5, 0.5)
    df['Kalman_HR'] = [kf.update(x) for x in df['Obs_HR']]
    
    # Hemodynamics
    df['MAP'] = (df['SBP'] + 2*df['DBP']) / 3
    df['PP'] = df['SBP'] - df['DBP'] # Pulse Pressure
    
    # Advanced Prognostics
    df['CEI'] = df['Kalman_HR'] / df['PP'] # Cardiac Effort Index
    df['Entropy'] = calc_entropy(df['Obs_HR']) # Complexity
    
    return df

# --- 4. VISUALIZATION ENGINES ---

def plot_command_strip(df, curr_time):
    """
    The Master Timeline:
    Track 1: Kalman-Filtered HR (Signal vs Noise).
    Track 2: MAP & Pulse Pressure (The "Life Force").
    """
    start = max(0, curr_time - 180)
    data = df.iloc[start:curr_time]
    
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # --- ROW 1: HR & Kalman ---
    # Raw Noise (Ghost)
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Obs_HR'],
        mode='markers', marker=dict(color='#9ca3af', size=2, opacity=0.4),
        name='Raw Sensor'
    ), row=1, col=1)
    
    # True State (Kalman)
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Kalman_HR'],
        mode='lines', line=dict(color=THEME['hr'], width=2.5),
        name='True HR (Kalman)'
    ), row=1, col=1)
    
    # --- ROW 2: Hemodynamics ---
    # MAP Line
    fig.add_trace(go.Scatter(
        x=data.index, y=data['MAP'],
        mode='lines', line=dict(color=THEME['map'], width=2),
        name='MAP'
    ), row=2, col=1)
    
    # Pulse Pressure Area (Band)
    fig.add_trace(go.Scatter(
        x=data.index, y=data['SBP'], line=dict(width=0), showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data['DBP'], line=dict(width=0), 
        fill='tonexty', fillcolor='rgba(139, 92, 246, 0.15)',
        name='Pulse Pressure'
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_white", height=400, margin=dict(l=0, r=0, t=20, b=0),
        hovermode="x unified", legend=dict(orientation="h", y=1.05, x=0)
    )
    fig.update_yaxes(title="Heart Rate", row=1, col=1, gridcolor=THEME['grid'])
    fig.update_yaxes(title="Pressures (mmHg)", row=2, col=1, gridcolor=THEME['grid'])
    
    return fig

def plot_hemo_loop(df, curr_time):
    """
    Frank-Starling Proxy: HR (Cost) vs PP (Benefit).
    """
    start = max(0, curr_time - 120)
    data = df.iloc[start:curr_time]
    
    fig = go.Figure()
    
    # Background Zones
    fig.add_shape(type="rect", x0=40, y0=40, x1=90, y1=100, fillcolor=THEME['zone_ok'], line_width=0, layer="below")
    fig.add_shape(type="rect", x0=100, y0=0, x1=160, y1=35, fillcolor=THEME['zone_crit'], line_width=0, layer="below")
    fig.add_annotation(x=130, y=15, text="CARDIAC FAILURE", showarrow=False, font=dict(color="red", size=10))

    # Trajectory
    fig.add_trace(go.Scatter(
        x=data['Kalman_HR'], y=data['PP'],
        mode='lines', line=dict(width=3, color=THEME['pp']), opacity=0.8,
        name='Work Loop'
    ))
    
    # Current Head
    fig.add_trace(go.Scatter(
        x=[data['Kalman_HR'].iloc[-1]], y=[data['PP'].iloc[-1]],
        mode='markers', marker=dict(size=12, color=THEME['map'], symbol='diamond'),
        name='Current'
    ))

    fig.update_layout(
        template="plotly_white", height=300, 
        title=dict(text="<b>Hemodynamic Work Loop</b>", font=dict(size=12)),
        xaxis=dict(title="HR (Input Cost)", showgrid=True, gridcolor=THEME['grid']),
        yaxis=dict(title="Pulse Press. (Output)", showgrid=True, gridcolor=THEME['grid']),
        margin=dict(l=10, r=10, t=40, b=10), showlegend=False
    )
    return fig

def plot_advanced_metrics(df, curr_time):
    """
    Strip chart for CEI (Effort) and Entropy (Complexity).
    """
    start = max(0, curr_time - 180)
    data = df.iloc[start:curr_time]
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    
    # CEI
    fig.add_trace(go.Scatter(
        x=data.index, y=data['CEI'], fill='tozeroy', 
        fillcolor='rgba(217, 119, 6, 0.1)', line=dict(color=THEME['cei']), name='Cardiac Effort'
    ), row=1, col=1)
    fig.add_hline(y=3.0, line_dash='dot', line_color='red', row=1, col=1)
    
    # Entropy
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Entropy'], line=dict(color=THEME['entropy']), name='Hemo-Entropy'
    ), row=2, col=1)
    
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=0,r=0,t=20,b=0), showlegend=False)
    fig.update_yaxes(title="Effort Idx", row=1, col=1, gridcolor=THEME['grid'])
    fig.update_yaxes(title="Entropy", row=2, col=1, gridcolor=THEME['grid'])
    
    return fig

def plot_3d_attractor(df, curr_time):
    """
    3D Phase Space Reconstruction (Light Mode Edition).
    """
    tau = 2
    data = df['Kalman_HR'].iloc[max(0, curr_time-200):curr_time].values
    if len(data) < 10: return go.Figure()
    
    x, y, z = data[:-2*tau], data[tau:-tau], data[2*tau:]
    colors = np.linspace(0, 1, len(x))
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z, mode='lines',
        line=dict(color=colors, colorscale='Bluered', width=5)
    )])
    
    # Current Point
    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]], mode='markers', marker=dict(size=8, color='red')
    ))

    fig.update_layout(
        template="plotly_white", height=300,
        title=dict(text="<b>Attractor (Phase Space)</b>", font=dict(size=12)),
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis=dict(title='', showticklabels=False, backgroundcolor=THEME['bg']),
            yaxis=dict(title='', showticklabels=False, backgroundcolor=THEME['bg']),
            zaxis=dict(title='', showticklabels=False, backgroundcolor=THEME['bg']),
        )
    )
    return fig

def plot_heatmap_fingerprint(df, curr_time):
    """
    The Genetic Barcode of Instability.
    """
    start = max(0, curr_time - 60)
    subset = df.iloc[start:curr_time][['Kalman_HR', 'MAP', 'PP', 'PI', 'Entropy']]
    
    # Z-Score Normalization
    z = (subset - subset.mean()) / subset.std()
    z = z.T # Transpose for heatmap
    
    fig = go.Figure(data=go.Heatmap(
        z=z.values, x=z.columns, y=z.index,
        colorscale='RdBu_r', zmid=0, showscale=False
    ))
    
    fig.update_layout(
        template="plotly_white", height=200, 
        title=dict(text="<b>Deviation Fingerprint</b>", font=dict(size=12)),
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(showticklabels=False),
        yaxis=dict(tickfont=dict(size=10))
    )
    return fig
