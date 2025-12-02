import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- 1. ADVANCED MATH CLASSES ---

class KalmanFilter1D:
    """
    A pure NumPy implementation of a 1D Kalman Filter.
    mathematically estimates the 'True' state of a system from noisy measurements.
    
    Why it saves lives: It ignores sensor artifacts (coughing, movement) 
    and shows the doctor the REAL trend, reducing alarm fatigue.
    """
    def __init__(self, process_variance, measurement_variance, estimated_measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_measurement_variance = estimated_measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def update(self, measurement):
        # 1. Prediction Update
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        # 2. Measurement Update
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.posteri_estimate

def calculate_sample_entropy(L, m=2, r=0.2):
    """
    Computes Sample Entropy (SampEn), a measure of physiological complexity.
    
    Math: Counts frequency of patterns of length m vs m+1 within tolerance r.
    Clinical Meaning: 
    - High Entropy = Healthy (Complex, adaptable system).
    - Low Entropy = Sepsis/Failure (Rigid, metronomic system).
    """
    # Simplified vectorization for speed in demo
    N = len(L)
    if N < 10: return 0
    
    # Normalize
    L = (L - np.mean(L)) / np.std(L)
    
    # This is a simplified proxy for SampEn for real-time visualization speed
    # Real SampEn is O(N^2), this uses standard deviation of differences as a proxy
    # for 'smoothness' or loss of complexity.
    diffs = np.diff(L)
    complexity = np.std(diffs) 
    
    return complexity

# --- 2. SIMULATION ENGINE (Stochastic Differential Equation Proxy) ---

def simulate_advanced_dynamics(mins=720):
    t = np.arange(mins)
    
    # 1. Generate "True" Biological State (Hidden Markov Model concept)
    # Baseline
    true_hr = np.ones(mins) * 75
    true_map = np.ones(mins) * 90
    
    # Event: Sepsis (loss of complexity + hemodynamic drift)
    start = 250
    
    # Drift dynamics
    drift_curve = np.linspace(0, 40, mins-start)**1.2
    true_hr[start:] += drift_curve  # Tachycardia
    true_map[start:] -= (drift_curve * 0.8) # Hypotension
    
    # 2. Add Biological Variability (1/f Noise)
    # Healthy people have HIGH variability (Pink Noise)
    # Sick people have LOW variability (Brownian/Red Noise or Sine wave)
    
    def generate_noise(n, complexity_level):
        # Complexity level 1.0 = High (Healthy), 0.1 = Low (Sick)
        noise = np.random.normal(0, 1, n)
        # Sickness dampens the high frequencies
        if complexity_level < 0.5:
             noise = np.convolve(noise, np.ones(10)/10, mode='same')
        return noise * 2.0
    
    # Apply variable noise based on patient state
    noise_vector = np.array([generate_noise(1, 1.0 if i < 350 else 0.2)[0] for i in range(mins)])
    
    # 3. Add Sensor Noise (Measurement Error)
    # Kalman filter target
    sensor_artifact = np.random.normal(0, 4, mins) # High variance noise
    
    # Combine
    observed_hr = true_hr + noise_vector + sensor_artifact
    observed_map = true_map + (noise_vector*0.5) + (sensor_artifact*0.5)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Observed_HR': observed_hr, 
        'Observed_MAP': observed_map,
        'True_HR': true_hr, # In real life, we don't know this (Kalman estimates it)
        'True_MAP': true_map
    }, index=t)
    
    return df

# --- 3. ANALYTICS PIPELINE ---

def run_kalman_smoothing(df):
    """
    Applies the Kalman Filter to the noisy observed data.
    """
    kf = KalmanFilter1D(process_variance=1e-5, measurement_variance=0.1, estimated_measurement_variance=0.1)
    
    kalman_hr = []
    kf.posteri_estimate = df['Observed_HR'].iloc[0]
    
    for val in df['Observed_HR']:
        kalman_hr.append(kf.update(val))
        
    df['Kalman_HR'] = kalman_hr
    return df

def run_complexity_analysis(df, window=60):
    """
    Rolling Entropy calculation.
    """
    entropy = df['Kalman_HR'].rolling(window).apply(lambda x: calculate_sample_entropy(x))
    df['Entropy'] = entropy.fillna(1.0)
    return df

# --- 4. VISUALIZATION ENGINES ---

THEME = {
    'bg': '#0f172a', # Deep Space Blue (Commercial Dark Mode)
    'paper': '#1e293b',
    'text': '#f1f5f9',
    'accent': '#38bdf8', # Cyan
    'kalman': '#f43f5e', # Rose
    'raw': '#475569',    # Slate (for noise)
    'grid': '#334155'
}

def plot_kalman_separation(df, curr_time):
    """
    Visualizes the power of the Kalman Filter.
    Shows Raw Data (Gray/Noisy) vs Mathematical Estimate (Red/Smooth).
    """
    start = max(0, curr_time - 120)
    data = df.iloc[start:curr_time]
    
    fig = go.Figure()
    
    # 1. Raw Sensor Data (Ghost Trace)
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Observed_HR'],
        mode='markers', marker=dict(color=THEME['raw'], size=3, opacity=0.5),
        name='Raw Sensor (Noisy)'
    ))
    
    # 2. Kalman Estimate (The Signal)
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Kalman_HR'],
        mode='lines', line=dict(color=THEME['accent'], width=3),
        name='Kalman Estimate (True State)'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME['paper'],
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        title=dict(text="<b>Bayesian State Estimation (Kalman Filter)</b>", font=dict(size=14)),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=True, gridcolor=THEME['grid'], title="Time (min)"),
        yaxis=dict(showgrid=True, gridcolor=THEME['grid'], title="Heart Rate (bpm)"),
        legend=dict(orientation="h", y=1.02, x=0)
    )
    return fig

def plot_phase_space_3d(df, curr_time):
    """
    3D Phase Space Reconstruction (Time-Delay Embedding).
    Math: (x(t), x(t-tau), x(t-2tau))
    Meaning: Visualizes the 'Attractor' of the heart.
    - Ball/Cloud = Healthy Chaos.
    - Flat Line/Loop = Pathological Rigidity.
    """
    tau = 2 # Time lag
    
    # Get recent history
    data = df['Kalman_HR'].iloc[max(0, curr_time-300):curr_time].values
    
    if len(data) < 10: return go.Figure()
    
    # Embed
    x = data[:-2*tau]
    y = data[tau:-tau]
    z = data[2*tau:]
    
    # Color mapping based on time (fade tail)
    colors = np.linspace(0, 1, len(x))
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(
            color=colors,
            colorscale='Viridis',
            width=4
        )
    )])
    
    # Add current head
    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode='markers', marker=dict(size=8, color='red')
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME['paper'],
        height=350,
        title=dict(text="<b>Attractor Geometry (3D Phase Space)</b>", font=dict(size=14)),
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis=dict(title='t', showgrid=True, gridcolor=THEME['grid'], backgroundcolor=THEME['bg']),
            yaxis=dict(title='t-1', showgrid=True, gridcolor=THEME['grid'], backgroundcolor=THEME['bg']),
            zaxis=dict(title='t-2', showgrid=True, gridcolor=THEME['grid'], backgroundcolor=THEME['bg']),
        )
    )
    return fig

def plot_entropy_monitor(df, curr_time):
    """
    Plots the breakdown of physiological complexity.
    """
    start = max(0, curr_time - 180)
    data = df.iloc[start:curr_time]
    
    fig = go.Figure()
    
    # Entropy Line
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Entropy'],
        mode='lines', line=dict(color='#a855f7', width=2), # Purple
        fill='tozeroy', fillcolor='rgba(168, 85, 247, 0.1)',
        name='Complexity (SampEn)'
    ))
    
    # Sepsis Threshold
    fig.add_hline(y=0.5, line_dash='dot', line_color='red', annotation_text="De-complexification Threshold (Sepsis)")
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME['paper'],
        plot_bgcolor='rgba(0,0,0,0)',
        height=250,
        title=dict(text="<b>System Complexity (Sample Entropy)</b>", font=dict(size=14)),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(title="Entropy (σ)", showgrid=True, gridcolor=THEME['grid']),
        xaxis=dict(showgrid=True, gridcolor=THEME['grid'])
    )
    return fig
