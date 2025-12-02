import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import welch

# --- 1. CLINICAL DESIGN SYSTEM ---
THEME = {
    'bg': '#ffffff',
    'grid': '#f1f5f9',
    'text': '#0f172a',
    'ok': '#10b981',       # Emerald (Stable)
    'warn': '#f59e0b',     # Amber (Compensating)
    'crit': '#ef4444',     # Red (Decompensated)
    'flow': '#0ea5e9',     # Blue (CI)
    'press': '#be185d',    # Magenta (MAP)
    'resist': '#d97706',   # Orange (SVR)
    'meta': '#8b5cf6',     # Violet (Lactate/O2)
    'ghost': '#cbd5e1'     # Slate (History)
}

# --- 2. ADVANCED MATH: KALMAN FILTER ---
class RobustKalman:
    """
    Adaptive Kalman Filter for vital signs. 
    Smooths data while preserving acute clinical events (jumps).
    """
    def __init__(self, process_noise=1e-4, sensor_noise=0.1):
        self.q = process_noise
        self.r = sensor_noise
        self.x = 0.0 # Estimate
        self.p = 1.0 # Error covariance

    def update(self, measurement):
        # Prediction
        p_pred = self.p + self.q
        # Update
        k = p_pred / (p_pred + self.r)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * p_pred
        return self.x

# --- 3. SIMULATION: BAROREFLEX FEEDBACK LOOP ---
def simulate_coupled_physiology(mins=720):
    """
    Simulates Septic Shock using a coupled differential equation approximation.
    Models the battle between Vasodilation (Pathology) and Sympathetic Tone (Defense).
    """
    t_axis = np.arange(mins)
    
    # State Vectors
    map_val = np.zeros(mins)
    co_val = np.zeros(mins)
    svr_val = np.zeros(mins)
    hr_val = np.zeros(mins)
    lactate = np.zeros(mins)
    
    # Initial Conditions (Healthy 70kg Male)
    # Target MAP = 90, Resting HR = 70, Base SVR = 1200
    current_map = 90.0
    current_hr = 70.0
    sympathetic_tone = 1.0 # Baseline multiplier
    
    # PATHOLOGY: Sepsis (Progressive Vasoplegia)
    # SVR capacity drops over time
    svr_capacity = np.linspace(1.0, 0.4, mins) # Lose 60% of vascular tone capacity
    
    # SIMULATION LOOP
    kf_map = RobustKalman(1e-3, 5.0)
    
    for t in range(mins):
        # 1. PATHOLOGY: Loss of Vascular Tone
        # Base resistance drops starting at T=120
        sepsis_factor = 1.0 if t < 120 else svr_capacity[t]
        
        # 2. PHYSIOLOGY: Baroreflex Feedback Control
        # Error signal: How far is MAP from setpoint (65-90)?
        pressure_error = 75 - current_map
        
        # Integrative Control (Accumulate sympathetic tone if pressure stays low)
        if pressure_error > 0:
            sympathetic_tone += 0.002 * (pressure_error / 10) # Ramp up
        else:
            sympathetic_tone -= 0.001 # Relax
            
        # Physiological limits (Catecholamine burnout)
        sympathetic_tone = np.clip(sympathetic_tone, 0.8, 2.5)
        
        # 3. COMPUTE HEMODYNAMICS
        # HR responds to Sympathetic Tone
        target_hr = 70 * sympathetic_tone
        current_hr += 0.1 * (target_hr - current_hr) # Smooth transition
        
        # SVR responds to Tone but limited by Sepsis (Alpha receptors fail)
        current_svr = 1200 * sympathetic_tone * sepsis_factor
        
        # Contractility (Inotropy) increases with Tone
        stroke_volume = 70 * (sympathetic_tone ** 0.5) 
        # But drops if HR is too high (filling time)
        if current_hr > 140: stroke_volume *= 0.85
        
        # Cardiac Output = HR * SV
        current_co = (current_hr * stroke_volume) / 1000 # L/min
        
        # MAP = CO * SVR (Ohms Law) + CVP (assume 5)
        raw_map = (current_co * current_svr / 80) + 5
        
        # Add biological noise (1/f) and sensor noise
        noise = np.random.normal(0, 2)
        current_map = kf_map.update(raw_map + noise)
        
        # 4. METABOLIC: Oxygen Debt
        # DO2 = CO * 1.34 * Hb * SpO2. Assume Hb=12, SpO2=98%
        do2 = current_co * 12 * 1.34 * 0.98 * 10
        
        # Lactate rises if DO2 < 400 (Supply Dependency)
        lac_change = 0
        if do2 < 400: lac_change = 0.02
        # Lactate clearance
        lac_change -= 0.005 * (lactate[t-1] if t>0 else 1.0)
        
        current_lac = (lactate[t-1] if t>0 else 1.0) + lac_change
        current_lac = max(0.5, current_lac)
        
        # Store
        map_val[t] = current_map
        co_val[t] = current_co
        svr_val[t] = current_svr
        hr_val[t] = current_hr + np.random.normal(0, 1) # Monitor jitter
        lactate[t] = current_lac

    # Create DataFrame
    df = pd.DataFrame({
        'MAP': map_val, 'CO': co_val, 'SVR': svr_val, 'HR': hr_val, 'Lactate': lactate
    }, index=t_axis)
    
    # Derived Metrics
    df['CI'] = df['CO'] / 1.8 # BSA approx
    df['SVRI'] = df['SVR'] * 1.8
    df['PP'] = (df['MAP'] / 3) # Crude approximation for visualization utility
    df['Eadyn'] = df['PP'] / (df['CO']/df['HR']*1000) # Dynamic Elastance Proxy
    
    return df

# --- 4. ACTIONABLE VISUALIZATIONS ---

def plot_kpi_spark(df, col, color):
    """Mini sparklines for KPI cards"""
    data = df[col].iloc[-60:]
    fig = go.Figure(go.Scatter(
        x=data.index, y=data.values, mode='lines', 
        line=dict(color=color, width=2), fill='tozeroy', fillcolor=f"rgba{color[3:-1]}, 0.1)"
    ))
    fig.update_layout(template="plotly_white", height=40, margin=dict(l=0,r=0,t=0,b=0), 
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

def plot_gdt_bullseye(df, curr_time):
    """
    GOAL DIRECTED THERAPY MAP
    Diagnostic tool to choose between Fluids (Volume) vs Pressors (SVR).
    """
    data = df.iloc[max(0, curr_time-60):curr_time]
    cur = data.iloc[-1]
    
    fig = go.Figure()
    
    # --- DIAGNOSTIC ZONES ---
    # 1. Vasoplegia (Septic) - High Flow, Low SVR
    fig.add_shape(type="rect", x0=3.0, y0=40, x1=6.0, y1=65, 
                  fillcolor="rgba(255, 165, 0, 0.15)", line_width=0, layer="below")
    fig.add_annotation(x=4.5, y=52, text="VASOPLEGIA<br>(Start Norepi)", font=dict(color="#d97706", size=10), showarrow=False)
    
    # 2. Hypovolemia/Cardiogenic - Low Flow, Low/High Press
    fig.add_shape(type="rect", x0=1.0, y0=40, x1=2.5, y1=100, 
                  fillcolor="rgba(239, 68, 68, 0.15)", line_width=0, layer="below")
    fig.add_annotation(x=1.75, y=70, text="LOW FLOW<br>(Fluids/Ino)", font=dict(color="#dc2626", size=10), showarrow=False)
    
    # 3. Target
    fig.add_shape(type="rect", x0=2.5, y0=65, x1=4.5, y1=90, 
                  fillcolor="rgba(16, 185, 129, 0.15)", line_width=2, line_color="#10b981", layer="below")
    fig.add_annotation(x=3.5, y=77, text="GOAL", font=dict(color="#059669", weight="bold"), showarrow=False)

    # Trajectory
    fig.add_trace(go.Scatter(
        x=data['CI'], y=data['MAP'], mode='lines', 
        line=dict(color=THEME['ghost'], width=2, dash='dot'), name='Trend'
    ))
    
    # Current
    fig.add_trace(go.Scatter(
        x=[cur['CI']], y=[cur['MAP']], mode='markers',
        marker=dict(color=THEME['text'], size=14, symbol='cross'), name='Now'
    ))

    fig.update_layout(template="plotly_white", height=300, margin=dict(l=10,r=10,t=40,b=10),
                      title="<b>Hemodynamic Phenotype (GDT)</b>",
                      xaxis=dict(title="Cardiac Index (L/min/m²)", range=[1.0, 6.0], gridcolor=THEME['grid']),
                      yaxis=dict(title="MAP (mmHg)", range=[40, 100], gridcolor=THEME['grid']), showlegend=False)
    return fig

def plot_coupling_curve(df, curr_time):
    """
    Ventriculo-Arterial Coupling.
    Visualizes the efficiency of the cardiovascular system.
    """
    data = df.iloc[max(0, curr_time-120):curr_time]
    
    fig = go.Figure()
    
    # Iso-Pressure Lines (MAP = CO * SVR)
    x = np.linspace(1, 8, 100)
    for p in [50, 65, 90]:
        y = (p / x) * 80
        col = 'red' if p < 65 else 'green' if p==65 else 'gray'
        dash = 'solid' if p==65 else 'dot'
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=col, width=1, dash=dash), hoverinfo='skip'))
    
    # Patient Data
    fig.add_trace(go.Scatter(
        x=data['CO'], y=data['SVR'], mode='lines',
        line=dict(color=THEME['resist'], width=3), name='Patient'
    ))
    fig.add_trace(go.Scatter(
        x=[data['CO'].iloc[-1]], y=[data['SVR'].iloc[-1]], mode='markers',
        marker=dict(color=THEME['text'], size=10), name='Now'
    ))
    
    # Annotation
    cur = data.iloc[-1]
    dx = "Balanced"
    if cur['SVR'] < 800: dx = "Vasodilation"
    if cur['SVR'] > 1500: dx = "Vasoconstriction"
    
    fig.add_annotation(x=5, y=2000, text=f"STATE: {dx}", showarrow=False, bgcolor="white", bordercolor="gray")

    fig.update_layout(template="plotly_white", height=250, margin=dict(l=10,r=10,t=30,b=10),
                      title="<b>V-A Coupling (Pump vs Pipes)</b>",
                      xaxis=dict(title="Cardiac Output (L/min)", range=[1, 8], gridcolor=THEME['grid']),
                      yaxis=dict(title="SVR (dyne)", range=[400, 2400], gridcolor=THEME['grid']), showlegend=False)
    return fig

def plot_renal_cliff(df, curr_time):
    """
    Renal Perfusion Pressure Analysis.
    Shows the 'Cliff' where autoregulation fails.
    """
    data = df.iloc[max(0, curr_time-180):curr_time]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # MAP Area
    fig.add_trace(go.Scatter(
        x=data.index, y=data['MAP'], fill='tozeroy', 
        fillcolor='rgba(190, 24, 93, 0.1)', line=dict(color=THEME['press']), name='MAP'
    ), secondary_y=False)
    
    # AKI Threshold
    fig.add_hline(y=65, line_color='red', line_dash='dot', annotation_text="AKI Cliff (65)", secondary_y=False)
    
    # Lactate (Metabolic consequence)
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Lactate'], line=dict(color=THEME['meta'], width=2), name='Lactate'
    ), secondary_y=True)
    
    fig.update_layout(template="plotly_white", height=250, margin=dict(l=10,r=10,t=30,b=10),
                      title="<b>Organ Perfusion & Metabolic Debt</b>", showlegend=False)
    fig.update_yaxes(title="MAP", secondary_y=False, gridcolor=THEME['grid'])
    fig.update_yaxes(title="Lactate", secondary_y=True, showgrid=False)
    return fig
