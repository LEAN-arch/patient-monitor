import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import welch

# --- 1. CLINICAL PALETTE ---
THEME = {
    'bg': '#ffffff',
    'grid': '#f8fafc',
    'text': '#334155',
    'safe': '#10b981',      # Emerald Green
    'warn': '#f59e0b',      # Amber
    'crit': '#ef4444',      # Red
    'flow': '#0ea5e9',      # Sky Blue
    'pressure': '#d946ef',  # Fuschia
    'trace_hist': '#cbd5e1' # Slate 300 (Ghost trails)
}

# --- 2. SIMULATION ENGINE (GDT Model) ---
def simulate_gdt_data(mins=720):
    t = np.arange(mins)
    
    def noise(n, amp=1):
        return np.convolve(np.random.normal(0,1,n), [0.1]*10, mode='same') * amp

    # Baseline (Healthy)
    hr = 75 + noise(mins, 3)
    sbp = 120 + noise(mins, 3)
    dbp = 75 + noise(mins, 2)
    spo2 = 98 + noise(mins, 0.5)
    hb = 12.0
    
    # SCENARIO: Septic Vasoplegia (Warm) -> Myocardial Depression (Cold)
    start = 240
    
    # 1. Vasodilation (SVR Drop)
    dbp_drop = np.linspace(0, 35, mins-start) # Diastolic drops fast in sepsis
    sbp_drop = np.linspace(0, 40, mins-start)
    
    # 2. Compensation
    hr_rise = np.linspace(0, 55, mins-start)
    
    hr[start:] += hr_rise
    sbp[start:] -= sbp_drop
    dbp[start:] -= dbp_drop
    
    df = pd.DataFrame({'HR': hr, 'SBP': sbp, 'DBP': dbp, 'SpO2': spo2}, index=t)
    
    # Physics
    df['MAP'] = (df['SBP'] + 2*df['DBP']) / 3
    df['PP'] = df['SBP'] - df['DBP']
    
    # Cardiac Index (Proxy)
    # Septic patients are initially Hyperdynamic (High CI)
    raw_co = df['PP'] * df['HR']
    df['CI'] = (raw_co / raw_co.mean()) * 3.2 
    # Add late stage myocardial depression
    depression = np.zeros(mins)
    depression[start+200:] = np.linspace(0, 1.5, mins-(start+200))
    df['CI'] -= depression
    
    # SVRI (Dyne)
    df['SVRI'] = ((df['MAP'] - 5) / df['CI']) * 80
    
    # Oxygen (DO2I)
    df['DO2I'] = df['CI'] * hb * 1.34 * (df['SpO2']/100) * 10
    
    # Lactate (Oxygen Debt)
    df['Lactate'] = 1.0
    # Lactate rises when DO2I < 400 or flow independent causes
    debt = np.maximum(0, (450 - df['DO2I']) * 0.02)
    df['Lactate'] += debt.cumsum() * 0.05
    
    # Renal (Autoregulation)
    # Urine matches MAP until MAP < 65, then drops off a cliff
    df['Urine'] = 1.0
    df['Urine'] = np.where(df['MAP'] > 65, 
                           1.0 + noise(mins, 0.1), 
                           np.maximum(0, 1.0 - (65 - df['MAP'])*0.05))

    return df

# --- 3. SME VISUALIZATION LIBRARY ---

def plot_shock_compass(df, curr_time):
    """
    SME INSIGHT: The 'Shock Compass' divides hemodynamics into 4 Phenotypes.
    This replaces mental math. The background quadrants tell the diagnosis.
    """
    data = df.iloc[max(0, curr_time-60):curr_time]
    cur_ci = data['CI'].iloc[-1]
    cur_map = data['MAP'].iloc[-1]
    
    fig = go.Figure()
    
    # --- A. DIAGNOSTIC ZONES (Background) ---
    # 1. VASODILATION (High Flow / Low Press) - Bottom Right
    fig.add_shape(type="rect", x0=2.5, y0=40, x1=6.0, y1=65, 
                  fillcolor="rgba(255, 165, 0, 0.1)", line_width=0, layer="below")
    fig.add_annotation(x=4.25, y=52, text="VASOPLEGIA<br>(Pressors)", showarrow=False, font=dict(color="orange", size=10))
    
    # 2. HYPOPERFUSION (Low Flow / Low Press) - Bottom Left
    fig.add_shape(type="rect", x0=1.0, y0=40, x1=2.5, y1=65, 
                  fillcolor="rgba(255, 0, 0, 0.1)", line_width=0, layer="below")
    fig.add_annotation(x=1.75, y=52, text="CARDIOGENIC<br>HYPOVOLEMIC<br>(Inotropes/Fluids)", showarrow=False, font=dict(color="red", size=10))
    
    # 3. VASOCONSTRICTION (Low Flow / High Press) - Top Left
    fig.add_shape(type="rect", x0=1.0, y0=65, x1=2.5, y1=100, 
                  fillcolor="rgba(0, 0, 255, 0.05)", line_width=0, layer="below")
    fig.add_annotation(x=1.75, y=82, text="VASOCONSTRICTION<br>(Check Volume)", showarrow=False, font=dict(color="blue", size=10))
    
    # 4. GOAL (Normal Flow / Normal Press) - Top Right
    fig.add_shape(type="rect", x0=2.5, y0=65, x1=6.0, y1=100, 
                  fillcolor="rgba(0, 128, 0, 0.1)", line_width=0, layer="below")
    fig.add_annotation(x=4.25, y=82, text="STABLE<br>TARGET", showarrow=False, font=dict(color="green", size=12, weight="bold"))

    # --- B. DATA ---
    # History Trail (Ghost)
    fig.add_trace(go.Scatter(x=data['CI'], y=data['MAP'], mode='lines', 
                             line=dict(color=THEME['trace_hist'], width=2), name='Trend'))
    
    # Current Head
    fig.add_trace(go.Scatter(x=[cur_ci], y=[cur_map], mode='markers', 
                             marker=dict(color='black', size=14, symbol='cross'), name='Current'))

    fig.update_layout(template="plotly_white", height=350, margin=dict(l=10,r=10,t=40,b=10),
                      title="<b>Hemodynamic Compass (Diagnosis)</b>",
                      xaxis=dict(title="Cardiac Index (L/min/m²)", range=[1.0, 6.0], gridcolor=THEME['grid']),
                      yaxis=dict(title="MAP (mmHg)", range=[40, 100], gridcolor=THEME['grid']),
                      showlegend=False)
    return fig

def plot_oxygen_mismatch(df, curr_time):
    """
    SME INSIGHT: Visualizes the 'Gap' between supply and demand.
    Red Fill = Oxygen Debt (The killer in sepsis).
    """
    data = df.iloc[max(0, curr_time-180):curr_time]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # DO2 (Supply)
    fig.add_trace(go.Scatter(x=data.index, y=data['DO2I'], 
                             line=dict(color=THEME['safe'], width=2),
                             fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.1)',
                             name='Delivery (DO2)'), secondary_y=False)
    
    # Critical Threshold Line
    fig.add_hline(y=400, line_dash="dot", line_color="gray", annotation_text="Critical Threshold", secondary_y=False)
    
    # Lactate (Debt)
    fig.add_trace(go.Scatter(x=data.index, y=data['Lactate'], 
                             line=dict(color=THEME['crit'], width=3), 
                             name='Lactate'), secondary_y=True)
    
    fig.update_layout(template="plotly_white", height=250, margin=dict(l=10,r=10,t=30,b=10),
                      title="<b>Metabolic Mismatch (Oxygen Debt)</b>", legend=dict(orientation="h", y=1.1))
    fig.update_yaxes(title="DO2 Index", secondary_y=False, gridcolor=THEME['grid'])
    fig.update_yaxes(title="Lactate", secondary_y=True, showgrid=False)
    return fig

def plot_renal_autoregulation(df, curr_time):
    """
    SME INSIGHT: Plots the patient against the theoretical Autoregulation Curve.
    This shows if the patient has fallen off the 'Cliff' where flow becomes pressure-dependent.
    """
    data = df.iloc[max(0, curr_time-120):curr_time]
    
    fig = go.Figure()
    
    # 1. Theoretical Autoregulation Curve (Sigmoid)
    x_theory = np.linspace(30, 120, 100)
    # Logistic function to mimic autoregulation plateau between 65 and 110
    y_theory = 1.0 / (1 + np.exp(-0.15 * (x_theory - 55))) 
    
    fig.add_trace(go.Scatter(x=x_theory, y=y_theory, mode='lines', 
                             line=dict(color='#94a3b8', dash='dot'), name='Autoregulation Limit'))
    
    # 2. Patient Trajectory
    fig.add_trace(go.Scatter(x=data['MAP'], y=data['Urine'], mode='lines+markers',
                             line=dict(color=THEME['flow'], width=2),
                             marker=dict(size=4, color=np.linspace(0,1,len(data)), colorscale='Blues'),
                             name='Patient'))
    
    # 3. Current State
    fig.add_trace(go.Scatter(x=[data['MAP'].iloc[-1]], y=[data['Urine'].iloc[-1]], mode='markers',
                             marker=dict(color=THEME['crit'], size=12, symbol='circle'), name='Now'))

    # Zone Annotation
    if data['MAP'].iloc[-1] < 65:
        fig.add_annotation(x=50, y=0.5, text="PRESSURE DEPENDENT<br>(AKI RISK)", showarrow=False, font=dict(color="red"))

    fig.update_layout(template="plotly_white", height=250, margin=dict(l=10,r=10,t=30,b=10),
                      title="<b>Renal Autoregulation Curve</b>",
                      xaxis=dict(title="Perfusion Pressure (MAP)", range=[30, 100], gridcolor=THEME['grid']),
                      yaxis=dict(title="Urine Output", range=[0, 1.5], gridcolor=THEME['grid']),
                      showlegend=False)
    return fig

def plot_fluid_vector(df, curr_time):
    """
    SME INSIGHT: Replaces static dots with Vectors (Arrows).
    Shows the MAGNITUDE and DIRECTION of the response to the last hemodynamic shift.
    """
    # Look at last 15 mins to get a "Vector"
    recent = df.iloc[curr_time-15:curr_time]
    start_pt = recent.iloc[0]
    end_pt = recent.iloc[-1]
    
    fig = go.Figure()
    
    # Ideal Starling Curve (Logarithmic)
    x_ideal = np.linspace(40, 100, 100)
    y_ideal = np.log(x_ideal)*25 - 50
    fig.add_trace(go.Scatter(x=x_ideal, y=y_ideal, line=dict(color='lightgray'), name='Ideal'))
    
    # The Vector (Arrow)
    fig.add_annotation(
        x=end_pt['DBP'], y=end_pt['PP'],
        ax=start_pt['DBP'], ay=start_pt['PP'],
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowwidth=3, arrowcolor=THEME['flow']
    )
    
    # Markers for context
    fig.add_trace(go.Scatter(x=[start_pt['DBP']], y=[start_pt['PP']], mode='markers', 
                             marker=dict(color='gray', size=8), name='Start'))
    fig.add_trace(go.Scatter(x=[end_pt['DBP']], y=[end_pt['PP']], mode='markers', 
                             marker=dict(color=THEME['flow'], size=12), name='End'))

    # Interpretation
    delta_pp = end_pt['PP'] - start_pt['PP']
    txt = "FLUID RESPONSIVE" if delta_pp > 2 else "NON-RESPONDER"
    col = "green" if delta_pp > 2 else "red"
    
    fig.add_annotation(x=70, y=10, text=txt, showarrow=False, font=dict(color=col, weight="bold"))

    fig.update_layout(template="plotly_white", height=250, margin=dict(l=10,r=10,t=30,b=10),
                      title="<b>Frank-Starling Vector (15m Delta)</b>",
                      xaxis=dict(title="Preload (DBP)", gridcolor=THEME['grid']),
                      yaxis=dict(title="Stroke Vol (PP)", gridcolor=THEME['grid']), showlegend=False)
    return fig

def plot_autonomic_spectrum(df, curr_time):
    """
    SME INSIGHT: A simplified Frequency Domain plot. 
    Shows loss of 'Power' (Height) and 'Complexity' (Width) in sepsis.
    """
    # Get last 64 mins for FFT
    window = df['HR'].iloc[max(0, curr_time-64):curr_time].values
    if len(window) < 64: return go.Figure()
    
    f, Pxx = welch(window, fs=1/60)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=Pxx, fill='tozeroy', line=dict(color=THEME['pressure'])))
    
    fig.update_layout(template="plotly_white", height=150, margin=dict(l=10,r=10,t=20,b=10),
                      title="<b>Autonomic Spectrum (HRV)</b>",
                      xaxis=dict(title="Frequency (Hz)", showticklabels=False),
                      yaxis=dict(showticklabels=False, showgrid=False))
    return fig
