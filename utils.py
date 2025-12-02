import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from scipy.signal import welch

# --- 1. THEME (Precision Clinical) ---
THEME = {
    'bg': '#ffffff',
    'grid': '#f1f5f9',
    'text': '#1e293b',
    # Systems
    'cardiac': '#0369a1',  # Blue
    'vascular': '#be185d', # Magenta
    'resp': '#059669',     # Green
    'neuro': '#7c3aed',    # Violet
    'renal': '#d97706',    # Amber
    'metabolic': '#dc2626' # Red
}

# --- 2. EXPANDED SIMULATION ---
def simulate_panopticon_data(mins=720):
    t = np.arange(mins)
    
    def noise(n, amp=1):
        return np.convolve(np.random.normal(0,1,n), [0.1]*10, mode='same') * amp

    # Base Signals
    hr = 70 + noise(mins, 4)
    sbp = 120 + noise(mins, 3)
    dbp = 80 + noise(mins, 2)
    rr = 16 + noise(mins, 1)
    spo2 = 98 + noise(mins, 0.5)
    
    # SCENARIO: Septic Shock (Distributive)
    # 1. Vasodilation (SVR Drops) -> 2. HR Compels -> 3. Tissue Hypoxia (Lactate) -> 4. Renal Hit
    
    start = 200
    
    # Pathophysiology
    svr_drop = np.linspace(0, 400, mins-start) # SVR dropping
    hr_rise = np.linspace(0, 50, mins-start)   # Tachycardia
    sbp_drop = np.linspace(0, 30, mins-start)  # Hypotension
    lactate_rise = np.linspace(0, 6, mins-start) # Metabolic Acidosis
    uo_drop = np.linspace(0, 0.5, mins-start)  # Oliguria
    
    # Apply
    hr[start:] += hr_rise
    sbp[start:] -= sbp_drop
    dbp[start:] -= (sbp_drop * 0.6)
    spo2[start:] = np.maximum(88, spo2[start:] - np.linspace(0, 5, mins-start))
    
    # Derived DataFrame
    df = pd.DataFrame({'HR': hr, 'SBP': sbp, 'DBP': dbp, 'RR': rr, 'SpO2': spo2}, index=t)
    
    # Advanced Physics
    df['MAP'] = (df['SBP'] + 2*df['DBP']) / 3
    df['PP'] = df['SBP'] - df['DBP']
    df['CO'] = (df['HR'] * df['PP']) / 1000 * 2.0 # Cardiac Output Proxy
    df['SVR'] = (df['MAP'] / df['CO']) * 80 # SVR Proxy
    df['Lactate'] = 1.0
    df['Lactate'].iloc[start:] += lactate_rise
    df['UrineOutput'] = 1.0 # mL/kg/hr
    df['UrineOutput'].iloc[start:] = np.maximum(0.1, 1.0 - uo_drop)
    
    # Oxygen Delivery (DO2) Proxy
    # DO2 = CO * Hb * SpO2 * 1.34
    df['DO2'] = df['CO'] * 13 * (df['SpO2']/100) * 1.34
    
    # Entropy (Neuro)
    df['Entropy'] = df['HR'].rolling(30).apply(lambda x: np.std(np.diff(x))).fillna(1.0)
    
    # Rolling Correlation (Hemodynamic Coherence)
    df['Hemo_Corr'] = df['HR'].rolling(60).corr(df['MAP']).fillna(0)
    
    return df

# --- 3. VISUALIZATION LIBRARY ---

# A. SPARKLINE GENERATOR (For Top Row)
def plot_sparkline(df, col, color):
    data = df[col].iloc[-60:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines', 
                             line=dict(color=color, width=2), fill='tozeroy', fillcolor=f"rgba{color[3:-1]}, 0.1)"))
    fig.update_layout(template="plotly_white", height=60, margin=dict(l=0,r=0,t=0,b=0), 
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

# B. MAIN MONITORS
def plot_command_center(df, curr_time):
    start = max(0, curr_time-180)
    data = df.iloc[start:curr_time]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # HR
    fig.add_trace(go.Scatter(x=data.index, y=data['HR'], line=dict(color=THEME['cardiac'], width=2), name='HR'), secondary_y=False)
    # MAP
    fig.add_trace(go.Scatter(x=data.index, y=data['MAP'], line=dict(color=THEME['vascular'], width=2), name='MAP'), secondary_y=True)
    # Forecast
    trend = np.poly1d(np.polyfit(np.arange(30), data['MAP'].tail(30), 1))(np.arange(30, 60))
    fig.add_trace(go.Scatter(x=np.arange(curr_time, curr_time+30), y=trend, 
                             line=dict(color=THEME['vascular'], dash='dot'), name='Forecast'), secondary_y=True)
    
    fig.update_layout(template="plotly_white", height=250, margin=dict(l=0,r=0,t=20,b=0), hovermode="x unified", legend=dict(orientation="h", y=1.1))
    fig.update_yaxes(title="HR", secondary_y=False, gridcolor=THEME['grid'])
    fig.update_yaxes(title="MAP", secondary_y=True, showgrid=False)
    return fig

# C. PHYSICS & MECHANICS
def plot_starling(df, curr_time):
    data = df.iloc[max(0, curr_time-120):curr_time]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['DBP'], y=data['PP'], mode='lines+markers', 
                             marker=dict(color=data.index, colorscale='Blues'), line=dict(color='rgba(0,0,0,0.2)')))
    fig.update_layout(template="plotly_white", height=200, margin=dict(l=0,r=0,t=30,b=0),
                      title="<b>Starling (Preload vs SV)</b>", xaxis_title="Preload (DBP)", yaxis_title="Stroke Vol (PP)")
    return fig

def plot_svr_co(df, curr_time):
    data = df.iloc[max(0, curr_time-120):curr_time]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['CO'], y=data['SVR'], mode='lines', line=dict(color=THEME['vascular'])))
    fig.add_trace(go.Scatter(x=[data['CO'].iloc[-1]], y=[data['SVR'].iloc[-1]], mode='markers', marker=dict(color='red', size=8)))
    fig.update_layout(template="plotly_white", height=200, margin=dict(l=0,r=0,t=30,b=0),
                      title="<b>V-A Coupling (Pump vs Pipe)</b>", xaxis_title="CO (Flow)", yaxis_title="SVR (Resist)")
    return fig

def plot_pv_proxy(df, curr_time):
    # Proxy PV Loop: Pressure (MAP) vs Stroke Volume (PP)
    data = df.iloc[max(0, curr_time-60):curr_time]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['PP'], y=data['MAP'], mode='lines', line=dict(color=THEME['cardiac'])))
    fig.update_layout(template="plotly_white", height=200, margin=dict(l=0,r=0,t=30,b=0),
                      title="<b>Work Loop (PV Proxy)</b>", xaxis_title="Stroke Vol (PP)", yaxis_title="Pressure (MAP)")
    return fig

# D. PERFUSION & METABOLIC
def plot_oxygen_delivery(df, curr_time):
    data = df.iloc[max(0, curr_time-180):curr_time]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=data.index, y=data['DO2'], fill='tozeroy', line=dict(color=THEME['resp']), name='DO2'), secondary_y=False)
    fig.add_trace(go.Scatter(x=data.index, y=data['Lactate'], line=dict(color=THEME['metabolic'], width=2), name='Lactate'), secondary_y=True)
    fig.update_layout(template="plotly_white", height=200, margin=dict(l=0,r=0,t=30,b=0), title="<b>Oxygen Supply (DO2) vs Debt</b>")
    return fig

def plot_renal_curve(df, curr_time):
    data = df.iloc[max(0, curr_time-180):curr_time]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['MAP'], y=data['UrineOutput'], mode='markers', marker=dict(color=THEME['renal'], opacity=0.6)))
    fig.add_vline(x=65, line_color='red', line_dash='dot')
    fig.update_layout(template="plotly_white", height=200, margin=dict(l=0,r=0,t=30,b=0), 
                      title="<b>Renal Perfusion Curve</b>", xaxis_title="MAP", yaxis_title="Urine (mL/kg/hr)")
    return fig

# E. ADVANCED MATH
def plot_spectral_density(df, curr_time):
    # PSD of Heart Rate (Autonomic Tone)
    window = df['HR'].iloc[max(0, curr_time-64):curr_time]
    if len(window) < 64: return go.Figure()
    f, Pxx = welch(window, fs=1/60) # 1 sample per min assumption
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=Pxx, line=dict(color=THEME['neuro']), fill='tozeroy'))
    fig.update_layout(template="plotly_white", height=200, margin=dict(l=0,r=0,t=30,b=0),
                      title="<b>Autonomic Spectrum (PSD)</b>", xaxis_title="Freq (Hz)", yaxis_title="Power")
    return fig

def plot_coherence(df, curr_time):
    data = df.iloc[max(0, curr_time-180):curr_time]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data.index, y=data['Hemo_Corr'], marker_color=np.where(data['Hemo_Corr']>0, 'red', 'green')))
    fig.update_layout(template="plotly_white", height=200, margin=dict(l=0,r=0,t=30,b=0),
                      title="<b>Hemodynamic Coherence</b>", yaxis_title="Correlation (HR vs MAP)")
    return fig
