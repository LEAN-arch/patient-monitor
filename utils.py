import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import welch

# --- 1. THEME: GOAL DIRECTED THERAPY (GDT) ---
THEME = {
    'bg': '#ffffff',
    'grid': '#f1f5f9',
    'text': '#1e293b',
    # Physiology
    'flow': '#0ea5e9',     # Blue (Cardiac Index)
    'pressure': '#be185d', # Magenta (MAP)
    'resist': '#d97706',   # Amber (SVR)
    'oxygen': '#10b981',   # Green (DO2)
    'debt': '#ef4444',     # Red (Lactate)
    'renal': '#8b5cf6',    # Violet
    # Zones
    'target': 'rgba(16, 185, 129, 0.1)',
    'danger': 'rgba(239, 68, 68, 0.1)'
}

# --- 2. HELPER: COLOR CONVERSION ---
def hex_to_rgba(hex_color, opacity=0.1):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{opacity})"
    return hex_color

# --- 3. SIMULATION: SEPTIC SHOCK CASCADE ---
def simulate_gdt_data(mins=720):
    t = np.arange(mins)
    
    def noise(n, amp=1):
        return np.convolve(np.random.normal(0,1,n), [0.1]*10, mode='same') * amp

    # Baseline Vitals (70kg Patient)
    hr = 75 + noise(mins, 3)
    sbp = 120 + noise(mins, 3)
    dbp = 75 + noise(mins, 2)
    spo2 = 98 + noise(mins, 0.5)
    hb = 12.0 # Hemoglobin (g/dL)
    
    # SCENARIO: Warm Sepsis (High CO, Low SVR) -> Cold Shock (Low CO, High Lactate)
    start = 240
    
    # 1. Vasodilation (SVR Crash)
    dbp_drop = np.linspace(0, 30, mins-start)
    sbp_drop = np.linspace(0, 35, mins-start)
    
    # 2. Compensatory Hyperdynamic State
    hr_rise = np.linspace(0, 50, mins-start)
    
    hr[start:] += hr_rise
    sbp[start:] -= sbp_drop
    dbp[start:] -= dbp_drop
    
    df = pd.DataFrame({'HR': hr, 'SBP': sbp, 'DBP': dbp, 'SpO2': spo2}, index=t)
    
    # --- ADVANCED PHYSIOLOGY ---
    df['MAP'] = (df['SBP'] + 2*df['DBP']) / 3
    df['PP'] = df['SBP'] - df['DBP']
    
    # Cardiac Index (CI) Proxy
    raw_co_proxy = df['PP'] * df['HR']
    df['CI'] = (raw_co_proxy / raw_co_proxy.mean()) * 3.0
    
    # SVRI Proxy
    df['SVRI'] = ((df['MAP'] - 8) / df['CI']) * 80
    
    # Oxygen Delivery (DO2I)
    df['DO2I'] = df['CI'] * hb * 1.34 * (df['SpO2']/100)
    
    # Lactate
    df['Lactate'] = 1.0
    septic_lac = np.zeros(mins)
    septic_lac[start:] = np.linspace(0, 5, mins-start)
    df['Lactate'] += septic_lac
    
    # Renal Perfusion
    df['Urine'] = 1.0 # mL/kg/hr
    renal_hit = np.maximum(0, (65 - df['MAP']) * 0.05)
    df['Urine'] = np.maximum(0.1, df['Urine'] - renal_hit)

    return df

# --- 4. ACTIONABLE VISUALIZATIONS ---

def plot_hemodynamic_bullseye(df, curr_time):
    """X: Cardiac Index (Flow), Y: MAP (Pressure)"""
    data = df.iloc[max(0, curr_time-60):curr_time]
    cur_ci = data['CI'].iloc[-1]
    cur_map = data['MAP'].iloc[-1]
    
    fig = go.Figure()
    
    # Goal Box
    fig.add_shape(type="rect", x0=2.5, y0=65, x1=4.0, y1=90, 
                  fillcolor=THEME['target'], line_width=1, line_color="green", layer="below")
    fig.add_annotation(x=3.25, y=77, text="GOAL", font=dict(color="green", size=12), showarrow=False)

    # Quadrant annotations
    if cur_map < 65:
        if cur_ci > 2.5:
            txt, col = "VASODILATED\n(Add Pressors)", "orange"
        else:
            txt, col = "HYPOPERFUSION\n(Fluids/Inotropes)", "red"
        fig.add_annotation(x=cur_ci, y=cur_map+10, text=txt, font=dict(color=col, weight="bold"))

    fig.add_trace(go.Scatter(x=data['CI'], y=data['MAP'], mode='lines', 
                             line=dict(color='gray', width=1, dash='dot'), name='History'))
    fig.add_trace(go.Scatter(x=[cur_ci], y=[cur_map], mode='markers', 
                             marker=dict(color=THEME['pressure'], size=15, symbol='cross'), name='YOU ARE HERE'))

    fig.update_layout(template="plotly_white", height=350, margin=dict(l=0,r=0,t=40,b=0),
                      title="<b>Hemodynamic Bullseye (Targeting)</b>",
                      xaxis=dict(title="Cardiac Index (Flow)", range=[1.5, 6.0], gridcolor=THEME['grid']),
                      yaxis=dict(title="MAP (Pressure)", range=[40, 110], gridcolor=THEME['grid']))
    return fig

def plot_oxygen_ledger(df, curr_time):
    """DO2 vs Lactate"""
    start = max(0, curr_time - 180)
    data = df.iloc[start:curr_time]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=data.index, y=data['DO2I'], fill='tozeroy', 
                             fillcolor=hex_to_rgba(THEME['oxygen'], 0.2),
                             line=dict(color=THEME['oxygen']), name='Oxygen Delivery'), secondary_y=False)
    
    fig.add_hline(y=400, line_dash="dot", line_color="green", annotation_text="Crit DO2", secondary_y=False)
    
    fig.add_trace(go.Scatter(x=data.index, y=data['Lactate'], 
                             line=dict(color=THEME['debt'], width=3), name='Lactate'), secondary_y=True)
    
    if data['Lactate'].iloc[-1] > 2.0:
         fig.add_annotation(x=data.index[-1], y=data['Lactate'].iloc[-1], 
                            text="METABOLIC FAILURE", arrowhead=1, ax=-40, ay=-40, bgcolor="red", font=dict(color="white"))

    fig.update_layout(template="plotly_white", height=250, margin=dict(l=0,r=0,t=30,b=0), 
                      title="<b>Oxygen Debt Ledger</b>", legend=dict(orientation="h", y=1.1))
    fig.update_yaxes(title="DO2I", secondary_y=False, gridcolor=THEME['grid'])
    fig.update_yaxes(title="Lactate", secondary_y=True, showgrid=False)
    return fig

def plot_renal_trajectory(df, curr_time):
    data = df.iloc[max(0, curr_time-120):curr_time]
    fig = go.Figure()
    
    # Danger Zone
    fig.add_shape(type="rect", x0=40, y0=0, x1=65, y1=0.5, 
                  fillcolor=THEME['danger'], line_width=0, layer="below")
    fig.add_annotation(x=52, y=0.25, text="AKI ZONE", font=dict(color="red"), showarrow=False)

    fig.add_trace(go.Scatter(x=data['MAP'], y=data['Urine'], mode='lines+markers',
                             marker=dict(color=np.linspace(0,1,len(data)), colorscale='Reds_r'),
                             line=dict(color='gray', width=1)))
    fig.add_trace(go.Scatter(x=[data['MAP'].iloc[-1]], y=[data['Urine'].iloc[-1]], mode='markers',
                             marker=dict(color=THEME['renal'], size=10)))

    fig.update_layout(template="plotly_white", height=250, margin=dict(l=0,r=0,t=30,b=0),
                      title="<b>Renal Function Trajectory</b>",
                      xaxis=dict(title="MAP", range=[40, 100], gridcolor=THEME['grid']),
                      yaxis=dict(title="Urine", range=[0, 1.5], gridcolor=THEME['grid']))
    return fig

def plot_frank_starling_vector(df, curr_time):
    data = df.iloc[max(0, curr_time-60):curr_time]
    fig = go.Figure()
    
    x = np.linspace(40, 100, 100)
    y = np.log(x)*25 - 50
    fig.add_trace(go.Scatter(x=x, y=y, line=dict(color='lightgray', dash='dot'), name='Ideal'))
    
    # Trajectory
    fig.add_trace(go.Scatter(x=data['DBP'], y=data['PP'], mode='lines', 
                             line=dict(color=THEME['flow'], width=3), name='History'))
    
    # Current Head (Vector Tip)
    fig.add_trace(go.Scatter(x=[data['DBP'].iloc[-1]], y=[data['PP'].iloc[-1]], mode='markers',
                             marker=dict(color=THEME['flow'], size=12, symbol='triangle-up'), name='Current'))
    
    slope = (data['PP'].iloc[-1] - data['PP'].iloc[0]) / (data['DBP'].iloc[-1] - data['DBP'].iloc[0] + 0.01)
    status, color = ("FLUID RESPONSIVE", "green") if slope > 0.5 else ("NON-RESPONDER", "red")
    
    fig.add_annotation(x=data['DBP'].mean(), y=data['PP'].max()+5, text=status, font=dict(color=color, weight="bold"), showarrow=False)

    fig.update_layout(template="plotly_white", height=250, margin=dict(l=0,r=0,t=30,b=0),
                      title="<b>Starling Vector</b>",
                      xaxis=dict(title="Preload (DBP)", gridcolor=THEME['grid']),
                      yaxis=dict(title="Stroke Vol (PP)", gridcolor=THEME['grid']))
    return fig

def plot_autonomic_psd(df, curr_time):
    window = df['HR'].iloc[max(0, curr_time-64):curr_time].values
    if len(window) < 64: return go.Figure()
    
    f, Pxx = welch(window, fs=1/60)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=Pxx, fill='tozeroy', line=dict(color=THEME['resist'])))
    
    fig.update_layout(template="plotly_white", height=150, margin=dict(l=0,r=0,t=20,b=0),
                      title="<b>Autonomic Stress (PSD)</b>", 
                      xaxis=dict(title="Freq (Hz)", showticklabels=False), yaxis=dict(showticklabels=False))
    return fig
