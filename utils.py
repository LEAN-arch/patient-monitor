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
    # DBP drops disproportionately
    dbp_drop = np.linspace(0, 30, mins-start)
    sbp_drop = np.linspace(0, 35, mins-start)
    
    # 2. Compensatory Hyperdynamic State (Early Sepsis)
    # HR rises significantly to maintain CO against low resistance
    hr_rise = np.linspace(0, 50, mins-start)
    
    # Apply Pathophysiology
    hr[start:] += hr_rise
    sbp[start:] -= sbp_drop
    dbp[start:] -= dbp_drop
    
    # Build DataFrame
    df = pd.DataFrame({'HR': hr, 'SBP': sbp, 'DBP': dbp, 'SpO2': spo2}, index=t)
    
    # --- ADVANCED PHYSIOLOGY ---
    
    # 1. Pressures
    df['MAP'] = (df['SBP'] + 2*df['DBP']) / 3
    df['PP'] = df['SBP'] - df['DBP']
    
    # 2. Flow (Cardiac Index - CI)
    # Estimate: (PP * HR) proportional to CO. Normalized to CI ~3.0 L/min/m2
    raw_co_proxy = df['PP'] * df['HR']
    df['CI'] = (raw_co_proxy / raw_co_proxy.mean()) * 3.0
    
    # 3. Resistance (SVRI)
    # SVRI = (MAP - CVP) / CI * 80. Assuming CVP=8.
    df['SVRI'] = ((df['MAP'] - 8) / df['CI']) * 80
    
    # 4. Oxygen Delivery (DO2I) vs Consumption
    # DO2I = CI * Hb * 1.34 * SpO2
    df['DO2I'] = df['CI'] * hb * 1.34 * (df['SpO2']/100)
    
    # Lactate (Oxygen Debt)
    # Rises when DO2I drops below critical threshold (~350) OR Sepsis metabolic derangement
    df['Lactate'] = 1.0
    # Add septic rise independent of flow (Cytopathic hypoxia)
    septic_lac = np.zeros(mins)
    septic_lac[start:] = np.linspace(0, 5, mins-start)
    df['Lactate'] += septic_lac
    
    # 5. Renal Perfusion
    # Urine Output drops as MAP drops < 65
    df['Urine'] = 1.0 # mL/kg/hr
    renal_hit = np.maximum(0, (65 - df['MAP']) * 0.05)
    df['Urine'] = np.maximum(0.1, df['Urine'] - renal_hit)

    return df

# --- 4. ACTIONABLE VISUALIZATIONS ---

def plot_hemodynamic_bullseye(df, curr_time):
    """
    ACTION: Tells you WHICH DRUG to give.
    X-Axis: Cardiac Index (Flow)
    Y-Axis: MAP (Pressure)
    
    Quadrants:
    - Low Flow / Low Pressure -> Needs FLUIDS/INOTROPES
    - High Flow / Low Pressure -> Needs VASOPRESSORS (Sepsis)
    - Center -> GOAL
    """
    data = df.iloc[max(0, curr_time-60):curr_time]
    cur_ci = data['CI'].iloc[-1]
    cur_map = data['MAP'].iloc[-1]
    
    fig = go.Figure()
    
    # 1. The "Goal Box" (Safety Corridor)
    fig.add_shape(type="rect", x0=2.5, y0=65, x1=4.0, y1=90, 
                  fillcolor=THEME['target'], line_width=1, line_color="green", layer="below")
    fig.add_annotation(x=3.25, y=77, text="GOAL", font=dict(color="green", size=12), showarrow=False)

    # 2. Quadrant logic annotations
    if cur_map < 65:
        if cur_ci > 2.5:
            txt = "VASODILATED\n(Add Pressors)"
            col = "orange"
        else:
            txt = "HYPOPERFUSION\n(Fluids/Inotropes)"
            col = "red"
        fig.add_annotation(x=cur_ci, y=cur_map+10, text=txt, font=dict(color=col, weight="bold"))

    # 3. Trajectory
    fig.add_trace(go.Scatter(x=data['CI'], y=data['MAP'], mode='lines', 
                             line=dict(color='gray', width=1, dash='dot'), name='History'))
    
    # 4. Current State
    fig.add_trace(go.Scatter(x=[cur_ci], y=[cur_map], mode='markers', 
                             marker=dict(color=THEME['pressure'], size=15, symbol='cross'), name='YOU ARE HERE'))

    fig.update_layout(template="plotly_white", height=350, margin=dict(l=0,r=0,t=40,b=0),
                      title="<b>Hemodynamic Bullseye (Targeting)</b>",
                      xaxis=dict(title="Cardiac Index (Flow)", range=[1.5, 6.0], gridcolor=THEME['grid']),
                      yaxis=dict(title="MAP (Pressure)", range=[40, 110], gridcolor=THEME['grid']))
    return fig

def plot_oxygen_ledger(df, curr_time):
    """
    ACTION: Shows if shock is developing at cellular level.
    Compares Oxygen Delivery (DO2) vs Lactate (Debt).
    """
    start = max(0, curr_time - 180)
    data = df.iloc[start:curr_time]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # DO2 Area
    fig.add_trace(go.Scatter(x=data.index, y=data['DO2I'], fill='tozeroy', 
                             fillcolor=hex_to_rgba(THEME['oxygen'], 0.2),
                             line=dict(color=THEME['oxygen']), name='Oxygen Delivery (DO2I)'), secondary_y=False)
    
    # Critical DO2 Line
    fig.add_hline(y=400, line_dash="dot", line_color="green", annotation_text="Crit DO2", secondary_y=False)
    
    # Lactate Line
    fig.add_trace(go.Scatter(x=data.index, y=data['Lactate'], 
                             line=dict(color=THEME['debt'], width=3), name='Lactate (Debt)'), secondary_y=True)
    
    # Crash Forecast
    if data['Lactate'].iloc[-1] > 2.0 and data['Lactate'].iloc[-1] > data['Lactate'].iloc[-10]:
         fig.add_annotation(x=data.index[-1], y=data['Lactate'].iloc[-1], 
                            text="METABOLIC FAILURE", arrowhead=1, ax=-40, ay=-40, bgcolor="red", font=dict(color="white"))

    fig.update_layout(template="plotly_white", height=250, margin=dict(l=0,r=0,t=30,b=0), 
                      title="<b>Oxygen Debt Ledger (Supply vs Demand)</b>", legend=dict(orientation="h", y=1.1))
    fig.update_yaxes(title="DO2I (mL/min/m2)", secondary_y=False, gridcolor=THEME['grid'])
    fig.update_yaxes(title="Lactate (mmol/L)", secondary_y=True, showgrid=False)
    return fig

def plot_renal_trajectory(df, curr_time):
    """
    ACTION: Predicts AKI. 
    Plots MAP vs Urine Output. 
    """
    data = df.iloc[max(0, curr_time-120):curr_time]
    
    fig = go.Figure()
    
    # Danger Zone
    fig.add_shape(type="rect", x0=40, y0=0, x1=65, y1=0.5, 
                  fillcolor=THEME['danger'], line_width=0, layer="below")
    fig.add_annotation(x=52, y=0.25, text="AKI ZONE", font=dict(color="red"), showarrow=False)

    # Trajectory
    fig.add_trace(go.Scatter(x=data['MAP'], y=data['Urine'], mode='lines+markers',
                             marker=dict(color=np.linspace(0,1,len(data)), colorscale='Reds_r'),
                             line=dict(color='gray', width=1)))
    
    # Arrow head
    fig.add_trace(go.Scatter(x=[data['MAP'].iloc[-1]], y=[data['Urine'].iloc[-1]], mode='markers',
                             marker=dict(color=THEME['renal'], size=10)))

    fig.update_layout(template="plotly_white", height=250, margin=dict(l=0,r=0,t=30,b=0),
                      title="<b>Renal Function Trajectory</b>",
                      xaxis=dict(title="MAP (Perfusion)", range=[40, 100], gridcolor=THEME['grid']),
                      yaxis=dict(title="Urine (mL/kg/hr)", range=[0, 1.5], gridcolor=THEME['grid']))
    return fig

def plot_frank_starling_vector(df, curr_time):
    """
    ACTION: Fluid Responsiveness.
    Uses PP (Stroke Vol) vs DBP (Preload).
    """
    data = df.iloc[max(0, curr_time-60):curr_time]
    
    fig = go.Figure()
    
    # Ideal Curve
    x = np.linspace(40, 100, 100)
    y = np.log(x)*25 - 50
    fig.add_trace(go.Scatter(x=x, y=y, line=dict(color='lightgray', dash='dot'), name='Ideal'))
    
    # Patient Vector
    fig.add_trace(go.Scatter(x=data['DBP'], y=data['PP'], mode='lines', 
                             line=dict(color=THEME['flow'], width=3, arrow='bar'), name='Patient'))
    
    # Interpretation
    slope = (data['PP'].iloc[-1] - data['PP'].iloc[0]) / (data['DBP'].iloc[-1] - data['DBP'].iloc[0] + 0.01)
    status = "FLUID RESPONSIVE" if slope > 0.5 else "NON-RESPONDER"
    color = "green" if slope > 0.5 else "red"
    
    fig.add_annotation(x=data['DBP'].mean(), y=data['PP'].max()+5, text=status, font=dict(color=color, weight="bold"), showarrow=False)

    fig.update_layout(template="plotly_white", height=250, margin=dict(l=0,r=0,t=30,b=0),
                      title="<b>Starling Vector (Fluid Status)</b>",
                      xaxis=dict(title="Preload Proxy (DBP)", gridcolor=THEME['grid']),
                      yaxis=dict(title="Stroke Vol Proxy (PP)", gridcolor=THEME['grid']))
    return fig

def plot_autonomic_psd(df, curr_time):
    """
    ACTION: Detects Sepsis Stress (High Sympathetic Tone).
    """
    window = df['HR'].iloc[max(0, curr_time-64):curr_time].values
    if len(window) < 64: return go.Figure()
    
    f, Pxx = welch(window, fs=1/60)
    
    # Split into LF (Sympathetic) and HF (Parasympathetic) approx
    # Very simplified for demo
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=Pxx, fill='tozeroy', line=dict(color=THEME['resist'])))
    
    fig.update_layout(template="plotly_white", height=150, margin=dict(l=0,r=0,t=20,b=0),
                      title="<b>Autonomic Stress (HRV Spectrum)</b>", 
                      xaxis=dict(title="Freq (Hz)", showticklabels=False), yaxis=dict(showticklabels=False))
    return fig
