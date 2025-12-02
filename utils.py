import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter

# --- 1. FUTURISTIC COLOR PALETTE (Cyberpunk/Medical) ---
COLORS = {
    'bg': '#0e1117',        # Deep Space
    'card': '#1a1f2e',      # Panel Grey
    'text': '#e2e8f0',
    'grid': '#2d3748',
    # Neon Signals
    'cyan': '#00f2ea',      # HR / Flow
    'magenta': '#ff0055',   # Pressure / Critical
    'amber': '#ffae00',     # Warning / CNS
    'lime': '#ccff00',      # Normal / Safe
    'purple': '#b026ff',    # SV / Volume
    'white': '#ffffff'
}

# --- 2. ADVANCED HEMODYNAMIC SIMULATION ---
def simulate_comprehensive_shock(mins=720):
    t = np.arange(mins)
    
    # Noise generators
    def noise(n, amp=1.0):
        w = np.random.normal(0, 1, n)
        return np.convolve(w, [0.1]*10, mode='same') * amp

    # BASELINE
    hr = 70 + noise(mins, 3)
    sbp = 120 + noise(mins, 3)
    dbp = 80 + noise(mins, 2)
    rr = 16 + noise(mins, 1)
    
    # SCENARIO: Hypovolemic Shock transitioning to Cardiogenic Failure
    start = 240
    
    # 1. Preload Failure (Bleeding/Dehydration)
    # Stroke Volume (PP) drops FIRST.
    pp_drop = np.linspace(0, 30, mins-start)
    
    # 2. Compensation
    # HR rises, SVR rises (Vasoconstriction)
    hr_rise = np.linspace(0, 55, mins-start)
    svr_rise = np.linspace(0, 15, mins-start) # SVR proxy
    
    # Apply Physics
    true_sbp = sbp.copy()
    true_dbp = dbp.copy()
    
    true_sbp[start:] -= (pp_drop * 1.5) # SBP drops fast
    true_dbp[start:] += (svr_rise * 0.5) # DBP held up by SVR initially
    hr[start:] += hr_rise
    
    # 3. Respiratory Compensation (Acidosis)
    rr[start:] += np.linspace(0, 14, mins-start)

    # 4. Create DataFrame
    df = pd.DataFrame({'HR': hr, 'SBP': true_sbp, 'DBP': true_dbp, 'RR': rr}, index=t)
    
    # --- DERIVED ADVANCED PHYSICS ---
    
    # MAP (Perfusion Pressure)
    df['MAP'] = (df['SBP'] + 2*df['DBP']) / 3
    
    # Pulse Pressure (Proxy for Stroke Volume)
    df['PP'] = df['SBP'] - df['DBP']
    
    # Cardiac Output (CO) = HR * SV (Proxy)
    # We normalize to approx liters/min for display
    df['CO'] = (df['HR'] * df['PP']) / 1000 * 2.0 
    
    # Systemic Vascular Resistance (SVR) = MAP / CO
    # Proxy unit
    df['SVR'] = df['MAP'] / df['CO']
    
    # Shock Index
    df['SI'] = df['HR'] / df['SBP']
    
    # Autonomic Tone (HRV proxy)
    # High variability = Healthy (0), Low = Stress (1)
    df['Sympathetic_Tone'] = df['HR'].rolling(60).std().fillna(1)
    # Invert so High StdDev = Low Tone Score
    df['Sympathetic_Tone'] = 1 / (df['Sympathetic_Tone'] + 0.1)
    df['Sympathetic_Tone'] = (df['Sympathetic_Tone'] - df['Sympathetic_Tone'].min()) / (df['Sympathetic_Tone'].max() - df['Sympathetic_Tone'].min())
    
    return df

# --- 3. VISUALIZATION ENGINES ---

def plot_neon_timeline(df, curr_time):
    """
    Main HUD: MAP and CO on dual axis with neon glow.
    """
    start = max(0, curr_time - 180)
    data = df.iloc[start:curr_time]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Glow Effect Helper (Shadow trace)
    def add_neon_trace(x, y, color, name, sec_y=False):
        # Glow
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines', line=dict(color=color, width=10),
            opacity=0.1, showlegend=False, hoverinfo='skip'
        ), secondary_y=sec_y)
        # Core
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines', line=dict(color=color, width=2.5),
            name=name
        ), secondary_y=sec_y)

    add_neon_trace(data.index, data['MAP'], COLORS['magenta'], 'MAP (Pressure)', False)
    add_neon_trace(data.index, data['CO'], COLORS['cyan'], 'Cardiac Output (Flow)', True)
    
    # Thresholds
    fig.add_hline(y=65, line_color=COLORS['amber'], line_dash="dot", secondary_y=False)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=350, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.05, x=0),
        hovermode="x unified"
    )
    fig.update_yaxes(title="MAP (mmHg)", gridcolor=COLORS['grid'], secondary_y=False)
    fig.update_yaxes(title="CO (L/min)", gridcolor=COLORS['grid'], secondary_y=True, showgrid=False)
    return fig

def plot_starling_curve(df, curr_time):
    """
    The "Holy Grail" of ICU: Fluid Responsiveness.
    X: Preload (Est. by Inverse Shock Index or DBP) -> We use DBP as preload proxy here.
    Y: Stroke Volume (PP).
    """
    start = max(0, curr_time - 120)
    data = df.iloc[start:curr_time]
    
    fig = go.Figure()
    
    # Theoretical Curve (Background)
    x_theory = np.linspace(40, 100, 100)
    y_theory = np.log(x_theory) * 20 - 40
    fig.add_trace(go.Scatter(x=x_theory, y=y_theory, mode='lines', 
                             line=dict(color='gray', dash='dot'), name='Ideal Starling'))
    
    # Patient Trajectory
    # Color mapping: Fade to white
    fig.add_trace(go.Scatter(
        x=data['DBP'], y=data['PP'],
        mode='lines+markers',
        marker=dict(size=4, color=np.linspace(0, 1, len(data)), colorscale='Purples'),
        line=dict(color=COLORS['purple'], width=2),
        name='Patient Trajectory'
    ))
    
    # Current Head
    fig.add_trace(go.Scatter(
        x=[data['DBP'].iloc[-1]], y=[data['PP'].iloc[-1]],
        mode='markers', marker=dict(size=15, color=COLORS['cyan'], symbol='cross'),
        name='Current State'
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        title="<b>Starling Curve (Fluid Responsiveness)</b>",
        xaxis=dict(title="Preload Proxy (DBP)", gridcolor=COLORS['grid']),
        yaxis=dict(title="Stroke Vol Proxy (PP)", gridcolor=COLORS['grid']),
        height=300, margin=dict(l=0,r=0,t=40,b=0), showlegend=False
    )
    return fig

def plot_svr_co_coupling(df, curr_time):
    """
    Visualizes Pump vs Pipes.
    X: Cardiac Output (Pump)
    Y: SVR (Pipes)
    """
    start = max(0, curr_time - 120)
    data = df.iloc[start:curr_time]
    
    fig = go.Figure()
    
    # Zones
    fig.add_annotation(x=3, y=40, text="VASOCONSTRICTION<br>(Cold Shock)", font=dict(color=COLORS['magenta'], size=10))
    fig.add_annotation(x=6, y=10, text="VASODILATION<br>(Warm Shock)", font=dict(color=COLORS['cyan'], size=10))

    fig.add_trace(go.Scatter(
        x=data['CO'], y=data['SVR'],
        mode='lines', line=dict(color=COLORS['white'], width=2),
        name='Trajectory'
    ))
    
    # Current
    fig.add_trace(go.Scatter(
        x=[data['CO'].iloc[-1]], y=[data['SVR'].iloc[-1]],
        mode='markers', marker=dict(size=12, color=COLORS['amber']),
        name='Current'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        title="<b>Hemodynamic Coupling (Pump vs Pipes)</b>",
        xaxis=dict(title="Cardiac Output", gridcolor=COLORS['grid']),
        yaxis=dict(title="SVR (Resistance)", gridcolor=COLORS['grid']),
        height=300, margin=dict(l=0,r=0,t=40,b=0), showlegend=False
    )
    return fig

def plot_organ_matrix(df, curr_time):
    """
    Sparklines for major organ systems.
    """
    start = max(0, curr_time - 60)
    data = df.iloc[start:curr_time]
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=("CNS (Sympathetic Tone)", "Respiratory (RR)", "Renal (Perfusion)", "Cardiac (Efficiency)"))
    
    # CNS
    fig.add_trace(go.Scatter(x=data.index, y=data['Sympathetic_Tone'], line=dict(color=COLORS['amber'])), row=1, col=1)
    # Resp
    fig.add_trace(go.Scatter(x=data.index, y=data['RR'], line=dict(color=COLORS['cyan'])), row=1, col=2)
    # Renal (Using MAP as proxy)
    fig.add_trace(go.Scatter(x=data.index, y=data['MAP'], line=dict(color=COLORS['magenta'])), row=2, col=1)
    # Cardiac (Using PP)
    fig.add_trace(go.Scatter(x=data.index, y=data['PP'], line=dict(color=COLORS['purple'])), row=2, col=2)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=250, margin=dict(l=0,r=0,t=20,b=0), showlegend=False
    )
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)
    
    return fig
