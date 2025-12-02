import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# --- 1. DESIGN SYSTEM (Commercial "Precision Light" Palette) ---
THEME = {
    'bg_paper': '#ffffff',
    'bg_plot': 'rgba(240, 242, 246, 0.5)',
    'text_main': '#1f2937',
    'text_sub': '#6b7280',
    'grid': '#e5e7eb',
    # Data Colors
    'hr_line': '#2563eb',       # Professional Blue
    'hr_fill': 'rgba(37, 99, 235, 0.1)',
    'bp_sys': '#db2777',        # Pink/Magenta
    'bp_dia': 'rgba(219, 39, 119, 0.3)',
    'si_line': '#d97706',       # Amber
    'pi_line': '#059669',       # Emerald
    # SPC Zones
    'zone_err': 'rgba(220, 38, 38, 0.08)',  # Red tint
    'zone_warn': 'rgba(245, 158, 11, 0.12)',# Amber tint
    'forecast': '#7c3aed'       # Violet for AI
}

# --- 2. ADVANCED PHYSIOLOGICAL SIMULATION ---
def simulate_clinical_course(mins=720):
    t = np.arange(mins)
    
    # Base Dynamics (Sinus Arrhythmia + Diurnal noise)
    noise = np.random.normal(0, 0.5, mins)
    hr = 72 + 2 * np.sin(t/40) + noise
    sbp = 118 + 1.5 * np.cos(t/60) + noise
    pi = 4.0 + 0.2 * np.sin(t/20) + 0.1 * noise
    
    # EVENT: "Cold Sepsis" (Vasoconstriction -> Tachycardia -> Hypotension)
    # Starts insidiously at T=300
    start = 300
    
    # 1. PI Drops FIRST (The hidden sign)
    drift_pi = np.linspace(0, 3.5, mins-start)
    pi[start:] = np.maximum(0.3, pi[start:] - drift_pi)
    
    # 2. HR Rises (Compensation)
    drift_hr = np.linspace(0, 45, mins-start) ** 1.1
    hr[start:] += drift_hr
    
    # 3. SBP Crashes LATE (Decompensation) - Starts at T=500
    crash = 500
    if mins > crash:
        drift_sbp = np.linspace(0, 35, mins-crash) ** 1.5
        sbp[crash:] -= drift_sbp

    df = pd.DataFrame({'HR': hr, 'SBP': sbp, 'PI': pi}, index=t)
    df['SI'] = df['HR'] / df['SBP'] # Shock Index
    df['MAP'] = (df['SBP'] + (df['SBP']*0.6))/1.6 # Approx MAP for visualization
    return df

# --- 3. ANALYTICS ENGINES ---

def run_western_electric_spc(series, window=60):
    """
    Calculates dynamic statistical zones.
    Returns the bands needed for the 'River' visualization.
    """
    roll = series.rolling(window=window)
    mean = roll.mean()
    std = roll.std()
    
    return {
        'mean': mean,
        'u3': mean + 3*std, 'l3': mean - 3*std, # Critical
        'u2': mean + 2*std, 'l2': mean - 2*std, # Warning
        'u1': mean + 1*std, 'l1': mean - 1*std  # Nominal
    }

def generate_prognosis(series, horizon=30):
    """
    Auto-Regressive projection with confidence cone.
    """
    y = series.values[-45:] # Use last 45 mins for momentum
    x = np.arange(len(y)).reshape(-1, 1)
    
    # Weighted regression (recent points matter more)
    weights = np.linspace(0.5, 1.0, len(y))
    model = LinearRegression()
    model.fit(x, y, sample_weight=weights)
    
    future_x = np.arange(len(y), len(y)+horizon).reshape(-1, 1)
    pred = model.predict(future_x)
    
    # Uncertainty expands with time
    sigma = np.std(y)
    expand = np.linspace(1, 2.5, horizon)
    upper = pred + (1.96 * sigma * expand)
    lower = pred - (1.96 * sigma * expand)
    
    return pred, upper, lower

def run_pca_compass(df, curr_time):
    """
    Computes the 'Clinical Vector' to show direction of deterioration.
    """
    # Baseline Reference (First 2 hours)
    ref = df.iloc[:120][['HR', 'SBP', 'PI']]
    scaler = StandardScaler().fit(ref)
    pca = PCA(n_components=2).fit(scaler.transform(ref))
    
    # Current Trajectory (Last 4 hours)
    recent = df.iloc[max(0, curr_time-240):curr_time][['HR', 'SBP', 'PI']]
    coords = pca.transform(scaler.transform(recent))
    
    # Loadings (Which variable pulls which way?)
    loadings = pd.DataFrame(pca.components_.T, columns=['x', 'y'], index=['HR', 'SBP', 'PI'])
    
    return coords, loadings

# --- 4. COMMERCIAL-GRADE PLOTTING ---

def plot_command_center(df, curr_time):
    """
    The 'Money Plot'. A synchronized, 3-track timeline.
    Row 1: HR + SPC Zones + AI Forecast
    Row 2: Blood Pressure (Area)
    Row 3: Perfusion + Shock Index
    """
    # Slice Data
    window = 180 # 3 hour lookback
    start = max(0, curr_time - window)
    data = df.iloc[start:curr_time]
    
    # Analytics
    spc = run_western_electric_spc(df['HR'].iloc[:curr_time])
    ai_pred, ai_up, ai_low = generate_prognosis(df['HR'].iloc[:curr_time])
    t_fut = np.arange(curr_time, curr_time+30)

    # Layout
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.45, 0.25, 0.3],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]]
    )

    # --- TRACK 1: HR + SPC + AI (The Primary Signal) ---
    # 1A. SPC Zones (Painted first as background)
    fig.add_trace(go.Scatter(x=data.index, y=spc['u3'].loc[data.index], line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=spc['u2'].loc[data.index], line=dict(width=0), 
                             fill='tonexty', fillcolor=THEME['zone_warn'], name='Warning Zone (2σ)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=spc['l2'].loc[data.index], line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=spc['l3'].loc[data.index], line=dict(width=0), 
                             fill='tonexty', fillcolor=THEME['zone_warn'], showlegend=False), row=1, col=1)
    
    # 1B. The AI Forecast Fan
    fig.add_trace(go.Scatter(x=t_fut, y=ai_up, line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_fut, y=ai_low, line=dict(width=0), 
                             fill='tonexty', fillcolor='rgba(124, 58, 237, 0.15)', name='AI Prognosis'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_fut, y=ai_pred, line=dict(color=THEME['forecast'], dash='dot', width=2), showlegend=False), row=1, col=1)

    # 1C. Actual Data
    fig.add_trace(go.Scatter(
        x=data.index, y=data['HR'], mode='lines', 
        line=dict(color=THEME['hr_line'], width=2.5), name='Heart Rate'
    ), row=1, col=1)

    # --- TRACK 2: Blood Pressure Dynamics ---
    fig.add_trace(go.Scatter(
        x=data.index, y=data['SBP'], line=dict(color=THEME['bp_sys'], width=2), name='Sys BP'
    ), row=2, col=1)
    # Fill to a baseline approx (MAP) to show "Pulse Pressure" visually
    fig.add_trace(go.Scatter(
        x=data.index, y=data['MAP'], line=dict(width=0), 
        fill='tonexty', fillcolor=THEME['bp_dia'], showlegend=False
    ), row=2, col=1)

    # --- TRACK 3: Perfusion & Shock Index ---
    fig.add_trace(go.Scatter(
        x=data.index, y=data['PI'], line=dict(color=THEME['pi_line'], width=2), 
        name='Perfusion (PI)', fill='tozeroy', fillcolor='rgba(5, 150, 105, 0.1)'
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=data.index, y=data['SI'], line=dict(color=THEME['si_line'], width=2), name='Shock Index'
    ), row=3, col=1, secondary_y=True)
    
    # Add Threshold Line for SI
    fig.add_hline(y=0.9, line_dash="dot", line_color=THEME['si_line'], row=3, col=1, secondary_y=True, opacity=0.5)

    # --- STYLING (The "Apple Health" Look) ---
    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.01, x=0, bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
        hovermode="x unified",
        plot_bgcolor=THEME['bg_plot']
    )
    
    # Axis Polish
    fig.update_xaxes(showgrid=True, gridcolor=THEME['grid'], showspikes=True, spikemode='across', spikethickness=1)
    fig.update_yaxes(showgrid=True, gridcolor=THEME['grid'], tickfont=dict(size=10))
    
    fig.update_yaxes(title_text="HR (BPM)", row=1, col=1)
    fig.update_yaxes(title_text="BP (mmHg)", row=2, col=1)
    fig.update_yaxes(title_text="PI %", row=3, col=1)
    fig.update_yaxes(title_text="SI", row=3, col=1, secondary_y=True, showgrid=False)

    return fig

def plot_pca_compass(coords, loadings):
    """
    The 'GPS' Plot.
    Shows the patient moving away from the 'Safe Harbor' (Center).
    """
    fig = go.Figure()

    # 1. The Safe Zone (Homeostasis)
    fig.add_shape(type="circle", x0=-2, y0=-2, x1=2, y1=2, 
                  line_color=THEME['pi_line'], fillcolor="rgba(5, 150, 105, 0.05)")

    # 2. The Trajectory (Comet Tail)
    # Fade out older points to show direction clearly
    opacity = np.linspace(0.1, 1, len(coords))
    fig.add_trace(go.Scatter(
        x=coords[:,0], y=coords[:,1], mode='markers+lines',
        marker=dict(color=THEME['hr_line'], size=4, opacity=opacity),
        line=dict(color=THEME['hr_line'], width=1, dash='solid'),
        showlegend=False
    ))

    # 3. The "Current State" Pulsing Dot
    fig.add_trace(go.Scatter(
        x=[coords[-1,0]], y=[coords[-1,1]], mode='markers',
        marker=dict(color=THEME['bp_sys'], size=14, symbol='cross'),
        name='YOU ARE HERE'
    ))

    # 4. The Compass Vectors (Loadings)
    # These show "North" for Tachycardia, "East" for Hypotension, etc.
    scale = 3.5
    for feature in loadings.index:
        x_vec = loadings.loc[feature, 'x'] * scale
        y_vec = loadings.loc[feature, 'y'] * scale
        
        # Draw Arrow
        fig.add_annotation(
            x=x_vec, y=y_vec, ax=0, ay=0, xref='x', yref='y', axref='x', ayref='y',
            text=f"<b>{feature}</b>", showarrow=True, arrowhead=2, arrowsize=1, 
            arrowcolor=THEME['text_sub'], font=dict(size=10, color=THEME['text_sub'])
        )

    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="<b>CLINICAL STATE VECTOR</b> (PCA)", font=dict(size=12, color=THEME['text_sub'])),
        xaxis=dict(showgrid=True, gridcolor=THEME['grid'], zeroline=False, visible=False),
        yaxis=dict(showgrid=True, gridcolor=THEME['grid'], zeroline=False, visible=False),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_causal_strip(df, curr_time):
    """
    A minimal heatmap strip that acts as a 'Genetic Barcode' of instability.
    """
    # Z-Score relative to baseline
    baseline = df.iloc[:120].mean()
    std = df.iloc[:120].std()
    
    # 2 Hour Window
    w_start = max(0, curr_time-120)
    subset = df.iloc[w_start:curr_time][['HR','SBP','PI']]
    z = ((subset - baseline) / std).T
    
    # High contrast colorscale (Blue=Low, White=Normal, Red=High)
    fig = go.Figure(data=go.Heatmap(
        z=z.values, x=z.columns, y=z.index,
        colorscale='RdBu_r', zmid=0, zmin=-4, zmax=4,
        showscale=False
    ))
    
    fig.update_layout(
        template="plotly_white",
        height=150,
        margin=dict(l=0, r=0, t=25, b=0),
        title=dict(text="<b>DEVIATION FINGERPRINT (Z-Score)</b>", font=dict(size=11, color=THEME['text_sub'])),
        xaxis=dict(showgrid=False, showticklabels=False), # Clean look
        yaxis=dict(showgrid=False, tickfont=dict(size=10, weight='bold'))
    )
    return fig
