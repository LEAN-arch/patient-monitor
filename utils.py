import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Simulation Logic (Unchanged for consistency) ---
def simulate_patient(mins_total=720):
    t = np.arange(mins_total)
    
    # Base Vitals (Stable) with slight noise
    hr = np.random.normal(75, 1.5, mins_total)
    sbp = np.random.normal(120, 2, mins_total)
    spo2 = np.random.normal(98, 0.5, mins_total)
    rr = np.random.normal(16, 1, mins_total)
    pi = np.random.normal(3.5, 0.1, mins_total)

    # SHOCK EVENT starts at min 400
    shock_start = 400
    decomp_start = 550
    
    # Phase 1: Compensation (400-550)
    # HR rises steadily
    hr[shock_start:] += np.linspace(0, 45, mins_total-shock_start) + np.random.normal(0, 2, mins_total-shock_start)
    # PI drops (Vasoconstriction) - The early sign
    pi[shock_start:] = np.maximum(0.2, pi[shock_start:] - np.linspace(0, 3.2, mins_total-shock_start))
    # RR increases (Compensatory tachypnea)
    rr[shock_start:] += np.linspace(0, 14, mins_total-shock_start)

    # Phase 2: Decompensation (550+)
    # SBP crashes
    sbp[decomp_start:] -= np.linspace(0, 45, mins_total-decomp_start) + np.random.normal(0, 2, mins_total-decomp_start)
    # SpO2 drops
    spo2[decomp_start:] = np.maximum(82, spo2[decomp_start:] - np.linspace(0, 12, mins_total-decomp_start))

    df = pd.DataFrame({'HR': hr, 'SBP': sbp, 'SpO2': spo2, 'RR': rr, 'PI': pi}, index=t)
    return df

def fit_var_and_residuals_full(df, baseline_window=120):
    baseline = df.iloc[:baseline_window]
    means = baseline.mean()
    stds = baseline.std()
    stds[stds == 0] = 1
    z_scores = (df - means) / stds
    cov_matrix = z_scores.iloc[:baseline_window].cov()
    return z_scores, cov_matrix

def compute_mahalanobis_risk(z_scores, cov_matrix):
    inv_cov = np.linalg.pinv(cov_matrix.values)
    values = z_scores.values
    md_sq = [np.dot(np.dot(row, inv_cov), row.T) for row in values]
    risk = np.sqrt(md_sq)
    return risk, inv_cov

# --- 2. Commercial-Grade Visualization Functions ---

# COLOR PALETTE (Medical Standard)
COLORS = {
    'HR': '#00ff00',      # Green
    'SBP': '#ff3333',     # Red
    'SpO2': '#00ccff',    # Cyan
    'RR': '#ffff00',      # Yellow
    'PI': '#d142f5',      # Purple
    'Risk': '#ffffff',    # White
    'Grid': '#1f2630',    # Dark Grey
    'Bg': '#0e1117'       # Streamlit Dark
}

def plot_monitor_strip(df, t_axis, shock_index):
    """
    Creates a synchronized multi-channel strip chart resembling a bedside monitor.
    Features: Dark mode, neon lines, synchronized crosshairs (spikelines).
    """
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02,
        row_heights=[0.25, 0.25, 0.25, 0.25],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # --- Track 1: Hemodynamics (HR & SBP) ---
    # HR Area
    fig.add_trace(go.Scatter(
        x=t_axis, y=df['HR'], name="HR", mode='lines',
        line=dict(color=COLORS['HR'], width=2),
        fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.1)'
    ), row=1, col=1)
    
    # SBP Line
    fig.add_trace(go.Scatter(
        x=t_axis, y=df['SBP'], name="SBP", mode='lines',
        line=dict(color=COLORS['SBP'], width=2)
    ), row=1, col=1, secondary_y=True)

    # --- Track 2: Shock Index (The Derived Metric) ---
    fig.add_trace(go.Scatter(
        x=t_axis, y=shock_index, name="Shock Index", mode='lines',
        line=dict(color='#ffa500', width=2),
        fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.1)'
    ), row=2, col=1)
    
    # Critical Threshold Line
    fig.add_hline(y=0.9, line_dash="dot", line_color="red", row=2, col=1, annotation_text="Risk > 0.9", annotation_position="top left")

    # --- Track 3: Perfusion (PI) ---
    fig.add_trace(go.Scatter(
        x=t_axis, y=df['PI'], name="Perfusion", mode='lines',
        line=dict(color=COLORS['PI'], width=2),
        fill='tozeroy', fillcolor='rgba(209, 66, 245, 0.1)'
    ), row=3, col=1)

    # --- Track 4: Respiration & Oxygen ---
    fig.add_trace(go.Scatter(
        x=t_axis, y=df['SpO2'], name="SpO2", mode='lines',
        line=dict(color=COLORS['SpO2'], width=2)
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=t_axis, y=df['RR'], name="RR", mode='lines',
        line=dict(color=COLORS['RR'], width=1, dash='dot')
    ), row=4, col=1)

    # --- Commercial Styling ---
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600,
        margin=dict(l=0, r=0, t=20, b=20),
        legend=dict(orientation="h", y=1.02, x=0, bgcolor='rgba(0,0,0,0)'),
        hovermode="x unified" # Key for synchronization
    )

    # Grid and Axis Styling (Medical Monitor Look)
    for i in range(1, 5):
        fig.update_yaxes(showgrid=True, gridcolor=COLORS['Grid'], zeroline=False, row=i, col=1)
        # Add Spikelines (The vertical cursor)
        fig.update_xaxes(
            showgrid=True, gridcolor=COLORS['Grid'], 
            showspikes=True, spikemode='across', spikesnap='cursor', 
            spikedash='solid', spikecolor='grey', spikethickness=1,
            row=i, col=1
        )

    # Specific Y-Axis Labels
    fig.update_yaxes(title_text="HR / SBP", row=1, col=1)
    fig.update_yaxes(title_text="Shock Idx", row=2, col=1)
    fig.update_yaxes(title_text="PI %", row=3, col=1)
    fig.update_yaxes(title_text="SpO2 / RR", row=4, col=1)

    return fig

def plot_heatmap_commercial(z_scores_view, vars_list):
    """
    High-contrast heatmap with custom colorscale to highlight anomalies.
    """
    # Custom colorscale: Black/Blue for low/normal, fiery Red/Yellow for high deviation
    
    # Clip Z-scores for visualization stability
    z_clipped = z_scores_view.clip(-4, 4)
    
    fig = go.Figure(data=go.Heatmap(
        z=z_clipped.T,
        x=z_scores_view.index,
        y=vars_list,
        colorscale=[
            [0.0, '#0000ff'],   # Deep Blue (Low)
            [0.35, '#000033'],  # Dark Blue
            [0.5, '#000000'],   # Black (Baseline)
            [0.65, '#330000'],  # Dark Red
            [1.0, '#ff0000']    # Bright Red (High)
        ],
        zmid=0,
        showscale=False # Remove colorbar to save space, rely on hover
    ))
    
    fig.update_layout(
        title=dict(text="DEVIATION HEATMAP (Z-Score)", font=dict(size=12, color="gray")),
        height=200,
        margin=dict(l=0, r=0, t=30, b=10),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified"
    )
    fig.update_xaxes(showgrid=False, showspikes=True, spikemode='across', spikecolor='white')
    fig.update_yaxes(showgrid=False)
    
    return fig

def plot_radar_commercial(current_row):
    """
    A 'Target' style radar chart with risk zones.
    """
    categories = ['Tachycardia', 'Hypotension', 'Hypoperfusion', 'Desaturation', 'Tachypnea']
    
    # Normalize to 0-1 scale where 1 is Max Criticality
    # Using clinical logic inversions where necessary
    val_hr = np.clip((current_row['HR'] - 60) / 100, 0, 1)
    val_sbp = np.clip((140 - current_row['SBP']) / 100, 0, 1) # Inverted
    val_pi = np.clip((4.0 - current_row['PI']) / 4.0, 0, 1)   # Inverted
    val_spo2 = np.clip((100 - current_row['SpO2']) / 20, 0, 1) # Inverted
    val_rr = np.clip((current_row['RR'] - 12) / 30, 0, 1)

    values = [val_hr, val_sbp, val_pi, val_spo2, val_rr]
    values += [values[0]]
    categories += [categories[0]]

    fig = go.Figure()

    # 1. Background Risk Zones
    fig.add_trace(go.Scatterpolar(
        r=[0.5]*6, theta=categories, fill='toself', 
        fillcolor='rgba(0, 255, 0, 0.1)', line=dict(color='green', width=1, dash='dot'),
        hoverinfo='skip', name='Safe Zone'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[0.8]*6, theta=categories, fill='toself', 
        fillcolor='rgba(255, 165, 0, 0.1)', line=dict(color='orange', width=1, dash='dot'),
        hoverinfo='skip', name='Warning Zone'
    ))

    # 2. Actual Patient Data
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.4)',
        line=dict(color='red', width=3),
        name='Current Status'
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(tickfont=dict(size=10, color='white'))
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=30, r=30, t=20, b=20),
        height=300
    )
    return fig

def plot_gauge_commercial(score):
    """
    Minimalist semi-circle gauge.
    """
    color = "#00ff00"
    if score > 15: color = "#ffa500"
    if score > 25: color = "#ff0000"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        number = {'font': {'size': 40, 'color': color}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 50], 'visible': False},
            'bar': {'color': color, 'thickness': 1},
            'bgcolor': "#333",
            'steps': [
                {'range': [0, 50], 'color': "#1e2630"} # Background track
            ],
        }
    ))
    fig.update_layout(
        height=200, 
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig
