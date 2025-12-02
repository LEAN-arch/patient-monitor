import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats

# --- 1. Robust Physiological Simulation ---
def simulate_patient(mins_total=720):
    """
    Simulates a patient transitioning from stable -> compensatory shock -> decompensated shock.
    """
    t = np.arange(mins_total)
    
    # Base Vitals (Stable)
    hr = np.random.normal(75, 2, mins_total)
    sbp = np.random.normal(120, 3, mins_total)
    spo2 = np.random.normal(98, 0.5, mins_total)
    rr = np.random.normal(16, 1, mins_total)
    pi = np.random.normal(3.5, 0.2, mins_total) # Perfusion Index

    # SHOCK EVENT starts at min 400
    # Phase 1: Compensation (400-550): HR UP, SBP Stable/Slight Up, PI DOWN (vasoconstriction)
    # Phase 2: Decompensation (550+): HR High, SBP CRASHES, SpO2 Drops
    
    shock_start = 400
    decomp_start = 550
    
    # HR Trend
    hr[shock_start:] += np.linspace(0, 40, mins_total-shock_start) + np.random.normal(0, 1, mins_total-shock_start)
    
    # SBP Trend (Maintained then crashes)
    sbp[decomp_start:] -= np.linspace(0, 40, mins_total-decomp_start)
    
    # PI Trend (Early indicator!)
    pi[shock_start:] = np.maximum(0.1, pi[shock_start:] - np.linspace(0, 3.0, mins_total-shock_start))
    
    # RR Trend (Compensating for acidosis)
    rr[shock_start:] += np.linspace(0, 12, mins_total-shock_start)

    # SpO2 (Late sign)
    spo2[decomp_start:] = np.maximum(80, spo2[decomp_start:] - np.linspace(0, 10, mins_total-decomp_start))

    df = pd.DataFrame({'HR': hr, 'SBP': sbp, 'SpO2': spo2, 'RR': rr, 'PI': pi}, index=t)
    return df

# --- 2. Analytics Helper ---
def fit_var_and_residuals_full(df, baseline_window=120):
    """
    Calculates Z-score deviations based on the first 2 hours (baseline).
    """
    baseline = df.iloc[:baseline_window]
    means = baseline.mean()
    stds = baseline.std()
    
    # Avoid division by zero
    stds[stds == 0] = 1
    
    # Calculate Z-scores (Standardized Residuals)
    z_scores = (df - means) / stds
    
    # Covariance of the baseline Z-scores
    cov_matrix = z_scores.iloc[:baseline_window].cov()
    
    return z_scores, cov_matrix

def compute_mahalanobis_risk(z_scores, cov_matrix):
    """
    Computes Mahalanobis Distance (Global Instability Score).
    """
    inv_cov = np.linalg.pinv(cov_matrix.values)
    values = z_scores.values
    
    # MD = sqrt( (x-u)T * inv_cov * (x-u) )
    # Since we z-scored, u=0.
    
    md_sq = [np.dot(np.dot(row, inv_cov), row.T) for row in values]
    risk = np.sqrt(md_sq)
    
    return risk, inv_cov

# --- 3. Clinical Visualizations ---

def plot_combined_vitals(df, t_axis, shock_index):
    """
    Plots vitals aligned with the Shock Index to show hemodynamic coupling.
    """
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05,
                        row_heights=[0.4, 0.3, 0.3],
                        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]])

    # Row 1: Hemodynamics (HR vs SBP)
    fig.add_trace(go.Scatter(x=t_axis, y=df['HR'], name="HR", line=dict(color='#ff4b4b', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_axis, y=df['SBP'], name="SBP", line=dict(color='#636efa', width=2)), row=1, col=1, secondary_y=True)
    
    # Row 2: Shock Index (The Actionable Metric)
    # SI > 0.9 is dangerous
    fig.add_trace(go.Scatter(x=t_axis, y=shock_index, name="Shock Index", 
                             fill='tozeroy', line=dict(color='#ffa15a', width=1)), row=2, col=1)
    fig.add_hline(y=0.9, line_dash="dot", line_color="red", annotation_text="Critical Threshold (>0.9)", row=2, col=1)

    # Row 3: Perfusion (Early Warning)
    fig.add_trace(go.Scatter(x=t_axis, y=df['PI'], name="Perfusion Index", line=dict(color='#00cc96', width=2)), row=3, col=1)

    fig.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark", title_text="Hemodynamic Profile & Shock Index")
    fig.update_yaxes(title_text="HR (bpm)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="SBP (mmHg)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Shock Index", row=2, col=1)
    fig.update_yaxes(title_text="PI %", row=3, col=1)
    
    return fig

def plot_temporal_contribution(z_scores_view, vars_list):
    """
    Heatmap showing WHICH variable is driving the instability over time.
    Red = High Deviation, Blue = Low Deviation (relative to baseline).
    """
    z_clipped = z_scores_view.clip(-4, 4) # Clip for color scaling
    
    fig = go.Figure(data=go.Heatmap(
        z=z_clipped.T,
        x=z_scores_view.index,
        y=vars_list,
        colorscale='RdBu_r', # Red=High, Blue=Low
        zmid=0,
        colorbar=dict(title="Z-Score Dev")
    ))
    
    fig.update_layout(
        title="Anatomy of Instability (Deviation Heatmap)",
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_dark"
    )
    return fig

def plot_clinical_radar(current_row, baseline_means):
    """
    Radar chart mapping the current state to clinical axes.
    We normalize everything so that 'Center' is baseline and 'Edge' is critical.
    """
    categories = ['Tachycardia (HR)', 'Hypotension (Inv SBP)', 'Hypoperfusion (Inv PI)', 'Desaturation (Inv SpO2)', 'Tachypnea (RR)']
    
    # Normalize values for visual impact (Arbitrary scaling for demo)
    # Higher value on chart = WORSE clinical state
    val_hr = max(0, (current_row['HR'] - 60) / 100) 
    val_sbp = max(0, (140 - current_row['SBP']) / 100) # Inverted: Lower BP = Higher Risk
    val_pi = max(0, (4.0 - current_row['PI']) / 4.0)   # Inverted: Lower PI = Higher Risk
    val_spo2 = max(0, (100 - current_row['SpO2']) / 20) # Inverted
    val_rr = max(0, (current_row['RR'] - 12) / 30)

    values = [val_hr, val_sbp, val_pi, val_spo2, val_rr]
    values += [values[0]] # Close the loop
    categories += [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current State',
        line_color='#ef553b'
    ))
    
    # Add a "Stable Zone" circle
    fig.add_trace(go.Scatterpolar(
        r=[0.2]*6,
        theta=categories,
        line_color='green',
        line_dash='dot',
        hoverinfo='skip',
        name='Stable Limit'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
        ),
        showlegend=False,
        title="Physiological Footprint",
        height=350,
        margin=dict(l=40, r=40, t=40, b=20),
        template="plotly_dark"
    )
    return fig

def plot_risk_gauge(current_risk):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = current_risk,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Multivariate Instability Score"},
        gauge = {
            'axis': {'range': [None, 50], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "white", 'thickness': 0.2},
            'steps': [
                {'range': [0, 15], 'color': "#00cc96"}, # Stable
                {'range': [15, 25], 'color': "#ffa15a"}, # Warning
                {'range': [25, 50], 'color': "#ef553b"}  # Critical
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 25
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), template="plotly_dark")
    return fig
