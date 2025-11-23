import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
import math

# --- 1. Simulation (Physiologically Realistic) ---
def simulate_patient(mins_total=720, seed=42):
    """
    Simulates a patient progressing from stable -> respiratory distress -> decompensated shock.
    """
    np.random.seed(seed)
    
    # 1. Define Baseline (Stable Patient)
    # HR, SBP, SpO2, RR, PI
    means = np.array([75., 120., 98., 16., 3.5])
    stds  = np.array([3.,  5.,   0.5, 1.0, 0.5])
    
    # Physiological correlations
    corr = np.array([
        [ 1.0, -0.2, -0.1,  0.5, -0.2], # HR
        [-0.2,  1.0,  0.1, -0.1,  0.4], # SBP
        [-0.1,  0.1,  1.0, -0.2, -0.1], # SpO2
        [ 0.5, -0.1, -0.2,  1.0, -0.2], # RR
        [-0.2,  0.4, -0.1, -0.2,  1.0]  # PI
    ])
    cov = np.outer(stds, stds) * corr
    
    # 2. Generate VAR(1) Baseline
    phi = 0.7 
    data = np.zeros((mins_total, 5))
    data[0] = means
    
    noise_cov = cov * (1 - phi**2)
    
    for t in range(1, mins_total):
        noise = np.random.multivariate_normal(np.zeros(5), noise_cov)
        data[t] = means + phi * (data[t-1] - means) + noise

    df = pd.DataFrame(data, columns=['HR','SBP','SpO2','RR','PI'])
    
    # 3. Inject Clinical Events
    
    # Event A: Transient Hypoxia (Mucus Plug / Position change) at min 180
    # SpO2 drops sharply, HR rises to compensate, then resolves
    t_event = 180
    df.loc[t_event:t_event+20, 'SpO2'] -= 8.0 
    df.loc[t_event:t_event+20, 'HR'] += 15.0
    
    # Event B: Decompensated Shock starting at min 420
    t_shock = 420
    for t in range(t_shock, mins_total):
        # Non-linear progression (gets worse faster)
        progress = ((t - t_shock) / (mins_total - t_shock)) ** 1.5
        
        # Clinical Pattern: Shock
        # 1. HR rises (Compensatory Tachycardia)
        df.at[t, 'HR']   += 50.0 * progress
        # 2. SBP drops (Hypotension - late sign)
        df.at[t, 'SBP']  -= 40.0 * progress
        # 3. RR rises (Metabolic Acidosis compensation)
        df.at[t, 'RR']   += 14.0 * progress
        # 4. PI drops (Peripheral vasoconstriction - early sign)
        df.at[t, 'PI']   -= 3.0 * progress
        # 5. SpO2 drops eventually
        df.at[t, 'SpO2'] -= 6.0 * progress
        
        # Add stress noise (variability increases in instability)
        df.iloc[t] += np.random.normal(0, 1.0 * progress, 5)

    # Clamps
    df['SpO2'] = df['SpO2'].clip(0, 100)
    df['SBP'] = df['SBP'].clip(40, 250)
    df['PI'] = df['PI'].clip(0.1, 20)
    
    return df

# --- 2. Analytics ---

def fit_var_and_residuals_full(df_full, baseline_window=120):
    """ Fits VAR on stable baseline, predicts residuals for full history. """
    train_data = df_full.iloc[:baseline_window]
    model = VAR(train_data)
    try:
        results = model.fit(maxlags=1)
        lag = results.k_ar
        
        c = results.params[0, :]
        A = results.params[1:, :]
        
        data_val = df_full.values
        residuals = np.zeros_like(data_val)
        
        for t in range(lag, len(data_val)):
            pred = c + np.dot(data_val[t-1], A)
            residuals[t] = data_val[t] - pred
            
        cov_est = np.cov(residuals[lag:baseline_window].T)
        return residuals, cov_est, lag
        
    except Exception:
        return (df_full - df_full.mean()).values, np.eye(5), 0

def compute_mahalanobis_risk(residuals, cov_est, lam=0.15):
    """ Computes Mahalanobis Distance + EWMA Smoothing. """
    try:
        inv_cov = np.linalg.pinv(cov_est + 1e-6 * np.eye(cov_est.shape[0]))
    except:
        inv_cov = np.eye(cov_est.shape[0])
        
    D2 = np.sum((residuals @ inv_cov) * residuals, axis=1)
    
    # EWMA Smoothing
    z = np.zeros_like(D2)
    z[0] = D2[0]
    for t in range(1, len(D2)):
        z[t] = lam * D2[t] + (1 - lam) * z[t-1]
        
    return z, inv_cov

# --- 3. Enhanced Visualization (Clinical Focus) ---

def plot_vitals(df_view, t_axis):
    """ Plots HR/SBP/SpO2 with clear reference cues. """
    fig = go.Figure()
    
    # Highlight "Normal" HR Range (60-100)
    fig.add_shape(type="rect",
        x0=t_axis[0], x1=t_axis[-1], y0=60, y1=100,
        fillcolor="green", opacity=0.05, layer="below", line_width=0,
    )

    fig.add_trace(go.Scatter(x=t_axis, y=df_view['HR'], name='HR (bpm)', line=dict(color='#d62728', width=2.5)))
    fig.add_trace(go.Scatter(x=t_axis, y=df_view['SBP'], name='SBP (mmHg)', line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=t_axis, y=df_view['SpO2'], name='SpO2 (%)', yaxis='y2', line=dict(color='#2ca02c', width=2, dash='dot')))
    
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.1, x=0),
        yaxis=dict(title="Hemodynamics", range=[40, 180], showgrid=True, gridcolor='lightgrey'),
        yaxis2=dict(title="SpO2", overlaying='y', side='right', range=[80, 101], showgrid=False),
        plot_bgcolor='white',
        hovermode="x unified"
    )
    return fig

def plot_risk_enhanced(risk_scores, t_axis):
    """ Risk plot with colored semantic zones (Safe/Warning/Critical). """
    fig = go.Figure()
    
    # Define semantic zones
    max_val = max(50, np.max(risk_scores) * 1.1)
    
    # Green Zone (Stable)
    fig.add_hrect(y0=0, y1=15, fillcolor="green", opacity=0.1, layer="below", line_width=0, annotation_text="STABLE", annotation_position="top left")
    # Yellow Zone (Warning)
    fig.add_hrect(y0=15, y1=30, fillcolor="orange", opacity=0.1, layer="below", line_width=0, annotation_text="WARNING", annotation_position="top left")
    # Red Zone (Critical)
    fig.add_hrect(y0=30, y1=max_val, fillcolor="red", opacity=0.1, layer="below", line_width=0, annotation_text="CRITICAL", annotation_position="top left")

    fig.add_trace(go.Scatter(x=t_axis, y=risk_scores, mode='lines', name='Risk Index', line=dict(color='black', width=2)))
    
    fig.update_layout(
        height=250,
        title="<b>Integrated Physiological Instability Score</b>",
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(title="Deviation Magnitude", range=[0, max_val]),
        showlegend=False,
        plot_bgcolor='white'
    )
    return fig

def plot_pca_clinical(df_full, curr_time, baseline_window=120):
    """ 
    PCA fitted ONLY on baseline. 
    This makes (0,0) the 'Safe Zone'. Deviation from center = Clinical Instability.
    """
    if len(df_full) < 2: return go.Figure()
    
    # 1. Fit Scaler & PCA on BASELINE ONLY (The "Stable" definition)
    baseline_data = df_full.iloc[:baseline_window]
    scaler = StandardScaler().fit(baseline_data)
    pca = PCA(n_components=2).fit(scaler.transform(baseline_data))
    
    # 2. Transform the FULL trajectory
    X_scaled = scaler.transform(df_full.iloc[:curr_time])
    coords = pca.transform(X_scaled)
    
    # 3. Plot
    fig = go.Figure()
    
    # Draw "Safe Zone" (Circle at 0,0)
    fig.add_shape(type="circle",
        xref="x", yref="y", x0=-2, y0=-2, x1=2, y1=2,
        line_color="green", fillcolor="green", opacity=0.1, line_dash="dot"
    )
    fig.add_annotation(x=0, y=0, text="Safe Baseline", showarrow=False, font=dict(color="green"))

    # Draw Trajectory
    # Color by time (fading) to show direction
    colors = np.arange(len(coords))
    
    fig.add_trace(go.Scatter(
        x=coords[:,0], y=coords[:,1],
        mode='markers+lines',
        marker=dict(
            size=5, 
            color=colors, 
            colorscale='Turbo', 
            colorbar=dict(title="Time (min)")
        ),
        line=dict(color='gray', width=0.5),
        name='Patient State'
    ))
    
    # Highlight Current State
    fig.add_trace(go.Scatter(
        x=[coords[-1,0]], y=[coords[-1,1]],
        mode='markers+text',
        marker=dict(size=15, color='red', symbol='x'),
        text=["CURRENT"], textposition="top center",
        name='Current'
    ))

    fig.update_layout(
        height=400,
        title="<b>Hemodynamic State Space</b><br><i>(Distance from center = Instability)</i>",
        xaxis=dict(title="Principal Component 1", showgrid=True),
        yaxis=dict(title="Principal Component 2", showgrid=True),
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='white'
    )
    return fig

def plot_deviation_matrix(residuals, vars_list):
    """ Heatmap showing Directionality of deviation (Red=High, Blue=Low). """
    if len(residuals) < 2: return go.Figure()

    # Standardize residuals (Z-score)
    scaler = StandardScaler()
    res_norm = scaler.fit_transform(residuals)
    
    # Clip for readability (-4 to +4 sigma)
    res_norm = np.clip(res_norm, -4, 4)
    
    fig = go.Figure(data=go.Heatmap(
        z=res_norm.T,
        x=np.arange(len(res_norm)),
        y=vars_list,
        colorscale='RdBu_r', # Red = High, Blue = Low
        zmid=0,
        colorbar=dict(title="Std Dev<br>Deviation")
    ))
    
    fig.update_layout(
        height=300, 
        title="<b>Clinical Deviation Matrix</b><br><i>(Red = Unexpectedly High, Blue = Unexpectedly Low)</i>",
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='white'
    )
    return fig
