import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from hmmlearn import hmm
import math

# --- 1. Simulation (Physiologically Realistic) ---
def simulate_patient(mins_total=720, seed=42):
    """
    Simulates a patient progressing from stable -> respiratory distress -> shock.
    """
    np.random.seed(seed)
    
    # 1. Define Baseline (Stable Patient)
    # HR, SBP, SpO2, RR, PI
    means = np.array([72., 120., 98., 14., 4.0])
    stds  = np.array([4.,  8.,   0.5, 1.5, 0.5])
    
    # Correlation Matrix (Physiological links)
    # HR/SBP neg correlated (baroreflex), HR/RR pos correlated
    corr = np.array([
        [ 1.0, -0.2, -0.1,  0.4, -0.1], # HR
        [-0.2,  1.0,  0.1, -0.1,  0.3], # SBP
        [-0.1,  0.1,  1.0, -0.2, -0.1], # SpO2
        [ 0.4, -0.1, -0.2,  1.0, -0.1], # RR
        [-0.1,  0.3, -0.1, -0.1,  1.0]  # PI
    ])
    cov = np.outer(stds, stds) * corr
    
    # 2. Generate VAR(1) Process (Baseline)
    phi = 0.6 # Autocorrelation strength
    data = np.zeros((mins_total, 5))
    data[0] = means + np.random.multivariate_normal(np.zeros(5), cov)
    
    noise_cov = cov * (1 - phi**2) # Scale noise to maintain variance
    
    for t in range(1, mins_total):
        noise = np.random.multivariate_normal(np.zeros(5), noise_cov)
        # Mean reversion formula
        data[t] = means + phi * (data[t-1] - means) + noise

    df = pd.DataFrame(data, columns=['HR','SBP','SpO2','RR','PI'])
    
    # 3. Inject Clinical Events
    
    # Event A: Minor Desaturation (Transient) at min 200
    # SpO2 drops, HR compensates slightly
    t_desat = 200
    df.loc[t_desat:t_desat+15, 'SpO2'] -= 6.0 
    df.loc[t_desat:t_desat+15, 'HR'] += 8.0
    
    # Event B: Progressive Shock (Decompensation) starting min 450
    # HR rises, SBP falls (narrowing pulse pressure), RR rises, PI falls
    t_shock = 450
    for t in range(t_shock, mins_total):
        progress = (t - t_shock) / (mins_total - t_shock) # 0 to 1
        intensity = progress * 1.2 # Scaling factor
        
        df.at[t, 'HR']   += 35.0 * intensity  # Tachycardia -> 110+
        df.at[t, 'SBP']  -= 30.0 * intensity  # Hypotension -> 90
        df.at[t, 'RR']   += 12.0 * intensity  # Tachypnea -> 26
        df.at[t, 'PI']   -= 2.5 * intensity   # Vasoconstriction -> <1.5
        df.at[t, 'SpO2'] -= 4.0 * intensity   # Mild hypoxia -> 94
        
        # Add jitter to the trend
        df.iloc[t] += np.random.normal(0, 0.5, 5)

    # 4. Post-processing and Cleanup
    # Clamp SpO2 <= 100
    df['SpO2'] = df['SpO2'].clip(upper=100.0)
    # Ensure SBP > 40
    df['SBP'] = df['SBP'].clip(lower=40.0)
    
    return df

# --- 2. Analytics (Robust) ---

def fit_var_and_residuals_full(df_full, baseline_window=180):
    """
    Fits VAR on the baseline period, then computes residuals for the ENTIRE dataset.
    This ensures we don't lose data when slicing views.
    """
    # Fit only on the "stable" beginning
    train_data = df_full.iloc[:baseline_window]
    model = VAR(train_data)
    try:
        results = model.fit(maxlags=1)
        lag = results.k_ar
        
        # Apply model to full dataset to get residuals
        # VAR results.resid only gives residuals for the training set.
        # We must manually compute residuals for the whole series using the coefficients.
        
        # Get coefficients (A) and intercept (c)
        # Equation: X_t = c + A * X_{t-1} + e_t
        c = results.params[0, :] # intercepts
        A = results.params[1:, :] # coefficient matrix
        
        data_val = df_full.values
        residuals = np.zeros_like(data_val)
        
        # Manual prediction for t=1 to end
        for t in range(lag, len(data_val)):
            pred = c + np.dot(data_val[t-1], A)
            residuals[t] = data_val[t] - pred
            
        # Covariance of residuals (from training set)
        cov_est = np.cov(residuals[lag:baseline_window].T)
        
        return residuals, cov_est, lag
        
    except Exception as e:
        print(f"VAR Fit failed: {e}")
        # Fallback: just subtract mean
        return (df_full - df_full.mean()).values, np.eye(df_full.shape[1]), 0

def compute_mahalanobis_risk(residuals, cov_est, lam=0.1):
    """
    Computes Mahalanobis distance on residuals, then applies EWMA smoothing.
    """
    # Inverse Covariance with Regularization (prevents singular matrix crashes)
    try:
        inv_cov = np.linalg.pinv(cov_est + 1e-6 * np.eye(cov_est.shape[0]))
    except:
        inv_cov = np.eye(cov_est.shape[0])
        
    # Calculate D^2 for every point
    # D^2 = r^T * S^-1 * r
    # Optimized calculation using einstein summation or diagonal dot
    # D2 shape: (N,)
    D2 = np.sum((residuals @ inv_cov) * residuals, axis=1)
    
    # Apply EWMA Smoothing to D2 to get "Risk Score"
    z = np.zeros_like(D2)
    z[0] = D2[0]
    for t in range(1, len(D2)):
        z[t] = lam * D2[t] + (1 - lam) * z[t-1]
        
    return z, inv_cov

# --- 3. Visualization ---

def plot_vitals(df_view, t_axis):
    fig = go.Figure()
    
    # HR and SBP
    fig.add_trace(go.Scatter(x=t_axis, y=df_view['HR'], name='HR (bpm)', 
                             line=dict(color='#ef553b', width=2)))
    fig.add_trace(go.Scatter(x=t_axis, y=df_view['SBP'], name='SBP (mmHg)', 
                             line=dict(color='#636efa', width=2)))
    
    # SpO2 on secondary axis
    fig.add_trace(go.Scatter(x=t_axis, y=df_view['SpO2'], name='SpO2 (%)', 
                             line=dict(color='#00cc96', width=2, dash='dot'), yaxis='y2'))
    
    fig.update_layout(
        height=400,
        xaxis_title="Time (minutes)",
        yaxis=dict(title="Hemodynamics", range=[40, 160]),
        yaxis2=dict(title="SpO2", overlaying='y', side='right', range=[85, 101], showgrid=False),
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified"
    )
    return fig

def plot_risk(risk_scores, t_axis, thresh=15.0):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_axis, y=risk_scores, fill='tozeroy', 
                             name='Risk Index', line=dict(color='purple')))
    
    # Add threshold line
    fig.add_hline(y=thresh, line_dash="dash", annotation_text="Warning Threshold")
    
    fig.update_layout(
        height=250,
        title="Multivariate Anomaly Score (Integrated Risk)",
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis_title="Mahalanobis Distance (Smoothed)"
    )
    return fig

def plot_heatmap(residuals, vars_list):
    # Normalize residuals for heatmap (Z-score)
    scaler = StandardScaler()
    res_norm = scaler.fit_transform(residuals)
    
    # Cap values for better visualization contrast
    res_norm = np.clip(res_norm, -3, 3)
    
    fig = go.Figure(data=go.Heatmap(
        z=res_norm.T,
        x=np.arange(len(res_norm)),
        y=vars_list,
        colorscale='RdBu',
        midpoint=0
    ))
    fig.update_layout(height=300, title="Residual Heatmap (Standardized)", margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_pca(df_full):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_full)
    
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    
    fig = px.scatter(x=components[:,0], y=components[:,1], 
                     color=np.arange(len(components)),
                     color_continuous_scale='Turbo',
                     labels={'color': 'Time'},
                     title="PCA Trajectory (Patient State Space)")
    fig.update_layout(height=350)
    return fig
