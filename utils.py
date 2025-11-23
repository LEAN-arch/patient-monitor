import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from hmmlearn import hmm  # <--- Switched library
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import math

# --- 1. Simulation ---
def simulate_patient(mins_total=720, seed=0):
    np.random.seed(seed)
    baseline_mean = {'HR':75., 'SBP':120., 'SpO2':98., 'RR':16., 'PI':5.}
    vars_ = ['HR','SBP','SpO2','RR','PI']
    corr = np.array([[1.0, -0.3, -0.2,  0.4,  0.1],
                     [-0.3, 1.0,  0.1, -0.2,  0.4],
                     [-0.2, 0.1,  1.0, -0.1, -0.1],
                     [0.4, -0.2, -0.1,  1.0,  0.0],
                     [0.1,  0.4, -0.1,  0.0,  1.0]])
    stds = np.array([5.,8.,0.5,1.5,0.8]); cov = np.outer(stds, stds)*corr
    phi = 0.55
    X = np.zeros((mins_total, len(vars_)))
    X[0] = np.array([baseline_mean[v] for v in vars_]) + np.random.multivariate_normal(np.zeros(len(vars_)), cov)
    for t in range(1, mins_total):
        noise = np.random.multivariate_normal(np.zeros(len(vars_)), cov)
        X[t] = phi*(X[t-1] - np.array([baseline_mean[v] for v in vars_])) + np.array([baseline_mean[v] for v in vars_]) + noise
    df = pd.DataFrame(X, columns=vars_)
    
    # Add Events
    t_abrupt = 300; t_grad = 480
    df.loc[t_abrupt:t_abrupt+5, 'SpO2'] -= 7.0
    df.loc[t_abrupt:t_abrupt+12, 'HR'] += 12.0
    for t in range(t_grad, mins_total):
        drift = (t - t_grad)
        df.loc[t:, 'HR'] += 0.07 * drift
        df.loc[t:, 'SBP'] -= 0.055 * drift
        df.loc[t:, 'SpO2'] -= 0.018 * drift
        df.loc[t:, 'PI'] -= 0.02 * drift
        df.loc[t:, 'RR'] += 0.02 * drift
        
    # Derived metrics
    df['DBP'] = (df['SBP'] - 40) + np.random.normal(0,2.0,mins_total)
    df['PP'] = df['SBP'] - df['DBP']
    df['SI'] = df['HR'] / df['SBP']
    
    # SQI and HRV surrogates
    df['SQI'] = 1.0
    drops = np.random.choice(np.arange(60, mins_total-60), size=3, replace=False)
    for d in drops:
        w = np.random.randint(5,25)
        df.loc[d:d+w, 'SQI'] = np.random.uniform(0.0, 0.5)
        
    df['SDNN'] = df['HR'].rolling(window=5, min_periods=2).std().fillna(method='bfill')
    df['RMSSD'] = df['HR'].diff().rolling(window=5, min_periods=2).apply(lambda x: np.sqrt(np.mean(x**2)) if len(x)>0 else 0).fillna(method='bfill')
    
    return df

# --- 2. Analytics ---
def fit_var_and_residuals(df_vars, train_end=180, maxlags=1):
    model = VAR(df_vars.iloc[:train_end])
    res = model.fit(maxlags=maxlags)
    lag = res.k_ar
    resid = res.resid
    residuals_full = np.zeros((len(df_vars), df_vars.shape[1]))
    residuals_full[lag:lag+len(resid), :] = resid
    cov_est = np.cov(residuals_full[lag:train_end].T)
    return residuals_full, cov_est

def mahalanobis_ewma(residuals_full, cov_est, lam=0.05):
    cov_inv = np.linalg.pinv(cov_est + 1e-8*np.eye(cov_est.shape[0]))
    D2 = np.array([residuals_full[t].T.dot(cov_inv).dot(residuals_full[t]) for t in range(len(residuals_full))])
    zD = np.zeros_like(D2); zD[0] = D2[0]
    for t in range(1, len(D2)):
        zD[t] = lam * D2[t] + (1-lam) * zD[t-1]
    return D2, zD, cov_inv

def mewma_T2(df_vars, baseline_end=180, lam_v=0.05):
    p = df_vars.shape[1]
    baseline_mean_vec = np.mean(df_vars.iloc[:baseline_end].values, axis=0)
    lam = lam_v
    z_vec = np.zeros((len(df_vars), p)); z_prev = np.zeros(p)
    for t in range(len(df_vars)):
        x_t = df_vars.iloc[t].values - baseline_mean_vec
        z_curr = lam * x_t + (1-lam) * z_prev
        z_vec[t,:] = z_curr; z_prev = z_curr
    Sigma_x = np.cov(df_vars.iloc[:baseline_end].T)
    Sigma_z = (lam**2 / (2 - lam)) * Sigma_x + 1e-8*np.eye(p)
    Sigma_z_inv = np.linalg.pinv(Sigma_z)
    T2 = np.array([z_vec[t].T.dot(Sigma_z_inv).dot(z_vec[t]) for t in range(len(df_vars))])
    return T2

def fit_hmm_multivariate(df_vars, n_states=3):
    """
    Updated to use hmmlearn instead of pomegranate.
    """
    data = df_vars.fillna(method='bfill').values
    # GaussianHMM with diagonal covariance is standard for this type of data
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100, random_state=42)
    model.fit(data)
    states = model.predict(data)
    return states

# --- 3. Visualization ---
def plot_pca_trajectory(df_vars):
    X = df_vars.fillna(method='bfill').values
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pc = pca.fit_transform(Xs)
    fig = px.scatter(x=pc[:,0], y=pc[:,1], color=np.arange(len(pc)), 
                     color_continuous_scale='Viridis', labels={'color':'Time (min)'}, 
                     title="PCA Trajectory")
    return fig

def plot_residuals_heatmap(residuals, varnames):
    base_end = max(10, min(180, residuals.shape[0]//3))
    mu_r = np.mean(residuals[:base_end,:], axis=0)
    sd_r = np.std(residuals[:base_end,:], axis=0, ddof=1) + 1e-9
    zres = (residuals - mu_r) / sd_r
    fig = go.Figure(data=go.Heatmap(z=zres.T, x=np.arange(residuals.shape[0]), 
                                    y=varnames, colorscale='RdBu', zmin=-3, zmax=3))
    fig.update_layout(title="Residuals Heatmap", xaxis_title="Time (min)")
    return fig
