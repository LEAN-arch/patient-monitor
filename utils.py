import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- 1. THEME: TITAN DARK ---
THEME = {
    'bg': '#000000',
    'card': '#111111',
    'grid': '#333333',
    'text': '#e0e0e0',
    # Signals
    'hr': '#00e5ff',       # Cyan (Electrical)
    'map': '#ff2975',      # Neon Red (Pressure)
    'ci': '#00ff33',       # Neon Green (Flow)
    'svr': '#ff9900',      # Neon Orange (Resist)
    'do2': '#8c1eff',      # Neon Purple (Metabolic)
    'pca': '#ffffff',      # White
    # Zones
    'zone_ok': 'rgba(0, 255, 51, 0.1)',
    'zone_crit': 'rgba(255, 41, 117, 0.1)'
}

# --- 2. HELPER ---
def hex_to_rgba(hex_color, opacity=0.1):
    hex_color = hex_color.lstrip('#')
    return f"rgba({int(hex_color[0:2], 16)},{int(hex_color[2:4], 16)},{int(hex_color[4:6], 16)},{opacity})"

# --- 3. PHYSICS ENGINE (DIGITAL TWIN) ---
class DigitalTwin:
    def __init__(self):
        self.preload = 12.0
        self.contractility = 1.0
        self.afterload = 1.0
        
    def step(self, sepsis_severity=0.0):
        # Pathology: Vasodilation + Capillary Leak
        eff_afterload = self.afterload * (1.0 - (sepsis_severity * 0.7))
        eff_preload = self.preload * (1.0 - (sepsis_severity * 0.4))
        
        # Starling Curve
        sv_max = 100 * self.contractility
        sv = sv_max * (eff_preload**2 / (eff_preload**2 + 8**2))
        
        # Baroreflex
        est_map = (sv * 75 * eff_afterload * 0.05) + 5
        hr_drive = np.clip((90 - est_map) * 1.8, 0, 90)
        hr = 70 + hr_drive
        
        # Output
        co = (hr * sv) / 1000
        svr_dyne = 1200 * eff_afterload
        map_val = (co * svr_dyne / 80) + 5
        
        # Oxygen
        do2 = co * 1.34 * 12 * 0.98 * 10
        lac_gen = max(0, (400 - do2) * 0.01)
        
        return {'HR': hr, 'SV': sv, 'CO': co, 'MAP': map_val, 'SVR': svr_dyne, 'DO2': do2, 'Lac_Gen': lac_gen}

def simulate_titan_data(mins=720):
    twin = DigitalTwin()
    history = []
    curr_lac = 1.0
    
    # Generate Sepsis Curve
    sepsis = np.linspace(0, 0.85, mins)
    noise_gen = np.random.normal(0, 0.02, mins)
    
    for t in range(mins):
        s = twin.step(sepsis[t] + noise_gen[t])
        curr_lac = (curr_lac * 0.99) + s['Lac_Gen']
        s['Lactate'] = max(0.8, curr_lac)
        s['Preload_Status'] = twin.preload * (1 - sepsis[t]*0.4)
        history.append(s)
        
    df = pd.DataFrame(history)
    
    # Derived Metrics
    df['CI'] = df['CO'] / 1.8
    df['SVRI'] = df['SVR'] * 1.8
    df['PP'] = df['SV'] / 1.5
    df['SI'] = df['HR'] / df['MAP']
    df['Entropy'] = df['HR'].rolling(60).apply(lambda x: np.std(np.diff(x))).fillna(1.0)
    df['Urine'] = np.where(df['MAP'] > 65, 1.0 + np.random.normal(0,0.1,mins), np.maximum(0, 1.0 - (65-df['MAP'])*0.05))
    
    # PCA
    scaler = StandardScaler()
    pca_data = df[['HR', 'MAP', 'CI', 'SVR']].fillna(0)
    pca_coords = PCA(n_components=2).fit_transform(scaler.fit_transform(pca_data))
    df['PC1'] = pca_coords[:,0]
    df['PC2'] = pca_coords[:,1]
    
    # Predictions
    last_sev = sepsis[-1]
    pred_nat, pred_fluid, pred_press = [], [], []
    for i in range(30):
        sev = last_sev + 0.001*i
        pred_nat.append(twin.step(sev)['MAP'])
        t_f = DigitalTwin(); t_f.preload *= 1.3
        pred_fluid.append(t_f.step(sev)['MAP'])
        t_p = DigitalTwin(); t_p.afterload *= 1.4; t_p.preload *= 1.1
        pred_press.append(t_p.step(sev)['MAP'])
        
    preds = {'time': np.arange(mins, mins+30), 'nat': pred_nat, 'fluid': pred_fluid, 'press': pred_press}
    return df, preds

# --- 4. VISUALIZATIONS ---

def plot_spark_spc(df, col, color, thresh_low, thresh_high):
    data = df[col].iloc[-60:]
    fig = go.Figure()
    fig.add_shape(type="rect", x0=data.index[0], x1=data.index[-1], y0=thresh_low, y1=thresh_high, 
                  fillcolor="rgba(255,255,255,0.05)", line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines', 
                             line=dict(color=color, width=2), fill='tozeroy', fillcolor=hex_to_rgba(color, 0.2)))
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      height=50, margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

def plot_predictive_compass(df, preds, curr_time):
    data = df.iloc[max(0, curr_time-60):curr_time]
    cur = data.iloc[-1]
    fig = go.Figure()
    
    # Zones
    fig.add_shape(type="rect", x0=0, y0=0, x1=2.5, y1=65, fillcolor=THEME['zone_crit'], line_width=0, layer="below")
    fig.add_shape(type="rect", x0=2.5, y0=65, x1=6.0, y1=100, fillcolor=THEME['zone_ok'], line_width=0, layer="below")
    
    # Data
    fig.add_trace(go.Scatter(x=data['CI'], y=data['MAP'], mode='lines', line=dict(color='#555', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=[cur['CI']], y=[cur['MAP']], mode='markers', marker=dict(color='white', size=12, symbol='cross')))
    
    # Vectors
    fig.add_annotation(x=cur['CI']*1.25, y=preds['fluid'][-1], ax=cur['CI'], ay=cur['MAP'], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor=THEME['ci'], text="FLUID", font=dict(color=THEME['ci']))
    fig.add_annotation(x=cur['CI']*1.05, y=preds['press'][-1], ax=cur['CI'], ay=cur['MAP'], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor=THEME['do2'], text="PRESSOR", font=dict(color=THEME['do2']))

    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      height=300, margin=dict(l=10,r=10,t=40,b=10), title="<b>Predictive Compass</b>",
                      xaxis=dict(title="CI", range=[1.5, 6], gridcolor=THEME['grid']), yaxis=dict(title="MAP", range=[40, 100], gridcolor=THEME['grid']), showlegend=False)
    return fig

def plot_multiverse(df, preds, curr_time):
    hist = df.iloc[max(0, curr_time-60):curr_time]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MAP'], line=dict(color='white', width=2)))
    fig.add_trace(go.Scatter(x=preds['time'], y=preds['nat'], line=dict(color='#555', dash='dot')))
    fig.add_trace(go.Scatter(x=preds['time'], y=preds['fluid'], line=dict(color=THEME['ci'])))
    fig.add_trace(go.Scatter(x=preds['time'], y=preds['press'], line=dict(color=THEME['do2'])))
    fig.add_hline(y=65, line_color='red', line_dash='dot')
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      height=300, margin=dict(l=10,r=10,t=30,b=10), title="<b>Intervention Horizon</b>",
                      xaxis=dict(gridcolor=THEME['grid']), yaxis=dict(title="MAP", gridcolor=THEME['grid']), showlegend=False)
    return fig

def plot_organ_radar(df, curr_time):
    cur = df.iloc[-1]
    # Normalize Risks
    risk_renal = np.clip((65 - cur['MAP'])/20, 0, 1)
    risk_cardiac = np.clip((cur['HR'] - 100)/50, 0, 1)
    risk_meta = np.clip((cur['Lactate'] - 1.5)/4, 0, 1)
    risk_perf = np.clip((2.2 - cur['CI'])/1.0, 0, 1)
    
    r = [risk_renal, risk_cardiac, risk_meta, risk_perf, risk_renal]
    theta = ['Renal', 'Cardiac', 'Metabolic', 'Perfusion', 'Renal']
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[0.2]*5, theta=theta, fill='toself', fillcolor=THEME['zone_ok'], line_width=0))
    fig.add_trace(go.Scatterpolar(r=r, theta=theta, fill='toself', fillcolor=THEME['zone_crit'], line=dict(color='red', width=2)))
    
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(radialaxis=dict(visible=False, range=[0, 1]), bgcolor='rgba(0,0,0,0)'),
        showlegend=False, height=250, margin=dict(l=30,r=30,t=30,b=20),
        title="<b>Organ Risk Topology</b>"
    )
    return fig

def plot_starling_vector(df, curr_time):
    data = df.iloc[max(0, curr_time-30):curr_time]
    fig = go.Figure()
    x=np.linspace(0,20,50); y=np.log(x+1)*30
    fig.add_trace(go.Scatter(x=x, y=y, line=dict(color='#333', dash='dot')))
    fig.add_trace(go.Scatter(x=data['Preload_Status'], y=data['SV'], mode='lines', line=dict(color=THEME['ci'])))
    fig.add_trace(go.Scatter(x=[data['Preload_Status'].iloc[-1]], y=[data['SV'].iloc[-1]], mode='markers', marker=dict(color='white', size=8, symbol='triangle-up')))
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      height=250, margin=dict(l=10,r=10,t=30,b=10), title="<b>Starling Vector</b>",
                      xaxis=dict(title="Preload", gridcolor=THEME['grid']), yaxis=dict(title="SV", gridcolor=THEME['grid']), showlegend=False)
    return fig

def plot_oxygen_debt(df, curr_time):
    data = df.iloc[max(0, curr_time-120):curr_time]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=data.index, y=data['DO2'], fill='tozeroy', fillcolor=hex_to_rgba(THEME['do2'],0.1), line=dict(color=THEME['do2']), name='DO2'), secondary_y=False)
    fig.add_trace(go.Scatter(x=data.index, y=data['Lactate'], line=dict(color='red', width=2), name='Lac'), secondary_y=True)
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      height=250, margin=dict(l=10,r=10,t=30,b=10), title="<b>O2 Supply/Demand</b>", showlegend=False)
    return fig

def plot_renal_cliff(df, curr_time):
    data = df.iloc[max(0, curr_time-120):curr_time]
    fig = go.Figure()
    fig.add_shape(type="rect", x0=30, y0=0, x1=65, y1=0.5, fillcolor=THEME['zone_crit'], line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=data['MAP'], y=data['Urine'], mode='lines+markers', marker=dict(color=np.linspace(0,1,len(data)), colorscale='Reds'), line=dict(color='#555')))
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      height=250, margin=dict(l=10,r=10,t=30,b=10), title="<b>Renal Autoregulation</b>",
                      xaxis=dict(title="MAP", gridcolor=THEME['grid']), yaxis=dict(title="Urine", gridcolor=THEME['grid']), showlegend=False)
    return fig
