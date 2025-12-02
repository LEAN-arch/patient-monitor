import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import welch

# --- 1. THEME: GLASSMORPHIC CLINICAL ---
THEME = {
    'bg': '#f8fafc',
    'text': '#0f172a',
    'grid': '#e2e8f0',
    # Clinical Signals
    'hr': '#0ea5e9',       # Cyan Blue
    'map': '#d946ef',      # Fuschia
    'ci': '#10b981',       # Emerald
    'svr': '#f59e0b',      # Amber
    'do2': '#6366f1',      # Indigo
    # Predictive
    'pred_natural': '#94a3b8', # Grey (Natural course)
    'pred_interv': '#10b981',  # Green (With Intervention)
    # Zones
    'zone_safe': 'rgba(16, 185, 129, 0.08)',
    'zone_warn': 'rgba(245, 158, 11, 0.08)',
    'zone_crit': 'rgba(239, 68, 68, 0.08)'
}

# --- 2. ADVANCED PHYSICS ENGINE (DIGITAL TWIN) ---
class DigitalTwin:
    """
    A 0-D Cardiovascular System Model.
    Simulates the interaction between:
    - Preload (Volume)
    - Contractility (Pump)
    - Afterload (Resistance)
    - Compliance (Stiffness)
    """
    def __init__(self):
        self.preload = 12.0  # CVP proxy (mmHg)
        self.contractility = 1.0 # Emax multiplier
        self.afterload = 1.0 # SVR multiplier
        self.compliance = 1.0 # Arterial compliance
        
    def step(self, sepsis_severity=0.0):
        """
        Calculates one hemodynamic state based on current parameters.
        sepsis_severity: 0.0 (Healthy) to 1.0 (Refractory Shock)
        """
        # Pathology: Sepsis causes Vasodilation (Low SVR) and Capillary Leak (Low Preload)
        effective_afterload = self.afterload * (1.0 - (sepsis_severity * 0.6))
        effective_preload = self.preload * (1.0 - (sepsis_severity * 0.3))
        
        # 1. Frank-Starling Mechanism (Sigmoid curve)
        # Stroke Volume depends on Preload and Contractility
        sv_max = 100 * self.contractility
        sv = sv_max * (effective_preload**2 / (effective_preload**2 + 8**2))
        
        # 2. Baroreflex (Compensatory Tachycardia)
        # Target MAP is 90. If estimated MAP drops, HR rises.
        est_map = (sv * 70 * effective_afterload * 0.05) + 5
        hr_drive = np.clip((90 - est_map) * 1.5, 0, 80)
        hr = 70 + hr_drive
        
        # 3. Final Hemodynamics
        co = (hr * sv) / 1000 # L/min
        # MAP = CO * SVR
        svr_dyne = 1200 * effective_afterload
        map_val = (co * svr_dyne / 80) + 5
        
        # 4. Metabolic (Lactate)
        # Supply/Demand Mismatch
        do2 = co * 1.34 * 12 * 0.98 * 10
        lactate_gen = max(0, (400 - do2) * 0.005)
        
        return {
            'HR': hr, 'SV': sv, 'CO': co, 'MAP': map_val, 
            'SVR': svr_dyne, 'DO2': do2, 'Lactate_Gen': lactate_gen
        }

def simulate_predictive_scenario(mins=720):
    """
    Generates the patient history AND the "What-If" future branches.
    """
    twin = DigitalTwin()
    history = []
    
    # --- PHASE 1: GENERATE HISTORY (0 to Current) ---
    current_lactate = 1.0
    sepsis_curve = np.linspace(0, 0.8, mins) # Progressive Sepsis
    
    for t in range(mins):
        # Add noise
        noise = np.random.normal(0, 0.02)
        
        # Simulate Step
        state = twin.step(sepsis_severity=sepsis_curve[t] + noise)
        
        # Integrate Lactate
        current_lactate = (current_lactate * 0.995) + state['Lactate_Gen'] # Clearance vs Generation
        state['Lactate'] = max(0.8, current_lactate)
        state['Time'] = t
        state['Preload_Status'] = twin.preload * (1 - sepsis_curve[t]*0.3)
        history.append(state)
        
    df_hist = pd.DataFrame(history)
    
    # Derived Metrics
    df_hist['CI'] = df_hist['CO'] / 1.8
    df_hist['SVRI'] = df_hist['SVR'] * 1.8
    df_hist['PP'] = df_hist['SV'] / 1.5 # Pulse Pressure proxy
    
    # --- PHASE 2: GENERATE PREDICTIONS (Current to +30 mins) ---
    # We branch the simulation into 3 futures:
    # 1. Natural Course (Do nothing)
    # 2. Fluid Bolus (+500mL -> Preload Boost)
    # 3. Vasopressor (+0.1mcg -> Afterload Boost)
    
    last_severity = sepsis_curve[-1]
    
    pred_natural = []
    pred_fluid = []
    pred_pressor = []
    
    for i in range(30): # 30 mins into future
        severity = last_severity + (0.001 * i) # Sepsis continues
        
        # Branch 1: Natural
        s_nat = twin.step(sepsis_severity=severity)
        pred_natural.append(s_nat['MAP'])
        
        # Branch 2: Fluids (Boost Preload by 20%)
        twin_fluid = DigitalTwin()
        twin_fluid.preload *= 1.3 # 30% boost
        s_fluid = twin_fluid.step(sepsis_severity=severity)
        pred_fluid.append(s_fluid['MAP'])
        
        # Branch 3: Pressors (Boost Afterload by 40%)
        twin_pressor = DigitalTwin()
        twin_pressor.afterload *= 1.4
        twin_pressor.preload *= 1.1
        s_press = twin_pressor.step(sepsis_severity=severity)
        pred_pressor.append(s_press['MAP'])

    predictions = {
        'time': np.arange(mins, mins+30),
        'natural': pred_natural,
        'fluid': pred_fluid,
        'pressor': pred_pressor
    }
    
    return df_hist, predictions

# --- 3. NEXT-GEN VISUALIZATIONS ---

def plot_predictive_bullseye(df, predictions, curr_time):
    """
    The 'Money Plot'. Shows current state AND vectors for interventions.
    """
    data = df.iloc[max(0, curr_time-60):curr_time]
    cur = data.iloc[-1]
    
    fig = go.Figure()
    
    # 1. Background Diagnostic Zones
    fig.add_shape(type="rect", x0=0, y0=0, x1=2.5, y1=65, fillcolor='rgba(239,68,68,0.1)', layer="below", line_width=0) # Crit
    fig.add_shape(type="rect", x0=2.5, y0=65, x1=6.0, y1=100, fillcolor='rgba(16,185,129,0.1)', layer="below", line_width=0) # Goal
    
    # 2. History Trend
    fig.add_trace(go.Scatter(x=data['CI'], y=data['MAP'], mode='lines', 
                             line=dict(color='gray', width=1, dash='dot'), name='History'))
    
    # 3. Current State
    fig.add_trace(go.Scatter(x=[cur['CI']], y=[cur['MAP']], mode='markers',
                             marker=dict(color='black', size=15, symbol='circle-x'), name='Current'))
    
    # 4. PREDICTIVE VECTORS
    
    # Fluid Vector (Increases CI mostly, MAP slightly)
    vec_fluid_x = cur['CI'] * 1.25
    vec_fluid_y = predictions['fluid'][-1]
    
    fig.add_annotation(
        x=vec_fluid_x, y=vec_fluid_y, ax=cur['CI'], ay=cur['MAP'],
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#3b82f6",
        text="FLUIDS", font=dict(color="#3b82f6", weight="bold")
    )
    
    # Pressor Vector (Increases MAP mostly, CI neutral/down)
    vec_press_x = cur['CI'] * 1.05
    vec_press_y = predictions['pressor'][-1]
    
    fig.add_annotation(
        x=vec_press_x, y=vec_press_y, ax=cur['CI'], ay=cur['MAP'],
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#d946ef",
        text="PRESSORS", font=dict(color="#d946ef", weight="bold")
    )

    fig.update_layout(template="plotly_white", height=350, margin=dict(l=10,r=10,t=40,b=10),
                      title="<b>Predictive Hemodynamic Compass</b>",
                      xaxis=dict(title="Cardiac Index (Flow)", range=[1.5, 6.0], gridcolor=THEME['grid']),
                      yaxis=dict(title="MAP (Pressure)", range=[40, 100], gridcolor=THEME['grid']), showlegend=False)
    return fig

def plot_organ_radar(df, curr_time):
    """
    Multivariate Risk Polygon.
    """
    cur = df.iloc[-1]
    
    # Normalize Risks (0 = Safe, 1 = Critical)
    risk_renal = np.clip((65 - cur['MAP'])/20, 0, 1) # Low MAP = AKI
    risk_cardiac = np.clip((cur['HR'] - 100)/50, 0, 1) # Tachy = Ischemia
    risk_meta = np.clip((cur['Lactate'] - 1.5)/4, 0, 1) # Lactate = Cell Death
    risk_perf = np.clip((2.2 - cur['CI'])/1.0, 0, 1) # Low CI = Shock
    
    r = [risk_renal, risk_cardiac, risk_meta, risk_perf, risk_renal]
    theta = ['Renal (AKI)', 'Cardiac (Stress)', 'Metabolic (Debt)', 'Perfusion (Shock)', 'Renal (AKI)']
    
    fig = go.Figure()
    
    # Safe Zone
    fig.add_trace(go.Scatterpolar(r=[0.2]*5, theta=theta, fill='toself', 
                                  fillcolor='rgba(16,185,129,0.2)', line=dict(width=0), hoverinfo='skip'))
    
    # Patient State
    fig.add_trace(go.Scatterpolar(r=r, theta=theta, fill='toself', 
                                  fillcolor='rgba(239,68,68,0.2)', line=dict(color='#ef4444', width=2)))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
        showlegend=False, height=250, margin=dict(l=30,r=30,t=20,b=20),
        title="<b>Organ Risk Topology</b>"
    )
    return fig

def plot_horizon_multiverse(df, predictions, curr_time):
    """
    Visualizes the "Multiverse" of outcomes based on intervention.
    """
    hist_data = df.iloc[max(0, curr_time-60):curr_time]
    
    fig = go.Figure()
    
    # History
    fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['MAP'], 
                             line=dict(color='black', width=2), name='History'))
    
    # Future 1: Natural (Crash)
    fig.add_trace(go.Scatter(x=predictions['time'], y=predictions['natural'],
                             line=dict(color='gray', dash='dot'), name='No Intervention'))
    
    # Future 2: Fluid (Blue)
    fig.add_trace(go.Scatter(x=predictions['time'], y=predictions['fluid'],
                             line=dict(color='#3b82f6', width=2), name='+500mL Bolus'))
    
    # Future 3: Pressor (Purple)
    fig.add_trace(go.Scatter(x=predictions['time'], y=predictions['pressor'],
                             line=dict(color='#d946ef', width=2), name='+Norepinephrine'))
    
    # Threshold
    fig.add_hline(y=65, line_color='red', line_dash='solid', opacity=0.3, annotation_text="CRITICAL MAP")

    fig.update_layout(template="plotly_white", height=250, margin=dict(l=10,r=10,t=30,b=10),
                      title="<b>Intervention Simulation (Next 30 min)</b>",
                      xaxis=dict(title="Timeline", gridcolor=THEME['grid']),
                      yaxis=dict(title="Projected MAP", gridcolor=THEME['grid']),
                      legend=dict(orientation="h", y=1.1))
    return fig

def plot_starling_dynamic(df, curr_time):
    """
    Frank-Starling with Dynamic Elastance (Eadyn) overlay.
    """
    data = df.iloc[max(0, curr_time-60):curr_time]
    
    fig = go.Figure()
    
    # Curve
    x = np.linspace(0, 20, 100)
    y = np.log(x+1)*30
    fig.add_trace(go.Scatter(x=x, y=y, line=dict(color='lightgray'), name='Ideal'))
    
    # Patient Trajectory
    # FIX: Removed invalid 'arrow' property. Used markers to show direction.
    fig.add_trace(go.Scatter(x=data['Preload_Status'], y=data['SV'], mode='lines',
                             line=dict(color=THEME['ci'], width=3), name='Trajectory'))
    
    # Current Head
    fig.add_trace(go.Scatter(x=[data['Preload_Status'].iloc[-1]], y=[data['SV'].iloc[-1]], 
                             mode='markers', marker=dict(color=THEME['ci'], size=10, symbol='diamond'), name='Current'))
    
    # Calculate Slope (Fluid Responsiveness)
    d_sv = data['SV'].iloc[-1] - data['SV'].iloc[0]
    d_pre = data['Preload_Status'].iloc[-1] - data['Preload_Status'].iloc[0] + 0.01
    slope = d_sv / d_pre
    
    txt = "FLUID RESPONDER" if slope > 2.0 else "NON-RESPONDER"
    col = "green" if slope > 2.0 else "red"
    
    fig.add_annotation(x=10, y=10, text=f"{txt}<br>(Slope: {slope:.1f})", 
                       font=dict(color=col, weight="bold"), showarrow=False)

    fig.update_layout(template="plotly_white", height=250, margin=dict(l=10,r=10,t=30,b=10),
                      title="<b>Frank-Starling Mechanics</b>",
                      xaxis=dict(title="Preload (CVP proxy)", gridcolor=THEME['grid']),
                      yaxis=dict(title="Stroke Volume", gridcolor=THEME['grid']), showlegend=False)
    return fig
