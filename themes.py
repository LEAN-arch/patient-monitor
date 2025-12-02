# themes.py
"""
Centralized theme definitions and CSS strings for Streamlit UI.
"""
THEME_CSS = """
<style>
:root{
  --bg: #00111a;
  --card-bg: #071629;
  --muted: #9aa7b2;
  --text: #dfeff6;
  --accent: #00c2ff;
  --warn: #ffb020;
  --danger: #ff3b5c;
}
body { background-color: var(--bg); color: var(--text); }
.stApp .titan-card {
  background: var(--card-bg);
  border-radius: 8px;
  padding: 10px;
  border: 1px solid rgba(255,255,255,0.04);
  margin-bottom: 10px;
}
.kpi-lbl { font-size:0.75rem; color:var(--muted); font-weight:700; text-transform:uppercase; }
.kpi-val { font-size:1.25rem; font-weight:800; font-family: 'Roboto Mono', monospace; }
.alert-header { background: #051622; border-left: 6px solid var(--accent); padding: 12px; margin-bottom: 12px; display:flex; justify-content:space-between; align-items:center; gap:8px; }
.section-header { font-size:1.05rem; font-weight:800; color:var(--text); border-bottom:1px solid rgba(255,255,255,0.04); margin-top:18px; margin-bottom:10px; padding-bottom:6px; }
.small { color: #9fb3c4; font-size:0.85rem; }
</style>
"""
