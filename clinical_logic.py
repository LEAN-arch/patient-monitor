# clinical_logic.py
"""
Clinical decision logic / prescriptive engine.
Translates alert list and metrics into prioritized actions (structured).
"""
from typing import Dict, Any, List
import utils

def build_action_plan(df: 'pd.DataFrame', alerts: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Return an ordered list of recommended actions with rationale and priority.
    """
    plan: List[Dict[str, str]] = []
    if not alerts.get("alerts"):
        plan.append({"priority": "low", "action": "Continue monitoring", "rationale": "No active alerts."})
        return plan

    # critical first
    for a in alerts["alerts"]:
        t = a["type"]
        sev = a["severity"]
        if t == "SUSTAINED_HYPOTENSION" and sev == "critical":
            plan.append({"priority": "critical", "action": "Start or titrate vasopressor (norepinephrine)", "rationale": a["message"]})
        elif t == "SUSTAINED_HYPOTENSION":
            plan.append({"priority": "high", "action": "Assess fluid responsiveness (PLR/echo); consider 250-500 mL crystalloid", "rationale": a["message"]})
        elif t == "LOW_CI":
            plan.append({"priority": "high", "action": "Consider inotrope (dobutamine) or optimize preload if responsive", "rationale": a["message"]})
        elif t == "LACTATE_RISING":
            plan.append({"priority": "high", "action": "Investigate perfusion/oxygen delivery and source control; recheck labs", "rationale": a["message"]})
        elif t == "ELEVATED_SHOCK_INDEX":
            plan.append({"priority": "medium", "action": "Rapid bedside assessment for bleeding/sepsis/arrhythmia", "rationale": a["message"]})
        else:
            plan.append({"priority": "low", "action": a.get("suggested_action", "Clinical review"), "rationale": a.get("message", "")})

    # deduplicate similar actions (simple pass)
    seen = set()
    dedup = []
    for p in plan:
        key = (p["action"], p["priority"])
        if key not in seen:
            seen.add(key)
            dedup.append(p)
    return dedup
