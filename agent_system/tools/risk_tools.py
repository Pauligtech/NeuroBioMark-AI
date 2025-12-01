from typing import Dict


def compute_risk(z_scores: dict, indices: dict) -> dict:
    """
    Compute a simple, research-oriented risk score based on:
    - hippocampal z-score
    - ventricle z-score
    - composite indices
    """

    score = 0.0

    # Hippocampal atrophy → negative hipp_z increases risk
    hipp_z = z_scores["hipp_z"]
    if hipp_z < 0:
        score += (-hipp_z) * 0.6   # weighted

    # Ventricular enlargement → positive ventricle_z increases risk
    vent_z = z_scores["ventricle_z"]
    if vent_z > 0:
        score += (vent_z) * 0.4

    # Composite index: Ventricle / Hippocampus ratio
    vh = indices["vh_ratio"]
    score += vh * 0.05

    # Composite index: Atrophy index (vent / icv)
    ai = indices["atrophy_index"]
    score += ai * 10.0   # scaled

    # Risk classification
    if score < 1.0:
        tier = "low"
    elif score < 2.0:
        tier = "moderate"
    else:
        tier = "high"

    return {"risk_score": float(score), "risk_tier": tier}

