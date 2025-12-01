"""
adk_tools.py
------------

Plain Python wrapper functions that expose your existing logic
as "tools" for the ADK / Agents framework.

In your ADK config / Tool declarations, you'll point at these
functions or mirror these signatures.
"""

from typing import Dict, Any, List

from agent_system.tools.data_tools import (
    load_biomarker_table,
    get_subject_timepoint,
    get_subject_history,
)
from agent_system.tools.biomarker_tools import (
    compute_z_scores,
    compute_composite_indices,
)
from agent_system.tools.risk_tools import compute_risk


def tool_load_subject_row(subject_id: str, timepoint: str) -> Dict[str, Any]:
    """
    Load a single subject/timepoint row as a dict.
    Intended as a data-access tool for the Biomarker Agent.
    """
    df = load_biomarker_table()
    row = get_subject_timepoint(df, subject_id, timepoint)
    return row


def tool_load_subject_history(subject_id: str) -> List[Dict[str, Any]]:
    """
    Load longitudinal history for a subject.
    Intended as a memory/longitudinal tool.
    """
    df = load_biomarker_table()
    return get_subject_history(df, subject_id)


def tool_compute_biomarkers(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute z-scores + composite indices for a single row.
    This is the core "biomarker analysis" tool.
    """
    z_scores = compute_z_scores(row)
    indices = compute_composite_indices(row)
    return {"z_scores": z_scores, "indices": indices}


def tool_compute_risk(biomarker_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute non-diagnostic risk pattern from z-scores + indices.
    """
    z = biomarker_info["z_scores"]
    idx = biomarker_info["indices"]
    return compute_risk(z, idx)


def tool_summarize_longitudinal(subject_id: str) -> Dict[str, Any]:
    """
    Very simple longitudinal summary tool.
    Can be expanded later, but for now just returns the history dict.
    """
    history = tool_load_subject_history(subject_id)
    return {"subject_id": subject_id, "history": history}
