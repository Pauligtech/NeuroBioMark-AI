"""
adk_agents.py
-------------

Agent role + tool configuration for the ADK / Agents framework.

You will map these into actual ADK Agent objects using the
course's templates (e.g., defining system prompts + attached tools).
"""

from typing import Dict, List


def get_agent_configs() -> Dict[str, Dict]:
    """
    Return a dictionary of agent configs.

    Keys are agent names; values describe:
      - role: high-level natural language description
      - tools: which Python tools this agent is allowed to call
      - output: what this agent is expected to return
    """
    return {
        "intake_agent": {
            "role": (
                "You are the Intake Agent. You take a user request "
                "(subject_id, timepoint, and analysis preference) "
                "and turn it into a structured job."
            ),
            "tools": [],  # usually no tools – just parsing / validation
            "expected_output": {
                "subject_id": "string",
                "timepoint": "string",
                "analysis_type": "single | longitudinal",
            },
        },
        "biomarker_agent": {
            "role": (
                "You are the Biomarker Agent. Given a subject_id and timepoint, "
                "you call tools to load the MRI-derived biomarkers and compute "
                "z-scores and composite indices. You do NOT give clinical advice."
            ),
            "tools": [
                "tool_load_subject_row",
                "tool_compute_biomarkers",
            ],
            "expected_output": {
                "z_scores": "dict of biomarker z-scores",
                "indices": "dict of composite indices",
            },
        },
        "risk_agent": {
            "role": (
                "You are the Risk Pattern Agent. You take biomarker z-scores and "
                "composite indices, and summarize them into a non-diagnostic risk "
                "score and tier (low, moderate, high). You never say someone "
                "has or does not have Alzheimer’s; you only describe patterns."
            ),
            "tools": [
                "tool_compute_risk",
            ],
            "expected_output": {
                "risk_score": "float",
                "risk_tier": "low | moderate | high",
            },
        },
        "report_agent": {
            "role": (
                "You are the Report Agent. You take biomarkers, risk, and optional "
                "longitudinal summary, and produce a structured plain-language "
                "report for researchers. You clearly emphasize this is a research "
                "tool, not a diagnostic system."
            ),
            # In ADK, this one will use Gemini as the LLM + maybe tools for formatting.
            "tools": [],
            "expected_output": {
                "text_report": "string",
            },
        },
    }
