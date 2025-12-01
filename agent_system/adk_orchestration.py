"""
adk_orchestration.py
--------------------

Defines a sequential multi-agent workflow that mirrors what
you will build in ADK (Intake → Biomarker → Risk → Report).

This is a "reference implementation" you can port into the
actual ADK run loop in your Kaggle notebook.
"""
from agent_system.memory import get_memory_store

from typing import Dict, Any

from agent_system.adk_tools import (
    tool_load_subject_row,
    tool_compute_biomarkers,
    tool_compute_risk,
    tool_summarize_longitudinal,
)
from agent_system.agents.intake_agent import IntakeAgent
from agent_system.agents.report_agent import ReportAgent


def run_adk_style_pipeline(
    subject_id: str,
    timepoint: str,
    analysis_type: str = "single",
    include_longitudinal: bool = True,
) -> Dict[str, Any]:

    memory = get_memory_store()

    # 1) Intake Agent
    intake = IntakeAgent()
    job = intake.run(subject_id=subject_id, timepoint=timepoint, analysis_type=analysis_type)

    # 2) Biomarker Agent
    row = tool_load_subject_row(job["subject_id"], job["timepoint"])
    biomarker_info = tool_compute_biomarkers(row)

    # 3) Risk Agent
    risk_info = tool_compute_risk(biomarker_info)

    # 4) Optional: Longitudinal Summary
    longitudinal_summary = None
    if include_longitudinal:
        # First check memory
        cached = memory.get_longitudinal(job["subject_id"])

        if cached:
            longitudinal_summary = cached
        else:
            # Compute fresh
            longitudinal_summary = tool_summarize_longitudinal(job["subject_id"])
            # Save to memory
            memory.put_longitudinal(job["subject_id"], longitudinal_summary)

    # 5) Report Agent
    report_agent = ReportAgent()
    report_text = report_agent.run(
        subject_id=job["subject_id"],
        timepoint=job["timepoint"],
        biomarkers=biomarker_info,
        risk=risk_info,
        longitudinal=longitudinal_summary,
    )

    return {
        "job": job,
        "row": row,
        "biomarkers": biomarker_info,
        "risk": risk_info,
        "longitudinal": longitudinal_summary,
        "report_text": report_text,
    }

