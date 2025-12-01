from agent_system.tools.data_tools import load_biomarker_table, get_subject_timepoint
from agent_system.agents.intake_agent import IntakeAgent
from agent_system.agents.biomarker_agent import BiomarkerAgent
from agent_system.agents.risk_agent import RiskAgent
from agent_system.agents.report_agent import ReportAgent


def run_pipeline(subject_id: str, timepoint: str) -> str:
    """End-to-end pipeline: load data, compute biomarkers, risk, and a text report."""
    df = load_biomarker_table()

    intake = IntakeAgent()
    job = intake.run(subject_id, timepoint)

    row = get_subject_timepoint(df, job["subject_id"], job["timepoint"])

    biomarker_agent = BiomarkerAgent()
    biomarker_info = biomarker_agent.run(row)

    risk_agent = RiskAgent()
    risk = risk_agent.run(biomarker_info)

    report_agent = ReportAgent()
    report = report_agent.run(job["subject_id"], job["timepoint"], biomarker_info, risk)

    return report
