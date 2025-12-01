from agent_system.tools.risk_tools import compute_risk


class RiskAgent:
    """Turns biomarker metrics into a non-diagnostic risk pattern."""

    def run(self, biomarker_info: dict) -> dict:
        return compute_risk(biomarker_info["z_scores"], biomarker_info["indices"])
