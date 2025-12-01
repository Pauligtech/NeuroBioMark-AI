class IntakeAgent:
    """Parses user intent into a structured job."""

    def run(self, subject_id: str, timepoint: str, analysis_type: str = "single") -> dict:
        return {
            "subject_id": subject_id,
            "timepoint": timepoint,
            "analysis_type": analysis_type,
        }
