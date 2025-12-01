class ReportAgent:
    """
    Clean, PDF-safe narrative report generator.
    Designed to avoid raw dicts/lists inside text output.
    """

    def run(
        self,
        subject_id: str,
        timepoint: str,
        biomarkers: dict,
        risk: dict,
        longitudinal: dict | None = None
    ) -> str:

        z = biomarkers["z_scores"]
        idx = biomarkers["indices"]

        # --------------------------------------
        # HEADER
        # --------------------------------------
        report = [
            "=== MRI Biomarker Research Report ===",
            f"Subject: {subject_id}",
            f"Timepoint: {timepoint}",
            ""
        ]

        # --------------------------------------
        # Z-SCORES
        # --------------------------------------
        report += [
            "Z-Scores (relative to reference):",
            f"  • Hippocampal volume z: {z['hipp_z']:.2f}",
            f"  • Ventricular volume z: {z['ventricle_z']:.2f}",
            f"  • Grey matter z:        {z['gm_z']:.2f}",
            f"  • White matter z:       {z['wm_z']:.2f}",
            ""
        ]

        # --------------------------------------
        # COMPOSITE INDICES
        # --------------------------------------
        report += [
            "Composite MRI Indices:",
            f"  • Ventricle/Hippocampus ratio: {idx['vh_ratio']:.2f}",
            f"  • Atrophy index (vent/ICV):    {idx['atrophy_index']:.4f}",
            f"  • GM/WM ratio:                 {idx['gm_wm_ratio']:.2f}",
            ""
        ]

        # --------------------------------------
        # RISK SUMMARY
        # --------------------------------------
        report += [
            "AI-Derived Risk Pattern (non-diagnostic):",
            f"  • Risk score: {risk['risk_score']:.2f}",
            f"  • Risk tier:  {risk['risk_tier']}",
            ""
        ]

        # --------------------------------------
        # LONGITUDINAL SUMMARY
        # --------------------------------------
        if longitudinal:
            history = longitudinal.get("history", None)

            report.append("Longitudinal Summary:")

            if isinstance(history, list):
                # Pretty bullets instead of raw list of dicts
                for visit in history:
                    sid = visit.get("subject_id", "?")
                    tp = visit.get("timepoint", "?")
                    age = visit.get("age", "?")
                    label = visit.get("label", "?")

                    report.append(
                        f"  • {sid} @ {tp} | Age: {age}, Clinical label: {label}"
                    )
            else:
                # Fallback
                report.append("  (No structured history available)")

            report.append("")

        # --------------------------------------
        # DISCLAIMER
        # --------------------------------------
        report += [
            "Disclaimer:",
            "  This is a research tool only.",
            "  It does not provide medical diagnosis or clinical decision-making.",
        ]

        return "\n".join(report)
