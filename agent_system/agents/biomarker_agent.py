from agent_system.tools.biomarker_tools import compute_z_scores, compute_composite_indices


class BiomarkerAgent:
    """Computes z-scores and composite indices from a biomarker row dict."""

    def run(self, row_dict: dict) -> dict:
        z_scores = compute_z_scores(row_dict)
        indices = compute_composite_indices(row_dict)
        return {"z_scores": z_scores, "indices": indices}
