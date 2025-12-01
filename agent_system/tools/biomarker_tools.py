import json
from typing import Dict, Any


def compute_z_scores(row, ref_path: str = None):
    """
    Compute z-scores using the structure in reference_stats.json:
    {
      "hipp_vol": {"mean": ..., "std": ...},
      "ventricle_vol": {"mean": ..., "std": ...},
      "gm_vol": {"mean": ..., "std": ...},
      "wm_vol": {"mean": ..., "std": ...}
    }
    """

    import os, json

    # robust path resolution
    if ref_path is None:
        tools_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(tools_dir, "..", ".."))
        ref_path = os.path.join(project_root, "data", "reference_stats.json")

    with open(ref_path, "r") as f:
        ref = json.load(f)

    # column names from your CSV
    hipp = row["hipp_vol"]
    vent = row["ventricle_vol"]
    gm = row["gm_vol"]
    wm = row["wm_vol"]

    z_scores = {
        "hipp_z": (hipp - ref["hipp_vol"]["mean"]) / ref["hipp_vol"]["std"],
        "ventricle_z": (vent - ref["ventricle_vol"]["mean"]) / ref["ventricle_vol"]["std"],
        "gm_z": (gm - ref["gm_vol"]["mean"]) / ref["gm_vol"]["std"],
        "wm_z": (wm - ref["wm_vol"]["mean"]) / ref["wm_vol"]["std"],
    }

    return z_scores





def compute_composite_indices(row: Dict[str, Any]) -> Dict[str, float]:
    """Compute simple composite indices for neurodegeneration patterns."""
    indices: Dict[str, float] = {}

    hipp = row["hipp_vol"]
    vent = row["ventricle_vol"]
    gm = row["gm_vol"]
    wm = row["wm_vol"]
    icv = row["icv"]

    indices["vh_ratio"] = float(vent / (hipp + 1e-8))
    indices["atrophy_index"] = float(vent / (icv + 1e-8))
    indices["gm_wm_ratio"] = float(gm / (wm + 1e-8))

    return indices

