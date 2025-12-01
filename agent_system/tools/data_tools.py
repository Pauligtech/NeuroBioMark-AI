import pandas as pd
import os


def load_biomarker_table(path: str = None) -> pd.DataFrame:
    """
    Load biomarkers CSV using an absolute robust path, so it works from VS Code,
    Kaggle, notebooks, and scripts.
    """
    if path is None:
        # find the project root relative to this file
        tools_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(tools_dir, "..", ".."))
        path = os.path.join(project_root, "data", "biomarkers.csv")

    return pd.read_csv(path)


def get_subject_timepoint(df: pd.DataFrame, subject_id: str, timepoint: str) -> dict:
    """
    Return a single row (as dict) for a given subject and timepoint.
    """
    row = df[(df["subject_id"] == subject_id) & (df["timepoint"] == timepoint)]
    if row.empty:
        raise ValueError(f"No data for subject {subject_id} at timepoint {timepoint}")
    return row.iloc[0].to_dict()


def get_subject_history(df: pd.DataFrame, subject_id: str) -> list[dict]:
    """
    Return all rows for a subject, sorted by timepoint, as a list of dicts.
    """
    sub_df = df[df["subject_id"] == subject_id].sort_values("timepoint")
    return sub_df.to_dict(orient="records")
