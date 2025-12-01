#!/usr/bin/env python3
"""
NeuroBioMark AI – Streamlit Hybrid Dashboard (with Rotating Logo)
"""

from __future__ import annotations
import os
import io
import tempfile
import base64
from io import BytesIO
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from fpdf import FPDF

from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing

from agent_system.adk_orchestration import run_adk_style_pipeline
from agent_system.tools.data_tools import load_biomarker_table

import google.genai as genai
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

from agent_system.memory import get_memory_store
import uuid

# ---- GEMINI SETUP ----
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_KEY:
    client = genai.Client(api_key=GEMINI_KEY)
else:
    client = None


# ==========================================================
# GEMINI CLIENT (google-genai)
# ==========================================================

def _load_gemini_api_key() -> str | None:
    """
    Load GEMINI_API_KEY from environment or from .gemini_api_key file.
    Supports either:
      - just the key
      - or a line like: GEMINI_API_KEY=xxxx
    """
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key.strip()

    # Fallback: local file
    path = ".gemini_api_key"
    if os.path.exists(path):
        with open(path, "r") as f:
            line = f.read().strip()
        if line.startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip()
        return line.strip()

    return None


GEMINI_API_KEY = _load_gemini_api_key()

if GEMINI_API_KEY:
    try:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        # If something goes wrong, just disable Gemini but don't crash app
        GEMINI_CLIENT = None
        print("Failed to initialize Gemini client:", e)
else:
    GEMINI_CLIENT = None

# Light, fast model – this name is in your list()
GEMINI_MODEL_NAME = "gemini-2.5-flash"





# ============================
# Unicode-safe helper functions
# ============================

def sanitize_unicode(text: str) -> str:
    """
    Replace disallowed UTF-8 characters for FPDF when not using a Unicode font.
    If using DejaVu fonts (recommended), most will render fine, but this ensures safety.
    """
    if text is None:
        return ""

    replacements = {
        "–": "-", "—": "-", "•": "*",
        "’": "'", "‘": "'", "“": '"', "”": '"',
        "→": "->", "×": "x", "±": "+/-",
        "≥": ">=", "≤": "<="
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    return text


def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0


def safe_str(x):
    if x is None:
        return ""
    return sanitize_unicode(str(x))

# ==========================================================

# Optional NIfTI preview
try:
    import nibabel as nib
    HAS_NIB = True
except Exception:
    HAS_NIB = False


# ==========================================================
# GLOBAL SETTINGS
# ==========================================================

ROTATING_LOGO_SIZE = 140  # <--- Change here to resize rotating logo

CUSTOM_CSS = """
<style>
body {
    background-color: #f3f6fb;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.block-container { max-width: 1250px; padding-top: 1rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1f3d, #123b7a);
    color: #e5e7eb;
}
[data-testid="stSidebar"] * { color: #e5e7eb !important; }

/* Cards */
.ns-card {
    background: #fff;
    padding: 1rem 1.2rem;
    border-radius: 14px;
    box-shadow: 0 3px 8px rgba(15,23,42,0.05);
    border: 1px solid #e5e7f5;
    margin-bottom: 0.9rem;
}

/* Chat bubbles */
.chat-user {
    background: #2563eb10;
    border-radius: 12px;
    padding: 0.5rem 0.7rem;
    margin-bottom: 0.4rem;
}
.chat-assistant {
    background: #10b98110;
    border-radius: 12px;
    padding: 0.5rem 0.7rem;
    margin-bottom: 0.4rem;
}

/* Topbar */
.ns-topbar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.75rem 1rem; background: #fff; border-radius: 14px;
    box-shadow: 0 4px 12px rgba(15,23,42,0.05);
    border: 1px solid #e5e7f5; margin-bottom: 1.2rem;
}
.ns-title { font-size: 1.2rem; font-weight: 700; }
.ns-subtitle { font-size: 0.85rem; color: #6b7280; }

/* Rotating logo */
#rotating-logo {
    position: fixed;
    top: 22px;
    left: 30px;
    z-index: 99999;
    animation: spin 7s linear infinite;
}
@keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
</style>
"""


# ==========================================================
# UTILITIES
# ==========================================================

@st.cache_data
def load_data() -> pd.DataFrame:
    return load_biomarker_table()


def load_logo() -> Optional[Image.Image]:
    path = os.path.join("assets", "neuroscan_logo.png")
    try:
        return Image.open(path)
    except:
        return None


def logo_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def get_timepoints(df: pd.DataFrame, subject: str) -> List[str]:
    return sorted(df[df["subject_id"] == subject]["timepoint"].unique().tolist())


# -------------------- MRI utilities --------------------

# MRI handling
def mri_to_volume(uploaded) -> Optional[np.ndarray]:
    """
    Return a 3D volume from an uploaded file.

    - If NIfTI (.nii / .nii.gz) and nibabel is available: load as 3D.
    - If PNG/JPG: convert to grayscale and treat as a single-slice volume (H, W, 1).
    """
    if uploaded is None:
        return None

    name = uploaded.name.lower()

    # ----- NIfTI path -----
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        if not HAS_NIB:
            return None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        img = nib.load(tmp_path)
        vol = img.get_fdata()
        if vol.ndim == 4:  # take first volume if 4D
            vol = vol[..., 0]
        os.remove(tmp_path)
        return vol.astype(np.float32)

    # ----- Image path (PNG/JPG) -----
    if name.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(uploaded).convert("L")  # grayscale
        arr = np.array(img, dtype=np.float32)

        # Normalize 0–1
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()

        # Make it (H, W, 1) so the viewer sees a "volume"
        vol = arr[..., np.newaxis]
        return vol

    # Unknown format
    return None


def load_demo_mri_volume() -> Optional[np.ndarray]:
    """
    Load the built-in demo MRI from assets/brain_demo_mri.png
    and return it as a (H, W, 1) float32 volume.
    """
    demo_path = os.path.join("assets", "brain_demo_mri.png")
    if not os.path.exists(demo_path):
        return None

    img = Image.open(demo_path).convert("L")
    arr = np.array(img, dtype=np.float32)

    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()

    vol = arr[..., np.newaxis]
    return vol



def slice_from_volume(vol: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0: sl = vol[index, :, :]
    elif axis == 1: sl = vol[:, index, :]
    else: sl = vol[:, :, index]

    sl = sl - sl.min()
    if sl.max() > 0:
        sl = sl / sl.max()
    return (sl * 255).astype(np.uint8)


# ==========================================================
#   IMAGE-DERIVED BIOMARKERS (CLASSICAL APPROXIMATION)
# ==========================================================

def compute_image_biomarkers(vol: np.ndarray) -> Dict[str, float]:
    """
    Very simple, non-clinical biomarker extraction from an MRI volume.

    - Works for 2D PNGs or 3D NIfTIs.
    - Returns fractions/indices in arbitrary units (no real mm³).
    - Meant for demo / capstone, not clinical use.
    """
    # Squeeze possible singleton axis (e.g. (H, W, 1))
    vol = np.squeeze(vol)
    if vol.ndim == 2:
        vol_proc = vol.astype(np.float32)
    elif vol.ndim == 3:
        vol_proc = vol.astype(np.float32)
    else:
        raise ValueError(f"Unsupported volume ndim={vol.ndim}")

    # Normalize to 0–1
    vmin, vmax = float(vol_proc.min()), float(vol_proc.max())
    if vmax > vmin:
        vol_proc = (vol_proc - vmin) / (vmax - vmin)
    else:
        vol_proc = np.zeros_like(vol_proc, dtype=np.float32)

    # Slight smoothing to reduce noise
    vol_smooth = gaussian_filter(vol_proc, sigma=1.0)

    # Only consider non-zero voxels for threshold estimation
    nonzero = vol_smooth[vol_smooth > 0]
    if nonzero.size == 0:
        raise ValueError("Volume appears to be empty / all zeros.")

    # Brain vs background via Otsu
    thr_brain = threshold_otsu(nonzero)
    brain_mask = vol_smooth > thr_brain

    # Clean mask a bit
    brain_mask = binary_opening(brain_mask, iterations=1)
    brain_mask = binary_closing(brain_mask, iterations=1)

    total_voxels = float(vol_proc.size)
    brain_voxels = float(brain_mask.sum())
    if brain_voxels == 0:
        raise ValueError("No brain voxels detected – threshold too high / wrong image type.")

    brain_ratio = brain_voxels / total_voxels

    # ------------------------------------------------------------------
    # Within-brain intensities → approximate CSF / GM / WM by tertiles
    # ------------------------------------------------------------------
    brain_vals = vol_smooth[brain_mask]
    p33, p66 = np.percentile(brain_vals, [33, 66])

    csf_mask = brain_mask & (vol_smooth <= p33)
    gm_mask  = brain_mask & (vol_smooth > p33) & (vol_smooth <= p66)
    wm_mask  = brain_mask & (vol_smooth > p66)

    csf_vox = float(csf_mask.sum())
    gm_vox  = float(gm_mask.sum())
    wm_vox  = float(wm_mask.sum())

    csf_frac = csf_vox / brain_voxels
    gm_frac  = gm_vox / brain_voxels
    wm_frac  = wm_vox / brain_voxels

    # ------------------------------------------------------------------
    # Crude ventricle proxy = CSF in central region
    # ------------------------------------------------------------------
    if vol_proc.ndim == 2:
        H, W = vol_proc.shape
        h0, h1 = int(0.3 * H), int(0.7 * H)
        w0, w1 = int(0.3 * W), int(0.7 * W)
        center_csf = csf_mask[h0:h1, w0:w1]
    else:  # 3D
        Z, H, W = vol_proc.shape
        z0, z1 = int(0.3 * Z), int(0.7 * Z)
        h0, h1 = int(0.3 * H), int(0.7 * H)
        w0, w1 = int(0.3 * W), int(0.7 * W)
        center_csf = csf_mask[z0:z1, h0:h1, w0:w1]

    vent_proxy = float(center_csf.sum()) / brain_voxels

    # ------------------------------------------------------------------
    # Crude hippocampal proxy = GM in central medial region
    # ------------------------------------------------------------------
    if vol_proc.ndim == 2:
        H, W = vol_proc.shape
        h0, h1 = int(0.4 * H), int(0.7 * H)
        w0, w1 = int(0.3 * W), int(0.7 * W)
        hipp_region = gm_mask[h0:h1, w0:w1]
    else:
        Z, H, W = vol_proc.shape
        z0, z1 = int(0.4 * Z), int(0.7 * Z)
        h0, h1 = int(0.4 * H), int(0.8 * H)
        w0, w1 = int(0.3 * W), int(0.7 * W)
        hipp_region = gm_mask[z0:z1, h0:h1, w0:w1]

    hipp_proxy = float(hipp_region.sum()) / brain_voxels

    # ------------------------------------------------------------------
    # Atrophy index = CSF fraction inside brain
    # ------------------------------------------------------------------
    atrophy_index = csf_frac

    biomarkers = {
        "brain_ratio": brain_ratio,
        "gm_fraction": gm_frac,
        "wm_fraction": wm_frac,
        "csf_fraction": csf_frac,
        "vent_proxy": vent_proxy,
        "hipp_proxy": hipp_proxy,
        "atrophy_index": atrophy_index,
    }
    return biomarkers


def compute_image_risk(bm: Dict[str, float]) -> Dict[str, Any]:
    """
    Simple composite risk from image-derived biomarkers.

    Higher with:
    - more CSF / ventricles
    - more atrophy
    Lower with:
    - larger hippocampal proxy
    """
    vent = bm["vent_proxy"]
    atrophy = bm["atrophy_index"]
    hipp = bm["hipp_proxy"]

    raw_score = 2.0 * atrophy + 1.5 * vent - 1.0 * hipp

    # Normalize into 0–3 (roughly)
    risk_score = max(0.0, min(3.0, raw_score * 3.0))

    if risk_score < 1.0:
        tier = "low"
    elif risk_score < 2.0:
        tier = "moderate"
    else:
        tier = "high"

    return {
        "risk_score": risk_score,
        "risk_tier": tier,
    }




from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter

def preprocess_mri_slice(img: np.ndarray) -> dict:
    """
    Simple MRI preprocessing pipeline:
    - Normalize
    - Gaussian denoise
    - Threshold-based brain masking
    - Extract proxy biomarkers
    """

    # --- Convert to float + normalize -------------------
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # --- Denoise ----------------------------------------
    img_smooth = gaussian_filter(img, sigma=1)

    # --- Brain mask using Otsu threshold ----------------
    try:
        t = threshold_otsu(img_smooth)
        brain_mask = (img_smooth > t).astype(np.uint8)
    except:
        brain_mask = np.ones_like(img_smooth)

    # --- Biomarker Proxies ------------------------------
    total_area = img.size
    brain_area = brain_mask.sum()
    brain_ratio = brain_area / total_area  # proxy for atrophy

    # Ventricular brightness proxy (central ROI)
    H, W = img.shape
    central = img[H//3:2*H//3, W//3:2*W//3]
    ventricle_intensity = float(central.mean())

    hippocampal_proxy = float(np.percentile(img[brain_mask.astype(bool)], 40))
    ventricle_proxy = float(1 - ventricle_intensity)
    atrophy_index = float(1 - brain_ratio)

    biomarkers = {
        "brain_ratio": brain_ratio,
        "hipp_proxy": hippocampal_proxy,
        "vent_proxy": ventricle_proxy,
        "atrophy_index": atrophy_index
    }

    return {
        "processed": img_smooth,
        "brain_mask": brain_mask,
        "biomarkers": biomarkers
    }


# ==========================================================
# ADVANCED MRI PROCESSING UTILITIES
# ==========================================================
from sklearn.cluster import KMeans
from scipy.ndimage import laplace, binary_dilation


def n4_bias_correction(img: np.ndarray) -> np.ndarray:
    """
    Simplified bias field correction using Gaussian smoothing.
    Not N4ITK, but a decent approximation for demos.
    """
    smooth = gaussian_filter(img, sigma=12)
    corrected = img / (smooth + 1e-6)
    corrected = corrected - corrected.min()
    corrected = corrected / (corrected.max() + 1e-6)
    return corrected


def skull_strip(img: np.ndarray) -> np.ndarray:
    """Generate a rough brain mask."""
    t = threshold_otsu(img)
    mask = img > t * 0.8
    mask = binary_closing(mask, iterations=2)
    mask = binary_opening(mask, iterations=1)
    mask = binary_dilation(mask, iterations=2)
    return mask.astype(np.uint8)


def segment_tissues(img: np.ndarray, mask: np.ndarray):
    """KMeans segmentation into 3 clusters: CSF, GM, WM."""
    vals = img[mask == 1].reshape(-1, 1)

    kmeans = KMeans(n_clusters=3, n_init=5, random_state=42)
    labels = kmeans.fit_predict(vals)

    gm_lvl = np.percentile(vals, 50)
    wm_lvl = np.percentile(vals, 80)

    csf = (img < gm_lvl) & (mask == 1)
    gm = (img >= gm_lvl) & (img < wm_lvl) & (mask == 1)
    wm = (img >= wm_lvl) & (mask == 1)

    return csf.astype(np.uint8), gm.astype(np.uint8), wm.astype(np.uint8)


def hippocampus_proxy_mask(img: np.ndarray, gm_mask: np.ndarray) -> np.ndarray:
    """Approximate hippocampus by GM in a medial temporal ROI."""
    H, W = img.shape
    h0, h1 = int(0.4 * H), int(0.8 * H)
    w0, w1 = int(0.3 * W), int(0.7 * W)
    roi = gm_mask[h0:h1, w0:w1]

    hippo = np.zeros_like(gm_mask)
    hippo[h0:h1, w0:w1] = roi
    return hippo * 255


def edge_enhance(img: np.ndarray) -> np.ndarray:
    """Laplacian edge detection."""
    edges = np.abs(laplace(img))
    edges = edges / (edges.max() + 1e-6)
    return edges






# ==========================================================
# PLOTS & PDF
# ==========================================================

def plot_risk_gauge(risk_score: float):
    norm = max(0.0, min(risk_score / 3.0, 1.0))
    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(aspect="equal"))

    cmap = plt.cm.get_cmap("RdYlGn_r")
    color = cmap(norm)

    ax.pie(
        [norm, 1 - norm],
        startangle=90, counterclock=False,
        wedgeprops=dict(width=0.3, edgecolor="white"),
        colors=[color, "#e5e7eb"],
    )
    ax.text(0, 0, f"{risk_score:.2f}", ha="center", va="center", fontsize=14, fontweight="bold")
    st.pyplot(fig)


from fpdf import FPDF
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np


def sanitize_unicode(text: str) -> str:
    """Replace characters that FPDF cannot render."""
    if text is None:
        return ""
    replacer = {
        "–": "-",  # en dash
        "—": "-",  # em dash
        "•": "-",  # bullets
        "’": "'",  # apostrophe
        "“": '"',
        "”": '"',
    }
    for bad, good in replacer.items():
        text = text.replace(bad, good)
    return text


def safe_float(value):
    """Convert None or non-floats to printable string."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2f}"
    except:
        return str(value)


def safe_float4(value):
    """Same but 4 decimals."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except:
        return str(value)


from fpdf import FPDF
import os


def pdf_write_field(pdf, key, value):
    """Robust PDF-safe field writer that handles numbers, strings, dicts, lists, and Unicode."""

    # ----------------------------------------
    # Handle None
    # ----------------------------------------
    if value is None:
        pdf.cell(0, 6, f"{key}: N/A", ln=True)
        return

    # ----------------------------------------
    # Try converting numeric values safely
    # ----------------------------------------
    try:
        # Remove common non-numeric chars
        clean = str(value).replace(",", "").replace("–", "-")
        num = float(clean)
        pdf.cell(0, 6, f"{key}: {num:.2f}", ln=True)
        return
    except:
        pass  # Not numeric → continue

    # ----------------------------------------
    # Handle structured values
    # ----------------------------------------
    if isinstance(value, (dict, list)):
        import json
        json_text = json.dumps(value, indent=2)
        json_text = sanitize_unicode(json_text)

        pdf.set_font("DejaVu", "B", 11)
        pdf.cell(0, 6, f"{key}:", ln=True)

        pdf.set_font("DejaVu", "", 11)
        pdf.multi_cell(0, 5, json_text)
        pdf.ln(2)
        return

    # ----------------------------------------
    # Handle plain strings
    # ----------------------------------------
    text = sanitize_unicode(str(value))
    pdf.cell(0, 6, f"{key}: {text}", ln=True)



def generate_pdf_bytes(result: Dict[str, Any]) -> bytes:
    """
    Generate a clean Unicode-safe PDF for the MRI biomarker report.
    Updated for FPDF 2.6+ (no deprecated params).
    """
    import json
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    # -------------------------------------------------
    # Safe extract fields
    # -------------------------------------------------
    job = result.get("job", {})
    biomarkers = result.get("biomarkers", {})
    z = biomarkers.get("z_scores", {})
    idx = biomarkers.get("indices", {})
    risk = result.get("risk", {})
    narrative = sanitize_unicode(result.get("report_text", ""))
    history = result.get("history", [])

    # -------------------------------------------------
    # Initialize PDF
    # -------------------------------------------------
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    # -------------------------------------------------
    # Load Unicode fonts  (NO uni=True — deprecated)
    # -------------------------------------------------
    font_dir = "assets/fonts"

    pdf.add_font("DejaVu", "", os.path.join(font_dir, "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", os.path.join(font_dir, "DejaVuSans-Bold.ttf"))
    pdf.add_font("DejaVu", "I", os.path.join(font_dir, "DejaVuSans-Oblique.ttf"))

    # -------------------------------------------------
    # Title
    # -------------------------------------------------
    pdf.set_font("DejaVu", "B", 16)
    pdf.cell(
        0, 10,
        "NeuroBioMark AI – MRI Biomarker Report",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )
    pdf.ln(4)

    # -------------------------------------------------
    # Subject info
    # -------------------------------------------------
    pdf.set_font("DejaVu", "", 12)
    pdf_write_field(pdf, "Subject", job.get("subject_id"))
    pdf_write_field(pdf, "Timepoint", job.get("timepoint"))
    pdf.ln(4)

    # -------------------------------------------------
    # Z-scores
    # -------------------------------------------------
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(
        0, 10,
        "Biomarker Z-Scores",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )

    pdf.set_font("DejaVu", "", 11)
    for key, val in z.items():
        pdf_write_field(pdf, key, val)
    pdf.ln(4)

    # -------------------------------------------------
    # Composite indices
    # -------------------------------------------------
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(
        0, 10,
        "Composite Indices",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )

    pdf.set_font("DejaVu", "", 11)
    pdf_write_field(pdf, "Ventricle/Hippocampus ratio", idx.get("vh_ratio"))
    pdf_write_field(pdf, "Atrophy index", idx.get("atrophy_index"))
    pdf_write_field(pdf, "GM/WM ratio", idx.get("gm_wm_ratio"))
    pdf.ln(4)

    # -------------------------------------------------
    # Risk summary
    # -------------------------------------------------
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(
        0, 10,
        "AI-Derived Risk Summary",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )

    pdf.set_font("DejaVu", "", 11)
    pdf_write_field(pdf, "Risk score (0–3)", risk.get("risk_score"))
    pdf_write_field(pdf, "Risk tier", risk.get("risk_tier"))
    pdf.ln(4)

    # -------------------------------------------------
    # Longitudinal history
    # -------------------------------------------------
    if isinstance(history, list) and len(history) > 0:
        pdf.set_font("DejaVu", "B", 14)
        pdf.cell(
            0, 10,
            "Longitudinal Summary",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT
        )

        pdf.set_font("DejaVu", "", 11)
        for visit in history:
            txt = sanitize_unicode(json.dumps(visit, indent=2))
            pdf.multi_cell(0, 5, txt)
            pdf.ln(1)
        pdf.ln(4)

    # -------------------------------------------------
    # Narrative
    # -------------------------------------------------
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(
        0, 10,
        "AI-Generated Narrative Interpretation",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )

    pdf.set_font("DejaVu", "", 11)
    pdf.multi_cell(0, 6, narrative)
    pdf.ln(4)

    # -------------------------------------------------
    # Disclaimer
    # -------------------------------------------------
    pdf.set_font("DejaVu", "I", 9)
    pdf.multi_cell(
        0, 5,
        "Disclaimer: This is a research tool only. "
        "It does not provide medical diagnosis or clinical decision-making."
    )

    # -------------------------------------------------
    # Export PDF bytes (convert bytearray → bytes)
    # -------------------------------------------------
    pdf_bytes = pdf.output()   # returns bytearray
    return bytes(pdf_bytes)    # Streamlit requires bytes, NOT bytearray









def generate_clinical_summary(result: dict) -> str:
    z = result["biomarkers"]["z_scores"]
    idx = result["biomarkers"]["indices"]
    risk = result["risk"]["risk_score"]

    summary = f"""
        CLINICAL SUMMARY
        =================

        • Hippocampal z-score: {z['hipp_z']:.2f}
        Lower values may reflect medial-temporal lobe atrophy associated with early Alzheimer’s.

        • Ventricular/Hippocampal ratio: {idx['vh_ratio']:.2f}
        Higher ratios suggest ventricular enlargement relative to hippocampal size.

        • Grey/White Matter Ratio: {idx['gm_wm_ratio']:.2f}
        Used as a proxy for cortical thinning and structural decline.

        • Atrophy Index: {idx['atrophy_index']:.4f}
        Higher values represent more CSF and thus more global atrophy.

        • Composite Neurodegeneration Risk Score: {risk:.2f}

        INTERPRETATION
        --------------
        { "Findings raise concern for early neurodegeneration." if risk >= 1.0 else "Findings remain within normal limits for age." }
            """
    return summary



# ==========================================================
# TOPBAR
# ==========================================================

def render_topbar(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="ns-topbar">
            <div>
                <div class="ns-title">{title}</div>
                <div class="ns-subtitle">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# PAGE DEFINITIONS
# ==========================================================

def page_dashboard(df: pd.DataFrame):
    """Landing page + dashboard view depending on analysis availability."""

    result = st.session_state.get("analysis_result")

    # ---------------------------------------------------------
    # BEAUTIFUL LANDING PAGE (when no analysis yet)
    # ---------------------------------------------------------
    if result is None:
        st.markdown("""
        <style>
            .landing-title {
                font-size: 3rem;
                font-weight: 800;
                text-align: center;
                letter-spacing: -1px;
                margin-top: -20px;
                color: #0f172a;
            }
            .landing-subtitle {
                font-size: 1.3rem;
                text-align: center;
                color: #334155;
                margin-bottom: 2rem;
            }
            .landing-card {
                background: white;
                border-radius: 18px;
                padding: 2rem 2.5rem;
                margin-top: 1rem;
                box-shadow: 0 4px 18px rgba(0,0,0,0.06);
                border: 1px solid #e2e8f0;
                width: 85%;
                margin-left: auto;
                margin-right: auto;
            }
            .landing-step {
                font-size: 1.1rem;
                margin-bottom: 0.6rem;
                color: #475569;
            }
            .cta-row {
                text-align: center;
                margin-top: 1.6rem;
            }
            .cta-btn {
                padding: 0.7rem 1.6rem;
                font-size: 1.1rem;
                font-weight: 600;
                color: white !important;
                border-radius: 50px;
                background: linear-gradient(135deg,#2563eb,#1d4ed8);
                display: inline-block;
                margin: 0 0.5rem;
            }
        </style>
        """, unsafe_allow_html=True)

        # Title + subtitle
        st.markdown("<div class='landing-title'>🧠 NeuroBioMark AI</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='landing-subtitle'>Automated MRI biomarker extraction and Alzheimer’s risk analysis.</div>",
            unsafe_allow_html=True,
        )

        # Info card
        st.markdown("<div class='landing-card'>", unsafe_allow_html=True)
        st.markdown("### What this tool does")
        st.markdown("""
            <div class="landing-step">• Extracts hippocampal volume, ventricular size, GM/WM ratios, and atrophy indices.</div>
            <div class="landing-step">• Computes standardized biomarker z-scores using a reference cohort.</div>
            <div class="landing-step">• Generates a composite neurodegeneration risk score.</div>
            <div class="landing-step">• Tracks longitudinal changes across MRI timepoints.</div>
            <div class="landing-step">• Produces an AI-generated research report.</div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # CTA buttons
        colA, colB = st.columns([1,1])
        with colA:
            if st.button("📤 Upload MRI", use_container_width=True):
                st.session_state["force_page"] = "MRI Upload"
                st.rerun()   # <-- FIXED
        with colB:
            if st.button("🧪 Use Demo MRI", use_container_width=True):
                with st.spinner("Processing demo MRI…"):
                    result = run_adk_style_pipeline(
                        subject_id="S001",
                        timepoint="T0",
                        analysis_type="longitudinal",
                        include_longitudinal=True,
                    )
                    st.session_state["analysis_result"] = result
                st.success("Demo MRI loaded and processed!")
                st.rerun()   # <-- FIXED
        return

    # ---------------------------------------------------------
    # DASHBOARD VIEW WHEN ANALYSIS EXISTS
    # ---------------------------------------------------------
    render_topbar("NeuroBioMark AI Dashboard", "MRI-derived biomarker overview")

    z = result["biomarkers"]["z_scores"]
    idx = result["biomarkers"]["indices"]
    job = result["job"]
    risk = result["risk"]

    # Row 1
    r1c1, r1c2, r1c3 = st.columns([1.4, 1.1, 1.1])

    with r1c1:
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.subheader("🧠 Risk Score")
        plot_risk_gauge(risk["risk_score"])
        st.caption("Higher scores reflect stronger neurodegeneration indicators.")
        st.markdown('</div>', unsafe_allow_html=True)

    with r1c2:
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.subheader("📌 Subject Details")
        st.metric("ID", job["subject_id"])
        st.metric("Timepoint", job["timepoint"])
        st.markdown('</div>', unsafe_allow_html=True)

    with r1c3:
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.subheader("📊 Key Indices")
        st.metric("Hippocampal z", f"{z['hipp_z']:.2f}")
        st.metric("V/H Ratio", f"{idx['vh_ratio']:.2f}")
        st.metric("GM/WM Ratio", f"{idx['gm_wm_ratio']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    if not result:
        st.info("Run an analysis from the MRI Upload page.")
        return



def page_mri_upload(df):
    render_topbar("Upload MRI", "Load MRI and select subject")

    left, right = st.columns([1.4, 1.1])

    with left:
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.subheader("📤 Upload MRI")

        uploaded = st.file_uploader("Upload MRI file", 
                                    type=["png","jpg","jpeg","nii","nii.gz"])

        if uploaded:
            vol = mri_to_volume(uploaded)
            if vol is not None:
                st.session_state["mri_volume"] = vol
                st.session_state["mri_name"] = uploaded.name
                st.success("MRI file loaded successfully.")
                st.image(slice_from_volume(vol, 2, vol.shape[2]//2), caption="Preview slice")
            else:
                st.error("Unsupported file or failed to read MRI.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.subheader("🧪 MRI Processing")

        subject = st.selectbox("Subject", sorted(df["subject_id"].unique()))
        timepoint = st.selectbox("Timepoint", get_timepoints(df, subject))

        if st.button("Run Analysis", use_container_width=True):
            if "mri_volume" not in st.session_state:
                st.warning("Upload an MRI first.")
            else:
                with st.spinner("Running AI pipeline…"):
                    res = run_adk_style_pipeline(
                        subject_id=subject,
                        timepoint=timepoint,
                        analysis_type="longitudinal",
                        include_longitudinal=True,
                    )
                    st.session_state["analysis_result"] = res
                st.success("Analysis complete!")
        st.markdown('</div>', unsafe_allow_html=True)

def page_mri_viewer():
    render_topbar("MRI Viewer", "Scroll through MRI slices")

    vol = st.session_state.get("mri_volume")
    name = st.session_state.get("mri_name", "")

    if vol is None:
        st.info("Upload or load a demo MRI first.")
        return

    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.subheader(name)

    # Detect 2D / 3D MRI
    is_single_slice = (vol.ndim == 2) or (vol.ndim == 3 and vol.shape[-1] == 1)

    if is_single_slice:
        plane_options = ["Axial"]
    else:
        plane_options = ["Axial", "Coronal", "Sagittal"]

    plane = st.radio("Viewing plane", plane_options, horizontal=True)

    # Block invalid planes for 2D
    if is_single_slice and plane != "Axial":
        st.warning("This MRI contains only one slice. Coronal and sagittal views require a 3D MRI volume.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Map planes to axes
    axis_map = {"Axial": 2, "Coronal": 1, "Sagittal": 0}
    axis = axis_map[plane]


    # Compute max index safely
    if vol.ndim == 2:
        max_idx = 0
    else:
        max_idx = vol.shape[axis] - 1
        max_idx = max(max_idx, 0)

    # SAFE SLIDER HANDLING
    if max_idx <= 0:
        idx = 0
        st.info("This MRI contains only one slice — scrolling disabled.")
    else:
        idx = st.slider(
            "Slice index",
            min_value=0,
            max_value=max_idx,
            value=max_idx // 2,
        )


    # Extract slice
    sl = slice_from_volume(vol, axis, idx)
    st.image(sl, use_column_width=True, clamp=True)

    st.markdown("</div>", unsafe_allow_html=True)



def page_mri_pipeline():
    render_topbar("MRI Pipeline", "Run preprocessing to extract biomarkers")

    vol = st.session_state.get("mri_volume")
    name = st.session_state.get("mri_name", "")

    if vol is None:
        st.info("Upload an MRI first.")
        return

    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.subheader(name)

    # Only use the middle slice for preprocessing (2D)
    slice_img = vol[..., 0]
    result = preprocess_mri_slice(slice_img)

    st.session_state["mri_preproc"] = result  # save for other pages

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Raw Slice")
        st.image(slice_img, clamp=True, use_column_width=True)

    with col2:
        st.markdown("### Denoised Slice")
        st.image(result["processed"], clamp=True, use_column_width=True)

    with col3:
        st.markdown("### Brain Mask")
        st.image(result["brain_mask"] * 255, clamp=True, use_column_width=True)

    st.markdown("---")
    st.markdown("## Extracted Biomarker Estimates")

    st.json(result["biomarkers"])

    st.markdown("</div>", unsafe_allow_html=True)


def page_mri_advanced():
    render_topbar("MRI Advanced Lab", "Bias correction • Skull stripping • Tissue segmentation • ROI analysis")

    vol = st.session_state.get("mri_volume")
    if vol is None:
        st.info("Upload or load a demo MRI first.")
        return

    # Use central slice
    sl = np.squeeze(vol[..., vol.shape[-1]//2]) if vol.ndim == 3 else np.squeeze(vol)

    # Preprocessing steps
    norm = (sl - sl.min()) / (sl.max() - sl.min() + 1e-6)
    bias = n4_bias_correction(norm)
    mask = skull_strip(bias)
    csf, gm, wm = segment_tissues(bias, mask)
    edges = edge_enhance(bias)
    hippo = hippocampus_proxy_mask(bias, gm)

    # Visualization layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Raw Slice")
        st.image(norm, clamp=True)
        st.markdown("### Brain Mask")
        st.image(mask * 255, clamp=True)

    with col2:
        st.markdown("### Bias Corrected")
        st.image(bias, clamp=True)
        st.markdown("### Edges")
        st.image(edges, clamp=True)

    with col3:
        st.markdown("### GM / WM / CSF")
        st.image(np.stack([csf*255, gm*255, wm*255], axis=-1))
        st.markdown("### Hippocampal ROI (Proxy)")
        st.image(hippo, clamp=True)

    st.markdown("---")
    st.subheader("Advanced Metrics")
    st.json({
        "brain_mask_fraction": float(mask.sum()) / float(mask.size),
        "gm_fraction": float(gm.sum()) / float(mask.sum()),
        "wm_fraction": float(wm.sum()) / float(mask.sum()),
        "csf_fraction": float(csf.sum()) / float(mask.sum()),
        "hippocampal_proxy_volume": float(hippo.sum()) / 255.0,
    })




def page_biomarkers():
    """
    Full image-derived biomarker analysis page.
    Uses the current MRI volume in session_state["mri_volume"].
    """
    render_topbar("Biomarker Analysis", "MRI-derived tissue fractions and atrophy proxies.")

    vol = st.session_state.get("mri_volume")

    if vol is None:
        st.info("Upload an MRI (or use the demo MRI) first, then open this page.")
        return

    # --- Compute biomarkers safely ---
    try:
        bm = compute_image_biomarkers(vol)
        risk = compute_image_risk(bm)
    except Exception as e:
        st.error(f"Could not compute biomarkers from this image: {e}")
        return

    # --- Layout: overview metrics ---
    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.subheader("Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Brain coverage (brain / FOV)",
            f"{bm['brain_ratio']:.3f}",
            help="Fraction of all voxels that belong to brain tissue."
        )

    with col2:
        st.metric(
            "GM fraction",
            f"{bm['gm_fraction']:.3f}",
            help="Grey matter voxels / all brain voxels."
        )

    with col3:
        st.metric(
            "WM fraction",
            f"{bm['wm_fraction']:.3f}",
            help="White matter voxels / all brain voxels."
        )

    with col4:
        st.metric(
            "CSF fraction",
            f"{bm['csf_fraction']:.3f}",
            help="CSF-like voxels / all brain voxels (proxy for global atrophy)."
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Risk panel + proxies ---
    row1 = st.columns([1.1, 1.4])

    with row1[0]:
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.subheader("Composite risk (demo only)")
        plot_risk_gauge(risk["risk_score"])
        st.caption(
            f"Risk tier: **{risk['risk_tier'].upper()}** – based on atrophy, CSF/ventricle proxy, and "
            f"a crude hippocampal proxy. For research demonstration only."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with row1[1]:
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.subheader("Atrophy & regional proxies")

        colA, colB = st.columns(2)
        with colA:
            st.metric(
                "Atrophy index (CSF fraction)",
                f"{bm['atrophy_index']:.3f}",
                help="Higher values → more CSF inside the skull, a rough marker of global atrophy."
            )
            st.metric(
                "Ventricle proxy",
                f"{bm['vent_proxy']:.3f}",
                help="CSF fraction in the central region (not a true ventricle segmentation)."
            )
        with colB:
            st.metric(
                "Hippocampal proxy",
                f"{bm['hipp_proxy']:.3f}",
                help="Grey-matter–like voxels in a medial temporal region (very crude hippocampal proxy)."
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # --- Tissue composition plot ---
    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.subheader("Tissue composition")

    fig, ax = plt.subplots(figsize=(4, 3))
    labels = ["GM", "WM", "CSF"]
    values = [bm["gm_fraction"], bm["wm_fraction"], bm["csf_fraction"]]
    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of brain volume")
    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

    # --- JSON-style dump at the bottom for debugging / technical readers ---
    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.subheader("Raw biomarker estimates (for debugging)")
    st.json(
        {
            "image_biomarkers": bm,
            "image_risk": risk,
        }
    )
    st.caption(
        "All values are in arbitrary units and are meant only for demonstration, "
        "not for clinical use."
    )
    st.markdown("### 🧠 Clinical Summary")
    if st.button("Generate Clinical Summary"):
        try:
            result = st.session_state.get("analysis_result")
            if result:
                summary_text = generate_clinical_summary(result)
                st.info(summary_text)
            else:
                st.warning("Run a full analysis first.")
        except Exception as e:
            st.error(f"Could not generate summary: {e}")

    st.markdown("</div>", unsafe_allow_html=True)



def page_trends(df):
    render_topbar("Longitudinal Trends", "Structural changes over time")

    result = st.session_state.get("analysis_result")
    if not result:
        st.info("Run an analysis first.")
        return

    subject = result["job"]["subject_id"]
    sub = df[df["subject_id"] == subject]

    st.markdown('<div class="ns-card">', unsafe_allow_html=True)

    if sub["timepoint"].nunique() < 2:
        st.info("Need at least two timepoints for trend analysis.")
    else:
        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot(sub["timepoint"], sub["hipp_vol"], "-o", label="Hippocampus")
        ax.plot(sub["timepoint"], sub["ventricle_vol"], "-o", label="Ventricle")
        ax.legend()
        ax.set_ylabel("Volume")
        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)


def page_report():
    render_topbar("MRI Report", "AI-generated narrative summary")

    result = st.session_state.get("analysis_result")
    if not result:
        st.info("Run an analysis first.")
        return

    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.subheader("📝 AI-Generated Report")
    st.write(result["report_text"])
    st.markdown('</div>', unsafe_allow_html=True)

    pdf_bytes = generate_pdf_bytes(result)
    st.download_button(
        "Download PDF Report",
        data=pdf_bytes,
        file_name="NeuroBioMark_Report.pdf",
        mime="application/pdf"
    )


def load_gemini_key():
    """
    Load the Gemini API key from environment variable OR from .gemini_api_key file.
    Safe and non-recursive.
    """
    # 1) Try environment variable
    key = os.getenv("GEMINI_API_KEY")
    if key and len(key.strip()) > 0:
        return key.strip()

    # 2) Try local file
    path = ".gemini_api_key"
    if os.path.exists(path):
        with open(path, "r") as f:
            file_key = f.read().strip()
            if file_key:
                return file_key

    return None



def page_chat(df):
    render_topbar("AI Assistant", "Ask NeuroBioMark questions")

    from agent_system.memory import get_memory_store
    import uuid

    result = st.session_state.get("analysis_result")
    memory = get_memory_store()

    # -----------------------------
    # 1. Create or load session ID
    # -----------------------------
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    session_id = st.session_state.session_id

    # -----------------------------
    # 2. Load conversation history from memory
    # -----------------------------
    stored_history = memory.get_session_history(session_id)

    # Streamlit still needs chat in its state for UI
    st.session_state.chat = [(t["role"], t["content"]) for t in stored_history]

    # -----------------------------
    # 3. Display chat history
    # -----------------------------
    for role, msg in st.session_state.chat:
        css = "chat-user" if role == "user" else "chat-assistant"
        who = "You" if role == "user" else "NeuroBioMark"
        st.markdown(
            f'<div class="{css}"><b>{who}:</b> {msg}</div>',
            unsafe_allow_html=True
        )

    # -----------------------------
    # 4. Input box
    # -----------------------------
    prompt = st.chat_input("Ask something about MRI, biomarkers, neurodegeneration…")

    if prompt:
        # -- Save user message to memory
        memory.append_session_turn(
            session_id=session_id,
            role="user",
            content=prompt
        )

        # -- Generate assistant answer
        ans = run_gemini_assistant(prompt, result)

        # -- Save assistant message to memory
        memory.append_session_turn(
            session_id=session_id,
            role="assistant",
            content=ans
        )

        # -- Refresh UI
        st.rerun()




def run_gemini_assistant(prompt: str, analysis_result: dict | None) -> str:
    """Sends user question + MRI biomarker context to Gemini 2.5 Flash."""
    
    if client is None:
        return "Gemini API key not configured. Please set GEMINI_API_KEY."

    # ----- Build context -----
    context = """
You are NeuroBioMark AI — a biomedical MRI research assistant.
You must:
- Use scientific, research-oriented language.
- NEVER provide a clinical diagnosis, medical advice, or treatment.
- Interpret MRI biomarkers cautiously and describe biological significance only.
    """

    # If MRI biomarkers exist, attach them
    if analysis_result:
        try:
            z = analysis_result["biomarkers"]["z_scores"]
            idx = analysis_result["biomarkers"]["indices"]
            risk = analysis_result["risk"]

            context += f"""

                MRI BIOMARKERS (research only):

                Z-SCORES:
                • Hippocampal z = {z['hipp_z']:.2f}
                • Ventricular z = {z['ventricle_z']:.2f}
                • GM z = {z['gm_z']:.2f}
                • WM z = {z['wm_z']:.2f}

                INDICES:
                • V/H ratio = {idx['vh_ratio']:.2f}
                • Atrophy index = {idx['atrophy_index']:.4f}
                • GM/WM ratio = {idx['gm_wm_ratio']:.2f}

                RISK SCORE (non-diagnostic):
                • {risk['risk_score']:.2f} ({risk['risk_tier']})
                """
        except Exception as e:
            context += f"\n(Note: some biomarker fields missing: {e})"

    full_prompt = context + "\n\nUser question: " + prompt

    # ----- Call Gemini 2.5 Flash -----
    try:
        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt
        )
        return res.text or "No text returned."
    except Exception as e:
        return f"Gemini error: {e}"





def page_explain_biomarkers():
    render_topbar("Explain Biomarkers", "Understand what each MRI biomarker means")

    st.markdown('<div class="ns-card">', unsafe_allow_html=True)

    st.subheader("Click a biomarker to reveal its meaning")

    explanations = {
        "Hippocampal Volume (proxy)": "Lower values may indicate degeneration in the medial temporal lobe linked to Alzheimer’s.",
        "GM Fraction": "Grey matter decreases with cortical thinning and neuronal loss.",
        "WM Fraction": "White matter reflects connectivity integrity; reductions occur in aging and disease.",
        "CSF Fraction": "Higher CSF fraction suggests greater brain atrophy.",
        "Ventricle Proxy": "Ventricle enlargement is associated with neurodegeneration.",
        "Atrophy Index": "Global loss of brain tissue represented by CSF expansion.",
    }

    for biomarker, text in explanations.items():
        if st.button(biomarker):
            st.success(text)

    st.markdown("</div>", unsafe_allow_html=True)

def page_compare_mri():
    render_topbar("Compare MRI Timepoints", "Baseline vs Follow-up comparison")

    st.markdown('<div class="ns-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        base = st.file_uploader("Baseline MRI", type=["png","jpg","jpeg","nii","nii.gz"])

    with col2:
        follow = st.file_uploader("Follow-up MRI", type=["png","jpg","jpeg","nii","nii.gz"])

    if base and follow:
        try:
            vol0 = mri_to_volume(base)
            vol1 = mri_to_volume(follow)

            bm0 = compute_image_biomarkers(vol0)
            bm1 = compute_image_biomarkers(vol1)

            st.subheader("Biomarker Changes")
            for key in bm0:
                delta = bm1[key] - bm0[key]
                st.metric(key, f"{delta:+.3f}")

        except Exception as e:
            st.error(f"Could not compare MRI: {e}")

    st.markdown("</div>", unsafe_allow_html=True)




# ==========================================================
# MAIN APP
# ==========================================================

def main():
    st.set_page_config(page_title="NeuroBioMark AI", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    df = load_data()
    logo = load_logo()

    # ---------------- Rotating logo (ALL pages except sidebar) ----------------
    if logo is not None:
        encoded = logo_base64(logo)
        st.markdown(
            f"""
            <img id="rotating-logo" 
                 src="data:image/png;base64,{encoded}"
                 width="{ROTATING_LOGO_SIZE}px">
            """,
            unsafe_allow_html=True,
        )

    # --------------------- SIDEBAR ---------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; margin-top:-60px;">
                <img src="data:image/png;base64,{}" width="500">
                <div style="font-size:1.3rem; font-weight:700; margin-top: -50px;">NeuroBioMark AI</div>
                <div style="font-size:0.85rem; color:#cbd5f5;">Alzheimer's MRI Biomarker Agent</div>
            </div>
            """.format(logo_base64(logo) if logo else ""),
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            [
                "Dashboard",
                "MRI Upload",
                "MRI Viewer",
                "MRI Pipeline",
                "Biomarker Analysis",
                "Explain Biomarkers",
                "MRI Advanced Lab",
                "Compare MRI",
                "Longitudinal Trends",
                "Report",
                "AI Assistant",
            ],
            label_visibility="collapsed",
        )

    # ---------------- Route pages ----------------
    if page == "Dashboard": page_dashboard(df)
    elif page == "MRI Upload": page_mri_upload(df)
    elif page == "MRI Viewer": page_mri_viewer()
    elif page == "MRI Pipeline": page_mri_pipeline()
    elif page == "Biomarker Analysis": page_biomarkers()
    elif page == "Explain Biomarkers": page_explain_biomarkers()
    elif page == "MRI Advanced Lab": page_mri_advanced()
    elif page == "Compare MRI": page_compare_mri()
    elif page == "Longitudinal Trends": page_trends(df)
    elif page == "Report": page_report()
    elif page == "AI Assistant": page_chat(df)


if __name__ == "__main__":
    main()

# ---------------- FORCE ROTATING LOGO OUTSIDE SIDEBAR ----------------
import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import os

def _load_rotating_logo():
    path = os.path.join("assets", "neuroscan_logo.png")
    try:
        img = Image.open(path)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        st.write("Rotating logo load error:", e)
        return None

logo_encoded = _load_rotating_logo()

if logo_encoded:
    st.markdown(
        f"""
        <style>
        /* Place logo ABOVE sidebar using body container */
        .rotating-logo {{
            position: fixed !important;
            top: 30px !important;
            left: 260px !important;  /* Push right so it’s OUTSIDE sidebar */
            width: 110px;
            z-index: 9999999 !important;
            pointer-events: none; /* Prevent blocking clicks */
            animation: spin 8s linear infinite;
        }}

        @keyframes spin {{
            from {{ transform: rotate(0deg); }}
            to   {{ transform: rotate(360deg); }}
        }}
        </style>

        <img class="rotating-logo" 
            src="data:image/png;base64,{logo_encoded}">
        """,
        unsafe_allow_html=True,
    )
