"""Pipeline composition - orchestrates the full skin scan analysis."""
import numpy as np
from typing import Dict

from .preprocess import preprocess
from .face_mesh import FaceMeshDetector, make_region_masks
from .maps.redness import redness_map
from .maps.oiliness import oiliness_map
from .maps.texture import texture_map
from .maps.pores import pores_map
from .maps.blemishes import blemish_map
from .maps.hydration import hydration_map
from .maps.pigment import pigment_map
from .visualize import generate_all_overlays
from ..app.utils_io import encode_png_base64


def score_from_map(map_data: np.ndarray, masks: dict[str, np.ndarray]) -> float:
    """
    Compute single score from map using region-weighted average.

    Args:
        map_data: Normalized map [0, 1]
        masks: Region masks

    Returns:
        Score [0, 1]
    """
    # Create combined face mask
    face_mask = np.zeros(map_data.shape, dtype=bool)
    for region_mask in masks.values():
        face_mask |= region_mask > 0

    if not face_mask.any():
        return 0.0

    # Compute weighted average
    face_vals = map_data[face_mask]
    return float(np.mean(face_vals))


def run_scan(img: np.ndarray) -> Dict:
    """
    Run complete skin scan pipeline.

    Args:
        img: Input BGR image

    Returns:
        Dict with keys:
        - scores: dict[str, float]
        - overlays: dict[str, str] (base64 PNG)
        - regions: list[str]
    """
    # Preprocess
    img_processed = preprocess(img, max_size=1024)

    # Detect face landmarks
    detector = FaceMeshDetector()
    landmarks = detector.detect(img_processed)

    if landmarks is None:
        raise ValueError("No face detected in image")

    # Create region masks
    masks = make_region_masks(landmarks, img_processed.shape)

    if not masks:
        raise ValueError("Could not create region masks from landmarks")

    # Run all map analyses
    maps = {
        "redness": redness_map(img_processed, masks),
        "oiliness": oiliness_map(img_processed, masks),
        "texture": texture_map(img_processed, masks),
        "pores": pores_map(img_processed, masks),
        "blemishes": blemish_map(img_processed, masks),
        "hydration": hydration_map(img_processed, masks),
        "pigment": pigment_map(img_processed, masks),
    }

    # Compute scores
    scores = {name: score_from_map(map_data, masks) for name, map_data in maps.items()}

    # Generate overlay visualizations
    overlay_images = generate_all_overlays(maps, alpha=0.6)

    # Convert overlays to base64 PNG
    overlays = {
        name: encode_png_base64(overlay_rgba)
        for name, overlay_rgba in overlay_images.items()
    }

    # Get region names
    regions = list(masks.keys())

    return {
        "scores": scores,
        "overlays": overlays,
        "regions": regions,
    }
