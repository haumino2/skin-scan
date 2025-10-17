"""Tests for skin analysis map algorithms."""
import pytest
import numpy as np
import cv2

from src.pipeline.maps.redness import redness_map
from src.pipeline.maps.oiliness import oiliness_map
from src.pipeline.maps.texture import texture_map
from src.pipeline.maps.pores import pores_map
from src.pipeline.maps.blemishes import blemish_map
from src.pipeline.maps.hydration import hydration_map
from src.pipeline.maps.pigment import pigment_map


@pytest.fixture
def sample_image():
    """Create a synthetic test image."""
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    return img


@pytest.fixture
def sample_masks():
    """Create sample region masks."""
    h, w = 512, 512
    masks = {
        "forehead": np.zeros((h, w), dtype=np.uint8),
        "cheeks": np.zeros((h, w), dtype=np.uint8),
        "nose": np.zeros((h, w), dtype=np.uint8),
        "chin": np.zeros((h, w), dtype=np.uint8),
    }

    # Create circular regions
    center = (w // 2, h // 2)
    cv2.circle(masks["forehead"], (center[0], h // 4), 50, 255, -1)
    cv2.circle(masks["cheeks"], center, 80, 255, -1)
    cv2.circle(masks["nose"], (center[0], center[1] + 20), 40, 255, -1)
    cv2.circle(masks["chin"], (center[0], 3 * h // 4), 60, 255, -1)

    return masks


def test_redness_map(sample_image, sample_masks):
    """Test redness map generation."""
    result = redness_map(sample_image, sample_masks)

    assert result.shape == sample_image.shape[:2]
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_oiliness_map(sample_image, sample_masks):
    """Test oiliness map generation."""
    result = oiliness_map(sample_image, sample_masks)

    assert result.shape == sample_image.shape[:2]
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_texture_map(sample_image, sample_masks):
    """Test texture map generation."""
    result = texture_map(sample_image, sample_masks)

    assert result.shape == sample_image.shape[:2]
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_pores_map(sample_image, sample_masks):
    """Test pore detection map."""
    result = pores_map(sample_image, sample_masks)

    assert result.shape == sample_image.shape[:2]
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_blemish_map(sample_image, sample_masks):
    """Test blemish detection map."""
    result = blemish_map(sample_image, sample_masks)

    assert result.shape == sample_image.shape[:2]
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_hydration_map(sample_image, sample_masks):
    """Test hydration map generation."""
    result = hydration_map(sample_image, sample_masks)

    assert result.shape == sample_image.shape[:2]
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_pigment_map(sample_image, sample_masks):
    """Test pigmentation map."""
    result = pigment_map(sample_image, sample_masks)

    assert result.shape == sample_image.shape[:2]
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_empty_masks(sample_image):
    """Test maps with empty masks."""
    empty_masks = {}

    result = redness_map(sample_image, empty_masks)
    assert np.all(result == 0.0)
