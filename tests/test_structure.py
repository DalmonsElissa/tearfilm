"""Structural branch and physical read-out."""

from __future__ import annotations

import pytest
import torch

from tearfilm_persistence.structure.meniscus import (
    AXIAL_UM,
    MeniscusSegmenter,
    PixelGeometry,
    StructuralBranch,
    read_geometry,
)


def test_geometry_converts_micrometres_to_millimetres() -> None:
    geometry = PixelGeometry()
    assert geometry.axial_mm == pytest.approx(AXIAL_UM / 1000.0)
    assert geometry.pixel_area_mm2 == pytest.approx((AXIAL_UM / 1000.0) ** 2)


def test_geometry_rejects_a_zero_pitch() -> None:
    with pytest.raises(ValueError):
        PixelGeometry(axial_um=0.0).validated()


def test_segmenter_returns_a_full_resolution_map() -> None:
    segmenter = MeniscusSegmenter().eval()
    frame = torch.zeros(2, 3, 32, 32)
    with torch.no_grad():
        logits = segmenter(frame)
    assert logits.shape == (2, 1, 32, 32)


def test_read_geometry_uses_the_pixel_pitch() -> None:
    probability = torch.zeros(1, 1, 16, 8)
    probability[:, :, 4:10, :] = 1.0
    reading = read_geometry(probability, (0, 16), PixelGeometry())
    assert reading.tmh_mm == pytest.approx(6 * 0.015)
    assert reading.tma_mm2 == pytest.approx(6 * 8 * 0.015 * 0.015)


def test_read_geometry_ignores_rows_outside_the_band() -> None:
    probability = torch.zeros(1, 1, 16, 8)
    probability[:, :, 0:4, :] = 1.0
    reading = read_geometry(probability, (8, 16), PixelGeometry())
    assert reading.tmh_mm == pytest.approx(0.0)


def test_branch_returns_one_value_per_sample() -> None:
    branch = StructuralBranch().eval()
    frame = torch.zeros(3, 3, 32, 32)
    with torch.no_grad():
        height, area = branch(frame, (16, 32))
    assert height.shape == (3,)
    assert area.shape == (3,)
    assert float(height.mean()) > 0
