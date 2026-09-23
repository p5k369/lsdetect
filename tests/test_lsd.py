"""The line segment detector tests."""

from __future__ import annotations

import lsdetect as lsd
import numpy as np


def edge_image(angle_deg: float, size: int = 96) -> np.ndarray:
    """A step edge through the center at the given direction."""
    ys, xs = np.mgrid[0:size, 0:size].astype(float)
    normal = np.deg2rad(angle_deg + 90.0)
    signed = (xs - size / 2) * np.cos(normal) + (ys - size / 2) * np.sin(
        normal
    )
    return np.where(signed > 0, 200.0, 50.0)


def test_a_vertical_edge_is_found_at_its_angle() -> None:
    """A clean vertical step edge yields one long vertical segment."""
    segments = lsd.detect(edge_image(90.0))
    assert segments, "the edge has to be detected"
    longest = max(segments, key=lambda s: s.length)
    assert abs(abs(longest.angle) - 90.0) < 2.0
    assert longest.length > 60


def test_an_oblique_edge_is_found_at_its_angle() -> None:
    """A 30-degree edge comes back at 30 degrees."""
    segments = lsd.detect(edge_image(30.0))
    assert segments
    longest = max(segments, key=lambda s: s.length)
    assert abs(longest.angle - 30.0) < 3.0


def test_pure_noise_yields_no_detections() -> None:
    """The a-contrario model keeps noise at about zero detections."""
    rng = np.random.default_rng(7)
    noise = rng.uniform(0.0, 255.0, size=(128, 128))
    segments = lsd.detect(noise)
    assert len(segments) <= 1


def test_a_flat_image_yields_nothing() -> None:
    """No gradients, no segments."""
    flat = np.full((64, 64), 128.0)
    assert lsd.detect(flat) == []


def test_coordinates_come_back_in_input_scale() -> None:
    """Detection at scale 0.8 still reports input coordinates."""
    image = edge_image(90.0, size=120)
    segments = lsd.detect(image, scale=0.8)
    longest = max(segments, key=lambda s: s.length)
    assert 45 < longest.x1 < 75
    assert 45 < longest.x2 < 75


def test_two_separate_bars_give_two_segments() -> None:
    """Two parallel bars are not merged into one segment."""
    image = np.full((96, 96), 40.0)
    image[:, 30:33] = 220.0
    image[:, 66:69] = 220.0
    segments = lsd.detect(image)
    verticals = [s for s in segments if abs(abs(s.angle) - 90.0) < 3.0]
    centres = sorted({round((s.x1 + s.x2) / 2 / 10) for s in verticals})
    assert len(centres) >= 2
