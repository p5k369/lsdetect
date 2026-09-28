"""The line segment detector tests."""

from __future__ import annotations

import lsdetect as lsd
import numpy as np
import pytest


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


def identity() -> tuple[float, ...]:
    """The inverse homography that maps every pixel onto itself."""
    return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def test_an_identity_warp_returns_the_image() -> None:
    """With the identity matrix every pixel lands on itself."""
    rng = np.random.default_rng(7)
    src = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
    out = lsd.warp_rgb(src, identity(), 1.0, 0.0, 0.0)
    assert np.array_equal(out, src)


def test_an_identity_warp_returns_the_16bit_image() -> None:
    """The 16-bit warp keeps every sample exactly on the identity."""
    rng = np.random.default_rng(7)
    src = rng.integers(0, 65536, (24, 32, 3), dtype=np.uint16)
    out = lsd.warp_rgb(src, identity(), 1.0, 0.0, 0.0)
    assert out.dtype == np.uint16
    assert np.array_equal(out, src)


def test_the_16bit_warp_keeps_16bit_precision() -> None:
    """Interpolated values use the full range, not an 8-bit ladder."""
    src = np.zeros((2, 4, 3), dtype=np.uint16)
    src[:, :, :] = np.array([0, 20000, 40000, 60000])[None, :, None]
    shifted = (1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    out = lsd.warp_rgb(src, shifted, 1.0, 0.0, 0.0)
    assert out[0, 0, 0] == 10000
    assert out[0, 1, 0] == 30000
    assert out[0, 2, 0] == 50000


def test_both_depths_resample_the_same_geometry() -> None:
    """The same content warps identically at 8 and at 16 bit."""
    rng = np.random.default_rng(11)
    src8 = rng.integers(0, 256, (20, 28, 3), dtype=np.uint8)
    src16 = (src8.astype(np.uint16)) * 257
    matrix = (0.98, 0.01, 0.3, -0.02, 1.01, 0.2, 0.00001, 0.00002, 1.0)
    out8 = lsd.warp_rgb(src8, matrix, 1.0, 0.0, 0.0)
    out16 = lsd.warp_rgb(src16, matrix, 1.0, 0.0, 0.0)
    diff = out8.astype(int) - (out16.astype(float) / 257).round().astype(int)
    assert np.abs(diff).max() <= 1


def test_out_of_frame_sources_stay_black_in_16bit() -> None:
    """A source outside the image leaves the output pixel at zero."""
    src = np.full((8, 8, 3), 60000, dtype=np.uint16)
    far = (1.0, 0.0, 100.0, 0.0, 1.0, 100.0, 0.0, 0.0, 1.0)
    out = lsd.warp_rgb(src, far, 1.0, 0.0, 0.0)
    assert int(out.max()) == 0


def test_a_wrong_dtype_is_refused_clearly() -> None:
    """Anything but uint8 or uint16 raises instead of guessing."""
    src = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(TypeError, match="uint8 or uint16"):
        lsd.warp_rgb(src, identity(), 1.0, 0.0, 0.0)


def test_an_8bit_image_detects_like_a_float_one() -> None:
    """uint8 input finds the same edge as the float version."""
    floats = edge_image(90.0)
    bytes_ = floats.astype(np.uint8)
    found = lsd.detect(bytes_)
    assert found, "the edge has to be detected"
    longest = max(found, key=lambda s: s.length)
    assert abs(abs(longest.angle) - 90.0) < 2.0


def test_a_16bit_image_detects_like_an_8bit_one() -> None:
    """uint16 is read on the same gray scale, so thresholds hold."""
    floats = edge_image(90.0)
    words = (floats * 257).astype(np.uint16)
    found = lsd.detect(words)
    assert found, "the edge has to be detected"
    longest = max(found, key=lambda s: s.length)
    assert abs(abs(longest.angle) - 90.0) < 2.0


def test_detect_refuses_an_unknown_dtype() -> None:
    """Integer types the detector does not know raise clearly."""
    with pytest.raises(TypeError, match="uint8"):
        lsd.detect(np.zeros((8, 8), dtype=np.int32))


def test_warp_refuses_an_alpha_channel() -> None:
    """A (h, w, 4) array is not silently misread as RGB."""
    src = np.zeros((4, 4, 4), dtype=np.uint8)
    with pytest.raises(TypeError, match="uint8 or uint16"):
        lsd.warp_rgb(src, identity(), 1.0, 0.0, 0.0)
