from __future__ import annotations

import numpy as np

from agrovision.features import (
    FEATURE_NAMES,
    _rgb_to_hsv,
    extract_features,
    leaf_mask,
    lesion_mask,
)
from agrovision.generate import render_leaf


def _mean_features(label: str, n: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = [extract_features(render_leaf(rng, label)) for _ in range(n)]
    return np.mean(vectors, axis=0)


def _value(name: str, vector: np.ndarray) -> float:
    return float(vector[FEATURE_NAMES.index(name)])


def test_feature_vector_matches_names():
    rng = np.random.default_rng(1)
    vector = extract_features(render_leaf(rng, "healthy"))
    assert vector.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(vector).all()


def test_rgb_to_hsv_primaries():
    rgb = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    hue, sat, val = _rgb_to_hsv(rgb)
    assert np.allclose(hue[0], [0.0, 1 / 3, 2 / 3], atol=1e-6)
    assert np.allclose(sat[0], [1.0, 1.0, 1.0])
    assert np.allclose(val[0], [1.0, 1.0, 1.0])


def test_leaf_mask_segments_leaf_from_background():
    rng = np.random.default_rng(2)
    image = render_leaf(rng, "healthy")
    mask = leaf_mask(np.asarray(image, dtype=float) / 255.0)
    fraction = mask.mean()
    # The leaf is an ellipse: a sizeable region, but far from the whole frame.
    assert 0.1 < fraction < 0.55


def test_healthy_is_green_with_no_lesions():
    h = _mean_features("healthy")
    assert _value("green_fraction", h) > 0.47
    assert _value("spot_coverage", h) < 0.03


def test_chlorosis_is_yellower_and_less_green_than_healthy():
    h = _mean_features("healthy")
    c = _mean_features("chlorosis")
    assert _value("yellow_index", c) > _value("yellow_index", h)
    assert _value("green_fraction", c) < _value("green_fraction", h)


def test_blight_and_rust_have_lesion_coverage():
    for label in ("blight", "rust"):
        v = _mean_features(label)
        assert _value("spot_coverage", v) > 0.05


def test_rust_is_many_small_spots_blight_is_few_large():
    rust = _mean_features("rust")
    blight = _mean_features("blight")
    # Rust: more connected components; blight: larger individual components.
    assert _value("n_spots", rust) > _value("n_spots", blight)
    assert _value("mean_spot_area", blight) > _value("mean_spot_area", rust)


def test_lesion_mask_is_near_empty_for_healthy():
    rng = np.random.default_rng(5)
    rgb = np.asarray(render_leaf(rng, "healthy"), dtype=float) / 255.0
    leaf = leaf_mask(rgb)
    lesion = lesion_mask(rgb, leaf)
    assert lesion.sum() / leaf.sum() < 0.03


def test_accepts_float_and_uint8_identically():
    rng = np.random.default_rng(6)
    image = render_leaf(rng, "rust")
    a = extract_features(image)  # uint8 path
    b = extract_features(np.asarray(image, dtype=float) / 255.0)  # float path
    assert np.allclose(a, b)
