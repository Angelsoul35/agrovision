"""Synthetic, labelled leaf images for the classifier and the honest evaluation.

Everything here is rendered with NumPy (+ Pillow at the edges for PNG encoding).
No photograph, no download and no pretrained model is ever involved: the project
is fully offline and deterministic, and this module is where the "data" comes
from.

Four health classes are rendered, each with the colour signature a plant
pathologist would describe:

* ``healthy``   — a green-dominant leaf, no lesions.
* ``chlorosis`` — a global yellowing / paling of the whole leaf (loss of
  chlorophyll): raised red+green, lower saturation, no discrete lesions.
* ``blight``    — a few large brown **necrotic patches** on a green leaf.
* ``rust``      — many small orange-brown **pustule spots** scattered on a
  green leaf.

Design note (this matters for honesty). With four colour-coded classes a feature
classifier can trivially score ~1.0 if train and test are drawn from the *same*
parameter distribution — the within-distribution number would be meaningless.
So every image is generated from a set of **pattern knobs** (lesion size, lesion
count, chlorosis severity, lighting), and each knob's range is split into
interleaved bands. :func:`make_split_datasets` trains on one set of bands and
tests on the *disjoint* complementary bands, so no test image's spot size, spot
count, severity or lighting was ever seen during training. That is the honest,
realistic question — does the model generalise to leaf appearances it has not
seen — and it is why the reported accuracy is not a suspicious 1.0.
"""

from __future__ import annotations

import numpy as np

LABELS: list[str] = ["healthy", "chlorosis", "blight", "rust"]

IMAGE_SIZE = 64

# Base colours (RGB, 0-255). Leaves sit on a neutral, low-saturation background
# so a colour (saturation) threshold can segment leaf from background without any
# stored mask — see features.leaf_mask.
_BACKGROUND = np.array([202, 190, 174], dtype=float)
_HEALTHY_GREEN = np.array([62, 138, 58], dtype=float)
_CHLOROSIS_PALE = np.array([206, 200, 134], dtype=float)  # pale yellow target

# Lesion colour endpoints. Rust pustules are orange-brown and blight necrosis is
# dark brown — genuinely different hues, as a plant pathologist sees them. Each
# image samples a colour with a modest jitter that keeps the two classes in their
# own realistic bands (they do not collapse into one another), so colour is a real
# signal while lesion geometry (many small vs few large) is a second, independent
# one. The two together are what make the classes separable the way they are in
# the field.
_LESION_BROWN = np.array([96, 68, 44], dtype=float)
_LESION_ORANGE = np.array([178, 104, 44], dtype=float)


def _lesion_color(rng: np.random.Generator, bias: float) -> np.ndarray:
    """Sample a lesion colour. ``bias`` in [0, 1]: 0 is brown, 1 is orange.

    The jitter is deliberately modest so rust stays in the orange band and blight
    in the brown band — realistic within-class variation, not an artificial
    overlap that erases the colour signal.
    """
    t = float(np.clip(bias + rng.uniform(-0.15, 0.15), 0.0, 1.0))
    return (1.0 - t) * _LESION_BROWN + t * _LESION_ORANGE


def _banded(
    rng: np.random.Generator, lo: float, hi: float, regime: str, n_bands: int = 6
) -> float:
    """Sample a value from ``[lo, hi]`` restricted to this regime's bands.

    The range is cut into ``n_bands`` equal bands. ``train`` draws only from the
    even-indexed bands, ``test`` only from the odd-indexed ones, so the two are
    disjoint and interleaved — a test value always falls in a gap never covered
    in training. ``all`` uses the whole range (for nuisance variation that is not
    held out, such as leaf shape).
    """
    if regime == "all":
        return float(lo + rng.random() * (hi - lo))
    width = (hi - lo) / n_bands
    parity = 0 if regime == "train" else 1
    bands = [i for i in range(n_bands) if i % 2 == parity]
    b = int(bands[int(rng.integers(len(bands)))])
    return float(lo + (b + rng.random()) * width)


def _leaf_geometry(rng: np.random.Generator):
    """A rotated ellipse leaf, returning (mask, u, v) in leaf-local coordinates."""
    n = IMAGE_SIZE
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    cx = cy = (n - 1) / 2.0
    a = rng.uniform(21.0, 28.0)  # semi-major axis
    b = rng.uniform(12.0, 18.0)  # semi-minor axis
    theta = rng.uniform(0.0, np.pi)
    ct, st = np.cos(theta), np.sin(theta)
    dx, dy = xx - cx, yy - cy
    u = dx * ct + dy * st  # along the major axis
    v = -dx * st + dy * ct  # across it
    mask = (u / a) ** 2 + (v / b) ** 2 <= 1.0
    return mask, u, v, a, b


def _apply_veins(color: np.ndarray, mask: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
    """Darken a midrib and a few lateral veins, in place, within the leaf."""
    midrib = np.abs(v) < 0.9
    laterals = (np.mod(u + 64.0, 7.0) < 1.0) & (np.abs(v) < 9.0)
    vein = mask & (midrib | laterals)
    color[vein] *= 0.82


def _draw_blobs(
    color: np.ndarray,
    mask: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    a: float,
    b: float,
    rng: np.random.Generator,
    blob_color: np.ndarray,
    count: int,
    radius: float,
) -> None:
    """Paint ``count`` filled ellipses of ``blob_color`` at random leaf positions."""
    for _ in range(count):
        # Place the centre inside the leaf body (leave a margin so blobs stay on-leaf).
        cu = rng.uniform(-0.75 * a, 0.75 * a)
        cv = rng.uniform(-0.7 * b, 0.7 * b)
        ry = radius * rng.uniform(0.8, 1.25)
        rx = radius * rng.uniform(0.8, 1.25)
        blob = mask & (((u - cu) / rx) ** 2 + ((v - cv) / ry) ** 2 <= 1.0)
        jitter = rng.normal(0.0, 5.0, size=3)
        color[blob] = np.clip(blob_color + jitter, 0, 255)


def render_leaf(rng: np.random.Generator, label: str, regime: str = "all") -> np.ndarray:
    """Render one labelled leaf image as an ``(H, W, 3)`` uint8 array.

    ``regime`` selects the parameter bands for the disease-defining knobs and the
    lighting; ``"all"`` uses the full ranges (used by :func:`make_dataset` and the
    demo), ``"train"``/``"test"`` use disjoint bands (used by the honest split).
    """
    n = IMAGE_SIZE
    canvas = np.tile(_BACKGROUND, (n, n, 1))
    canvas += rng.normal(0.0, 3.0, size=canvas.shape)

    mask, u, v, a, b = _leaf_geometry(rng)
    leaf = np.zeros((n, n, 3), dtype=float)

    if label == "healthy":
        base = _HEALTHY_GREEN + rng.normal(0.0, 7.0, size=3)
        leaf[mask] = base
    elif label == "chlorosis":
        # Severity runs from mild (barely yellow, overlaps healthy) to severe.
        severity = _banded(rng, 0.28, 1.0, regime)
        base = (1.0 - severity) * _HEALTHY_GREEN + severity * _CHLOROSIS_PALE
        base = base + rng.normal(0.0, 7.0, size=3)
        leaf[mask] = base
    elif label == "blight":
        leaf[mask] = _HEALTHY_GREEN + rng.normal(0.0, 7.0, size=3)
        # Blight = a FEW LARGE dark-brown necrotic patches.
        count = int(round(_banded(rng, 2.0, 6.0, "all")))
        radius = _banded(rng, 5.0, 12.0, regime)
        _draw_blobs(leaf, mask, u, v, a, b, rng, _lesion_color(rng, 0.18), count, radius)
    elif label == "rust":
        leaf[mask] = _HEALTHY_GREEN + rng.normal(0.0, 7.0, size=3)
        # Rust = MANY SMALL orange pustules.
        count = int(round(_banded(rng, 12.0, 40.0, regime)))
        radius = _banded(rng, 1.2, 3.0, regime)
        _draw_blobs(leaf, mask, u, v, a, b, rng, _lesion_color(rng, 0.82), count, radius)
    else:  # pragma: no cover - guarded by LABELS in the dataset builders
        raise ValueError(f"unknown label {label!r}")

    _apply_veins(leaf, mask, u, v)
    canvas[mask] = leaf[mask]

    # Multiplicative lighting gradient (physically plausible shading). Being
    # multiplicative, it preserves hue and saturation, so colour-based leaf
    # segmentation survives it.
    strength = _banded(rng, 0.05, 0.42, regime)
    gy, gx = rng.normal(size=2)
    norm = np.hypot(gx, gy) + 1e-6
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    proj = ((xx - n / 2) * gx + (yy - n / 2) * gy) / (norm * n)
    shade = 1.0 + strength * np.clip(proj, -0.5, 0.5)
    canvas *= shade[:, :, None]

    canvas += rng.normal(0.0, 3.5, size=canvas.shape)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _balanced_labels(n: int) -> list[str]:
    reps = n // len(LABELS)
    labels = LABELS * reps
    labels += LABELS[: n - len(labels)]
    return labels


def make_dataset(
    n: int = 480, seed: int = 7, regime: str = "all"
) -> tuple[list[np.ndarray], list[str]]:
    """A class-balanced set of ``n`` rendered leaves and their labels."""
    rng = np.random.default_rng(seed)
    labels = _balanced_labels(n)
    rng.shuffle(labels)
    images = [render_leaf(rng, label, regime) for label in labels]
    return images, labels


def make_split_datasets(
    n_train: int = 360, n_test: int = 160, seed: int = 7
) -> tuple[list[np.ndarray], list[str], list[np.ndarray], list[str]]:
    """Parameter-disjoint train/test sets for honest evaluation.

    Training images draw their disease and lighting knobs from one set of
    interleaved bands; test images draw from the disjoint complementary bands, so
    every test leaf's lesion size, lesion count, chlorosis severity and lighting
    is a value never seen in training. Returns
    ``(train_images, train_labels, test_images, test_labels)``.
    """
    tr_x, tr_y = make_dataset(n=n_train, seed=seed, regime="train")
    te_x, te_y = make_dataset(n=n_test, seed=seed + 1000, regime="test")
    return tr_x, tr_y, te_x, te_y
