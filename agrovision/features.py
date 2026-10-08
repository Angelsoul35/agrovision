"""Classical colour and texture features for a leaf image.

No deep network and no pretrained weights: every feature is a hand-defined,
documented statistic a plant pathologist could read off the leaf. The feature
vector turns a raw ``(H, W, 3)`` image into the small, interpretable
representation the classifier reasons over, and it is computed identically for a
synthetic sample and for a real uploaded PNG.

The pipeline is:

1. Segment the leaf from the (neutral, low-saturation) background using a
   saturation threshold — colour-based, so it survives the multiplicative
   lighting gradient the generator applies.
2. Over the leaf pixels only, compute colour statistics (channel means, a green
   fraction, yellow and brown indices, HSV-like hue/saturation stats).
3. Build a lesion mask by colour thresholding (reddish-brown over green, low
   blue), and measure its coverage and its connected-component geometry with
   :func:`scipy.ndimage.label` — this is what separates *many small* rust spots
   from *few large* blight patches.
4. Add a simple texture feature: the variance of the image gradient magnitude.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# Order matters: this is the column order of the feature vector and must stay
# stable (the classifier and its explanations index into it).
FEATURE_NAMES: list[str] = [
    "mean_r",  # mean red over the leaf (0-1)
    "mean_g",  # mean green over the leaf (0-1)
    "mean_b",  # mean blue over the leaf (0-1)
    "green_fraction",  # mean of G/(R+G+B): how green-dominant the leaf is
    "yellow_index",  # mean of min(r,g)-b: high for yellowing (chlorosis)
    "brown_index",  # mean of max(r-g, 0): red-over-green excess of lesions
    "mean_hue",  # mean hue (0-1) of leaf pixels, HSV computed from RGB
    "mean_saturation",  # mean saturation (0-1): chlorosis is desaturated
    "std_saturation",  # saturation spread: lesions raise it
    "spot_coverage",  # fraction of leaf area that is lesion-coloured
    "n_spots",  # connected lesion components, scaled (rust: many; blight: few)
    "mean_spot_area",  # mean lesion-component area / leaf area (blight: large)
    "max_spot_area",  # largest lesion-component area / leaf area
    "texture_gradient_var",  # variance of gradient magnitude: rust is speckled
]

# A leaf pixel is anything clearly more saturated than the neutral background.
# Multiplicative lighting leaves saturation unchanged, so this is robust.
_SAT_LEAF_THRESHOLD = 0.18

# Lesion colour test, on channel fractions (scale-invariant to lighting):
# reddish-brown/orange over green, with blue the smallest channel.
_LESION_RG_EXCESS = 0.045
_LESION_GB_EXCESS = 0.030


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised RGB->HSV on a float array in [0, 1]. Returns (h, s, v) in [0, 1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    chroma = mx - mn
    value = mx
    saturation = np.where(mx > 0, chroma / np.where(mx == 0, 1, mx), 0.0)

    hue = np.zeros_like(mx)
    nz = chroma > 1e-9
    # Piecewise hue by which channel is the maximum.
    rmax = nz & (mx == r)
    gmax = nz & (mx == g) & ~rmax
    bmax = nz & (mx == b) & ~rmax & ~gmax
    hue[rmax] = ((g[rmax] - b[rmax]) / chroma[rmax]) % 6.0
    hue[gmax] = (b[gmax] - r[gmax]) / chroma[gmax] + 2.0
    hue[bmax] = (r[bmax] - g[bmax]) / chroma[bmax] + 4.0
    hue = (hue / 6.0) % 1.0
    return hue, saturation, value


def leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of leaf (vs background) pixels, from a float RGB image."""
    _, saturation, _ = _rgb_to_hsv(rgb)
    mask = saturation > _SAT_LEAF_THRESHOLD
    if mask.sum() < 0.02 * mask.size:
        # Degenerate image (almost no saturated pixels): treat all as leaf so the
        # statistics are still defined rather than dividing by zero.
        mask = np.ones_like(mask)
    return mask


def lesion_mask(rgb: np.ndarray, leaf: np.ndarray) -> np.ndarray:
    """Boolean mask of lesion-coloured (necrotic / rust) leaf pixels."""
    total = rgb.sum(axis=-1) + 1e-6
    r = rgb[..., 0] / total
    g = rgb[..., 1] / total
    b = rgb[..., 2] / total
    return leaf & (r - g > _LESION_RG_EXCESS) & (g - b > _LESION_GB_EXCESS)


def extract_features(image: np.ndarray) -> np.ndarray:
    """Turn an ``(H, W, 3)`` uint8 (or float) image into the feature vector.

    The returned vector's columns are :data:`FEATURE_NAMES`, in that order.
    """
    rgb = np.asarray(image, dtype=float)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    leaf = leaf_mask(rgb)
    hue, saturation, _ = _rgb_to_hsv(rgb)

    leaf_px = rgb[leaf]  # (N, 3)
    r, g, b = leaf_px[:, 0], leaf_px[:, 1], leaf_px[:, 2]
    denom = leaf_px.sum(axis=1) + 1e-6

    mean_r, mean_g, mean_b = float(r.mean()), float(g.mean()), float(b.mean())
    green_fraction = float((g / denom).mean())
    yellow_index = float((np.minimum(r, g) - b).mean())
    brown_index = float(np.clip(r - g, 0.0, None).mean())
    mean_hue = float(hue[leaf].mean())
    mean_saturation = float(saturation[leaf].mean())
    std_saturation = float(saturation[leaf].std())

    lesion = lesion_mask(rgb, leaf)
    leaf_area = float(leaf.sum())
    spot_coverage = float(lesion.sum()) / leaf_area

    labels, n = ndimage.label(lesion)
    if n > 0:
        areas = np.bincount(labels.ravel())[1:].astype(float)
        mean_spot_area = float(areas.mean()) / leaf_area
        max_spot_area = float(areas.max()) / leaf_area
    else:
        mean_spot_area = 0.0
        max_spot_area = 0.0
    # Scale the raw component count to a comparable magnitude for the model.
    n_spots = n / 40.0

    luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    gy, gx = np.gradient(luminance)
    grad_mag = np.hypot(gx, gy)
    texture_gradient_var = float(grad_mag[leaf].var())

    return np.array(
        [
            mean_r,
            mean_g,
            mean_b,
            green_fraction,
            yellow_index,
            brown_index,
            mean_hue,
            mean_saturation,
            std_saturation,
            spot_coverage,
            n_spots,
            mean_spot_area,
            max_spot_area,
            texture_gradient_var,
        ],
        dtype=float,
    )


def extract_many(images: list[np.ndarray]) -> np.ndarray:
    """Feature matrix ``(n_images, n_features)`` for a list of images."""
    return np.vstack([extract_features(img) for img in images])
