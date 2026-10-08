"""Application service: trains the classifier and assesses leaf images.

It also owns the PNG <-> array edge (base64 for the HTTP API) and a small bank
of deterministic sample leaves so the API and the CLI can be exercised without
any external image.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image

from .classifier import Assessment, LeafHealthClassifier
from .config import Settings, get_settings
from .generate import LABELS, render_leaf


def encode_png(image: np.ndarray) -> str:
    """Base64-encode an ``(H, W, 3)`` uint8 array as a PNG string."""
    buffer = BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_png(data: str) -> np.ndarray:
    """Decode a base64 PNG string into an ``(H, W, 3)`` uint8 RGB array."""
    raw = base64.b64decode(data)
    with Image.open(BytesIO(raw)) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


@dataclass
class Sample:
    sample_id: str
    true_label: str
    image: np.ndarray


class LeafService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.classifier = LeafHealthClassifier(
            min_confidence=self.settings.classifier_min_confidence,
            rule_green_fraction=self.settings.rule_green_fraction,
            rule_max_spot_coverage=self.settings.rule_max_spot_coverage,
        ).fit()

    def assess_image(self, image: np.ndarray) -> Assessment:
        return self.classifier.assess(image)

    def assess_base64(self, data: str) -> Assessment:
        return self.assess_image(decode_png(data))

    def sample(self, label: str, seed: int = 0) -> Sample:
        """A deterministic in-distribution sample leaf of the given class."""
        if label not in LABELS:
            raise KeyError(f"unknown label {label!r}; expected one of {LABELS}")
        # Built-in hash() is randomised per process (PYTHONHASHSEED), which would
        # make "the same" sample differ run to run. A stable digest keeps the
        # sample bank reproducible.
        digest = hashlib.sha256(f"{label}:{seed}".encode()).hexdigest()
        rng = np.random.default_rng(int(digest[:8], 16))
        image = render_leaf(rng, label, regime="all")
        return Sample(sample_id=f"{label}:{seed}", true_label=label, image=image)

    def assess_sample(self, label: str, seed: int = 0) -> tuple[Sample, Assessment]:
        sample = self.sample(label, seed)
        return sample, self.assess_image(sample.image)
