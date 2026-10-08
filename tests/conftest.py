from __future__ import annotations

import numpy as np
import pytest

from agrovision.generate import LABELS, render_leaf
from agrovision.service import LeafService


@pytest.fixture(scope="session")
def service() -> LeafService:
    # Fitting trains the feature-based classifier once for the whole session.
    return LeafService()


@pytest.fixture(scope="session")
def samples() -> dict[str, np.ndarray]:
    """One deterministic in-distribution image per class."""
    rng = np.random.default_rng(123)
    return {label: render_leaf(rng, label) for label in LABELS}
