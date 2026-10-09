"""Load the real Beans leaf-disease images and turn them into feature vectors.

This is the real-world counterpart to :mod:`agrovision.generate`. The synthetic
generator gives a controlled, 4-class, parameter-disjoint evaluation; this module
loads real field photographs (Makerere Beans dataset, vendored under
``datasets/beans/``) and runs them through the *same* classical feature extractor,
so the real benchmark measures the same approach on real imagery.

See ``datasets/beans/README.md`` for provenance. ``fetch_beans`` can refresh or
enlarge the vendored subset from the Hugging Face datasets-server.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import extract_features

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BEANS_DIR = _REPO_ROOT / "datasets" / "beans"
_CLASSES = ("angular_leaf_spot", "bean_rust", "healthy")
_HF_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=AI-Lab-Makerere%2Fbeans"
    "&config=default&split=train&offset={offset}&length=100"
)


def load_beans(beans_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return ``(X, y, class_names)`` for the vendored real leaf images.

    ``X`` is the classical feature matrix (one row per image, columns are
    :data:`agrovision.features.FEATURE_NAMES`); ``y`` holds the class index into
    ``class_names``. Raises if the vendored images are missing.
    """
    from PIL import Image

    root = Path(beans_dir) if beans_dir is not None else _BEANS_DIR
    classes = [c for c in _CLASSES if (root / c).is_dir()]
    if not classes:
        raise FileNotFoundError(
            f"No vendored Beans images under {root}. Run "
            "`python -c 'from agrovision.real_data import fetch_beans; fetch_beans()'`."
        )

    features: list[np.ndarray] = []
    labels: list[int] = []
    for idx, cls in enumerate(classes):
        paths = sorted((root / cls).glob("*.jpg"))
        if not paths:
            raise FileNotFoundError(f"No images for class {cls!r} under {root}.")
        for path in paths:
            image = np.asarray(Image.open(path).convert("RGB"))
            features.append(extract_features(image))
            labels.append(idx)
    return np.asarray(features), np.asarray(labels), classes


def fetch_beans(n_per_class: int = 60, beans_dir: str | Path | None = None) -> int:
    """Download a balanced subset of the real Beans train split into ``datasets/beans``.

    Needs network access to the Hugging Face datasets-server. Returns the number of
    images written. Only needed to create or enlarge the vendored subset.
    """
    import json
    import urllib.request

    root = Path(beans_dir) if beans_dir is not None else _BEANS_DIR
    pool: dict[int, list[str]] = {}
    names: list[str] | None = None
    for offset in range(0, 1034, 100):
        with urllib.request.urlopen(_HF_ROWS_URL.format(offset=offset), timeout=60) as resp:
            payload = json.load(resp)
        if names is None:
            names = next(f for f in payload["features"] if f["name"] == "labels")["type"]["names"]
        for row in payload["rows"]:
            cell = row["row"]
            pool.setdefault(cell["labels"], []).append(cell["image"]["src"])

    assert names is not None
    written = 0
    for label, srcs in pool.items():
        target = root / names[label]
        target.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(srcs[:n_per_class]):
            with urllib.request.urlopen(src, timeout=60) as resp:  # noqa: S310 - signed HF URL
                (target / f"{i:03d}.jpg").write_bytes(resp.read())
            written += 1
    return written
