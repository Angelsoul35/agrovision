"""An explainable leaf-health classifier: feature vector -> health class.

A ``StandardScaler`` feeds a multinomial logistic regression over the classical
colour/texture features. Linear by design, so every verdict comes with the
features that drove it (standardised feature value times the class weight) — an
agronomist can see *why* a leaf was called ``rust`` and overrule it.

On top of the model sits a deterministic **rule fallback** for the unambiguous
case: a leaf that is strongly green-dominant with essentially no lesion coverage
is reported ``healthy`` directly, without consulting the model. This mirrors how
a human triages — the obvious healthy leaf needs no classifier — and makes the
clear cases robust and explainable by a stated rule rather than a weight vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, extract_features, extract_many
from .generate import make_dataset

# Plain-language verdict and the recommended agronomic action per class.
ACTIONS: dict[str, str] = {
    "healthy": "leaf appears healthy; no action needed, continue routine monitoring",
    "chlorosis": (
        "chlorosis (yellowing) detected; check nitrogen/iron nutrition, soil pH "
        "and watering before it spreads"
    ),
    "blight": (
        "blight (necrotic patches) detected; remove affected leaves, improve "
        "drainage and consider a protective fungicide"
    ),
    "rust": (
        "rust detected; consider a targeted fungicide and improve air "
        "circulation to limit spore spread"
    ),
}


@dataclass
class Assessment:
    label: str
    confidence: float
    method: str  # "rule" or "model"
    needs_review: bool
    recommended_action: str
    top_features: list[tuple[str, float]]


class LeafHealthClassifier:
    def __init__(
        self,
        min_confidence: float = 0.55,
        rule_green_fraction: float = 0.47,
        rule_max_spot_coverage: float = 0.02,
    ) -> None:
        self.min_confidence = min_confidence
        self.rule_green_fraction = rule_green_fraction
        self.rule_max_spot_coverage = rule_max_spot_coverage
        self._pipeline: Pipeline | None = None

    def fit(
        self, images: list[np.ndarray] | None = None, labels: list[str] | None = None
    ) -> LeafHealthClassifier:
        if images is None or labels is None:
            images, labels = make_dataset()
        x = extract_many(images)
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, C=2.0)),
            ]
        )
        pipeline.fit(x, labels)
        self._pipeline = pipeline
        return self

    def _require_fitted(self) -> Pipeline:
        if self._pipeline is None:
            raise RuntimeError("classifier is not fitted; call fit() first")
        return self._pipeline

    def _green_fraction(self, feats: np.ndarray) -> float:
        return float(feats[FEATURE_NAMES.index("green_fraction")])

    def _spot_coverage(self, feats: np.ndarray) -> float:
        return float(feats[FEATURE_NAMES.index("spot_coverage")])

    def assess(self, image: np.ndarray) -> Assessment:
        pipeline = self._require_fitted()
        feats = extract_features(image)

        # Deterministic rule fallback for the obviously healthy leaf.
        if (
            self._green_fraction(feats) >= self.rule_green_fraction
            and self._spot_coverage(feats) <= self.rule_max_spot_coverage
        ):
            return Assessment(
                label="healthy",
                confidence=1.0,
                method="rule",
                needs_review=False,
                recommended_action=ACTIONS["healthy"],
                top_features=[
                    ("green_fraction", round(self._green_fraction(feats), 4)),
                    ("spot_coverage", round(self._spot_coverage(feats), 4)),
                ],
            )

        clf: LogisticRegression = pipeline.named_steps["clf"]
        proba = pipeline.predict_proba([feats])[0]
        best = int(np.argmax(proba))
        label = str(clf.classes_[best])
        confidence = float(proba[best])
        return Assessment(
            label=label,
            confidence=confidence,
            method="model",
            needs_review=confidence < self.min_confidence,
            recommended_action=ACTIONS.get(label, "unknown class"),
            top_features=self._explain(pipeline, feats, best),
        )

    def _explain(
        self, pipeline: Pipeline, feats: np.ndarray, class_index: int, k: int = 4
    ) -> list[tuple[str, float]]:
        """Per-instance feature contributions to the predicted class.

        Each contribution is the standardised feature value times that feature's
        weight for the class. The feature name is prefixed with ``high``/``low``
        (whether the value was above or below the training average) so the reason
        reads correctly even when it is a *low* value that supports the class
        (e.g. ``low spot_coverage`` supports ``healthy``).
        """
        scaler: StandardScaler = pipeline.named_steps["scale"]
        clf: LogisticRegression = pipeline.named_steps["clf"]
        scaled = scaler.transform([feats])[0]
        coef = clf.coef_[class_index] if clf.coef_.shape[0] > 1 else clf.coef_[0]
        contributions = scaled * coef
        order = np.argsort(contributions)[::-1]
        out: list[tuple[str, float]] = []
        for idx in order[:k]:
            if contributions[idx] <= 0:
                break
            direction = "high" if scaled[idx] >= 0 else "low"
            out.append((f"{direction} {FEATURE_NAMES[idx]}", round(float(contributions[idx]), 4)))
        return out
