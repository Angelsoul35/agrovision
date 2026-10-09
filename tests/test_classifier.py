from __future__ import annotations

import numpy as np
import pytest

from agrovision.classifier import ACTIONS, LeafHealthClassifier
from agrovision.features import FEATURE_NAMES
from agrovision.generate import LABELS, render_leaf
from agrovision.service import LeafService


def test_unfitted_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError):
        LeafHealthClassifier().assess(render_leaf(rng, "healthy"))


def test_each_class_classified_in_distribution(service: LeafService):
    # Healthy and chlorosis carry no lesion ambiguity, so they are near-perfect.
    # Rust and blight deliberately overlap (colour band and lesion geometry) and
    # early-stage lesions look healthy, so each is only recovered most of the time.
    # The leakage-free headline accuracy is measured in test_evaluation.
    per_class = {
        label: sum(service.assess_sample(label, seed=s)[1].label == label for s in range(10))
        for label in LABELS
    }
    assert per_class["healthy"] >= 9, per_class
    assert per_class["chlorosis"] >= 9, per_class
    assert per_class["blight"] >= 5, per_class
    assert per_class["rust"] >= 5, per_class
    # The overlapping disease pair is genuinely harder than the clear pair; if the
    # confounders were removed every class would be ~perfect and this would fail.
    assert per_class["blight"] + per_class["rust"] < per_class["healthy"] + per_class["chlorosis"]
    assert sum(per_class.values()) / (4 * 10) >= 0.78, per_class


def test_healthy_rule_fallback_fires(service: LeafService):
    _, assessment = service.assess_sample("healthy", seed=4)
    assert assessment.label == "healthy"
    assert assessment.method == "rule"
    assert assessment.confidence == 1.0
    assert not assessment.needs_review


def test_low_confidence_is_flagged_for_review():
    # An impossibly high threshold forces even a confident model verdict to review.
    clf = LeafHealthClassifier(min_confidence=0.999).fit()
    rng = np.random.default_rng(3)
    assessment = clf.assess(render_leaf(rng, "rust"))
    assert assessment.method == "model"
    assert assessment.needs_review


def test_explanation_is_populated_and_named(service: LeafService):
    # Use a blight leaf the model actually classifies (an early-stage one can fall
    # to the healthy rule), then check the explanation is well-formed.
    for seed in range(20):
        _, assessment = service.assess_sample("blight", seed=seed)
        if assessment.method == "model":
            break
    else:  # pragma: no cover - a model-classified blight always exists in 0..19
        raise AssertionError("no model-classified blight sample found")
    assert assessment.top_features
    for name, _contribution in assessment.top_features:
        direction, feature = name.split(" ", 1)
        assert direction in ("high", "low")
        assert feature in FEATURE_NAMES


def test_recommended_action_matches_label(service: LeafService):
    for label in LABELS:
        _, assessment = service.assess_sample(label, seed=3)
        assert assessment.recommended_action == ACTIONS[assessment.label]
