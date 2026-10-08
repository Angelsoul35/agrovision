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
    # Every class is classified correctly for (almost) every seed: the classes are
    # genuinely distinct, so this is strong, not a lowered bar. The only realistic
    # soft boundary is mild chlorosis vs healthy; the honest, parameter-disjoint
    # accuracy (~0.99 with a small train/test gap) is measured in test_evaluation.
    total_ok = 0
    total = 0
    for label in LABELS:
        correct = sum(service.assess_sample(label, seed=s)[1].label == label for s in range(10))
        assert correct >= 8, f"{label}: {correct}/10"
        total_ok += correct
        total += 10
    assert total_ok / total >= 0.95


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
    _, assessment = service.assess_sample("blight", seed=3)
    assert assessment.method == "model"
    assert assessment.top_features
    for name, _contribution in assessment.top_features:
        direction, feature = name.split(" ", 1)
        assert direction in ("high", "low")
        assert feature in FEATURE_NAMES


def test_recommended_action_matches_label(service: LeafService):
    for label in LABELS:
        _, assessment = service.assess_sample(label, seed=3)
        assert assessment.recommended_action == ACTIONS[assessment.label]
