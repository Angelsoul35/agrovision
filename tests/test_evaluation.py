from __future__ import annotations

from agrovision.evaluation import aggregate_eval, held_out_eval


def test_held_out_eval_is_realistic_and_generalises():
    report = held_out_eval(seed=7)
    # The classes overlap on purpose (rust vs blight colour/geometry, early-stage
    # lesions near healthy), so accuracy sits in a realistic band, NOT at ~1.0.
    assert report.n_test > 0
    # Not underfit: well above the 0.25 four-class chance level — the features
    # genuinely separate the classes.
    assert report.accuracy >= 0.80
    assert report.macro_f1 >= 0.80
    # Not a toy: if the overlapping confounders were removed the accuracy would
    # jump back toward 1.0, so this upper bound is the regression guard.
    assert report.accuracy <= 0.95
    # Not overfit: parameter-disjoint train and test scores stay close.
    assert abs(report.gap) <= 0.10


def test_held_out_str_reports_metrics():
    text = str(held_out_eval(seed=1))
    assert "accuracy" in text
    assert "macro-F1" in text
    assert "train-test gap" in text


def test_aggregate_eval_runs_over_seeds():
    text = aggregate_eval(seeds=range(2))
    assert "mean+/-std over 2 seeds" in text
    assert "macro-F1" in text
    assert "train-test gap" in text
