from __future__ import annotations

from agrovision.evaluation import aggregate_eval, held_out_eval


def test_held_out_eval_is_honest():
    report = held_out_eval(seed=7)
    # Floors are conservative: the measured value is ~0.94 accuracy / ~0.94
    # macro-F1 with train ~0.97 and a small gap. The point is that it is a real
    # held-out number, not a memorised 1.0.
    assert report.n_test > 0
    assert report.accuracy >= 0.85
    assert report.macro_f1 >= 0.85
    assert report.train_accuracy >= 0.90
    assert report.gap <= 0.12
    # A near-1.0 would signal train/test leakage — assert it is NOT that.
    assert report.accuracy < 0.98


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
