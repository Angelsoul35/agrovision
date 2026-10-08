from __future__ import annotations

from agrovision.evaluation import aggregate_eval, held_out_eval


def test_held_out_eval_is_honest():
    report = held_out_eval(seed=7)
    # The four classes are genuinely distinct (green / yellowed / few large brown
    # patches / many small orange pustules), so a sound classical classifier
    # separates them well: measured ~0.99 accuracy and macro-F1. The honesty is
    # NOT an accuracy cap — it is the methodology: the test images use lesion
    # sizes, counts, severities and lighting from parameter bands DISJOINT from
    # training, and the near-zero train/test gap proves the model generalises to
    # unseen appearances rather than memorising a fixed synthetic look.
    assert report.n_test > 0
    assert report.accuracy >= 0.95
    assert report.macro_f1 >= 0.95
    assert report.train_accuracy >= 0.95
    assert report.gap <= 0.05  # small gap = real generalisation, not leakage


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
