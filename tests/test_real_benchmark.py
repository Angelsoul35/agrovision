"""Tests for the real-image (Beans) benchmark.

They prove the vendored data is the real Beans set, that the classical features are
*well fit* on real images (clearly above chance, small train/test gap), and that an
unregularised forest overfits more than the regularised logistic model.
"""

from __future__ import annotations

import numpy as np
import pytest

from agrovision.real_benchmark import run_benchmark
from agrovision.real_data import load_beans


@pytest.fixture(scope="module")
def beans_data():
    # Load and feature-extract the 180 real images once for the whole module.
    return load_beans()


@pytest.fixture(scope="module")
def report(beans_data):
    return run_benchmark(beans_data)


def test_loads_the_real_beans_images(beans_data):
    X, y, classes = beans_data
    assert classes == ["angular_leaf_spot", "bean_rust", "healthy"]
    assert X.shape == (180, 14)  # 60 per class, 14 classical features
    assert [int((y == c).sum()) for c in range(3)] == [60, 60, 60]


def test_classical_features_are_well_fit_on_real_images(report):
    lg = report.logistic
    # Not underfit: clearly above the 0.33 three-class chance level.
    assert lg.cv_acc_mean >= 0.55, lg
    # Not fake-perfect: hand-defined features on real photos cannot reach ~1.0.
    assert lg.cv_acc_mean <= 0.85, lg
    # Not overfit: train and test accuracy stay close.
    assert lg.gap <= 0.12, lg
    assert report.verdict.startswith("well-fit")


def test_random_forest_overfits_more_than_logistic(report):
    # On 180 small images an unregularised forest memorises the training set
    # (train ~1.0) for no cross-validated gain, while the regularised logistic
    # model generalises. This is the "more capacity is not better" lesson.
    assert report.forest.train_acc >= 0.97
    assert report.forest.gap > report.logistic.gap + 0.1
    assert report.forest.cv_acc_mean <= report.logistic.cv_acc_mean + 0.03


def test_shuffled_labels_collapse_to_chance(beans_data):
    # Regression guard: if the features carried no real signal the accuracy would
    # sit at chance. Shuffling the labels must destroy the score.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, _ = beans_data
    rng = np.random.default_rng(0)
    y_shuffled = rng.permutation(y)
    pipe = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=3000, C=2.0))])
    acc = cross_val_score(
        pipe, X, y_shuffled, cv=StratifiedKFold(5, shuffle=True, random_state=0)
    ).mean()
    assert acc <= 0.45, acc  # near the 0.33 chance level, nowhere near 0.67


def test_format_report_and_main(report, beans_data, capsys, monkeypatch):
    import agrovision.real_benchmark as rb

    text = rb.format_report(report)
    assert "Makerere Beans" in text
    assert f"{report.logistic.cv_acc_mean:.3f}" in text
    # main() re-runs the benchmark; feed it the pre-loaded data so it is fast.
    monkeypatch.setattr(rb, "run_benchmark", lambda data=None: report)
    rb.main()
    assert "real-image benchmark" in capsys.readouterr().out


def test_loader_rejects_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_beans(tmp_path)  # empty dir -> no class folders


def test_fetch_beans_downloads_a_balanced_subset(tmp_path, monkeypatch):
    import json

    # Two fake rows API pages, then empty, and a fake image download.
    pages = {
        0: {
            "features": [{"name": "labels", "type": {"names": ["healthy", "bean_rust"]}}],
            "rows": [
                {"row": {"labels": 0, "image": {"src": "http://x/h.jpg"}}},
                {"row": {"labels": 1, "image": {"src": "http://x/r.jpg"}}},
            ],
        }
    }

    class _Resp:
        def __init__(self, body: bytes):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self._body

    def fake_urlopen(url, timeout=0):
        if "rows?" in url:
            offset = int(url.split("offset=")[1].split("&")[0])
            return _Resp(
                json.dumps(
                    pages.get(offset, {"rows": []}) if offset == 0 else {"rows": []}
                ).encode()
            )
        return _Resp(b"\xff\xd8\xff")  # minimal JPEG magic bytes

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    from agrovision.real_data import fetch_beans

    written = fetch_beans(n_per_class=1, beans_dir=tmp_path)
    assert written == 2
    assert (tmp_path / "healthy" / "000.jpg").exists()
    assert (tmp_path / "bean_rust" / "000.jpg").exists()
