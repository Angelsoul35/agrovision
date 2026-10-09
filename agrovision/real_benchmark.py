"""Benchmark the classical-CV approach on the REAL Beans leaf images.

Extracts the same hand-defined colour/texture features used on synthetic leaves
and trains on real field photographs, reporting whether the model is correctly fit
on real data: strong on unseen folds and with its training score close to its test
score. It compares a regularised logistic model against an unregularised random
forest to make the overfitting contrast explicit — the forest memorises 180 small
images (train accuracy ~1.0) without any cross-validated gain.

Run it with ``python -m agrovision.real_benchmark``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .real_data import load_beans

_SEED = 42
_N_SPLITS = 5


@dataclass
class ModelResult:
    name: str
    cv_acc_mean: float
    cv_acc_std: float
    train_acc: float
    test_acc: float
    gap: float = field(init=False)

    def __post_init__(self) -> None:
        self.gap = round(self.train_acc - self.test_acc, 4)


@dataclass
class RealBenchmarkReport:
    n_images: int
    classes: list[str]
    chance: float
    logistic: ModelResult
    forest: ModelResult
    confusion: list[list[int]]

    @property
    def verdict(self) -> str:
        lg = self.logistic
        if lg.cv_acc_mean <= self.chance + 0.08:
            return "underfit: barely above chance on real images"
        if lg.gap > 0.12:
            return "overfit: training accuracy runs well ahead of test"
        return "well-fit: clearly above chance with train and test accuracy close"


def _logistic() -> Pipeline:
    return Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, C=2.0))]
    )


def _forest() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=6, random_state=_SEED)),
        ]
    )


def _evaluate(name: str, pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> ModelResult:
    cv = StratifiedKFold(n_splits=_N_SPLITS, shuffle=True, random_state=_SEED)
    cv_acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=_SEED
    )
    pipe.fit(X_train, y_train)
    return ModelResult(
        name=name,
        cv_acc_mean=round(float(cv_acc.mean()), 4),
        cv_acc_std=round(float(cv_acc.std()), 4),
        train_acc=round(float(accuracy_score(y_train, pipe.predict(X_train))), 4),
        test_acc=round(float(accuracy_score(y_test, pipe.predict(X_test))), 4),
    )


def run_benchmark(
    data: tuple[np.ndarray, np.ndarray, list[str]] | None = None,
) -> RealBenchmarkReport:
    # Loading and feature-extracting 180 images is the slow step; callers (and
    # tests) can pass pre-loaded ``(X, y, classes)`` to do it only once.
    X, y, classes = data if data is not None else load_beans()
    logistic = _evaluate("Logistic (regularised)", _logistic(), X, y)
    forest = _evaluate("Random forest", _forest(), X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=_SEED
    )
    pipe = _logistic().fit(X_train, y_train)
    cm = confusion_matrix(y_test, pipe.predict(X_test))

    return RealBenchmarkReport(
        n_images=int(len(y)),
        classes=classes,
        chance=round(1.0 / len(classes), 4),
        logistic=logistic,
        forest=forest,
        confusion=cm.tolist(),
    )


def format_report(r: RealBenchmarkReport) -> str:
    lines = [
        "AgroVision — real-image benchmark (Makerere Beans)",
        "=" * 54,
        f"images: {r.n_images}   classes: {r.classes}   chance: {r.chance:.2f}",
        "",
        f"{'model':<24} {'CV accuracy':<16} {'train':>7} {'test':>7} {'gap':>7}",
    ]
    for m in (r.logistic, r.forest):
        lines.append(
            f"{m.name:<24} {m.cv_acc_mean:.3f} +/- {m.cv_acc_std:.3f}   "
            f"{m.train_acc:>7.3f} {m.test_acc:>7.3f} {m.gap:>7.3f}"
        )
    lines += ["", f"confusion (logistic, rows=true {r.classes}):"]
    lines += [f"  {row}" for row in r.confusion]
    lines += ["", f"verdict: {r.verdict}"]
    return "\n".join(lines)


def main() -> None:
    print(format_report(run_benchmark()))


if __name__ == "__main__":  # pragma: no cover
    main()
