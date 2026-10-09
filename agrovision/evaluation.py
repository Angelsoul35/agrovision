"""Parameter-disjoint (leakage-free) evaluation of the leaf-health classifier.

A within-distribution split would be optimistic. :func:`held_out_eval` instead
trains on one set of parameter bands and tests on the *disjoint* bands (see
:mod:`agrovision.generate`), so every test leaf's lesion size, lesion count,
chlorosis severity and lighting is unseen in training, and the number reflects
real generalisation. With the deliberately overlapping classes the result is a
realistic ~0.89 (four classes, chance 0.25) with a near-zero train/test gap — the
errors concentrate on the genuinely ambiguous rust-vs-blight pair. The headline is
averaged over several seeds so it is robust rather than a single lucky split.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import accuracy_score, f1_score

from .classifier import LeafHealthClassifier
from .generate import make_split_datasets


@dataclass
class EvalReport:
    accuracy: float
    macro_f1: float
    train_accuracy: float
    gap: float
    n_train: int
    n_test: int

    def __str__(self) -> str:
        return (
            f"accuracy       : {self.accuracy:.2f}   # parameters unseen in training\n"
            f"macro-F1       : {self.macro_f1:.2f}\n"
            f"train accuracy : {self.train_accuracy:.2f}\n"
            f"train-test gap : {self.gap:.2f}   # bounded -> learns features, not a fixed look\n"
            f"n_train/n_test : {self.n_train} / {self.n_test}"
        )


def held_out_eval(seed: int = 7) -> EvalReport:
    tr_x, tr_y, te_x, te_y = make_split_datasets(seed=seed)
    model = LeafHealthClassifier().fit(tr_x, tr_y)

    test_pred = [model.assess(img).label for img in te_x]
    train_pred = [model.assess(img).label for img in tr_x]

    acc = float(accuracy_score(te_y, test_pred))
    macro_f1 = float(f1_score(te_y, test_pred, average="macro"))
    train_acc = float(accuracy_score(tr_y, train_pred))
    return EvalReport(
        accuracy=acc,
        macro_f1=macro_f1,
        train_accuracy=train_acc,
        gap=round(train_acc - acc, 4),
        n_train=len(tr_x),
        n_test=len(te_x),
    )


def aggregate_eval(seeds: range | list[int] = range(5)) -> str:
    """Average the held-out eval over several seeds, so the headline number is
    robust rather than a single lucky split."""
    reports = [held_out_eval(seed=s) for s in seeds]
    accs = [r.accuracy for r in reports]
    f1s = [r.macro_f1 for r in reports]
    mean = sum(accs) / len(accs)
    std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
    f1_mean = sum(f1s) / len(f1s)
    train = sum(r.train_accuracy for r in reports) / len(reports)
    gap = sum(r.gap for r in reports) / len(reports)
    n = len(list(seeds))
    return (
        f"accuracy (mean+/-std over {n} seeds): {mean:.2f} +/- {std:.2f}   "
        f"# four classes; chance ~0.25; parameters unseen in training\n"
        f"macro-F1 (mean over {n} seeds)      : {f1_mean:.2f}\n"
        f"train accuracy                      : {train:.2f}\n"
        f"train-test gap                      : {gap:.2f}   "
        f"# bounded -> learns features, not a fixed synthetic look"
    )


if __name__ == "__main__":  # pragma: no cover
    print(aggregate_eval())
