"""Command-line demo and entry point."""

from __future__ import annotations

import sys

from .generate import LABELS
from .service import LeafService


def demo() -> None:
    svc = LeafService()

    print("AgroVision demo\n" + "=" * 64)
    print("\nGenerating one leaf of each class and assessing it (classical CV only):\n")

    correct = 0
    for label in LABELS:
        sample, a = svc.assess_sample(label, seed=3)
        ok = a.label == sample.true_label
        correct += int(ok)
        mark = "OK " if ok else "XX "
        review = "  [NEEDS REVIEW]" if a.needs_review else ""
        print(f"{mark} true={sample.true_label:<10} -> predicted={a.label:<10} "
              f"p={a.confidence:.2f} via {a.method}{review}")
        why = ", ".join(f"{name}={value}" for name, value in a.top_features[:3])
        print(f"      why: {why}")
        print(f"      verdict: {a.recommended_action}")
        print()

    print("-" * 64)
    print(f"demo samples classified correctly: {correct}/{len(LABELS)}")
    print(
        "\nNote: these are in-distribution samples. The honest, parameter-disjoint\n"
        "accuracy is reported by `python -m agrovision.evaluation`."
    )


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "demo":
        demo()
        return 0
    print("usage: python -m agrovision.cli demo")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
