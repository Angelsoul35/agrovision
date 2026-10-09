# ADR 0004 — A deterministic rule fronts the model, low confidence goes to review

## Status
Accepted.

## Context
A classifier is useful but fallible, and two different failure modes matter here.
First, the obviously healthy leaf — strongly green, no lesions — should not depend
on a learned weight vector that could drift; it is a case a human decides by a
glance and a simple rule. Second, a genuinely ambiguous leaf (the rust/blight
boundary, mild chlorosis) should not be asserted with false confidence.

## Decision
- **Rule fallback.** Before the model runs, a deterministic rule checks for the
  clearly-healthy case: if the leaf's `green_fraction` is at or above
  `AV_RULE_GREEN_FRACTION` *and* its `spot_coverage` is at or below
  `AV_RULE_MAX_SPOT_COVERAGE`, the verdict is `healthy`, reported as coming from
  the rule (confidence 1.0). The thresholds are chosen with margin: no diseased
  class in the benchmark reaches that green fraction at near-zero lesion coverage.
- **Confidence gating.** For every other leaf the model decides, and a verdict
  below `AV_CLASSIFIER_MIN_CONFIDENCE` is marked `needs_review` for an agronomist
  to confirm rather than being presented as firm.

## Consequences
- The clear case is robust and explained by a stated rule, not a weight vector.
- The ambiguous case is surfaced for a human instead of guessed, which is the
  rigorous behaviour for a tool that advises on crop treatment.
- Both thresholds are operator dials: raise the confidence threshold to send more
  to review, adjust the rule thresholds to trade rule coverage against safety.
- The rule is intentionally minimal (two features). It is a triage shortcut, not
  a second model; anything it does not clearly catch falls through to the
  classifier.
