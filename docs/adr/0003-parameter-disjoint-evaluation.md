# ADR 0003 — Evaluate on held-out parameter ranges

## Status
Accepted.

## Context
Four colour-coded leaf classes are trivially separable: a random train/test
split of the synthetic data reports ~1.0 accuracy, which is meaningless — it only
shows that train and test were drawn from the same distribution. Shipping that
number would be dishonest, and it is exactly the trap that makes "99% accurate"
plant-disease demos worthless. (A first version of this project did score ~0.99
on an interleaved split; that was the signal that the task was too easy and the
number untrustworthy.)

## Decision
Two things together make the evaluation honest:

1. **Make the classes genuinely overlap**, the way real disease does: chlorosis
   severity runs down into near-healthy; rust spot size/count and blight patch
   size overlap; lesion colour is sampled from overlapping brown↔orange ranges.
   This introduces irreducible ambiguity, so even in-distribution accuracy is
   ~0.95, not 1.0.
2. **Split by generative parameter, not at random.** Every knob (lesion size,
   lesion count, chlorosis severity, lighting) is cut into interleaved bands;
   training draws from one set of bands and testing from the disjoint complement.
   No test leaf's parameters were seen in training. The headline is averaged over
   several seeds so it is robust, not a lucky split.

## Consequences
- The reported accuracy (~0.93 ± 0.02, macro-F1 ~0.93, train ~0.95, near-zero
  train-test gap) is a real generalisation number on unseen leaf appearances.
- The bounded gap is the evidence that the model learned the features rather than
  a fixed synthetic look; a large gap would have meant the opposite.
- It is still a synthetic benchmark. The honest baseline to beat in production is
  this held-out score, measured against **real** labelled field imagery — never
  the within-distribution ~1.0.
