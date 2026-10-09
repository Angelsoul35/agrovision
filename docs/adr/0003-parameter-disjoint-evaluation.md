# ADR 0003 — Evaluate on held-out parameter ranges

## Status
Accepted.

## Context
A classifier trained and tested on synthetic data split *at random* can score high
for the wrong reason: train and test are drawn from the same parameter
distribution, so even a model that merely memorised the fixed synthetic look would
look good. That is the trap behind many worthless "99% accurate" plant-disease
demos. We need a number that reflects generalisation, not memorisation.

A tempting but wrong fix is to make the number smaller by *crippling the classes*
— sampling lesion colours from heavily overlapping ranges so rust and blight blur
together. That lowers the score, but it is disrigorous in the other direction: real
rust (small orange pustules) and real blight (large brown necrotic patches) are
genuinely distinguishable, so a benchmark that pretends they are not is just as
misleading as one that is trivially easy.

## Decision
Model the classes **faithfully** and make the *evaluation* rigorous through
methodology, not through artificial ambiguity:

1. **Classes reflect reality.** Each disease is rendered with its true colour and
   geometry signature (rust: many small orange pustules; blight: a few large brown
   patches; chlorosis: a global yellowing that, at mild severity, does shade toward
   healthy — a real soft boundary we keep). The classes are therefore genuinely
   separable, as they are in the field.
2. **Split by generative parameter, not at random.** Every knob (lesion size,
   lesion count, chlorosis severity, lighting) is cut into interleaved bands;
   training draws from one set and testing from the disjoint complement, so no test
   leaf's parameters were seen in training. The headline is averaged over several
   seeds so it is robust.

## Consequences
- The reported accuracy (~0.90 ± 0.03, macro-F1 ~0.89) is realistic because the classes overlap
  is genuinely separable — and it is **trustworthy because the train/test gap is
  ~0 on unseen parameters**. The gap, not the absolute score, is the trustworthy
  signal: a realistic score with a near-zero gap is real generalisation; a high score
  with a large gap would be the memorisation trap.
- The synthetic measurement is complemented by a **real-image benchmark** on the
  Makerere Beans dataset (`agrovision.real_benchmark`): the same features on real
  field photos, scored held-out. The leakage-free synthetic figure and the real
  figure together describe the system — reported against unseen parameters and real
  leaves, never a within-distribution number.
