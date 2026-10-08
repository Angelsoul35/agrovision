# ADR 0002 — Render leaves synthetically with NumPy

## Status
Accepted.

## Context
A supervised classifier needs labelled images. The offline/no-download constraint
rules out fetching a field dataset, and licensing a real one is out of scope for a
portfolio piece. We need labelled data that is reproducible and that we fully
understand.

## Decision
Render every leaf synthetically with NumPy (and Pillow only to encode PNGs at the
API edge). Each image is an ellipse leaf with veins, a class-specific colouring
(green; pale-yellow chlorosis; brown blight patches; orange-brown rust spots),
lesion size/count variation, a multiplicative lighting gradient and pixel noise.
Each image is produced from an explicit set of generative knobs seeded by a
`numpy.random.Generator`.

## Consequences
- Data is deterministic and reproducible from a seed, with exact control over
  class balance and difficulty.
- Because we control the generative parameters, we can hold whole parameter
  ranges out of training for an honest generalisation test (ADR 0003) — something
  a fixed real dataset does not give for free.
- The decisive limitation: synthetic leaves are **not** field photographs. They
  lack cluttered backgrounds, occlusion, co-occurring diseases, wet/dry and
  camera variation. The reported accuracy is therefore a benchmark on this
  synthetic distribution, explicitly not a claim about real crops. This is
  repeated wherever a number is reported, so it cannot be read out of context.
