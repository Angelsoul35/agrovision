# ADR 0001 — Classical computer vision, not a pretrained CNN

## Status
Accepted.

## Context
The task is to assess crop-leaf health from an image. The obvious high-accuracy
route is a convolutional network, typically fine-tuned from pretrained weights.
But the project has a hard constraint — fully offline, deterministic, no
downloads, no network — and a portfolio goal: demonstrate understanding, not
import it. A pretrained model would also make the headline accuracy a property of
someone else's dataset, not of this work.

## Decision
Use only **classical computer vision**: hand-defined colour and texture features
(channel statistics, a green fraction, yellow/brown indices, HSV hue/saturation,
lesion coverage, connected-component geometry, gradient-variance texture) feeding
a small linear classifier. No neural network, no pretrained weights, no network
access at any point.

## Consequences
- Every step is inspectable: the leaf is segmented by a stated rule, each feature
  has a documented meaning, and the linear model explains each verdict by feature
  contribution.
- Accuracy on *real* leaves will be lower than a good CNN's. That is accepted and
  stated: the leakage-free contribution is a transparent pipeline and a trustworthy
  evaluation, with a CNN on real imagery named as the production path.
- The whole system runs in seconds on CPU with no assets to fetch, so it is
  trivially reproducible from a seed.
