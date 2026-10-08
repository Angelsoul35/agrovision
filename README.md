# AgroVision

Offline, deterministic crop-leaf health assessment from leaf images, using
**classical computer vision** — no pretrained model, no downloads, no network.
It renders synthetic leaves, extracts hand-defined colour and texture features,
and classifies each leaf as `healthy`, `chlorosis` (yellowing), `blight`
(necrotic patches) or `rust` (orange-brown spots), with an explanation and a
recommended action. The evaluation is honest: it holds out whole parameter
ranges so the reported number reflects real generalisation, not a memorised 1.0.

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Why it matters

Most "AI plant doctor" demos reach for a pretrained CNN and report ~99%
accuracy on a split of the same curated dataset — a number that says little about
a leaf photographed in a real field. AgroVision takes the opposite, honest
stance for a portfolio piece: it stays fully offline and deterministic, uses only
features a plant pathologist could name and read off the leaf, and it measures
itself on **unseen** leaf appearances. The result is a lower but *trustworthy*
number, and a system whose every verdict can be explained and overruled.

## How it works

```mermaid
flowchart TD
    IMG["leaf image<br/>(RGB PNG or synthetic sample)"] -->|segment + measure| FEAT["classical features<br/>colour: green fraction, yellow/brown<br/>index, HSV stats; lesion coverage +<br/>connected components; gradient texture"]
    FEAT --> RULE{clearly healthy?<br/>high green fraction &<br/>~zero lesion coverage}
    RULE -->|yes| HEALTHY["verdict: healthy<br/>(deterministic rule)"]
    RULE -->|no| MODEL["logistic-regression classifier<br/>class + confidence + top features"]
    MODEL --> CONF{confidence >= threshold?}
    CONF -->|no| REVIEW["manual review<br/>(agronomist confirms)"]
    CONF -->|yes| REPORT["health report<br/>class + explanation +<br/>recommended action"]
    HEALTHY --> REPORT
```

1. **Generator (`generate.py`).** Renders RGB leaves with NumPy: an ellipse leaf
   with veins, class-specific colouring, lesions (size/count varied), lighting
   and noise. Disease pattern knobs are drawn from banded ranges so an honest
   split can hold whole bands out of training.
2. **Features (`features.py`).** Segments the leaf from the background by
   saturation (robust to lighting), then measures colour statistics (green
   fraction, a yellowing index, HSV hue/saturation), lesion coverage and its
   connected-component geometry (`scipy.ndimage` — *many small* rust spots vs
   *few large* blight patches), and a gradient-variance texture feature.
3. **Classifier (`classifier.py`).** A `StandardScaler` + multinomial logistic
   regression over those features. Being linear, each verdict ships with the
   features that drove it. A deterministic **rule** short-circuits the obviously
   healthy leaf (high green fraction, no lesions) without consulting the model.
4. **Service (`service.py`).** Decodes a leaf (base64 PNG or a sample id),
   assesses it, and returns the class, confidence, explanation and the
   recommended agronomic action.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m agrovision.cli demo          # render one leaf per class, classify, show verdicts
python -m agrovision.evaluation        # the honest, parameter-disjoint accuracy
uvicorn agrovision.main:app --reload   # HTTP API
```

Example API calls:

```bash
# Generate a sample leaf and assess it (true label echoed back)
curl -s -X POST localhost:8000/assess-sample -H 'content-type: application/json' \
  -d '{"label": "rust", "seed": 3}'

# Assess your own leaf image (base64-encoded PNG)
curl -s -X POST localhost:8000/assess -H 'content-type: application/json' \
  -d "{\"image_base64\": \"$(base64 -w0 leaf.png)\"}"

# A base64 PNG sample you can feed back into /assess
curl -s localhost:8000/sample/chlorosis
```

## Configuration

Settings load from environment variables with the `AV_` prefix (see
`.env.example`):

| Variable                       | Default | Meaning                                                  |
| ------------------------------ | ------- | -------------------------------------------------------- |
| `AV_CLASSIFIER_MIN_CONFIDENCE` | `0.55`  | Below this, a verdict is routed to manual review         |
| `AV_RULE_GREEN_FRACTION`       | `0.47`  | Green fraction at/above which the healthy rule may fire  |
| `AV_RULE_MAX_SPOT_COVERAGE`    | `0.02`  | Lesion coverage at/below which the healthy rule may fire |

## Evaluation: honest generalisation, not a memorised 1.0

With four colour-coded classes, a within-distribution split would report a
meaningless ~1.0 — and it would be easy to ship that number and call it done.
AgroVision instead makes the classes genuinely overlap the way real disease does
(mild chlorosis shades into healthy; a few large rust spots look like blight) and
then evaluates on **unseen parameters**: every generation knob (lesion size,
lesion count, chlorosis severity, lighting) is split into interleaved bands, and
`agrovision.evaluation` trains on one set of bands and tests on the *disjoint*
bands. No test leaf's spot size, count, severity or lighting was seen in
training.

```
$ python -m agrovision.evaluation
  accuracy (mean+/-std over 5 seeds): 0.93 +/- 0.02   # four classes; chance ~0.25; parameters unseen in training
  macro-F1 (mean over 5 seeds)      : 0.93
  train accuracy                      : 0.95
  train-test gap                      : 0.01   # bounded -> learns features, not a fixed synthetic look
```

The command averages over several seeds on purpose: a single split can land on a
lucky 0.96, and reporting that would be the same self-deception as a random
split. What this number does and does not claim, stated plainly:

- It **does** claim the model generalises to leaf appearances it has not seen, by
  leaning on the colour/texture features rather than a fixed synthetic look — the
  train accuracy is **0.95**, not 1.0, because the classes genuinely overlap, and
  the train-test gap is near zero.
- It does **not** claim field-ready performance. This is a **synthetic
  benchmark**: real leaves bring occlusion, soil and sky background, wet/dry
  variation, mixed and co-occurring diseases, and camera differences that no
  renderer captures. The errors that remain here are the honest ones — rust and
  blight at their overlapping boundary (see `docs/estudio/04-...`). A production
  system would train a CNN on **real, labelled field imagery** and report against
  that, treating this held-out score as the baseline to beat, not the ceiling.

Because the model can be wrong, low-confidence verdicts are routed to review, and
the clearly-healthy leaf is handled by a stated deterministic rule rather than a
weight vector.

## Testing

```bash
ruff check agrovision tests
pytest --cov=agrovision --cov-report=term-missing --cov-fail-under=80
```

The suite covers the feature extraction (one assertion per class signature, the
leaf/lesion masks and the HSV conversion), the classifier (each class, the
healthy rule fallback, low-confidence review, the explanation), the honest
held-out thresholds with a bounded gap (and an assertion that the accuracy is
*not* a leaky ~1.0), the HTTP API, and the CLI demo.

## Documentation

- `docs/architecture.md` — components, data flow, and the design rationale.
- `docs/adr/` — the main design decisions and their trade-offs.
- `docs/estudio/` — a study guide in Spanish, for learning the project in depth.

## A note on scope

AgroVision is a reference implementation, not a field diagnostic tool. It
demonstrates a clean, explainable classical-CV pipeline with an honest
evaluation. A production system would be trained and evaluated on real labelled
field photographs with a convolutional network, add proper leaf segmentation for
cluttered scenes, handle multiple co-occurring diseases and severity grading, and
report against that real-world held-out set.

## License

MIT. See [LICENSE](LICENSE).
