# Datasets — real leaf images (Beans)

## `beans/<class>/*.jpg` — Makerere Beans leaf-disease dataset

A **real**, public dataset of field photographs of bean leaves, used to validate
the classical-CV approach on real images (not only on the synthetic generator).

| | |
|---|---|
| **Origin** | Makerere University AI Lab — *Beans* dataset (`AI-Lab-Makerere/beans` on Hugging Face) |
| **Classes** | `healthy`, `bean_rust` (rust), `angular_leaf_spot` (necrotic angular lesions) |
| **Images here** | 60 per class (180 total), 500×500 RGB JPEG, a balanced subset of the ~1,034-image train split |
| **Retrieved** | 2026-10-09, via the Hugging Face datasets-server |
| **License** | MIT (per the dataset card) |

The images are **vendored** (committed) so the benchmark and its tests are
deterministic and run offline. Only a balanced subset is kept to bound repository
size; `agrovision.real_data.fetch_beans()` can pull more.

### Correspondence to AgroVision's classes

AgroVision's synthetic classes are `healthy / chlorosis / blight / rust`. The Beans
dataset covers three of these in real form: `healthy` → healthy, `bean_rust` → rust,
and `angular_leaf_spot` → a necrotic-lesion disease analogous to blight. There is no
real chlorosis class here, so the real benchmark is a 3-class problem; the synthetic
generator remains the 4-class, fully-controlled evaluation.

### What the real benchmark shows

`python -m agrovision.real_benchmark` extracts the **same** classical colour/texture
features used on synthetic leaves and trains on these real photos. The result is a
working, fully interpretable classifier: **~0.67 accuracy** (3 classes, chance 0.33)
with a well-fit logistic model (small train/test gap), where every verdict still
names the features that drove it. An unregularised random forest overfits (train ≈
1.0, large gap) for no cross-validated gain — so the regularised logistic model is
the right choice here, the same lesson as on tabular data.

The ~0.70 reflects a deliberate **interpretability trade-off**, not a tuning miss:
enriching the vector to 66 features (HSV histograms, Local Binary Patterns, grey-level
co-occurrence texture) and switching to an RBF-SVM reaches only ~0.72, so the simple
readable features are not leaving much on the table. A convolutional network would
score higher by learning its own features, but it gives up the per-feature
explanation an agronomist can check and overrule — which is exactly why this project
chooses classical CV (`docs/adr/0001-classical-cv-not-pretrained-cnn.md`). This is
the interpretable, offline baseline; a CNN is the accuracy-first alternative, not a
verdict that this one is unusable.
