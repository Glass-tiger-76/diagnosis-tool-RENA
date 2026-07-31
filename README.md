# Medical Image Classification — Transfer Learning Across Two Modalities

Two classifiers built with the same approach — ImageNet-pretrained ResNet, frozen backbone, retrained head — applied to different medical imaging data. The point of running both was to see what changes with the data and what doesn't.

**Learning project, not a diagnostic tool.** Numbers below are honest, including the failures.

| | Brain MRI | Chest X-ray |
|---|---|---|
| Classes | 4 (glioma, meningioma, notumor, pituitary) | 4 (COVID-19, normal, pneumonia, TB) |
| Train / Test | 5,600 / 1,600 | ~4,800 / 771 |
| Test accuracy | **94.0%** | **88.3%** |
| Weakest class | glioma (recall 0.82) | normal (recall 0.77) |

---

## Brain MRI

```
              precision  recall  f1   support
glioma           0.96     0.82  0.88     400
meningioma       0.90     0.96  0.93     400
notumor          0.92     1.00  0.96     400
pituitary        1.00     1.00  1.00     400
accuracy                        0.94    1600
```

```
            glioma  mening  notumor  pituit
glioma      [ 327     41      32       0  ]
meningioma  [  11    386       1       2  ]
notumor     [   0      0     400       0  ]
pituitary   [   1      0       0     399  ]
```

Pituitary is essentially solved (1.00/1.00). Glioma is the problem: 73 of 400 missed, split between meningioma (41) and notumor (32). Precision stays high at 0.96 — when the model says glioma it's right, it just doesn't say it often enough.

## Chest X-ray

```
              precision  recall  f1   support
COVID-19         0.96     0.86  0.91     106
NORMAL           0.91     0.77  0.83     234
PNEUMONIA        0.85     0.96  0.90     390
TUBERCULOSIS     0.92     0.83  0.87      41
accuracy                        0.88     771
```

53 healthy scans classified as pneumonia. The training set is pneumonia-heavy, so the model defaults to it under uncertainty — over-calling illness rather than missing it.

---

## What both models have in common

**Each has exactly one weak class, and in both cases it's the class that visually overlaps with its neighbours.** Glioma competes with meningioma and notumor; normal competes with pneumonia. Classes with distinctive presentation — pituitary tumours, TB — sit near-perfect. The models learn *distinctive* well and *similar* poorly, and that pattern held across two unrelated datasets.

Both also fail asymmetrically rather than uniformly. Overall accuracy hides this: 88% sounds acceptable until you see that 23% of healthy patients get flagged.

## Methodology notes

**Full fine-tuning overfit; head-only didn't.** An early X-ray run with ResNet50 and `layer4` unfrozen drove training loss to 0.0000 — complete memorisation — while test accuracy sat at 86.8% and normal recall at 0.59. Freezing the backbone entirely and training only the classifier head gave better generalisation with a fraction of the trainable parameters. Less capacity, better results, on datasets this size.

**Random splits leak on medical imaging.** A 15% random split of the MRI training data reported 97.98% accuracy with glioma at 0.98 — a 16-point jump over every previous run. Medical datasets contain multiple slices per patient, so random splitting puts near-identical images on both sides. Using the dataset's own held-out test folder brought it back to 94%. The inflated number was plausible enough to believe, which is what made it dangerous.

**Class weighting trades precision for recall, and it isn't free.** Upweighting COVID-19 4:1 moved its recall from 0.71 to 0.91, but precision dropped 0.98 → 0.81 and TB recall fell 0.97 → 0.90 as cases bled into the COVID column. Overall accuracy barely moved (0.9293 → 0.9304). Same headline, different model.

**Validation and test disagreed by 4.7 points** on the X-ray model (93.0% vs 88.3%), partly because the splits had different class balances. The test figure is the one reported here.

## Limitations

- Single dataset per modality; no external validation
- Normal recall of 0.77 means roughly 1 in 4 healthy chest scans is misflagged
- Glioma recall of 0.82 means roughly 1 in 5 gliomas is missed
- No calibration analysis — confidence scores are not probabilities
- 2D slices only; no 3D volume context, which is what radiologists actually use

## Files

- `train_xray.py`, `train_mri.py`
- `app.py` — Gradio demo, tabbed by scan type
- `weights/`
- NOTE:WEIGHTS WILL BE PROVIDED ON REQUEST OR WITH RELEASE LATER
- 
