# ChestMNIST multi-label classification

Small experiment notebook for **multi-label thoracic pathology classification** on [MedMNIST](https://github.com/MedMNIST/MedMNIST) **ChestMNIST**: 28×28 grayscale chest X-ray thumbnails with **14 binary labels per image** (labels are not mutually exclusive).

This repository is for **education and research prototyping** only—not clinical or diagnostic software.

## What’s in the repo

- [`chest_classification.ipynb`](chest_classification.ipynb) — end-to-end data loading, model, training, and evaluation.

## Notebook overview

- **Data**: `medmnist.ChestMNIST` splits `train` / `val` / `test`, with augmentation on training (e.g. RandomResizedCrop, rotation, affine, light horizontal flip) and fixed normalization using dataset mean/std.
- **Imbalance**: `pos_weight` for `BCEWithLogitsLoss` derived from label frequencies (with clamping and a small manual adjustment on one class index in the notebook).
- **Model**: `torchvision.models.resnet18` trained from scratch, adapted for **1 input channel**, `maxpool` replaced with `Identity()`, and a custom MLP head (512 → 256 → 128 → 14 logits).
- **Optional**: learning-rate range test with [`torch_lr_finder`](https://github.com/davidtvs/pytorch-lr-finder).
- **Training**: `AdamW` + `OneCycleLR`; validation **mean ROC-AUC** across classes; best weights saved as `best_model.pth` when validation AUC improves.
- **Metrics helpers**: per-class accuracy, micro-averaged precision/recall/F1, per-class and mean AUC (scikit-learn).

Exact metric numbers depend on your run, hardware, and random seed; treat any numbers in saved notebook outputs as **example runs**, not guarantees.

## Prerequisites

- **Python** 3.10+ recommended.
- **Accelerator**: CUDA or Apple MPS if available; the notebook falls back to CPU. Device selection order is `cuda` → `mps` → `cpu`.

## Setup

### Option A — `requirements.txt` (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B — minimal install (matches inline notebook cells)

```bash
pip install medmnist torch torchvision matplotlib numpy scikit-learn
pip install torch_lr_finder   # only if you run the LR-finder cells
```

For interactive use, install Jupyter or use the notebook in VS Code / Cursor with a Python environment that has the packages above.

## Running

1. Activate your virtual environment.
2. Open [`chest_classification.ipynb`](chest_classification.ipynb) and run cells from top to bottom (or **Run All**).

The first run downloads ChestMNIST into the standard MedMNIST cache. Training may write **`best_model.pth`** in the notebook’s working directory when validation AUC improves.

## Credit to MedMNIST

**Credit:** Data and tooling come from the [MedMNIST](https://github.com/MedMNIST/MedMNIST) project. Thank you to the MedMNIST authors for the dataset, benchmark, and open-source tooling.

## License
MIT License. 
