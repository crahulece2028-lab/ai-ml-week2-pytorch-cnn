# Week 2 — Deep Learning: PyTorch Fundamentals, CNNs & Experimentation

Complete solutions for the Week‑2 assignments of the **Skill Set Go EduTech — AI/ML** roadmap,
organized **task-wise** under `week2/`.

```
week2/
├── README.md
├── task2_1_pytorch_nn/                → 2.1 Neural Network in PyTorch
├── task2_2_cnn_classifier/            → 2.2 CNN Image Classifier
├── task2_3_hyperparameter_experiments/→ 2.3 Hyperparameter Experimentation
└── task2_4_technical_report/          → 2.4 Deep Learning Technical Report
```

| Task | Topic | Level | Location | Status |
|------|-------|-------|----------|--------|
| 2.1 | Neural Network in PyTorch | Intermediate | `task2_1_pytorch_nn/pytorch_nn_fashion_mnist.ipynb` | Done — 88.44% test acc |
| 2.2 | CNN Image Classifier | Intermediate | `task2_2_cnn_classifier/cnn_fashion_mnist.ipynb` | Done — 90.67% test acc |
| 2.3 | Hyperparameter Experimentation | Intermediate | `task2_3_hyperparameter_experiments/` | Done — 13 runs, 76 min |
| 2.4 | Deep Learning Technical Report | Intermediate | `task2_4_technical_report/technical_report_week2.pdf` | Done — 2 pages |

**Dataset:** Fashion-MNIST throughout — 60,000 train / 10,000 test, 28×28 greyscale, 10 classes.
Chosen over MNIST, which tops out near 99% with almost no effort and leaves nothing to read off
the loss curves. The 60k training set is split 54,000 / 6,000 into train and validation **per
class**; the official test set is not touched until final evaluation.

## 2.1 Neural Network in PyTorch — `task2_1_pytorch_nn/`

Feed-forward network built from scratch: `784 → 256 → 128 → 10`, ReLU, dropout 0.2,
**235,146 parameters**, `CrossEntropyLoss`, Adam, 8 epochs. Final test accuracy **0.8844**.

The notebook walks the concepts in order rather than jumping to a model:

- **Tensors** — creation, dtype promotion, `view` vs `reshape` vs `permute` (and when each one
  raises), indexing and `gather`, the broadcasting rule, `.to(DEVICE)` with a GPU path that
  activates if CUDA ever appears, and autograd — including a demonstration that gradients
  *accumulate* unless you call `zero_grad()`.
- **Custom `Dataset` / `DataLoader`** — a hand-written `FashionFlatDataset`, verified to match
  torchvision's `ToTensor + Normalize` pipeline to zero absolute difference. Shows that a
  `Dataset` returns unbatched items and the `DataLoader` is what adds the batch dimension.
- **Model** — `nn.Module` with `__init__`/`forward`, per-layer shape walkthrough, and why
  `nn.Linear`'s default init is already correct.
- **Losses** — `CrossEntropyLoss` on raw logits (including why a model-ending `Softmax` is a bug),
  `MSELoss` on one-hot targets as an ablation, and a genuine regression head on image brightness
  that overfits within two epochs.
- **Training loop** — written out inline once, then factored. Final curves, confusion matrix,
  per-class accuracy, misclassified samples, saved weights.

## 2.2 CNN Image Classifier — `task2_2_cnn_classifier/`

`2 × (Conv3×3 + BatchNorm + ReLU + MaxPool2) → Flatten → 128 → 10`, **421,738 parameters**.
Final test accuracy **0.9067** (best validation 0.9043 at epoch 8).

- **Why a CNN** — the first convolution costs 288 parameters against the MLP's 200,960, and the
  weight sharing is what buys translation robustness. Receptive field tracked stage by stage.
- **Transforms** — separate train and eval pipelines. `RandomCrop(28, padding=2)` +
  `RandomHorizontalFlip(0.5)` on train only, with the augmentation drawn out visually and a check
  that four draws from one index differ while two eval draws are identical.
- **Stratified split** — per class, so validation keeps the training class balance.
- **Training loop** — per-epoch train/validation loss and accuracy, per-epoch timing, best
  checkpoint tracking, then reverted to the best weights before the test set is touched.
- **Report** — test accuracy, per-class accuracy (worst: Shirt 0.7060), confusion matrix with
  row-normalized values, and sample predictions with confidence. The model turns out to be
  *under*confident when wrong (0.6704 vs 0.9362), the opposite of the usual failure mode.
- **Ablation** — augmentation **lost** (0.8872 vs 0.9005 at two epochs). Reported as a negative
  result rather than quietly kept, with the reasoning for why it needs a longer run to pay off.

## 2.3 Hyperparameter Experimentation — `task2_3_hyperparameter_experiments/`

13 runs, 76 minutes of CPU training, on the 2.2 CNN. A **greedy one-factor-at-a-time** search
rather than the full 81-run grid:

| Stage | Variable | Runs |
|---|---|---|
| A | optimizer × learning rate (3 × 3) | 9 |
| B | batch size (16 / 32 / 64) | 3 |
| C | epoch count (12-epoch trajectory) | 1 |

`run_experiments.py` does the training and appends to `results/experiment_results.csv` after every
run, skipping `run_id`s already present — so it is safe to interrupt and restart.
`experiment_log.ipynb` loads that log and presents it; it does not retrain anything.

What the log actually shows:

- **Optimizer choice was not the deciding factor.** The top three configurations — RMSprop 0.001,
  Adam 0.01, Adam 0.001 — span **0.12** accuracy points, a quarter of the noise floor. A tie.
- **Fragility was.** Across two orders of magnitude of learning rate: SGD varied 0.45 points,
  Adam 10.3, RMSprop 80.1. RMSprop at lr=0.1 collapsed to a uniform prediction — sitting at
  exactly 0.1000 with loss 2.3133 against `ln(10) = 2.3026` — **without tripping the divergence
  check**, because the loss never went non-finite.
- **Batch size: no defensible effect.** 1.03 points of spread, non-monotonic (batch 32 worst).
- **Epochs: the one large effect.** 3 → 12 epochs gained 1.20 points; peak 0.9218 at epoch 9.
- **A noise floor of 0.41 points** was measured from epoch-to-epoch jitter on the plateaued
  12-epoch run, and it is what stops the batch-size result being reported as a finding.
- One wall-clock outlier (4.2× on Adam lr=0.1) was investigated and **did not reproduce** in a
  controlled timing test, so the report draws no timing conclusions and says so.

## 2.4 Technical Report — `task2_4_technical_report/`

`technical_report_week2.pdf` — two pages, A4. Intro → Setup → Experiment design and results
(full 13-row table) → Key takeaways → Limitations.

`build_report.py` generates it and reads **every number from the artifacts** of tasks 2.1–2.3
(`artifacts/task2_1_summary.md`, the 2.2 checkpoint and confusion-matrix CSVs, the 2.3 result
CSVs), so the prose cannot drift from the runs it describes. Regenerate with:

```bash
cd task2_4_technical_report
python build_report.py
```

## Reproducing

```bash
cd week2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib seaborn scipy scikit-learn reportlab jupyter

# 2.1 — downloads Fashion-MNIST into task2_1_pytorch_nn/data, ~4 min
cd task2_1_pytorch_nn   && jupyter lab pytorch_nn_fashion_mnist.ipynb

# 2.2 — reuses the same data dir, ~20 min
cd ../task2_2_cnn_classifier && jupyter lab cnn_fashion_mnist.ipynb

# 2.3 — training first (~76 min), then the notebook (seconds)
cd ../task2_3_hyperparameter_experiments
python run_experiments.py && jupyter lab experiment_log.ipynb

# 2.4 — needs 2.1–2.3 artifacts present
cd ../task2_4_technical_report && python build_report.py
```

## Environment

- Python 3.14.4, PyTorch 2.14.0+cpu, torchvision 0.29.0+cpu, NumPy 2.5, pandas 3.0,
  matplotlib 3.11, seaborn 0.13, scikit-learn 1.9, reportlab 5.0
- No GPU on this machine, so everything ran on CPU with 6 threads. The `.to(DEVICE)` calls in the
  notebooks are written so the same code runs unchanged on a CUDA box.
- All notebooks are **executed with outputs embedded**; re-run them to confirm from scratch.
  The 2.3 sweep is resumable and its CSVs are checked in, so the notebook can be re-run without
  repeating the 76 minutes of training.
