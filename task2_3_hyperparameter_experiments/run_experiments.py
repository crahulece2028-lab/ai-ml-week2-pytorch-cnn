"""
Task 2.3 - controlled hyperparameter experiments on the 2.2 CNN.

One factor at a time against a fixed baseline, so every row differs from the baseline
in exactly one setting. Results append to CSV after each run, so the file is readable
while the sweep is still going.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SEED = 42
DATA_DIR = Path("../task2_1_pytorch_nn/data")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")          # no CUDA here; kept so results match the notebooks

EPOCHS_SWEEP = 3                       # budget per configuration
BATCH_SIZE = 64
LR_GRID = [0.1, 0.01, 0.001]
BATCH_GRID = [16, 32, 64]
OPT_GRID = ["sgd", "adam", "rmsprop"]


# --------------------------------------------------------------------------- data
def stratified_split(labels, val_fraction=0.1, seed=SEED):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in range(10):
        idx = rng.permutation(np.where(labels.numpy() == cls)[0])
        n_val = int(round(len(idx) * val_fraction))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return sorted(train_idx), sorted(val_idx)


raw = datasets.FashionMNIST(root=DATA_DIR, train=True, download=False)
train_idx, val_idx = stratified_split(raw.targets)

pixels = raw.data[train_idx].reshape(-1).float() / 255.0
MEAN, STD = pixels.mean().item(), pixels.std().item()

# Augmentation is OFF here. The 2.2 ablation showed RandomCrop + flip cost 1.33 accuracy
# points at this budget, and with it on, training accuracy is measured on harder images
# than validation accuracy, which makes the two columns hard to compare across runs.
TRAIN_TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])

train_set = Subset(datasets.FashionMNIST(root=DATA_DIR, train=True, download=False, transform=TRAIN_TF), train_idx)
val_set = Subset(datasets.FashionMNIST(root=DATA_DIR, train=True, download=False, transform=TRAIN_TF), val_idx)

print(f"train {len(train_set):,}  val {len(val_set):,}  mean {MEAN:.4f}  std {STD:.4f}", flush=True)


def loader_for(dataset, batch_size, seed, shuffle):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g, num_workers=0)


# --------------------------------------------------------------------------- model
class SmallCNN(nn.Module):
    def __init__(self, n_classes=10, dropout=0.3, width=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1, bias=False), nn.BatchNorm2d(width),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1, bias=False), nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(width * 2 * 7 * 7, 128), nn.ReLU(inplace=True), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_optimizer(name, params, lr):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    raise ValueError(name)


# --------------------------------------------------------------------------- one run
@torch.no_grad()
def validate(model, loader):
    model.eval()
    correct, loss_sum, seen = 0, 0.0, 0
    for xb, yb in loader:
        logits = model(xb)
        loss_sum += nn.functional.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
        seen += xb.size(0)
    return loss_sum / seen, correct / seen


def run(sweep, optimizer_name, lr, batch_size, epochs, note=""):
    """Train one configuration and return its summary row plus per-epoch history."""
    torch.manual_seed(SEED)
    model = SmallCNN()
    opt = make_optimizer(optimizer_name, model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    val_loader = loader_for(val_set, 512, SEED + 999, shuffle=False)
    steps_per_epoch = -(-len(train_set) // batch_size)

    history, t_start = [], time.perf_counter()
    diverged = False

    for epoch in range(1, epochs + 1):
        # reseed per epoch so every run sees the same batch order at a given batch size
        train_loader = loader_for(train_set, batch_size, SEED + epoch, shuffle=True)
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += xb.size(0)

        train_loss = loss_sum / seen
        if not np.isfinite(train_loss):
            diverged = True
            history.append({"sweep": sweep, "optimizer": optimizer_name, "lr": lr, "batch_size": batch_size,
                            "epoch": epoch, "train_loss": float("nan"), "train_acc": float("nan"),
                            "val_loss": float("nan"), "val_acc": float("nan")})
            print(f"  {optimizer_name:<8} lr={lr:<7} bs={batch_size:<4} epoch {epoch}: DIVERGED (non-finite loss)", flush=True)
            break

        val_loss, val_acc = validate(model, val_loader)
        history.append({"sweep": sweep, "optimizer": optimizer_name, "lr": lr, "batch_size": batch_size,
                        "epoch": epoch, "train_loss": train_loss, "train_acc": correct / seen,
                        "val_loss": val_loss, "val_acc": val_acc})
        print(f"  {optimizer_name:<8} lr={lr:<7} bs={batch_size:<4} epoch {epoch}/{epochs}: "
              f"train loss {train_loss:.4f} acc {correct/seen:.4f} | val acc {val_acc:.4f}", flush=True)

    seconds = time.perf_counter() - t_start
    last = history[-1]
    row = {
        "run_id": f"{sweep}-{optimizer_name}-lr{lr}-bs{batch_size}",
        "sweep": sweep,
        "optimizer": optimizer_name,
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "final_train_loss": last["train_loss"],
        "final_train_acc": last["train_acc"],
        "final_val_loss": last["val_loss"],
        "final_val_acc": last["val_acc"],
        "best_val_acc": max(h["val_acc"] for h in history),
        "wall_clock_sec": round(seconds, 1),
        "sec_per_epoch": round(seconds / max(len(history), 1), 1),
        "diverged": diverged,
        "note": note,
    }
    return row, history


RESULTS_CSV = OUT / "experiment_results.csv"
HISTORY_CSV = OUT / "experiment_history.csv"
rows, all_history = [], []

if RESULTS_CSV.exists():
    print("resuming from existing results file", flush=True)
    rows = pd.read_csv(RESULTS_CSV).to_dict("records")
    done = {r["run_id"] for r in rows}
else:
    done = set()


def record(row, history):
    if row["run_id"] in done:
        return
    rows.append(row)
    all_history.extend(history)
    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
    pd.DataFrame(all_history).to_csv(HISTORY_CSV, index=False)
    done.add(row["run_id"])
    print(f"  -> recorded {row['run_id']}  val {row['final_val_acc']:.4f}", flush=True)


# --------------------------------------------------------------------- Sweep A
print("\n=== Sweep A: optimizer x learning rate (batch 64, 3 epochs) ===", flush=True)
for opt_name in OPT_GRID:
    for lr in LR_GRID:
        if f"A-{opt_name}-lr{lr}-bs{BATCH_SIZE}" in done:
            continue
        row, hist = run("A", opt_name, lr, BATCH_SIZE, EPOCHS_SWEEP)
        record(row, hist)

sweep_a = pd.DataFrame([r for r in rows if r["sweep"] == "A"])
if len(sweep_a):
    print("\nSweep A ranking:", flush=True)
    print(sweep_a.sort_values("final_val_acc", ascending=False)[
        ["optimizer", "learning_rate", "final_val_acc", "best_val_acc", "final_train_loss", "diverged"]
    ].to_string(index=False), flush=True)
    best = sweep_a.sort_values("final_val_acc", ascending=False).iloc[0]
    BEST_OPT, BEST_LR = str(best["optimizer"]), float(best["learning_rate"])
else:
    BEST_OPT, BEST_LR = "adam", 0.001

print(f"\nbest from sweep A: {BEST_OPT} at lr={BEST_LR}", flush=True)

# --------------------------------------------------------------------- Sweep B
print(f"\n=== Sweep B: batch size (holding {BEST_OPT} lr={BEST_LR} fixed, 3 epochs) ===", flush=True)
for bs in BATCH_GRID:
    if f"B-{BEST_OPT}-lr{BEST_LR}-bs{bs}" in done:
        continue
    row, hist = run("B", BEST_OPT, BEST_LR, bs, EPOCHS_SWEEP,
                    note=f"batch size sweep at {BEST_OPT} lr={BEST_LR}")
    record(row, hist)

sweep_b = pd.DataFrame([r for r in rows if r["sweep"] == "B"])
if len(sweep_b):
    print("\nSweep B ranking:", flush=True)
    print(sweep_b.sort_values("final_val_acc", ascending=False)[
        ["batch_size", "steps_per_epoch", "final_val_acc", "best_val_acc", "sec_per_epoch"]
    ].to_string(index=False), flush=True)
    BEST_BS = int(sweep_b.sort_values("final_val_acc", ascending=False).iloc[0]["batch_size"])
else:
    BEST_BS = BATCH_SIZE
print(f"\nbest batch size: {BEST_BS}", flush=True)

# --------------------------------------------------------------------- Sweep C
print(f"\n=== Sweep C: number of epochs ({BEST_OPT} lr={BEST_LR} bs={BEST_BS}, 12 epochs) ===", flush=True)
row, hist = run("C", BEST_OPT, BEST_LR, BEST_BS, 12, note="long run, epoch count is the variable")
record(row, hist)

# --------------------------------------------------------------------- write out
summary = {
    "seed": SEED,
    "epochs_per_config_in_sweeps_a_b": EPOCHS_SWEEP,
    "best_optimizer": BEST_OPT,
    "best_learning_rate": BEST_LR,
    "best_batch_size": BEST_BS,
    "augmentation": False,
    "batch_size": BATCH_SIZE,
    "device": str(DEVICE),
    "torch": torch.__version__,
    "n_train": len(train_set),
    "n_val": len(val_set),
}
(OUT / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\n" + "=" * 78, flush=True)
print("ALL RUNS", flush=True)
print("=" * 78, flush=True)
print(pd.DataFrame(rows)[["sweep", "optimizer", "learning_rate", "batch_size", "epochs",
                           "final_val_acc", "best_val_acc", "final_train_loss",
                           "sec_per_epoch", "diverged"]].to_string(index=False), flush=True)
print(f"\nwrote {RESULTS_CSV}, {HISTORY_CSV}, {OUT / 'experiment_summary.json'}", flush=True)
