| Item | Value |
|---|---|
| Model | MLP 784 → 256 → 128 → 10, ReLU, dropout 0.2 |
| Parameters | 235,146 |
| Loss | `nn.CrossEntropyLoss` on raw logits |
| Optimizer | Adam, lr 1e-3, batch 64, 8 epochs |
| Final train loss | 0.2846 |
| Final test loss | 0.3295 |
| Best test accuracy | **0.8844** (epoch 8) |
| Chance baseline | 0.1000 |
