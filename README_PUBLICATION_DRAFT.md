# SpiLiFormer: Event-Based Visual Classification and Compression

> Independent research reproduction project — work in progress. This is not an official release of the paper authors' code or weights, and exact reproduction has not yet been established.

## Research objective

This project studies **Event-Based Visual Classification and Compression: Accuracy, Efficiency, and Generalization**.

Goals:
1. Reproduce the training and evaluation protocol reported in the SpiLiFormer paper.
2. Audit the implementation against the paper and available official source code.
3. Compare fixed Beta weighting with the Beta scheduling behavior found in the official source-code path.
4. Evaluate accuracy, generalization, and computational efficiency.
5. Perform targeted optimization only after the baseline is validated.

Paper: https://arxiv.org/abs/2503.15986

## Current experiments

- **Experiment 1 — Fixed Beta:** Beta kept constant at 0.5. Values previously reported in the project discussion are provisional until checked against the original run log and checkpoint.
- **Experiment 2 — Scheduled Beta:** Beta increased from 0.5 to 0.8. Its final checkpoint and epoch-129 test log were inspected.

### Verified Experiment 2 results

| Metric | Value |
|---|---:|
| Best test Top-1 | 79.7% |
| Top-5 at best Top-1 | 96.0% |
| Best checkpoint epoch index | 122 |
| Beta at best checkpoint | 0.783846 |
| Final-epoch Top-1 | 78.1% |
| Final-epoch Top-5 | 96.2% |
| Final-epoch test loss | 0.9330589 |
| Last completed epoch index | 129 |
| Completed epochs | 130 |
| Beta scheduler | 0.5 → 0.8 |

See `docs/RESULTS.md` and `docs/results.csv`.

## Current training configuration

| Setting | Value |
|---|---|
| Dataset representation | Prepared CIFAR10-DVS, 16 time steps |
| Model entry in training arguments | `SEWResNet` |
| Classes | 10 |
| Batch size | 16 |
| Simulation steps (`T`) | 16 |
| Optimizer | AdamW |
| Learning rate | 0.005 |
| Weight decay | 0.06 |
| LR schedule | Cosine |
| Warmup / cooldown | 10 / 10 epochs |
| Mixup | 0.5 |
| Label smoothing | 0.1 |
| AMP | Enabled |
| Hardware | NVIDIA A100-SXM4-80GB MIG 1g.20gb |
| Software | PyTorch 1.12.1+cu116, CUDA 11.6, CuPy 11.4.0, timm 0.6.12 |

### Epoch Configuration Explanation

The CLI specifies `epochs=120`, which is passed to the CosineLRScheduler as `t_initial=120`.

In the inspected timm 0.6.12 implementation, the effective training duration is calculated as:

    effective_epochs = scheduler.get_cycle_length() + cooldown_epochs

With a cosine cycle of 120 epochs and 10 cooldown epochs:

    effective_epochs = 120 + 10 = 130

Therefore, training completed epoch indices 0–129.

The 10 warmup epochs are included within the 120-epoch cosine schedule; they are not added separately.

The Beta scheduler uses the effective epoch count returned by create_scheduler(), explaining its total_epochs=130 and final Beta value of 0.8.

**Version note:** The upstream factory.py contains a comment referencing timm==0.5.4, whereas the inspected HPC environment uses timm==0.6.12. This version difference is documented for reproducibility transparency.

## Beta-weighted loss

The inspected implementation combines two classification losses:

```python
loss = (1 - beta) * criterion(x1, target) + beta * criterion(output, target)
```

Experiment 1 uses fixed `beta=0.5`; Experiment 2 schedules Beta from 0.5 to 0.8.

## Dataset and responsible sharing

The prepared CIFAR10-DVS dataset is not included. Dataset source, split manifest, and preprocessing provenance must be documented before claiming faithful reproduction. Do not redistribute data unless its license explicitly permits it.

Do not upload SSH keys, access tokens, passwords, HPC credentials, or unredacted sensitive logs.

## Checkpoints

Host model weights and optional full training checkpoints on Hugging Face. Keep source code and small result summaries on GitHub.

Hugging Face model link: https://huggingface.co/mustaphaelkady/spiliformer-cifar10dvs-reproduction

## Research integrity

This repository distinguishes checkpoint/log-verified results from previously reported values and open audit items. No result is presented as an official SpiLiFormer result.
