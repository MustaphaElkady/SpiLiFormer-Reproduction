---
language:
- en
library_name: pytorch
tags:
- event-based-vision
- spiking-neural-networks
- cifar10-dvs
- research-reproduction
- research-in-progress
---

# SpiLiFormer CIFAR10-DVS Reproduction — Experimental Checkpoints

## Model status

Experimental checkpoints from an independent research reproduction study inspired by the SpiLiFormer paper. These are **not official weights** from the paper authors, and this work is not yet a verified exact reproduction.

**Project:** Event-Based Visual Classification and Compression: Accuracy, Efficiency, and Generalization.

**Paper:** https://arxiv.org/abs/2503.15986

**Source code:** `<GITHUB_REPOSITORY_URL>` (replace after publication).

## Research goals
1. Reproduce the paper's event-based visual classification protocol.
2. Audit implementation against the paper and available official source code.
3. Compare fixed Beta weighting with scheduled Beta weighting.
4. Evaluate accuracy, generalization, and computational efficiency.
5. Optimize only after validating the baseline.

## Verified Scheduled-Beta checkpoint

- Experiment: `official-cifar10dvs-t16-beta-v1`
- Model argument in training code: `SEWResNet`
- Dataset representation: prepared CIFAR10-DVS, 16 time steps
- Best test Top-1: **79.7%**
- Top-5 at best Top-1: **96.0%**
- Final-epoch Top-1: **78.1%**
- Final-epoch Top-5: **96.2%**
- Final-epoch test loss: **0.9330589**
- Beta schedule: **0.5 → 0.8**
- Scheduler steps: **130**
- Last completed epoch index: **129**

## Training configuration

| Setting | Value |
|---|---|
| Batch size | 16 |
| Time steps (`T`) | 16 |
| Optimizer | AdamW |
| Learning rate | 0.005 |
| Weight decay | 0.06 |
| LR schedule | Cosine |
| Warmup / cooldown | 10 / 10 epochs |
| Mixup | 0.5 |
| Label smoothing | 0.1 |
| AMP | Enabled |
| Hardware | NVIDIA A100-SXM4-80GB MIG 1g.20gb |
| PyTorch | 1.12.1+cu116 |
| CUDA | 11.6 |
| CuPy | 11.4.0 |

### Epoch-count note
CLI arguments displayed `--epochs=120`, but the run completed indices 0–129 and Beta stored 130 steps. This discrepancy remains under source-level reproducibility review.

## Intended use and limitations
For research inspection and reproduction only. Not a production-ready classifier and not official SpiLiFormer weights. Dataset provenance/split, architecture mapping, preprocessing, and evaluation protocol remain under audit. The dataset is not redistributed.

## Loading
These are custom PyTorch training artifacts, not a Transformers model. Load using the matching project model implementation and configuration. The repository should include a state-dict export and manifest; a generic `from_pretrained` interface is not currently provided.

## Citation
Cite the original paper when using the underlying method. Cite this repository as an independent reproduction effort after its public release URLs and version are finalized.

## Research integrity
Only metrics supported by a final log or checkpoint are marked verified. Previously reported Fixed-Beta metrics must be rechecked against original artifacts before being used as verified comparison results.
