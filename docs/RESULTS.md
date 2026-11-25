# Experimental Results

**Status:** preliminary independent reproduction study; not an official SpiLiFormer result.

## Experiment 2 — Scheduled Beta (verified)

Experiment: `official-cifar10dvs-t16-beta-v1`

The final checkpoint inspection reported:
- Last completed epoch index: `129` (130 completed epochs).
- Best test Top-1: `79.7%`.
- Top-5 at best Top-1: `96.0%`.
- Beta state: initial `0.5`, final `0.8`, `total_epochs=130`, `current_epochs=130`.
- Final stored Beta: `0.8000000000000047` (normal floating-point representation).

Epoch 129 test log:
- Top-1: `78.1%`
- Top-5: `96.2%`
- Test loss: `0.9330588867740025`

### Configuration
- Prepared CIFAR10-DVS, `T=16`
- Model argument: `SEWResNet`
- Batch size: 16
- AdamW, learning rate 0.005, weight decay 0.06
- Cosine LR schedule; warmup 10; cooldown 10
- Mixup 0.5; label smoothing 0.1; AMP enabled
- NVIDIA A100-SXM4-80GB MIG 1g.20gb
- PyTorch 1.12.1+cu116; CUDA 11.6; CuPy 11.4.0

The job was interrupted by the Slurm time limit after saving a checkpoint, then resumed from `checkpoint_latest.pth`. The resumed log confirmed Beta restoration and continuation from epoch 85.

## Experiment 1 — Fixed Beta (provisional)

Previously reported values from the project discussion:

| Metric | Reported value | Status |
|---|---:|---|
| Best test Top-1 | 82.1% | Recheck original log/checkpoint |
| Final test Top-1 | 79.6% | Recheck original log |
| Beta | 0.5 fixed | Confirm from run arguments/checkpoint |

Do not present these as fully verified until the original artifacts are inspected. Extract best epoch, best Top-5, final Top-5, final loss, configuration, and commit.

## Comparison and interpretation

The previously reported best Top-1 values are 82.1% (Fixed Beta) and 79.7% (Scheduled Beta). If Experiment 1 artifacts confirm the reported value and all other settings are equivalent, Fixed Beta is higher by 2.4 percentage points.

This is an observation about these two runs only. It does not prove that fixed Beta is generally superior. Dataset, split, initialization, random seed, implementation, and evaluation settings must be checked before attributing the difference to Beta scheduling.

## Open audit items

1. Verify Experiment 1 directly from its original log and checkpoint.
2. Explain why CLI `--epochs=120` resulted in 130 completed epochs and 130 Beta steps.
3. Verify dataset source, split manifest, and preprocessing against the paper.
4. Verify `SEWResNet` architecture/model entry against the paper.
5. Confirm evaluation protocol, metric definitions, and random seeds.
6. Record the exact Git commit used for each experiment.
