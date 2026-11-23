# SpiLiFormer Reproduction

**Research project:** *Event-Based Visual Classification and Compression: Accuracy, Efficiency, and Generalization*

This repository documents the current reproduction of **SpiLiFormer** on the **CIFAR10-DVS** event-based vision dataset.

The purpose of this snapshot is to record the current implementation, training configuration, computing environment, measured result, and known differences from the original paper. The current project stage is reproduction and verification. Optimization is outside the scope of this version.

> **Status:** Initial reproduction snapshot. The current result is documented, but complete equivalence with every experimental detail in the paper has not yet been established.

## 1. Project and Upstream Reference

SpiLiFormer is a spiking-neural-network-based architecture for event-based visual classification.

- **Original implementation:** https://github.com/KirinZheng/SpiLiFormer
- **Reference commit used for this reproduction:** `23c6bf9fa927ae1edc593df56193ad46a43d7034`
- **This reproduction repository:** https://github.com/MustaphaElkady/SpiLiFormer-Reproduction
- **Original paper:** https://arxiv.org/abs/2503.15986
- **ICCV 2025 paper page:** https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_SpiLiFormer_Enhancing_Spiking_Transformers_with_Lateral_Inhibition_ICCV_2025_paper.html

The upstream repository and reference commit are recorded in [`UPSTREAM.txt`](UPSTREAM.txt).

## 2. Dataset and Input Preparation

The current experiment uses the original **CIFAR10-DVS** recordings.

| Property | Current preparation |
|---|---|
| Dataset | CIFAR10-DVS |
| Number of classes | 10 |
| Original recordings | 1,000 per class |
| Total recordings | 10,000 |
| Input representation | Frame-based |
| Number of frames (`T`) | 16 |
| Frame tensor shape | `(16, 2, 128, 128)` |
| Split configuration | `split_by="number"` |

The local preparation workflow converts the original `.aedat` recordings into the frame representation used by the training pipeline. The preparation code checks the expected number of recordings and validates the generated frame tensors.

**Dataset files are not included in this repository.** Users must obtain CIFAR10-DVS from its original source and configure local data paths before running the preparation and training scripts.

The table describes the current local preparation. Exact equivalence of every preprocessing and split detail with the paper's protocol has not yet been fully verified.

## 3. Model and Training Configuration

The local training code calls the model factory as follows:

```python
model = create_model(
    "SpiLiFormer",
    pretrained=False,
    drop_rate=0.,
    drop_path_rate=0.1,
    drop_block_rate=None,
    T=args.T,
    aug=args.aug,
)
```

The recorded run used this configuration:

| Setting | Value |
|---|---:|
| Model factory name | `SpiLiFormer` |
| Temporal steps (`T`) | 16 |
| Batch size | 16 |
| Data-loader workers | 4 |
| Optimizer | AdamW |
| Initial learning rate | 0.005 |
| Weight decay | 0.06 |
| Optimizer epsilon | `1e-8` |
| Scheduler | Cosine |
| `--epochs` argument | 120 |
| Warmup epochs | 10 |
| Cooldown epochs | 10 |
| Minimum learning rate | `1e-5` |
| Warmup learning rate | `1e-5` |
| Label smoothing | 0.1 |
| Mixup alpha | 0.5 |
| CutMix alpha | 0.0 |
| Mixup probability | 0.5 |
| Mixup switch probability | 0.5 |
| Mixup mode | `batch` |
| Mixup disabled from epoch | 75 |
| Augmentation argument | 1.0 |
| Connection function | `ADD` |
| Automatic mixed precision (AMP) | Enabled |
| Distributed training | Disabled |
| GPU count | 1 |

### Effective epoch count

Although the training script passes `--epochs 120`, the installed `timm==0.6.12` scheduler implementation adds the configured cooldown period to the cosine scheduler cycle length:

```python
num_epochs = lr_scheduler.get_cycle_length() + args.cooldown_epochs
```

For this run, the effective loop covered **130 epochs**, indexed from **0 through 129**. This explains why the completed run contains 130 epochs even though the command-line argument is 120.

This describes the current scheduler behavior. It does not, by itself, establish that the paper used the same effective epoch count.

## 4. Software and MOHESR-HPC Computing Environment

The experiment was run on the **MOHESR-HPC** cluster using Slurm. The details below describe the resources and software observed for this training job; they are not a specification of every resource available on the full cluster node.

### Allocated compute resources

| Component | Recorded value |
|---|---|
| HPC environment | MOHESR-HPC |
| Slurm partition | `main` |
| Compute node reported by job | `gpu1` |
| Nodes allocated | 1 |
| CPUs allocated to the training task | 4 |
| GPUs allocated | 1 |
| GPU model | NVIDIA A100-SXM4-80GB |
| GPU allocation profile | NVIDIA MIG `1g.20gb` |
| Training mode | Single GPU; distributed training disabled |
| Peak job RSS reported by Slurm | `20,056,076 KB` (approximately 19.1 GiB) |
| Training log `max mem` field | `3971` (as printed by the training script) |

**Important interpretation:** The A100 is an 80 GB GPU model, but this job was assigned a MIG `1g.20gb` instance, not the entire 80 GB device. The approximately 19.1 GiB figure is the job's observed peak resident system memory (`MaxRSS`), not the total RAM installed in the server. The training log's `max mem: 3971` value is reproduced as printed; no unit is asserted here.

The available job report does not establish the server's CPU model, total physical system RAM, or the complete GPU/MIG topology. Those specifications are therefore intentionally not inferred or claimed.

### Software environment

| Component | Version |
|---|---|
| Python | 3.10 |
| PyTorch | `1.12.1+cu116` |
| TorchVision | `0.13.1+cu116` |
| CUDA | 11.6 |
| CuPy | 11.4.0 |
| timm | 0.6.12 |
| SpikingJelly | 0.0.0.0.12 |
| NumPy | 1.23.5 |

Pinned dependencies are listed in [`requirements-cu116.txt`](requirements-cu116.txt).

The GPU compatibility check initialized the SpikingJelly CuPy backend and successfully ran a CUDA operation before training.

### HPC-specific files are intentionally excluded

The public repository does **not** include account-specific Slurm job scripts, HPC logs, quota reports, local environment files, or cluster-specific output files. These remain in the private working environment.

The `.gitignore` file excludes common HPC job and runtime artifacts. The server configuration is documented here for experiment transparency without publishing account-specific job scripts or filesystem paths.

## 5. Current Reproduction Results

The completed run reached the following best evaluation result:

| Metric | Current result |
|---|---:|
| Best Top-1 accuracy | **82.1%** |
| Top-5 accuracy at best Top-1 | **96.8%** |
| Best checkpoint epoch | **116** |
| Last completed epoch | **129** |
| Total epochs completed | **130** |

The best Top-1 result was obtained at epoch 116. The final epoch was 129 and did not exceed the best recorded Top-1 accuracy.

### Checkpoint files

The training run produced:

- `checkpoint_latest.pth` — checkpoint from the latest completed epoch (129).
- `checkpoint_max_test_acc1.pth` — checkpoint associated with the best Top-1 accuracy (epoch 116).
- `checkpoint_129.pth` — epoch-specific checkpoint from the final epoch.

The latest checkpoint was inspected and found to contain the model state, optimizer state, learning-rate scheduler state, AMP scaler state, and random-number-generator state. This supports resuming training with the saved training state rather than loading model weights alone.

**Checkpoint associated with the best reported result:** `checkpoint_max_test_acc1.pth`.

Checkpoint files are not included in the public repository.

## 6. Comparison with the Original Paper

The original SpiLiFormer paper reports **86.7% Top-1 accuracy** on CIFAR10-DVS for the **SpiLiFormer-2-256** model, with **1.57M parameters** and **16 time steps**.

The current reproduction obtains **82.1% Top-1 accuracy**. The local model audit reported **1,702,282 parameters (1.702282M)**.

| Metric | Original paper | Current reproduction |
|---|---:|---:|
| CIFAR10-DVS Top-1 | **86.7%** | **82.1%** |
| Parameters | 1.57M | 1.702282M |
| Time steps | 16 | 16 |
| Top-5 | Not reported here | 96.8% |
| Difference in Top-1 | — | **4.6 percentage points lower** |

The parameter-count difference is approximately **0.132M parameters**. It is recorded as an implementation difference requiring investigation; this README does not assume its cause.

The numerical comparison is based on the paper's reported CIFAR10-DVS result and the current experiment's recorded result. It is **not yet a fully controlled, apples-to-apples comparison**. The exact architecture configuration, parameter count, data split, preprocessing, evaluation protocol, and all training settings still need to be checked against the authors' implementation and paper.

### Current adaptations and open verification points

These points describe the current reproduction setup. They should not all be interpreted as changes to the SpiLiFormer architecture.

1. **HPC execution:** Training was executed on MOHESR-HPC using Slurm. Account-specific job scripts are excluded from this public repository.
2. **Dataset preparation:** Local scripts prepare CIFAR10-DVS in the frame-based, 16-step representation used by this run.
3. **Single-GPU execution:** The recorded experiment ran on one allocated A100 MIG GPU, with distributed training disabled.
4. **Checkpoint resumption:** The local workflow supports restoring model, optimizer, scheduler, AMP scaler, and RNG states from a checkpoint.
5. **Effective training duration:** The current `timm` cosine-scheduler setup produced 130 effective epochs from `--epochs 120` plus 10 cooldown epochs.
6. **Parameter count:** The local audit reports 1.702282M parameters, compared with 1.57M for the paper's reported SpiLiFormer-2-256. The reason for this difference has not yet been established.
7. **Code-level equivalence:** The model and training files originate from the upstream implementation, but a complete line-by-line audit of every local change against the referenced upstream commit has not yet been recorded.

The reproduction stage will focus on verifying these differences before any experimental optimization is attempted.

## 7. Training Workflow

The recorded workflow was:

1. Obtain the original CIFAR10-DVS recordings.
2. Prepare the dataset in the 16-frame representation.
3. Verify the prepared dataset and the Python/CUDA/CuPy environment.
4. Submit the training job to MOHESR-HPC through Slurm.
5. Monitor the training log and evaluation metrics.
6. Preserve the latest checkpoint and the best-Top-1 checkpoint.
7. If a job reaches its time limit, resume from the latest valid checkpoint.
8. Evaluate and report the best checkpoint rather than assuming the last epoch is the best.

The HPC-specific Slurm scripts and account-specific commands are intentionally not distributed in this public repository. The experiment's allocated server resources are summarized in Section 4.

## 8. Main Repository Files

| File or directory | Purpose |
|---|---|
| `cifar10dvs/model.py` | Model implementation |
| `cifar10dvs/train.py` | Training and evaluation entry point |
| `cifar10dvs/utils.py` | Training utilities and checkpoint-related functionality |
| `cifar10dvs/autoaugment.py` | Augmentation implementation |
| `prepare_original.py` | Original dataset preparation |
| `prepare_original_resumable.py` | Resumable dataset preparation |
| `audit_conversion.py` | Dataset conversion audit |
| `requirements-cu116.txt` | Pinned software dependencies |
| `UPSTREAM.txt` | Original repository and reference commit |

Dataset files, checkpoints, logs, HPC scripts, and local environment directories should not be committed.

## 9. Project Scope

This repository records the current reproduction state and its measured results.

- **Stage 1 — Reproduction:** Verify the implementation and experimental protocol against the original paper and source code, and identify the cause of the remaining accuracy and parameter-count differences.
- **Stage 2 — Optimization:** To be considered after the reproduction stage has been verified.

No optimization proposals or optimization changes are included in this initial snapshot.

## 10. Acknowledgment and Licensing Note

This reproduction is based on the public SpiLiFormer implementation:

- **Original repository:** https://github.com/KirinZheng/SpiLiFormer
- **Reference commit:** `23c6bf9fa927ae1edc593df56193ad46a43d7034`
- **Paper:** Zheng et al., *SpiLiFormer: Enhancing Spiking Transformers with Lateral Inhibition*, ICCV 2025.

The original repository's root file listing did not show a `LICENSE` file at the time this README was prepared. No license is asserted here for upstream-derived code. Confirm the authors' redistribution terms before publishing or relicensing copied upstream source files. This repository is an independent reproduction effort and is not the authors' official implementation.
