# Reproducibility Notes

## Confirmed for Scheduled-Beta Experiment 2
- Branch: `reproduction-beta-v1`
- Commit recorded during the project: `3f2e383` (`Restore official beta schedule and resume support`)
- Loss: `(1 - beta) * CE(x1, target) + beta * CE(output, target)`
- Beta range: 0.5 to 0.8
- Final checkpoint epoch index: 129
- Scheduler steps: 130
- Best test Top-1: 79.7%
- Final test Top-1: 78.1%

## Epoch-count discrepancy
The run arguments displayed `epochs=120`, but training reached epoch index 129 and the checkpoint stores `total_epochs=130`. The Beta scheduler is constructed using the `num_epochs` value returned by the scheduler creation function. Inspect that function and the installed `timm` version to document why it returns 130. Preserve the historical run record.

## Before claiming exact reproduction
- Compare architecture, layers, and parameter count with the paper.
- Confirm dataset source, train/test split, temporal conversion, and preprocessing.
- Confirm augmentation, mixup, loss, scheduler, defaults, and random seeds.
- Confirm evaluation protocol and checkpoint-selection rule.
- Verify Experiment 1 from raw artifacts.
- Record the Git commit for each run.

## Public-release security
- Never commit SSH private keys, tokens, passwords, or HPC credentials.
- Inspect every untracked file before staging; do not use `git add .` blindly.
- Sanitize HPC scripts to remove account-specific paths.
- Do not upload the dataset unless redistribution is explicitly permitted.
- Check the upstream code license and preserve required notices.
