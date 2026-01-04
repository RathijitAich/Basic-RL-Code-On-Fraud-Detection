# Credit Card Fraud Detection via DQN (Notebook)

This folder contains a single notebook, `fraud_014.ipynb`, which trains a Deep Q-Network (DQN) to classify credit card transactions as **legit (0)** or **fraud (1)** using the well-known Kaggle *Credit Card Fraud Detection* dataset.

The notebook frames fraud detection as a reinforcement learning problem:
- **Observation**: a single transaction feature vector
- **Action**: `0` (predict legit) or `1` (predict fraud)
- **Reward**: cost-sensitive shaping (TP good, TN small positive, FP negative, FN most negative)

It also reports standard ML metrics on a held-out test set (confusion matrix, precision/recall/F1, PR-AUC, ROC-AUC), and selects a decision threshold on a validation split.

## Requirements

- Python 3.9+ recommended
- Jupyter (VS Code Notebook or JupyterLab)
- Packages used in the notebook:
  - `gymnasium`
  - `numpy`, `pandas`
  - `scikit-learn`
  - `torch`
  - `matplotlib`

The first code cell in the notebook installs these with pip.

The notebook is intended to be run end-to-end as a single workflow (cells build on earlier variables).

## Dataset location (important)

The notebook searches upward from the current working directory until it finds:

```
dataset/creditcard.csv
```

So you must have the dataset in a `dataset` folder somewhere **above** this `Deep RL code` folder.

Example structure:

```
gym-fraud/
  dataset/
    creditcard.csv
  Deep RL code/
    fraud_014.ipynb
    README.md
```

If the dataset is not found, the notebook will raise:

> `FileNotFoundError: Could not find dataset/creditcard.csv above current folder`

The dataset is loaded with `pandas.read_csv(...)` and expects a `Class` label column (0/1).

## How to run (VS Code on Windows)

1. Open this folder in VS Code.
2. Open `fraud_014.ipynb`.
3. Select a Python kernel.
4. Run cells from top to bottom.

Notes:
- The notebook uses `torch.device("cuda" if torch.cuda.is_available() else "cpu")` and will automatically use a GPU if available.
- Feature scaling is done using `StandardScaler` fit on the training split only (then applied to val/test).

## What the notebook does

High level flow:

1. **Install dependencies** via pip.
2. **Load data** from `dataset/creditcard.csv`.
3. **Train/test split** (stratified) to preserve the fraud ratio.
4. **Standardize features** based on training statistics.
5. **Create an RL environment** (`FraudWindowEnv`) where each step is one transaction classification.
6. **Train a DQN** (policy + target network) using replay buffer + epsilon-greedy exploration.
7. **Validation**: plot PR and ROC curves, compute AP/ROC-AUC, and choose the best F1 threshold on the validation set.
8. **Test**: evaluate on the held-out test set using the validation-chosen threshold.

## Environment details

The custom Gymnasium environment is `FraudWindowEnv`:
- **Episode**: a short window of transactions (`episode_len=200` during training).
- **Step**: observe one transaction, choose action `0/1`, receive a reward, advance to the next transaction.
- **Rewards** (default `reward_cfg`):
  - TP (`y=1`, action=1): `+1.0`
  - TN (`y=0`, action=0): `+0.05`
  - FP (`y=0`, action=1): `-0.5`
  - FN (`y=1`, action=0): `-2.0`

## DQN setup (as in the notebook)

- Network: MLP `obs_dim → 128 → 128 → 2` with ReLU
- Discount: `gamma=0.99`
- Optimizer: Adam, `lr=1e-3`
- Replay buffer: `100_000`
- Batch size: `256`
- Target network update: every `1000` environment steps
- Gradient clipping: `5.0`
- Exploration: epsilon-greedy linearly decayed from `1.0 → 0.05` over `50_000` steps
- Training loop:
  - `total_steps=30_000`
  - `warmup_steps=2_000` (collect experience before learning)
  - Progress printed every `2_000` steps (includes sampled train/val accuracy)

Data splits:
- Train/Test: `80/20` stratified split
- RL Train/Validation: `90/10` stratified split from the training portion

## Metrics reported

On validation and test, the notebook reports:
- Confusion matrix `[[TN, FP], [FN, TP]]`
- Precision / Recall / F1
- PR-AUC (Average Precision)
- ROC-AUC

Scoring and thresholding:
- The notebook uses a continuous fraud “score” defined as `score = Q(fraud) - Q(legit)`.
- “Argmax” prediction is equivalent to `score >= 0`.
- It sweeps thresholds on the validation set to maximize F1, then uses that best threshold for test evaluation.

Important: the test cell contains a hard-coded `best_th = 0.668646` (copied from a prior validation run). If your validation output prints a different `best_th`, update that value before running the test cell.

## Reproducibility

The notebook sets seeds for Python `random`, NumPy, and PyTorch in the training loop cell. Results can still vary slightly due to GPU nondeterminism (if using CUDA).



