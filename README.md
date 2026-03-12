# CMPE591 - Homework 1 & Homework 2

This repository contains implementations for all HW1 deliverables:

1. Object position prediction from initial image + action with an MLP
2. Object position prediction from initial image + action with a CNN
3. Post-action image reconstruction from initial image + action

## Implementation Files

- Deliverable 1: `src/hw1_mlp_position.py`
- Deliverable 2: `src/hw1_cnn_position.py`
- Deliverable 3: `src/hw1_reconstruction.py`
- Homework 2 (DQN): `src/hw2_dqn.py`

### Direct Source Links

- [hw1_cnn_position.py](boun_dl_robotics/cmpe591.github.io/src/hw1_cnn_position.py)
- [hw1_mlp_position.py](boun_dl_robotics/cmpe591.github.io/src/hw1_mlp_position.py)
- [hw1_reconstruction.py](boun_dl_robotics/cmpe591.github.io/src/hw1_reconstruction.py)
- [hw2_dqn.py](boun_dl_robotics/cmpe591.github.io/src/hw2_dqn.py)

Each script supports `collect`, `train`, and `test` commands.

## Data Collection

```bash
python boun_dl_robotics/cmpe591.github.io/src/hw1_mlp_position.py collect \
  --num-samples 1250 \
  --workers 1 \
  --out-dir data/hw1 \
  --seed 42
```

## Train and Test Commands

### Deliverable 1 (MLP Position)

```bash
python boun_dl_robotics/cmpe591.github.io/src/hw1_mlp_position.py train \
  --data-path data/hw1 \
  --run-dir runs/hw1/mlp_pos

python boun_dl_robotics/cmpe591.github.io/src/hw1_mlp_position.py test \
  --data-path data/hw1 \
  --checkpoint-path runs/hw1/mlp_pos/best.pt \
  --run-dir runs/hw1/mlp_pos
```

### Deliverable 2 (CNN Position)

```bash
python boun_dl_robotics/cmpe591.github.io/src/hw1_cnn_position.py train \
  --data-path data/hw1 \
  --run-dir runs/hw1/cnn_pos

python boun_dl_robotics/cmpe591.github.io/src/hw1_cnn_position.py test \
  --data-path data/hw1 \
  --checkpoint-path runs/hw1/cnn_pos/best.pt \
  --run-dir runs/hw1/cnn_pos
```

### Deliverable 3 (Image Reconstruction)

```bash
python boun_dl_robotics/cmpe591.github.io/src/hw1_reconstruction.py train \
  --data-path data/hw1 \
  --run-dir runs/hw1/reconstruction

python boun_dl_robotics/cmpe591.github.io/src/hw1_reconstruction.py test \
  --data-path data/hw1 \
  --checkpoint-path runs/hw1/reconstruction/best.pt \
  --run-dir runs/hw1/reconstruction
```

## Results Report

Reported results are read from:

- `runs/hw1/mlp_pos/test_results.json`
- `runs/hw1/cnn_pos/test_results.json`
- `runs/hw1/reconstruction/test_results.json`

### Test Errors

| Deliverable | MSE | MAE / L1 | RMSE / PSNR |
| --- | ---: | ---: | ---: |
| D1 - MLP Position | 0.0528538 | MAE: 0.1818886 | RMSE: 0.2298995 |
| D2 - CNN Position | 0.0194302 | MAE: 0.1077741 | RMSE: 0.1393923 |
| D3 - Reconstruction | 0.0064120 | L1: 0.0161466 | PSNR: 21.9301 dB |

### Loss Curves

#### D1 - MLP Position

![D1 Epoch Loss](runs/hw1/mlp_pos/loss_epoch_plot.png)
![D1 Step Loss](runs/hw1/mlp_pos/loss_step_plot.png)

#### D2 - CNN Position

![D2 Epoch Loss](runs/hw1/cnn_pos/loss_epoch_plot.png)
![D2 Step Loss](runs/hw1/cnn_pos/loss_step_plot.png)

#### D3 - Reconstruction

![D3 Epoch Loss](runs/hw1/reconstruction/loss_epoch_plot.png)
![D3 Step Loss](runs/hw1/reconstruction/loss_step_plot.png)

### Deliverable 3 Reconstruction Samples

Each sample image is formatted as:
**Before | Ground Truth | Prediction**

| Sample 0 | Sample 1 | Sample 2 | Sample 3 |
| --- | --- | --- | --- |
| ![Recon Sample 0](runs/hw1/reconstruction/reconstruction_samples/sample_0000.png) | ![Recon Sample 1](runs/hw1/reconstruction/reconstruction_samples/sample_0001.png) | ![Recon Sample 2](runs/hw1/reconstruction/reconstruction_samples/sample_0002.png) | ![Recon Sample 3](runs/hw1/reconstruction/reconstruction_samples/sample_0003.png) |

| Sample 4 | Sample 5 | Sample 6 | Sample 7 |
| --- | --- | --- | --- |
| ![Recon Sample 4](runs/hw1/reconstruction/reconstruction_samples/sample_0004.png) | ![Recon Sample 5](runs/hw1/reconstruction/reconstruction_samples/sample_0005.png) | ![Recon Sample 6](runs/hw1/reconstruction/reconstruction_samples/sample_0006.png) | ![Recon Sample 7](runs/hw1/reconstruction/reconstruction_samples/sample_0007.png) |

## Saved Checkpoints

- `runs/hw1/mlp_pos/best.pt`
- `runs/hw1/cnn_pos/best.pt`
- `runs/hw1/reconstruction/best.pt`

---

# CMPE591 - Homework 2 (DQN)

Assignment 2 implementation is provided in:

- `src/hw2_dqn.py`

The script includes separate `train()` and `test()` functions with CLI commands.
By default, training uses `high_level_state` (as requested in the assignment) and supports optional pixel-state training.

## HW2 Setup

```bash
source robotic_env/bin/activate
```

Python compatibility:
- Python 3.9 and 3.11 are supported by the homework scripts.

## HW2 Train

```bash
python boun_dl_robotics/cmpe591.github.io/src/hw2_dqn.py train \
  --state-mode high_level
```

Example run directories used in this report:

```bash
python boun_dl_robotics/cmpe591.github.io/src/hw2_dqn.py train \
  --state-mode high_level \
  --run-dir runs/hw2/dqn_1

python boun_dl_robotics/cmpe591.github.io/src/hw2_dqn.py train \
  --state-mode high_level \
  --run-dir runs/hw2/dqn_2
```

## HW2 Test

```bash
python boun_dl_robotics/cmpe591.github.io/src/hw2_dqn.py test \
  --state-mode high_level \
  --checkpoint-path runs/hw2/dqn_2/best.pt \
  --run-dir runs/hw2/dqn_2
```

## HW2 Outputs

Training artifacts:

- `<run_dir>/best.pt`
- `<run_dir>/last.pt`
- `<run_dir>/train_metrics.json`
- `<run_dir>/reward_plot.png`
- `<run_dir>/rps_plot.png`

Test artifact:

- `<run_dir>/test_results.json`

## HW2 Report Section

### What Instructor Asks (Per Run)

For each run, include these 3 items in `README.md`:

1. What hyperparameters you changed
2. How performance changed (reward / reward-per-step / success rate)
3. Short discussion of why this change produced that effect

### Instructor Reference Hyperparameters (From Email)

The instructor shared the following state-based set as a reference:

| Hyperparameter | Value |
| --- | ---: |
| memory_size (`n_replay_buffer`) | 10000 |
| num_episodes (`n_episodes`) | 2500 |
| batch_size | 128 |
| eps_decay | 10000 |
| eps_end (`epsilon_min`) | 0.05 |
| eps_start (`epsilon`) | 0.9 |
| gamma | 0.99 |
| learning_rate (`lr`) | 0.0001 |
| tau (soft target update) | 0.005 |

### Current Code Defaults (`src/hw2_dqn.py`)

The current defaults in code (used for the latest trial setup) are:

| Hyperparameter | Value |
| --- | ---: |
| memory_size (`n_replay_buffer`) | 20000 |
| num_episodes (`n_episodes`) | 2500 |
| batch_size | 128 |
| eps_decay | 20000 |
| eps_end (`epsilon_min`) | 0.10 |
| eps_start (`epsilon`) | 1.0 |
| gamma | 0.995 |
| learning_rate (`lr`) | 0.00005 |
| tau (soft target update) | 0.01 |
| warmup_episodes (`n_warmup_episodes`) | 100 |
| num_collectors | 16 |
| collector_sync_updates | 25 |

### Recommended Train Command (Current Defaults)

```bash
python boun_dl_robotics/cmpe591.github.io/src/hw2_dqn.py train \
  --state-mode high_level \
  --render-mode offscreen
```

### Reproducing Instructor Set (Optional)

```bash
python boun_dl_robotics/cmpe591.github.io/src/hw2_dqn.py train \
  --state-mode high_level \
  --n-episodes 2500 \
  --batch-size 128 \
  --epsilon 0.9 \
  --epsilon-min 0.05 \
  --epsilon-decay 10000 \
  --gamma 0.99 \
  --lr 1e-4 \
  --tau 0.005 \
  --n-replay-buffer 10000
```

### Baseline Result (Before This Update)

This baseline is from the previous run file:
- `runs/hw2/dqn_1/train_metrics.json`

| Metric | Value |
| --- | ---: |
| Episodes | 3000 |
| Best Reward | 30.8804 |
| Final Epsilon | 0.0209 |
| Total Updates | 121864 |
| Reward Mean (all episodes) | 10.6222 |
| Reward/Step Mean (all episodes) | 0.3579 |
| Reward Mean (last 50 episodes) | 13.9764 |
| Reward/Step Mean (last 50 episodes) | 0.4789 |
| Reward Mean (last 100 episodes) | 13.7801 |
| Reward/Step Mean (last 100 episodes) | 0.4659 |
| Reward Mean (last 200 episodes) | 14.4158 |
| Reward/Step Mean (last 200 episodes) | 0.4866 |

### Multi-Run Comparison (HW2)

I trained each run in a separate folder and evaluated with 100 test episodes for a fair comparison.

| Run | What I changed | What happened | My take |
| --- | --- | --- | --- |
| Run-1 | More aggressive setup: `n_episodes=3000`, `batch_size=256`, `gamma=0.95`, `lr=1e-3`, `tau=0.001`, fast exploration decay (`epsilon=0.4` to `0.01`), and `n_warmup_episodes=50`. | Test (`runs/hw2/dqn_1/test_results.json`): mean reward `23.84`, mean RPS `0.4940`, success rate `5%`. | It learns dense reward quickly, but this does not consistently turn into goal-reaching behavior. |
| Run-2 | Mostly instructor-style setup: `n_episodes=2500`, `batch_size=128`, `gamma=0.99`, `lr=1e-4`, `tau=0.005`, `epsilon=0.9` to `0.05` (`decay=10000`), with a larger replay size `50000`. | Test (`runs/hw2/dqn_2/test_results.json`): mean reward `23.98`, mean RPS `0.4869`, success rate `2%`, lower reward variance (`std 12.15 -> 8.62`). | Compared to Run-1, behavior is more stable but also more conservative; it keeps reward variance lower, yet converts less often to terminal success. |
| Run-3 | More exploration-heavy/stable setup: `n_episodes=2500`, `batch_size=128`, `gamma=0.995`, `lr=5e-5`, `tau=0.01`, `epsilon=1.0` to `0.1` (`decay=20000`), `n_replay_buffer=20000`, `n_warmup_episodes=100`, `collector_sync_updates=25`. | Test (`runs/hw2/dqn_3/test_results.json`): mean reward `7.51`, mean RPS `0.1503`, success rate `0%`, mean steps `50.0`. | This setting stayed too exploratory and conservative for this task. The policy did not converge to goal-reaching behavior, and episodes almost always timed out. |

### Test Results

Source file (latest run): `runs/hw2/dqn_3/test_results.json`

| Metric | Value |
| --- | ---: |
| Evaluation Episodes | 100 |
| Epsilon (test) | 0.0 |
| Mean Total Reward | 7.5141 |
| Std Total Reward | 4.8327 |
| Mean Reward per Step | 0.1503 |
| Std Reward per Step | 0.0967 |
| Mean Steps | 50.0 |
| Success Rate | 0.0000 (0.0%) |

### Reward Curves

Run-1 (`runs/hw2/dqn_1`)

![HW2 Reward Run-1](runs/hw2/dqn_1/reward_plot.png)
![HW2 Reward per Step Run-1](runs/hw2/dqn_1/rps_plot.png)

Run-2 (`runs/hw2/dqn_2`)

![HW2 Reward Run-2](runs/hw2/dqn_2/reward_plot.png)
![HW2 Reward per Step Run-2](runs/hw2/dqn_2/rps_plot.png)

Run-3 (`runs/hw2/dqn_3`)

![HW2 Reward Run-3](runs/hw2/dqn_3/reward_plot.png)
![HW2 Reward per Step Run-3](runs/hw2/dqn_3/rps_plot.png)
