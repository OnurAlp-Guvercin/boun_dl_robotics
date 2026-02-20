# CMPE591 - Homework 1

This repository contains my implementation for all Assignment 1 deliverables:

1. Object position prediction from initial image and action using MLP
2. Object position prediction from initial image and action using CNN
3. Post-action image reconstruction from initial image and action

## Project Files

- `src/hw1_mlp_position.py` -> Deliverable 1 (`collect()`, `train()`, `test()`)
- `src/hw1_cnn_position.py` -> Deliverable 2 (`collect()`, `train()`, `test()`)
- `src/hw1_reconstruction.py` -> Deliverable 3 (`collect()`, `train()`, `test()`)

## Environment

Example activation:

```bash
source robotic_env/bin/activate
```

## Dataset Collection

```bash
python boun_dl_robotics/cmpe591.github.io/src/hw1_mlp_position.py collect \
  --num-samples 1250 --workers 1 --out-dir data/hw1 --seed 42
```

## Training and Testing

### Deliverable 1: MLP Position Prediction

```bash
python boun_dl_robotics/cmpe591.github.io/src/hw1_mlp_position.py train \
  --data-path data/hw1 --run-dir runs/hw1/mlp_pos

python boun_dl_robotics/cmpe591.github.io/src/hw1_mlp_position.py test \
  --data-path data/hw1 --checkpoint-path runs/hw1/mlp_pos/best.pt --run-dir runs/hw1/mlp_pos
```

### Deliverable 2: CNN Position Prediction

```bash
python boun_dl_robotics/cmpe591.github.io/src/hw1_cnn_position.py train \
  --data-path data/hw1 --run-dir runs/hw1/cnn_pos

python boun_dl_robotics/cmpe591.github.io/src/hw1_cnn_position.py test \
  --data-path data/hw1 --checkpoint-path runs/hw1/cnn_pos/best.pt --run-dir runs/hw1/cnn_pos
```

### Deliverable 3: Image Reconstruction

```bash
python boun_dl_robotics/cmpe591.github.io/src/hw1_reconstruction.py train \
  --data-path data/hw1 --run-dir runs/hw1/reconstruction

python boun_dl_robotics/cmpe591.github.io/src/hw1_reconstruction.py test \
  --data-path data/hw1 --checkpoint-path runs/hw1/reconstruction/best.pt --run-dir runs/hw1/reconstruction
```

## Report

Metrics below are taken from:

- `runs/hw1/mlp_pos/test_results.json`
- `runs/hw1/cnn_pos/test_results.json`
- `runs/hw1/reconstruction/test_results.json`

### Test Errors

| Deliverable | Metrics |
| --- | --- |
| MLP Position Prediction | MSE: **0.0528538**, MAE: **0.1818886**, RMSE: **0.2298995** |
| CNN Position Prediction | MSE: **0.0194302**, MAE: **0.1077741**, RMSE: **0.1393923** |
| Reconstruction | MSE: **0.0064120**, L1: **0.0161466**, PSNR: **21.9301** |

### Loss Curves

#### MLP Position

![MLP Epoch Loss](docs/images/hw1_report/mlp_loss_epoch.png)
![MLP Step Loss](docs/images/hw1_report/mlp_loss_step.png)

#### CNN Position

![CNN Epoch Loss](docs/images/hw1_report/cnn_loss_epoch.png)
![CNN Step Loss](docs/images/hw1_report/cnn_loss_step.png)

#### Reconstruction

![Reconstruction Epoch Loss](docs/images/hw1_report/recon_loss_epoch.png)
![Reconstruction Step Loss](docs/images/hw1_report/recon_loss_step.png)

### Reconstructed Images (Ground Truth Comparison)

Each saved sample image includes labels and is shown as:
**Before | Ground Truth | Prediction**.

![Recon Sample 0](docs/images/hw1_report/recon_sample_0000.png)
![Recon Sample 1](docs/images/hw1_report/recon_sample_0001.png)
![Recon Sample 2](docs/images/hw1_report/recon_sample_0002.png)
![Recon Sample 3](docs/images/hw1_report/recon_sample_0003.png)
![Recon Sample 4](docs/images/hw1_report/recon_sample_0004.png)
![Recon Sample 5](docs/images/hw1_report/recon_sample_0005.png)
![Recon Sample 6](docs/images/hw1_report/recon_sample_0006.png)
![Recon Sample 7](docs/images/hw1_report/recon_sample_0007.png)

## Saved Models

- `runs/hw1/mlp_pos/best.pt`
- `runs/hw1/cnn_pos/best.pt`
- `runs/hw1/reconstruction/best.pt`
