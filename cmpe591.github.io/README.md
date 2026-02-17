# CMPE591 - Homework 1

This repository includes an implementation for Assignment 1 deliverables:

1. Object position prediction from initial image and action using MLP
2. Object position prediction from initial image and action using CNN
3. Post-action image reconstruction from initial image and action

## Files

- `src/hw1_data.py`: dataset collection and loading utilities
- `src/hw1_mlp_position.py`: deliverable 1 (has `train()` and `test()`)
- `src/hw1_cnn_position.py`: deliverable 2 (has `train()` and `test()`)
- `src/hw1_reconstruction.py`: deliverable 3 (has `train()` and `test()`)

## Environment

Activate your environment first (example):

```powershell
.\robotic_env\Scripts\activate
cd cmpe591.github.io\src
```

## 1) Collect Dataset

This collects tuples `(img_before, action_id, pos_after, img_after)`.

```powershell
python hw1_data.py --num-samples 1000 --workers 4 --out-dir data/hw1 --seed 42
```

Output:

- `data/hw1/hw1_dataset.pt`

## 2) Train/Test Deliverable 1 (MLP Position)

```powershell
python hw1_mlp_position.py train --data-path data/hw1 --run-dir runs/hw1/mlp_pos --epochs 25 --batch-size 32
python hw1_mlp_position.py test --data-path data/hw1 --checkpoint-path runs/hw1/mlp_pos/best.pt --run-dir runs/hw1/mlp_pos
```

Outputs:

- `runs/hw1/mlp_pos/best.pt`
- `runs/hw1/mlp_pos/metrics.json`

## 3) Train/Test Deliverable 2 (CNN Position)

```powershell
python hw1_cnn_position.py train --data-path data/hw1 --run-dir runs/hw1/cnn_pos --epochs 25 --batch-size 32
python hw1_cnn_position.py test --data-path data/hw1 --checkpoint-path runs/hw1/cnn_pos/best.pt --run-dir runs/hw1/cnn_pos
```

Outputs:

- `runs/hw1/cnn_pos/best.pt`
- `runs/hw1/cnn_pos/metrics.json`

## 4) Train/Test Deliverable 3 (Reconstruction)

```powershell
python hw1_reconstruction.py train --data-path data/hw1 --run-dir runs/hw1/reconstruction --epochs 25 --batch-size 32
python hw1_reconstruction.py test --data-path data/hw1 --checkpoint-path runs/hw1/reconstruction/best.pt --run-dir runs/hw1/reconstruction
```

Outputs:

- `runs/hw1/reconstruction/best.pt`
- `runs/hw1/reconstruction/metrics.json`
- `runs/hw1/reconstruction/reconstruction_samples.png`

## Report Section (Fill After Training)

Include these in your final README report:

- Test errors for all three deliverables
- Loss curves (from per-epoch values in each `metrics.json`)
- Reconstructed image samples from deliverable 3

Example template:

| Deliverable | Test Metric |
| --- | --- |
| MLP position | MSE=?, MAE=?, RMSE=? |
| CNN position | MSE=?, MAE=?, RMSE=? |
| Reconstruction | MSE=?, L1=?, PSNR=? |
