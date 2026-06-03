# Final Project – VLM-Guided Robot Reaching

The robot arm sends the camera image to a VLM (Qwen3), which returns the target object's bounding box. An MLP predicts the next 5 waypoints from this bbox + current ee_pos. The robot arm executes these waypoints sequentially; a successful episode is one where the arm touches the object.

---

## VLM Server Setup

Start the vLLM server before running inference:

```bash
vllm serve /mnt/beegfs/LLM/onuralpguvercin/HAVL-RL-SWIFT/ZZ_trained_models/Qwen3-VL-4B-Instruct --served-model-name Qwen3 --tensor-parallel-size 8 --max-model-len 50000 --allowed-local-media-path /mnt --max-num-batched-tokens 150000 --max-num-seqs 50 --port 8000
```

To stop the server:
```bash
pkill -f "vllm serve"
```

Or press `Ctrl+C` in the terminal.

---

## Project Structure

```
final_project/
├── src/
│   ├── env.py          – MuJoCo environment (scenes with random objects)
│   ├── utils.py        – Camera projection, EE normalization, bbox computation
│   ├── collect.py      – Data collection
│   ├── model.py        – NavigationMLP (behaviour cloning, configurable horizon)
│   ├── train.py        – Training loop (--horizon parameter)
│   ├── vlm_client.py   – VLM HTTP client (vLLM / OpenAI-compatible)
│   ├── inference.py    – Closed-loop inference + evaluation (--horizon parameter)
│   └── visualize.py    – Visualization
├── data/trajectories/  – Collected trajectory files (*.pt)
└── runs/
    ├── nav_h1/, nav_h2/, ...  – Horizon-specific model checkpoints
    ├── vis_h1/, vis_h2/, ...  – Horizon-specific inference results
    ├── train_summary.json     – Training ablation summary
    └── inference_summary.json – Inference ablation summary
```

All commands are run from the project root (`/mnt/beegfs/LLM/onuralpguvercin/ROBOTICS`).

---

## Step-by-Step Usage

### Step 1 — Collect Data

**What it does:** Creates random scenes in MuJoCo. Each scene contains 2–4 colored boxes/spheres. The robot arm approaches each object in sequence; at every step the camera image, end-effector position (ee_pos), and ground-truth bounding box (gt_bbox) are recorded.

```bash
python final_project/src/collect.py --n-scenes 300 --n-workers 8 --seed 200 --out-dir final_project/data/trajectories
```

**Output:**
- `data/trajectories/traj_XXXXXX_<color_shape_i>.pt` — one file per successful trajectory
- `data/trajectories/metadata.json` — statistics

---

### Step 2 — Visualize Collected Data

**What it does:** Selects random samples from successful trajectories and draws 6 frames + GT bbox from each.

```bash
python final_project/src/visualize.py --mode trajectories --data-dir final_project/data/trajectories --n-samples 4 --out-dir final_project/runs/vis
```

**Output:** `runs/vis/<traj_name>_frames.png`

---

### Step 3 — Train the Model

**What it does:** Learns the `(bbox + ee_pos) → horizon-step delta` mapping using GT bboxes.

```bash
# Single horizon
python final_project/src/train.py --horizon 1 --data-dir final_project/data/trajectories --run-dir final_project/runs/nav_h1 --epochs 200
```

**Or multiple horizons (1–5):**
```bash
python final_project/src/train.py --horizon 1 2 3 4 5 --data-dir final_project/data/trajectories --run-dir final_project/runs/navigation --epochs 200 --batch-size 128
```

**Output:**
- `runs/nav_h1/best.pt`, `runs/nav_h2/best.pt`, ... — model per horizon
- `runs/train_summary.json` — summary (test_mse per horizon)

---

### Step 4 — Run Inference

**What it does:** Tests the trained model in closed-loop. In the default mode, bboxes come from the VLM; passing `--use-gt-bbox` skips the VLM and uses ground-truth bboxes instead. The model predicts raw deltas; during inference each delta component is clamped by `--max-delta` (set to `0` to disable clamping).

```bash
# Single horizon, VLM bbox
python final_project/src/inference.py --checkpoint final_project/runs/nav_h1/best.pt --horizon 1 --n-episodes 50 --vlm-url http://localhost:8000 --out-dir final_project/runs/vis_h1 --save-vis --max-delta 0.05
```

```bash
# Single horizon, GT bbox ablation
python final_project/src/inference.py --checkpoint final_project/runs/nav_h1/best.pt --horizon 1 --n-episodes 50 --use-gt-bbox --out-dir final_project/runs/vis_h1_gt --save-vis --max-delta 0.05
```

```bash
# Multiple horizons (1–5), VLM bbox
python final_project/src/inference.py --horizon 1 2 3 4 5 --n-episodes 20 --vlm-url http://localhost:8000 --out-dir final_project/runs --save-vis --max-delta 0.05
```

```bash
# Multiple horizons (1–5), GT bbox ablation
python final_project/src/inference.py --horizon 1 2 3 4 5 --n-episodes 20 --use-gt-bbox --out-dir final_project/runs --save-vis --max-delta 0.05
```

**Output:**
- `runs/vis_h1/eval_results.json`, `runs/vis_h2/eval_results.json`, ... — results per horizon
- `runs/inference_summary.json` — summary (success_rate vs horizon)

---

### Step 5 — Visualize Results

**a) Horizon comparison (success rate / distance / steps across all horizons):**

```bash
python final_project/src/visualize.py --mode horizon-compare --run-dir final_project/runs --out-dir final_project/runs/vis
```

**Output:** `runs/vis/horizon_comparison.png` — 3-panel plot comparing all `vis_h*/eval_results.json` files.

---

**b) Başarı özeti:**

```bash
python final_project/src/visualize.py --mode eval-summary --eval-json final_project/runs/vis_h1/eval_results.json --out-dir final_project/runs/vis
```

**Çıktı:** `runs/vis/eval_summary.png`

---

**c) Eğitim eğrisi:**

```bash
python final_project/src/visualize.py --mode training --run-dir final_project/runs/nav_h1 --out-dir final_project/runs/vis
```

**Çıktı:** `runs/vis/training_curves.png`

---

**d) Episode top-down trajektori:**

```bash
python final_project/src/visualize.py --mode episodes --eval-json final_project/runs/vis_h1/eval_results.json --n-samples 5 --out-dir final_project/runs/vis
```

**Çıktı:** `runs/vis/ep000_traj.png`

---

## System Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                      INFERENCE LOOP                          │
│                    (Configurable HORIZON)                    │
│                                                              │
│  env.reset() → select target object                          │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────┐   image + object name        ┌──────────┐       │
│  │ MuJoCo  │ ----------------------->     │  Qwen3   │       │
│  │   Env   │ <-- bbox (cx,cy,w,h) ──       │  :8000   │       │
│  └────┬────┘                              └──────────┘       │
│       │ ee_pos                                               │
│       ▼                                                      │
│  ┌──────────────────────────────────────┐                    │
│  │     NavigationMLP(horizon=H)         │                    │
│  │  7 → 256 → 256 → 256 → (3*H)         │                    │
│  │  input  : bbox(4) + ee_pos(3)        │                    │
│  │  output : H × EE delta (metres)      │                    │
│  └────────────┬─────────────────────────┘                    │
│               │                                              │
│               ▼                                              │
│   Apply H deltas sequentially (clamped) → contact/distance   │
│   contact → SUCCESS                                          │
│   after H steps → query VLM again (H steps = 1 query)        │
└──────────────────────────────────────────────────────────────┘
```

---

## Metrics

| Metric | Description |
|---|---|
| Success rate | Percentage of episodes where the arm touches the object |
| Mean final distance | EE-to-object distance at the end of the episode |
| Mean steps (success) | Average number of steps in successful episodes |
| Test MSE | MLP prediction error on the test set |
| Test RMSE | √(MSE) — delta-space error in metres |
