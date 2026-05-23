# Final Project – VLM-Guided Robot Reaching

Robot kolu, kameradan aldığı görüntüyü VLM'e (Qwen3) gönderir. VLM hedef nesnenin bbox'ını döner. MLP bu bbox + mevcut ee_pos'tan sonraki 5 waypoint'i tahmin eder. Robot kolu bu waypoint'leri sırayla uygular, nesneye dokunursa başarılı sayılır.

---

## VLM Server Kurulumu

Inference öncesi vLLM sunucusunu başlat:

```bash
vllm serve /mnt/beegfs/LLM/onuralpguvercin/HAVL-RL-SWIFT/ZZ_trained_models/Qwen3-VL-4B-Instruct \
  --served-model-name Qwen3 \
  --tensor-parallel-size 8 \
  --max-model-len 50000 \
  --allowed-local-media-path /mnt \
  --max-num-batched-tokens 150000 \
  --max-num-seqs 50 \
  --port 8000
```

Sunucuyu durdurmak için:
```bash
pkill -f "vllm serve"
```

Veya terminal'de `Ctrl+C` basın.

---

## Proje Yapısı

```
final_project/
├── src/
│   ├── env.py          – MuJoCo ortamı (rastgele nesneli sahneler)
│   ├── utils.py        – Kamera projeksiyonu, EE normalizasyonu, bbox hesaplama
│   ├── collect.py      – Veri toplama
│   ├── model.py        – NavigationMLP (behaviour cloning, configurable horizon)
│   ├── train.py        – Eğitim döngüsü (--horizon / --horizons parameter)
│   ├── vlm_client.py   – VLM HTTP istemcisi (vLLM / OpenAI uyumlu)
│   ├── inference.py    – Kapalı döngü inference + değerlendirme (--horizon / --horizons)
│   └── visualize.py    – Görselleştirme
├── data/trajectories/  – Toplanan trajektori dosyaları (*.pt)
└── runs/
    ├── nav_h1/, nav_h2/, ...  – Horizon-specific model checkpointleri
    ├── vis_h1/, vis_h2/, ...  – Horizon-specific inference sonuçları
    ├── train_summary.json     – Training ablation özeti
    └── inference_summary.json – Inference ablation özeti
```

Tüm komutlar proje kökünden (`/mnt/beegfs/LLM/onuralpguvercin/ROBOTICS`) çalıştırılır.

---

## Adım Adım Kullanım

### Adım 1 — Veri Topla

**Ne yapar:** MuJoCo'da rastgele sahneler oluşturur. Her sahnede 2–4 renkli kutu/küre var. Robot kolu her nesneye sırayla yaklaşır; her adımda kameradan görüntü, robot kolunun pozisyonu (ee_pos) ve nesnenin bbox'ı (gt_bbox) kaydedilir.

```bash
MUJOCO_GL=egl /trinity/home/onuralpguvercin/.conda/envs/robotic_env_311/bin/python \
  final_project/src/collect.py \
  --n-scenes 300 \
  --n-workers 8 \
  --seed 200 \
  --out-dir final_project/data/trajectories
```

**Çıktı:**
- `data/trajectories/traj_XXXXXX_<renk_şekil_i>.pt` — her başarılı trajektori için bir dosya
- `data/trajectories/metadata.json` — istatistikler

---

### Adım 2 — Toplanan Veriyi Görselleştir

**Ne yapar:** Başarılı trajektorilerden rastgele örnekler seçer, her birinden 6 frame + GT bbox çizer.

```bash
/trinity/home/onuralpguvercin/.conda/envs/robotic_env_311/bin/python \
  final_project/src/visualize.py --mode trajectories \
  --data-dir final_project/data/trajectories \
  --n-samples 4 \
  --out-dir final_project/runs/vis
```

**Çıktı:** `runs/vis/<traj_adı>_frames.png`

---

### Adım 3 — Modeli Eğit

**Ne yapar:** GT bbox'lar ile `(bbox + ee_pos) → horizon adım delta` ilişkisini öğrenir.

```bash
# Tek horizon
python final_project/src/train.py \
  --horizon 1 \
  --data-dir final_project/data/trajectories \
  --run-dir final_project/runs/nav_h1 \
  --epochs 200
```

**Ya da çoklu horizons (1-5):**
```bash
python final_project/src/train.py \
  --horizons 1 2 3 4 5 \
  --data-dir final_project/data/trajectories \
  --run-dir final_project/runs/navigation \
  --epochs 200 \
  --batch-size 128
```

**Çıktı:**
- `runs/nav_h1/best.pt`, `runs/nav_h2/best.pt`, ... — her horizon için model
- `runs/train_summary.json` — özet (test_mse per horizon)

---

### Adım 4 — Inference Al

**Ne yapar:** Eğitilen modeli VLM ile kapalı döngüde test eder.

```bash
# Tek horizon
MUJOCO_GL=egl python final_project/src/inference.py \
  --checkpoint final_project/runs/nav_h1/best.pt \
  --horizon 1 \
  --n-episodes 50 \
  --vlm-url http://localhost:8000 \
  --out-dir final_project/runs/vis_h1 \
  --save-vis
```

**Ya da çoklu horizons (1-5):**
```bash
MUJOCO_GL=egl python final_project/src/inference.py \
  --horizons 1 2 3 4 5 \
  --n-episodes 20 \
  --vlm-url http://localhost:8000 \
  --out-dir final_project/runs \
  --save-vis
```

**Çıktı:**
- `runs/vis_h1/eval_results.json`, `runs/vis_h2/eval_results.json`, ... — her horizon sonuçları
- `runs/inference_summary.json` — özet (success_rate vs horizon)

---

### Adım 5 — Sonuçları Görselleştir

**a) Başarı özeti:**

```bash
/trinity/home/onuralpguvercin/.conda/envs/robotic_env_311/bin/python \
  final_project/src/visualize.py --mode eval-summary \
  --eval-json final_project/runs/vis/eval_results.json \
  --out-dir final_project/runs/vis
```

**Çıktı:** `runs/vis/eval_summary.png`

---

**b) Eğitim eğrisi:**

```bash
/trinity/home/onuralpguvercin/.conda/envs/robotic_env_311/bin/python \
  final_project/src/visualize.py --mode training \
  --run-dir final_project/runs/navigation \
  --out-dir final_project/runs/vis
```

**Çıktı:** `runs/vis/training_curves.png`

---

**c) Episode top-down trajektori:**

```bash
/trinity/home/onuralpguvercin/.conda/envs/robotic_env_311/bin/python \
  final_project/src/visualize.py --mode episodes \
  --eval-json final_project/runs/vis/eval_results.json \
  --n-samples 5 \
  --out-dir final_project/runs/vis
```

**Çıktı:** `runs/vis/ep000_traj.png`

---

## Sistem Şeması

```
┌──────────────────────────────────────────────────────────────┐
│                     INFERENCE DÖNGÜSÜ                        │
│                    (Configurable HORIZON)                    │
│                                                              │
│  env.reset() → hedef nesne seç                               │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────┐   görüntü + nesne adı     ┌──────────┐          │
│  │ MuJoCo  │ --------------------->    │  Qwen3   │          │
│  │   Env   │ <-- bbox (cx,cy,w,h) ─    │  :8000   │          │
│  └────┬────┘                           └──────────┘          │
│       │ ee_pos                                               │
│       ▼                                                      │
│  ┌──────────────────────────────────────┐                    │
│  │     NavigationMLP(horizon=H)         │                    │
│  │  7 → 256 → 256 → 256 → (3*H)         │                    │
│  │  giriş : bbox(4) + ee_pos(3)         │                    │
│  │  çıkış : H × EE delta                │                    │
│  └────────────┬─────────────────────────┘                    │
│               │                                              │
│               ▼                                              │
│   H delta'sı sırayla uygula → temas/distance kontrolü        │
│   başarılı → BAŞARILI                                        │
│   H adım sonra → VLM'i tekrar sorgula (H adım = 1 query)     │
└──────────────────────────────────────────────────────────────┘
```

---

## Metrikler

| Metrik | Açıklama |
|---|---|
| Success rate | Nesneye dokunan episode yüzdesi |
| Mean final distance | Episode sonundaki EE-nesne mesafesi |
| Mean steps (success) | Başarılı episodelarda ortalama adım sayısı |
| Test MSE | MLP'nin test setindeki tahmin hatası |
| Test RMSE | √(MSE) — normalize koordinat uzayında |
