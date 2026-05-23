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

## Düzeltilen Hatalar

| Hata | Sonuç | Düzeltme |
|------|-------|----------|
| `world_to_pixel` yanlış depth işareti | `gt_bbox` her zaman `(0.5, 0.5)` → model konumu hiç öğrenmedi | `depth = -p_cam[2]` yapıldı |
| Eğitim GT bbox, inference VLM bbox | Distribution shift → %0 başarı | Eğitim GT bbox ile, VLM robustluğu inference tarafından sağlanır |
| Yetersiz veri (639 sample) | 48 trajektori az | Daha fazla sahne toplanmalı |

---

## Proje Yapısı

```
final_project/
├── src/
│   ├── env.py          – MuJoCo ortamı (rastgele nesneli sahneler)
│   ├── utils.py        – Kamera projeksiyonu, EE normalizasyonu, bbox hesaplama
│   ├── collect.py      – Veri toplama
│   ├── dataset.py      – Dataset loader
│   ├── model.py        – NavigationMLP (behaviour cloning)
│   ├── train.py        – Eğitim döngüsü
│   ├── vlm_client.py   – VLM HTTP istemcisi (vLLM / OpenAI uyumlu)
│   ├── inference.py    – Kapalı döngü inference + değerlendirme
│   └── visualize.py    – Görselleştirme
├── data/trajectories/  – Toplanan trajektori dosyaları (*.pt)
└── runs/navigation/    – Model checkpointleri ve metrikler
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

**Ne yapar:** GT bbox'lar ile `(bbox + ee_pos) → sonraki 5 waypoint` ilişkisini öğrenir. MLP mimarisi: 7 → 256 → 256 → 256 → 15.

```bash
/trinity/home/onuralpguvercin/.conda/envs/robotic_env_311/bin/python \
  final_project/src/train.py \
  --data-dir final_project/data/trajectories \
  --run-dir final_project/runs/navigation \
  --epochs 200
```

**Çıktı:**
- `runs/navigation/best.pt` — en iyi checkpoint
- `runs/navigation/train_metrics.json` — epoch bazlı metrikler
- `runs/navigation/loss_plot.png` — eğitim eğrisi

---

### Adım 4 — Inference Al

**Ne yapar:** Eğitilen modeli VLM ile kapalı döngüde test eder. Her 5 adımda VLM'e bbox sorar, MLP 5 waypoint tahmin eder, robot kolu oraya gider.

```bash
MUJOCO_GL=egl /trinity/home/onuralpguvercin/.conda/envs/robotic_env_311/bin/python \
  final_project/src/inference.py \
  --checkpoint final_project/runs/navigation/best.pt \
  --n-episodes 50 \
  --vlm-url http://localhost:8000 \
  --vlm-model Qwen3 \
  --out-dir final_project/runs/vis \
  --save-vis
```

**Çıktı:**
- `runs/vis/eval_results.json` — her episode: başarı/başarısız, adım sayısı, son mesafe
- `runs/vis/vis_episodes/ep000_red_box_0_frames.png` — her episode için 10 frame (sarı = VLM sorgu anı, kırmızı = sonraki adımlar)

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
│                                                              │
│  env.reset() → hedef nesne seç                               │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────┐   görüntü + nesne adı   ┌──────────┐           │
│  │ MuJoCo  │ ──────────────────────▶ │  Qwen3   │           │
│  │   Env   │ ◀─── bbox (cx,cy,w,h) ─ │  :8000   │           │
│  └────┬────┘                          └──────────┘           │
│       │ ee_pos                                               │
│       ▼                                                      │
│  ┌──────────────────────────┐                                │
│  │     NavigationMLP        │  giriş : bbox(4) + ee_pos(3)  │
│  │  7 → 256 → 256 → 256→15  │  çıkış : 5 × EE waypoint     │
│  └────────────┬─────────────┘                                │
│               │                                              │
│               ▼                                              │
│   waypoint'leri sırayla uygula → temas kontrolü             │
│   temas varsa → BAŞARILI                                     │
│   5 adım sonra → VLM'i tekrar sorgula                       │
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
