import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from hw1_data import N_ACTIONS, IMG_SHAPE, build_loaders, load_hw1_dataset, resolve_device, set_seeds


class ActionConditionedReconstructor(nn.Module):
    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.ReLU(),
        )
        self.to_latent = nn.Linear(256 * 4 * 4, latent_dim)
        self.from_latent = nn.Linear(latent_dim + N_ACTIONS, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, img_before: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        x = self.encoder(img_before)
        x = x.flatten(start_dim=1)
        z = self.to_latent(x)
        z = torch.cat([z, action_onehot], dim=1)
        y = self.from_latent(z).view(-1, 256, 8, 8)
        return self.decoder(y)


def evaluate(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_l1 = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            img_before = batch["img_before"].to(device)
            action_onehot = batch["action_onehot"].to(device)
            target = batch["img_after"].to(device)
            pred = model(img_before, action_onehot)
            b = img_before.size(0)
            total_mse += F.mse_loss(pred, target, reduction="sum").item()
            total_l1 += F.l1_loss(pred, target, reduction="sum").item()
            total_count += b

    mse = total_mse / (total_count * np.prod(IMG_SHAPE))
    l1 = total_l1 / (total_count * np.prod(IMG_SHAPE))
    psnr = 10 * math.log10(1.0 / max(mse, 1e-12))
    return {"mse": mse, "l1": l1, "psnr": psnr}


def save_examples(model: nn.Module, loader, device: torch.device, out_path: Path, n_samples: int = 8) -> None:
    model.eval()
    with torch.no_grad():
        for batch in loader:
            img_before = batch["img_before"].to(device)
            action_onehot = batch["action_onehot"].to(device)
            target = batch["img_after"].to(device)
            pred = model(img_before, action_onehot)
            n = min(n_samples, img_before.size(0))
            panel = torch.cat([img_before[:n], target[:n], pred[:n]], dim=0).cpu()
            save_image(panel, out_path, nrow=n)
            return


def train(
    data_path: str = "data/hw1",
    run_dir: str = "runs/hw1/reconstruction",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "auto",
) -> Dict[str, float]:
    set_seeds(seed)
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(device)

    dataset = load_hw1_dataset(Path(data_path))
    loaders = build_loaders(dataset, batch_size=batch_size, seed=seed)

    model = ActionConditionedReconstructor().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch in loaders.train:
            img_before = batch["img_before"].to(dev)
            action_onehot = batch["action_onehot"].to(dev)
            target = batch["img_after"].to(dev)
            pred = model(img_before, action_onehot)
            loss = 0.5 * F.mse_loss(pred, target) + 0.5 * F.l1_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * img_before.size(0)
            train_count += img_before.size(0)

        train_loss = train_loss_sum / max(train_count, 1)
        val_metrics = evaluate(model, loaders.val, dev)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_mse": val_metrics["mse"]})
        print(
            f"[RECON] epoch={epoch}/{epochs} train_loss={train_loss:.6f} "
            f"val_mse={val_metrics['mse']:.6f} val_psnr={val_metrics['psnr']:.2f}"
        )

        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            torch.save(model.state_dict(), out_dir / "best.pt")

    test_metrics = test(
        data_path=data_path,
        checkpoint_path=str(out_dir / "best.pt"),
        batch_size=batch_size,
        seed=seed,
        device=device,
        run_dir=run_dir,
        history=history,
    )
    return test_metrics


def test(
    data_path: str = "data/hw1",
    checkpoint_path: str = "runs/hw1/reconstruction/best.pt",
    batch_size: int = 32,
    seed: int = 42,
    device: str = "auto",
    run_dir: str = "runs/hw1/reconstruction",
    history: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, float]:
    set_seeds(seed)
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(device)

    dataset = load_hw1_dataset(Path(data_path))
    loaders = build_loaders(dataset, batch_size=batch_size, seed=seed)

    model = ActionConditionedReconstructor().to(dev)
    model.load_state_dict(torch.load(checkpoint_path, map_location=dev))
    metrics = evaluate(model, loaders.test, dev)
    save_examples(model, loaders.test, dev, out_dir / "reconstruction_samples.png")
    print(f"[RECON] test metrics: {metrics}")

    payload: Dict[str, object] = {"test": metrics}
    if history is not None:
        payload["history"] = history
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Homework 1 - image reconstruction.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data-path", type=str, default="data/hw1")
    p_train.add_argument("--run-dir", type=str, default="runs/hw1/reconstruction")
    p_train.add_argument("--epochs", type=int, default=25)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--device", type=str, default="auto")

    p_test = sub.add_parser("test")
    p_test.add_argument("--data-path", type=str, default="data/hw1")
    p_test.add_argument("--checkpoint-path", type=str, default="runs/hw1/reconstruction/best.pt")
    p_test.add_argument("--run-dir", type=str, default="runs/hw1/reconstruction")
    p_test.add_argument("--batch-size", type=int, default=32)
    p_test.add_argument("--seed", type=int, default=42)
    p_test.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    if args.command == "train":
        train(
            data_path=args.data_path,
            run_dir=args.run_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
        )
    else:
        test(
            data_path=args.data_path,
            checkpoint_path=args.checkpoint_path,
            run_dir=args.run_dir,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )


if __name__ == "__main__":
    main()
