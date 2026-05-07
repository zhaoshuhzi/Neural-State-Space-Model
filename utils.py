from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str | Path, model: torch.nn.Module, config: Dict, extra: Dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": config,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, map_location: str | torch.device = "cpu") -> Dict:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["state_dict"])
    return payload


@torch.no_grad()
def compute_activation_scores(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Estimate stimulation-driven activation strength per ROI."""

    model.eval()
    roi_dim = model.roi_dim
    total = np.zeros((roi_dim,), dtype=np.float64)
    count = 0

    for batch in dataloader:
        x = batch["x"].to(device)
        u = batch["u"].to(device)

        stimulated_mask = (u.sum(dim=(1, 2)) > 0)
        if stimulated_mask.sum().item() == 0:
            continue

        x = x[stimulated_mask]
        u = u[stimulated_mask]
        pred_with = model(x, u)["y_hat"]
        pred_without = model(x, torch.zeros_like(u))["y_hat"]
        diff = (pred_with - pred_without).abs()

        for b in range(diff.size(0)):
            onset_idx = torch.where(u[b, :, 0] > 0)[0]
            start = int(onset_idx[0].item()) if len(onset_idx) else 0
            total += diff[b, start:, :].mean(dim=0).cpu().numpy()
            count += 1

    if count == 0:
        raise RuntimeError("No stimulated samples found in dataloader.")
    return (total / count).astype(np.float32)


def save_activation_csv(path: str | Path, roi_names: Sequence[str], scores: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ranking = sorted(zip(roi_names, scores.tolist()), key=lambda x: x[1], reverse=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "roi_name", "activation_score"])
        for rank, (roi_name, score) in enumerate(ranking, start=1):
            writer.writerow([rank, roi_name, f"{score:.6f}"])


def save_json(path: str | Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def plot_activation_scores(
    path: str | Path,
    roi_names: Sequence[str],
    scores: np.ndarray,
    top_k: int = 10,
) -> None:
    ranking = sorted(zip(roi_names, scores.tolist()), key=lambda x: x[1], reverse=True)[:top_k]
    labels = [x[0] for x in ranking]
    values = [x[1] for x in ranking]

    plt.figure(figsize=(12, 5), dpi=150)
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=45, ha="right")
    plt.ylabel("Activation score")
    plt.title(f"Top-{top_k} stimulation-activated ROIs")
    plt.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
