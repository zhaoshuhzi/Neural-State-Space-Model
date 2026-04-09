from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, random_split


class NpzSequenceDataset(Dataset):
    """Dataset for real source / ROI sequences saved in NPZ format."""

    def __init__(self, npz_path: str | Path) -> None:
        payload = np.load(npz_path, allow_pickle=True)
        self.x = payload["X"].astype(np.float32)
        self.u = payload["U"].astype(np.float32)
        self.y = payload["Y"].astype(np.float32) if "Y" in payload else self.x.copy()
        self.adj = payload["A"].astype(np.float32) if "A" in payload else None
        if "roi_names" in payload:
            self.roi_names = [str(x) for x in payload["roi_names"].tolist()]
        else:
            self.roi_names = [f"ROI_{i:02d}" for i in range(self.x.shape[-1])]

        if self.x.ndim != 3 or self.u.ndim != 3 or self.y.ndim != 3:
            raise ValueError("X/U/Y must be 3D arrays: [num_samples, seq_len, dim]")
        if self.x.shape[:2] != self.u.shape[:2] or self.x.shape[:2] != self.y.shape[:2]:
            raise ValueError("X/U/Y must share [num_samples, seq_len]")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        item: Dict[str, Tensor] = {
            "x": torch.from_numpy(self.x[idx]),
            "u": torch.from_numpy(self.u[idx]),
            "y": torch.from_numpy(self.y[idx]),
        }
        if self.adj is not None:
            item["adj"] = torch.from_numpy(self.adj[idx])
        return item


class SyntheticStimActivationDataset(Dataset):
    """Synthetic dataset for end-to-end demo."""

    def __init__(
        self,
        num_samples: int = 512,
        seq_len: int = 80,
        num_rois: int = 16,
        stim_dim: int = 1,
        stimulated_fraction: float = 0.5,
        active_rois: Optional[Sequence[int]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        self.seq_len = seq_len
        self.num_rois = num_rois
        self.stim_dim = stim_dim
        self.roi_names = [f"ROI_{i:02d}" for i in range(num_rois)]

        if active_rois is None:
            default_rois = [1, 4, 7]
            active_rois = [r for r in default_rois if r < num_rois]
            if not active_rois:
                active_rois = list(range(min(2, num_rois)))
        self.active_rois = list(active_rois)

        base = rng.normal(0.0, 0.08, size=(num_rois, num_rois))
        np.fill_diagonal(base, rng.uniform(0.65, 0.82, size=num_rois))
        spectral_radius = max(np.abs(np.linalg.eigvals(base)))
        self.A = (base / max(spectral_radius, 1.0)) * 0.92

        self.B = np.zeros((num_rois, stim_dim), dtype=np.float32)
        for roi in self.active_rois:
            self.B[roi, 0] = rng.uniform(1.1, 1.6)

        self.x = np.zeros((num_samples, seq_len, num_rois), dtype=np.float32)
        self.u = np.zeros((num_samples, seq_len, stim_dim), dtype=np.float32)
        self.y = np.zeros((num_samples, seq_len, num_rois), dtype=np.float32)
        self.adj = np.zeros((num_samples, seq_len, num_rois, num_rois), dtype=np.float32)
        self.labels = np.zeros((num_samples,), dtype=np.int64)

        for n in range(num_samples):
            stimulated = rng.random() < stimulated_fraction
            self.labels[n] = int(stimulated)

            stim = np.zeros((seq_len, stim_dim), dtype=np.float32)
            onset = rng.integers(low=seq_len // 4, high=seq_len // 2)
            duration = rng.integers(low=3, high=8)
            if stimulated:
                stim[onset:onset + duration, 0] = 1.0
            self.u[n] = stim

            x = np.zeros((seq_len, num_rois), dtype=np.float32)
            state = rng.normal(0.0, 0.08, size=(num_rois,))
            for t in range(seq_len):
                local_noise = rng.normal(0.0, 0.05, size=(num_rois,))
                driven = self.B @ stim[t]
                state = self.A @ state + driven + local_noise
                x[t] = state.astype(np.float32)
                self.adj[n, t] = np.abs(np.outer(state, state))
            self.x[n] = x
            self.y[n] = x

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {
            "x": torch.from_numpy(self.x[idx]),
            "u": torch.from_numpy(self.u[idx]),
            "y": torch.from_numpy(self.y[idx]),
            "adj": torch.from_numpy(self.adj[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def split_dataset(dataset: Dataset, val_ratio: float = 0.2, seed: int = 42) -> Tuple[Dataset, Dataset]:
    total = len(dataset)
    val_size = int(round(total * val_ratio))
    train_size = total - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)
