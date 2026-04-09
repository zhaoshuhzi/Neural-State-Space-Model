from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nssm.data import NpzSequenceDataset, SyntheticStimActivationDataset
from nssm.models import NeuralStateSpaceModel
from nssm.utils import compute_activation_scores, plot_activation_scores, save_activation_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer stimulation-activated regions using a trained neural SSM.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", choices=["synthetic", "npz"], default="synthetic")
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--outdir", type=str, default="runs/infer")
    parser.add_argument("--num-samples", type=int, default=256, help="Synthetic dataset only")
    parser.add_argument("--seq-len", type=int, default=80, help="Synthetic dataset only")
    parser.add_argument("--num-rois", type=int, default=16, help="Synthetic dataset only")
    return parser.parse_args()


def build_dataset(args: argparse.Namespace):
    if args.dataset == "synthetic":
        return SyntheticStimActivationDataset(
            num_samples=args.num_samples,
            seq_len=args.seq_len,
            num_rois=args.num_rois,
            seed=42,
        )
    if not args.data_path:
        raise ValueError("--data-path is required when --dataset=npz")
    return NpzSequenceDataset(args.data_path)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]

    model = NeuralStateSpaceModel(
        roi_dim=config["roi_dim"],
        stim_dim=config["stim_dim"],
        encoder_dim=config["encoder_dim"],
        latent_dim=config["latent_dim"],
        dropout=config["dropout"],
        predict_network=config.get("predict_network", False),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    scores = compute_activation_scores(model, loader, device=device)
    roi_names = dataset.roi_names
    save_activation_csv(outdir / "activation_scores.csv", roi_names, scores)
    plot_activation_scores(outdir / "activation_scores.png", roi_names, scores, top_k=min(10, len(roi_names)))
    print(f"Saved activation ranking to: {outdir}")


if __name__ == "__main__":
    main()
