from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nssm.data import NpzSequenceDataset, SyntheticStimActivationDataset, split_dataset
from nssm.models import NeuralStateSpaceModel
from nssm.utils import (
    compute_activation_scores,
    plot_activation_scores,
    save_activation_csv,
    save_checkpoint,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a neural state-space model for stimulation-activation discovery.")
    parser.add_argument("--dataset", choices=["synthetic", "npz"], default="synthetic")
    parser.add_argument("--data-path", type=str, default="", help="Path to .npz file when --dataset=npz")
    parser.add_argument("--outdir", type=str, default="runs/demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--encoder-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--predict-network", action="store_true")
    parser.add_argument("--lambda-smooth", type=float, default=1e-3)
    parser.add_argument("--lambda-sparse", type=float, default=1e-3)
    parser.add_argument("--lambda-adj", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=512, help="Synthetic dataset only")
    parser.add_argument("--seq-len", type=int, default=80, help="Synthetic dataset only")
    parser.add_argument("--num-rois", type=int, default=16, help="Synthetic dataset only")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def build_datasets(args: argparse.Namespace):
    if args.dataset == "synthetic":
        dataset = SyntheticStimActivationDataset(
            num_samples=args.num_samples,
            seq_len=args.seq_len,
            num_rois=args.num_rois,
            seed=args.seed,
        )
    else:
        if not args.data_path:
            raise ValueError("--data-path is required when --dataset=npz")
        dataset = NpzSequenceDataset(args.data_path)
    train_set, val_set = split_dataset(dataset, val_ratio=args.val_ratio, seed=args.seed)
    return dataset, train_set, val_set


def build_model(args: argparse.Namespace, roi_dim: int, stim_dim: int) -> NeuralStateSpaceModel:
    return NeuralStateSpaceModel(
        roi_dim=roi_dim,
        stim_dim=stim_dim,
        encoder_dim=args.encoder_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        predict_network=args.predict_network,
    )


def run_epoch(
    model: NeuralStateSpaceModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    lambda_smooth: float,
    lambda_sparse: float,
    lambda_adj: float,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_recon = 0.0
    total_smooth = 0.0
    total_sparse = 0.0
    total_adj = 0.0
    n_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        u = batch["u"].to(device)
        y = batch["y"].to(device)
        adj_target = batch.get("adj")
        if adj_target is not None:
            adj_target = adj_target.to(device)

        out = model(x, u)
        recon_loss = F.mse_loss(out["y_hat"], y)
        smooth_loss = model.smoothness_penalty(out["z_seq"])
        sparse_loss = model.stimulus_sparsity_penalty()

        adj_loss = torch.tensor(0.0, device=device)
        if model.predict_network and lambda_adj > 0 and adj_target is not None:
            adj_loss = F.mse_loss(out["adj_hat"], adj_target)

        loss = recon_loss + lambda_smooth * smooth_loss + lambda_sparse * sparse_loss + lambda_adj * adj_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_smooth += smooth_loss.item()
        total_sparse += sparse_loss.item()
        total_adj += adj_loss.item()
        n_batches += 1

    scale = max(1, n_batches)
    return {
        "loss": total_loss / scale,
        "recon": total_recon / scale,
        "smooth": total_smooth / scale,
        "sparse": total_sparse / scale,
        "adj": total_adj / scale,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, train_set, val_set = build_datasets(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    sample = dataset[0]
    roi_dim = sample["x"].shape[-1]
    stim_dim = sample["u"].shape[-1]
    model = build_model(args, roi_dim=roi_dim, stim_dim=stim_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = {
        "dataset": args.dataset,
        "data_path": args.data_path,
        "roi_dim": roi_dim,
        "stim_dim": stim_dim,
        "encoder_dim": args.encoder_dim,
        "latent_dim": args.latent_dim,
        "dropout": args.dropout,
        "predict_network": args.predict_network,
        "seed": args.seed,
    }
    save_json(outdir / "config.json", config)

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            lambda_smooth=args.lambda_smooth,
            lambda_sparse=args.lambda_sparse,
            lambda_adj=args.lambda_adj,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            lambda_smooth=args.lambda_smooth,
            lambda_sparse=args.lambda_sparse,
            lambda_adj=args.lambda_adj,
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"recon={val_metrics['recon']:.4f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(outdir / "best_model.pt", model, config=config, extra={"history": history})

    save_json(outdir / "history.json", {"history": history})

    checkpoint = torch.load(outdir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    eval_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    try:
        scores = compute_activation_scores(model, eval_loader, device=device)
    except RuntimeError:
        eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        scores = compute_activation_scores(model, eval_loader, device=device)
    roi_names = dataset.roi_names
    save_activation_csv(outdir / "activation_scores.csv", roi_names, scores)
    plot_activation_scores(outdir / "activation_scores.png", roi_names, scores, top_k=min(10, len(roi_names)))

    direct_weights = model.direct_stimulus_roi_weights().detach().cpu().numpy().squeeze(-1).tolist()
    save_json(
        outdir / "direct_stimulus_weights.json",
        {
            "roi_names": roi_names,
            "weights": direct_weights,
        },
    )

    print(f"Saved model and activation results to: {outdir}")


if __name__ == "__main__":
    main()
