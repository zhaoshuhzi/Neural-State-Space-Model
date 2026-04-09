from __future__ import annotations

import math
from typing import Dict

import torch
from torch import Tensor, nn


class SpatiotemporalEncoder(nn.Module):
    """Temporal encoder for ROI / source-space EEG sequences.

    Input shape: [batch, time, roi_dim]
    Output shape: [batch, time, hidden_dim]
    """

    def __init__(
        self,
        roi_dim: int,
        hidden_dim: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(roi_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)  # [B, N, T]
        h = self.net(x)
        h = h.transpose(1, 2)  # [B, T, H]
        return self.norm(h)


class DynamicNetworkDecoder(nn.Module):
    """Decode latent states into a dynamic functional network."""

    def __init__(self, latent_dim: int, roi_dim: int, node_embed_dim: int = 8) -> None:
        super().__init__()
        self.roi_dim = roi_dim
        self.node_embed_dim = node_embed_dim
        self.node_proj = nn.Linear(latent_dim, roi_dim * node_embed_dim)

    def forward(self, z_seq: Tensor) -> Tensor:
        batch_size, seq_len, _ = z_seq.shape
        node_emb = self.node_proj(z_seq).view(batch_size, seq_len, self.roi_dim, self.node_embed_dim)
        adj = torch.matmul(node_emb, node_emb.transpose(-1, -2)) / math.sqrt(self.node_embed_dim)
        adj = 0.5 * (adj + adj.transpose(-1, -2))
        return torch.sigmoid(adj)


class NeuralStateSpaceModel(nn.Module):
    """Input-driven neural state-space model.

    State update:
        z_{t+1} = (1 - g_t) * z_t + g_t * tanh(W_z z_t + W_e e_t + W_u u_t)

    Readout:
        y_t = C z_t + D u_t

    The explicit stimulus pathway (W_u and D) is useful for discovering
    stimulation-driven activation regions.
    """

    def __init__(
        self,
        roi_dim: int,
        stim_dim: int = 1,
        encoder_dim: int = 64,
        latent_dim: int = 32,
        dropout: float = 0.1,
        predict_network: bool = False,
        node_embed_dim: int = 8,
    ) -> None:
        super().__init__()
        self.roi_dim = roi_dim
        self.stim_dim = stim_dim
        self.encoder_dim = encoder_dim
        self.latent_dim = latent_dim
        self.predict_network = predict_network

        self.encoder = SpatiotemporalEncoder(roi_dim=roi_dim, hidden_dim=encoder_dim, dropout=dropout)
        self.init_state = nn.Sequential(
            nn.Linear(roi_dim, latent_dim),
            nn.Tanh(),
        )

        self.state_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.enc_proj = nn.Linear(encoder_dim, latent_dim)
        self.stim_to_state = nn.Linear(stim_dim, latent_dim, bias=False)

        self.gate_state = nn.Linear(latent_dim, latent_dim, bias=False)
        self.gate_enc = nn.Linear(encoder_dim, latent_dim)
        self.gate_stim = nn.Linear(stim_dim, latent_dim, bias=False)

        self.state_to_roi = nn.Linear(latent_dim, roi_dim)
        self.stim_to_roi = nn.Linear(stim_dim, roi_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        if predict_network:
            self.network_decoder = DynamicNetworkDecoder(
                latent_dim=latent_dim,
                roi_dim=roi_dim,
                node_embed_dim=node_embed_dim,
            )
        else:
            self.network_decoder = None

    def step(self, z_prev: Tensor, e_t: Tensor, u_t: Tensor) -> Tensor:
        proposal = torch.tanh(
            self.state_proj(z_prev) + self.enc_proj(e_t) + self.stim_to_state(u_t)
        )
        gate = torch.sigmoid(
            self.gate_state(z_prev) + self.gate_enc(e_t) + self.gate_stim(u_t)
        )
        z_t = (1.0 - gate) * z_prev + gate * proposal
        return self.dropout(z_t)

    def forward(self, x: Tensor, u: Tensor) -> Dict[str, Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x must be [batch, time, roi], got shape={tuple(x.shape)}")
        if u.ndim != 3:
            raise ValueError(f"u must be [batch, time, stim_dim], got shape={tuple(u.shape)}")
        if x.size(0) != u.size(0) or x.size(1) != u.size(1):
            raise ValueError("x and u must share batch/time dimensions")

        encoded = self.encoder(x)
        _, seq_len, _ = x.shape
        z_t = self.init_state(x[:, 0, :])

        z_seq = []
        y_seq = []
        for t in range(seq_len):
            e_t = encoded[:, t, :]
            u_t = u[:, t, :]
            z_t = self.step(z_t, e_t, u_t)
            y_t = self.state_to_roi(z_t) + self.stim_to_roi(u_t)
            z_seq.append(z_t)
            y_seq.append(y_t)

        z_seq = torch.stack(z_seq, dim=1)
        y_hat = torch.stack(y_seq, dim=1)

        out: Dict[str, Tensor] = {
            "y_hat": y_hat,
            "z_seq": z_seq,
            "encoded": encoded,
        }
        if self.network_decoder is not None:
            out["adj_hat"] = self.network_decoder(z_seq)
        return out

    def smoothness_penalty(self, z_seq: Tensor) -> Tensor:
        if z_seq.size(1) < 2:
            return z_seq.new_tensor(0.0)
        return ((z_seq[:, 1:, :] - z_seq[:, :-1, :]) ** 2).mean()

    def stimulus_sparsity_penalty(self) -> Tensor:
        return self.stim_to_state.weight.abs().mean() + self.stim_to_roi.weight.abs().mean()

    @torch.no_grad()
    def direct_stimulus_roi_weights(self) -> Tensor:
        state_path = self.state_to_roi.weight @ self.stim_to_state.weight
        direct_path = self.stim_to_roi.weight
        return state_path + direct_path
