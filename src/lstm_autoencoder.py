"""src/lstm_autoencoder.py — the CONDITIONED LSTM autoencoder from the
pitch deck: telemetry + telecommand/state channels in, reconstruction out.

Owned by: Role B. The model class and forward pass are real and testable
today. The training loop is intentionally minimal — get this training on
real ESA-ADB data during the boilerplate-to-solution execution phase, not
as part of Day 1-10 boilerplate.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np


class ConditionedLSTMAutoencoder(nn.Module):
    """Encodes telemetry conditioned on mode/telecommand channels, then
    reconstructs the telemetry. Conditioning is done by concatenating the
    mode channels onto the telemetry input at every timestep — simple,
    and enough to test whether conditioning measurably reduces false
    alarms during mode transitions (the actual claim to test, per the
    execution plan's Milestone 3)."""

    def __init__(self, n_telemetry_channels: int, n_mode_channels: int = 0,
                 hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.n_telemetry_channels = n_telemetry_channels
        self.n_mode_channels = n_mode_channels
        input_size = n_telemetry_channels + n_mode_channels

        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, n_telemetry_channels)

    def forward(self, x_telemetry: torch.Tensor, x_mode: torch.Tensor | None = None) -> torch.Tensor:
        """x_telemetry: [batch, seq_len, n_telemetry_channels]
        x_mode: [batch, seq_len, n_mode_channels] or None (unconditioned mode)
        Returns reconstruction: [batch, seq_len, n_telemetry_channels]
        """
        if x_mode is not None and self.n_mode_channels > 0:
            x = torch.cat([x_telemetry, x_mode], dim=-1)
        else:
            x = x_telemetry

        _, (h, c) = self.encoder(x)
        # repeat the final hidden state across the sequence length as the
        # decoder's input — a standard, simple seq2seq-autoencoder pattern
        seq_len = x_telemetry.shape[1]
        decoder_input = h[-1].unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(decoder_input)
        return self.output_layer(decoded)


def reconstruction_error(model: ConditionedLSTMAutoencoder, x_telemetry: torch.Tensor,
                          x_mode: torch.Tensor | None = None) -> torch.Tensor:
    """Per-window reconstruction error (mean squared error), the raw signal
    that thresholding.py and events.py operate on."""
    model.eval()
    with torch.no_grad():
        recon = model(x_telemetry, x_mode)
        error = ((recon - x_telemetry) ** 2).mean(dim=(1, 2))
    return error


def train_minimal(model: ConditionedLSTMAutoencoder, x_telemetry: np.ndarray,
                   x_mode: np.ndarray | None = None, epochs: int = 10,
                   lr: float = 1e-3, batch_size: int = 32) -> list[float]:
    """Minimal training loop — trains on windows assumed to be mostly
    normal (per the deck's 'works with mostly-normal data' claim, i.e. no
    labeled anomaly examples required). Returns the loss history so you can
    sanity-check it's actually decreasing before trusting the model.

    NOT tuned — this is deliberately the simplest thing that trains, so
    Role B can verify the architecture works before spending real time on
    hyperparameters during the execution phase.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    x_t = torch.tensor(x_telemetry, dtype=torch.float32)
    x_m = torch.tensor(x_mode, dtype=torch.float32) if x_mode is not None else None

    n = len(x_t)
    losses = []
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)  # shuffle WINDOWS (not within-window order) — fine, since
                                    # each window is already a valid chronological sequence
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch_t = x_t[idx]
            batch_m = x_m[idx] if x_m is not None else None
            optimizer.zero_grad()
            recon = model(batch_t, batch_m)
            loss = loss_fn(recon, batch_t)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        losses.append(epoch_loss / n)
    return losses
