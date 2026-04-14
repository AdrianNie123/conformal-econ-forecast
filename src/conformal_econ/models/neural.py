"""LSTM forecasting model (PyTorch).

2-layer LSTM with a linear output head. Trained with Adam and early stopping.
Normalizes the training series to zero mean and unit variance before training,
un-normalizes on predict(). Device is auto-detected: CUDA → MPS → CPU.

predict_gaussian raises NotImplementedError by design — LSTM is a conformal-only
model. ConformalWrapper in conformal/wrappers.py handles the interval generation.

Spec: PRD Section 5.2 — 2 layers, hidden_size=64, dropout=0.2, 24-step lookback,
Adam lr=1e-3, early stopping patience=10.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from conformal_econ.models.base import ForecastModel

_DEFAULT_HIDDEN = 64
_DEFAULT_LAYERS = 2
_DEFAULT_DROPOUT = 0.2
_DEFAULT_LOOKBACK = 24
_DEFAULT_LR = 1e-3
_DEFAULT_PATIENCE = 10
_DEFAULT_MAX_EPOCHS = 200
_VAL_FRAC = 0.10  # Last 10% of training data used for early-stopping validation
_BATCH_SIZE = 32


def _get_device() -> torch.device:
    """Auto-detect best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _LSTMNet(nn.Module):
    """2-layer LSTM with a linear output head.

    Input shape: (batch, seq_len, 1) — univariate series, one feature.
    Output shape: (batch,) — single next-step prediction.

    Args:
        hidden_size: Number of features in the LSTM hidden state.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout applied between LSTM layers (ignored if num_layers=1).
    """

    def __init__(
        self,
        hidden_size: int = _DEFAULT_HIDDEN,
        num_layers: int = _DEFAULT_LAYERS,
        dropout: float = _DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, 1).

        Returns:
            Tensor of shape (batch,) — predicted next value for each sequence.
        """
        out, _ = self.lstm(x)
        # Only the last hidden state feeds into the output head.
        return self.linear(out[:, -1, :]).squeeze(-1)


def _build_sequences(
    series: np.ndarray, lookback: int
) -> tuple[np.ndarray, np.ndarray]:
    """Slice a series into overlapping input windows and corresponding targets.

    Args:
        series: Normalized 1-D float array.
        lookback: Length of each input window.

    Returns:
        Tuple (X, y) where X has shape (n - lookback, lookback, 1) and
        y has shape (n - lookback,).
    """
    n = len(series)
    x_seqs = np.array(
        [series[i : i + lookback] for i in range(n - lookback)], dtype=np.float32
    )
    y = series[lookback:].astype(np.float32)
    return x_seqs[:, :, np.newaxis], y


class LSTMModel(ForecastModel):
    """2-layer LSTM for univariate economic time series forecasting.

    Normalizes training data (z-score), trains with early stopping on a held-out
    validation tail, then un-normalizes predictions. Multi-step forecasting is
    recursive: each predicted value is fed back as the next input.

    predict_gaussian raises NotImplementedError — use ConformalWrapper for
    valid prediction intervals.

    Args:
        hidden_size: LSTM hidden dimension. Default 64 per PRD.
        num_layers: Number of LSTM layers. Default 2 per PRD.
        dropout: Dropout between layers. Default 0.2 per PRD.
        lookback: Input sequence length. Default 24 per PRD.
        lr: Adam learning rate. Default 1e-3 per PRD.
        patience: Early stopping patience (epochs without val improvement).
        max_epochs: Maximum training epochs. Set low (e.g. 3) for fast tests.
    """

    def __init__(
        self,
        hidden_size: int = _DEFAULT_HIDDEN,
        num_layers: int = _DEFAULT_LAYERS,
        dropout: float = _DEFAULT_DROPOUT,
        lookback: int = _DEFAULT_LOOKBACK,
        lr: float = _DEFAULT_LR,
        patience: int = _DEFAULT_PATIENCE,
        max_epochs: int = _DEFAULT_MAX_EPOCHS,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lookback = lookback
        self.lr = lr
        self.patience = patience
        self.max_epochs = max_epochs

        self._device = _get_device()
        self._net: _LSTMNet | None = None
        self._mean: float = 0.0
        self._std: float = 1.0
        self._last_window: np.ndarray | None = None  # Last lookback values after fit

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return "LSTM"

    def fit(self, y_train: np.ndarray) -> None:
        """Train the LSTM on a univariate series.

        Normalizes the series, splits off the last 10% as a validation set for
        early stopping, then trains until val loss stops improving.

        Args:
            y_train: 1-D array of training observations, oldest to newest.
                Must have length > lookback + a few observations for val split.
        """
        if len(y_train) < self.lookback + 2:
            raise ValueError(
                f"Training series too short: need at least {self.lookback + 2} "
                f"observations, got {len(y_train)}."
            )

        # Normalize: z-score using training statistics.
        self._mean = float(np.mean(y_train))
        self._std = float(np.std(y_train)) or 1.0
        z = (y_train - self._mean) / self._std

        # Train/val split: last _VAL_FRAC of sequences go to validation.
        x_all, y_all = _build_sequences(z, self.lookback)
        n_val = max(1, int(len(x_all) * _VAL_FRAC))
        x_tr, y_tr = x_all[:-n_val], y_all[:-n_val]
        x_val, y_val = x_all[-n_val:], y_all[-n_val:]

        # If training set after val split is empty, use everything for training.
        if len(x_tr) == 0:
            x_tr, y_tr = x_all, y_all
            x_val, y_val = x_all, y_all  # Overfit check on same data; acceptable for tiny series

        # Build DataLoader for training.
        train_ds = TensorDataset(
            torch.from_numpy(x_tr).to(self._device),
            torch.from_numpy(y_tr).to(self._device),
        )
        train_loader = DataLoader(train_ds, batch_size=_BATCH_SIZE, shuffle=False)
        x_val_t = torch.from_numpy(x_val).to(self._device)
        y_val_t = torch.from_numpy(y_val).to(self._device)

        # Build fresh network.
        self._net = _LSTMNet(self.hidden_size, self.num_layers, self.dropout).to(
            self._device
        )
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state: dict[str, torch.Tensor] = {}

        self._net.train()
        for _ in range(self.max_epochs):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(self._net(xb), yb)
                loss.backward()
                optimizer.step()

            # Validation loss for early stopping.
            self._net.eval()
            with torch.no_grad():
                val_loss = float(criterion(self._net(x_val_t), y_val_t).item())
            self._net.train()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        # Restore best weights.
        if best_state:
            self._net.load_state_dict(best_state)
        self._net.eval()

        # Store last lookback window for recursive prediction.
        self._last_window = z[-self.lookback :].copy()

    def predict(self, horizon: int) -> np.ndarray:
        """Recursive multi-step forecast.

        Feeds each predicted value back as the next input, just like the tree
        models. Predictions are in normalized space; un-normalization is applied
        at the end.

        Args:
            horizon: Number of steps ahead to forecast.

        Returns:
            1-D array of length `horizon` in original (un-normalized) scale.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._net is None or self._last_window is None:
            raise RuntimeError("Call fit() before predict().")

        self._net.eval()
        window = self._last_window.copy().astype(np.float32)
        preds_z: list[float] = []

        with torch.no_grad():
            for _ in range(horizon):
                x = torch.tensor(window, dtype=torch.float32, device=self._device)
                x = x.unsqueeze(0).unsqueeze(-1)  # (1, lookback, 1)
                pred_z = float(self._net(x).item())
                preds_z.append(pred_z)
                # Slide window forward.
                window = np.append(window[1:], pred_z)

        # Un-normalize.
        return np.array(preds_z, dtype=np.float64) * self._std + self._mean

    def predict_gaussian(
        self, horizon: int, alpha: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Not implemented — LSTM has no parametric interval.

        Use ConformalWrapper from conformal.wrappers for valid intervals.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "LSTM has no parametric prediction interval. "
            "Use ConformalWrapper for conformal intervals."
        )
