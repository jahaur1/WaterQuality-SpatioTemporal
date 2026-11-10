"""Native XLinear plus known future covariates for exact +24-hour targets.

The runner owns all timestamp alignment and training-only external scaling.
This module consumes nine historical channels (five easy + four chemical),
and optionally five easy covariates observed at the exact target timestamp.
The original XLinear Model, including its historical-window normalization,
is used unchanged.  Both branches output the runner's standardized chemical
coordinates; the runner must invert the chemical scaler after inference.
"""

from types import SimpleNamespace

import torch
from torch import nn

try:
    from .vendor_XLinear import Model as NativeXLinear
except ImportError:
    from vendor_XLinear import Model as NativeXLinear


class XLinearConditional24(nn.Module):
    """Forecast four chemical values from history and known +24h easy values.

    ``history`` has shape [B, L, 9], with water temperature, pH, dissolved
    oxygen, conductivity, turbidity followed by permanganate index, ammonia
    nitrogen, total phosphorus, total nitrogen.  Its last timestamp is the
    forecast origin. ``future_easy`` is [B, 1, 5] at exactly origin +24h.

    With ``use_future=False``, forward never reads ``future_easy``.  The same
    branch parameters are retained so ablations can share architecture and
    model-state structure.  Temporal validity is checked by the runner, not
    inferred from tensor positions.
    """

    history_channels = 9
    future_channels = 5
    output_channels = 4

    def __init__(
        self,
        seq_len=6,
        d_model=16,
        t_ff=16,
        c_ff=16,
        dropout=0.1,
        use_future=True,
    ):
        super().__init__()
        if seq_len <= 0 or d_model <= 0 or t_ff <= 0 or c_ff <= 0:
            raise ValueError("Sequence and hidden dimensions must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.seq_len = seq_len
        self.use_future = bool(use_future)
        configs = SimpleNamespace(
            seq_len=seq_len,
            pred_len=1,
            d_model=d_model,
            enc_in=self.history_channels,
            t_ff=t_ff,
            c_ff=c_ff,
            usenorm=1,
            embed_dropout=dropout,
            head_dropout=dropout,
            t_dropout=dropout,
            c_dropout=dropout,
            features="M",
        )
        self.native = NativeXLinear(configs)
        self.future_branch = nn.Sequential(
            nn.Linear(self.future_channels, 16),
            nn.ReLU(),
            nn.Linear(16, self.output_channels),
        )
        # Begin at the native model prediction, then learn a correction.
        nn.init.zeros_(self.future_branch[-1].weight)
        nn.init.zeros_(self.future_branch[-1].bias)

    def forward(self, history, future_easy=None):
        if (
            history.ndim != 3
            or history.shape[1:] != (self.seq_len, self.history_channels)
        ):
            raise ValueError(
                f"Expected history [batch, {self.seq_len}, 9], "
                f"got {tuple(history.shape)}."
            )
        # The native model has already undone its own history-window RevIN.
        # The values remain in the runner's externally standardized space.
        chemical = self.native(history)[:, 0, 5:9]
        if self.use_future:
            if (
                future_easy is None
                or future_easy.ndim != 3
                or future_easy.shape != (history.shape[0], 1, self.future_channels)
            ):
                actual = None if future_easy is None else tuple(future_easy.shape)
                raise ValueError(
                    f"Expected future_easy [batch, 1, 5] for exact +24h; got {actual}."
                )
            chemical = chemical + self.future_branch(future_easy[:, 0, :])
        return chemical


if __name__ == "__main__":
    import json

    torch.manual_seed(2025)
    model = XLinearConditional24().eval()
    history = torch.randn(4, 6, 9)
    future = torch.randn(4, 1, 5)
    with torch.no_grad():
        result = model(history, future)
        assert result.shape == (4, 4)
        assert torch.isfinite(result).all()
        native_result = model.native(history)[:, 0, 5:9]
        assert torch.equal(result, native_result)
        assert torch.count_nonzero(model.future_branch[-1].weight) == 0
        assert torch.count_nonzero(model.future_branch[-1].bias) == 0
        model.use_future = False
        without_future = model(history, None)
        shuffled_future = model(history, future.flip(0))
        assert torch.equal(without_future, shuffled_future)
        assert torch.equal(without_future, native_result)
    print(json.dumps({
        "history_shape": list(history.shape),
        "future_easy_shape": list(future.shape),
        "output_shape": list(result.shape),
        "parameters": sum(p.numel() for p in model.parameters()),
        "native_parameters": sum(p.numel() for p in model.native.parameters()),
        "future_branch_parameters": sum(
            p.numel() for p in model.future_branch.parameters()
        ),
        "future_correction_zero_initialized": True,
        "no_future_mode_ignores_future_input": True,
        "training_performed": False,
    }, indent=2))
