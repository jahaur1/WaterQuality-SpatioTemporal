"""XLinear backbone adapted for same-time five-input/four-target soft sensing.

This adapter is not the repository's original future-forecasting task.  It uses
only ``Forcast_multi`` from a byte-for-byte vendor copy of the original model
file.  The caller must fit X/Y scalers using training data only and invert the
four target scales after inference.  No target history, target-derived feature,
RevIN, or input-channel inverse normalization is used by this module.
"""

import torch
from torch import nn

try:
    from .vendor_XLinear import Forcast_multi
except ImportError:
    from vendor_XLinear import Forcast_multi


class XLinearSoftSensor(nn.Module):
    """Map easy-variable histories ``[B, L, 5]`` to four targets ``[B, 4]``.

    Input channel order: water temperature, pH, dissolved oxygen, conductivity,
    turbidity.  Output order: permanganate index, ammonia nitrogen, total
    phosphorus, total nitrogen.  The history may include the easy measurements
    at the target's time; its target values must never enter this input.
    """

    input_channels = 5
    output_channels = 4

    def __init__(
        self,
        seq_len=6,
        d_model=16,
        t_ff=16,
        c_ff=16,
        dropout=0.1,
    ):
        super().__init__()
        if seq_len <= 0 or d_model <= 0 or t_ff <= 0 or c_ff <= 0:
            raise ValueError("Sequence and hidden dimensions must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.seq_len = seq_len
        self.d_model = d_model
        self.backbone = Forcast_multi(
            seq_len=seq_len,
            d_model=d_model,
            channel=self.input_channels,
            t_ff=t_ff,
            c_ff=c_ff,
            t_dropout=dropout,
            c_dropout=dropout,
            embed_dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(self.input_channels * 2 * d_model, self.output_channels),
        )

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.input_channels):
            raise ValueError(
                f"Expected [batch, {self.seq_len}, {self.input_channels}] easy "
                f"measurements only; got {tuple(x.shape)}."
            )
        return self.head(self.backbone(x.permute(0, 2, 1)))


if __name__ == "__main__":
    # Shape and interface smoke test only; this performs no optimization.
    import inspect
    import json

    torch.manual_seed(2025)
    model = XLinearSoftSensor().eval()
    easy_history = torch.randn(4, 6, 5)
    with torch.no_grad():
        result = model(easy_history)
    assert result.shape == (4, 4)
    assert torch.isfinite(result).all()
    assert list(inspect.signature(model.forward).parameters) == ["x"]
    print(json.dumps({
        "input_shape": list(easy_history.shape),
        "output_shape": list(result.shape),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "only_five_easy_input_channels": True,
        "target_history_input": False,
        "uses_original_model_revin": False,
        "training_performed": False,
    }, indent=2))
