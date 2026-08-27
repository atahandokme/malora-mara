"""
SSM Channel Implementation

A diagonal State Space Model channel with learnable timescale.
Each channel maintains a hidden state that evolves over the sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def inverse_softplus(x):
    """Inverse of softplus for initialization."""
    return torch.log(torch.exp(torch.tensor(x)) - 1.0).item()


class SSMChannel(nn.Module):
    """
    A single SSM channel with diagonal state matrix.

    State evolution:
        h_t = A_bar * h_{t-1} + B * x_t * delta
        y_t = C * h_t

    where:
        A_bar = exp(delta * A), A is negative (stable)
        delta controls timescale (small = slow decay, large = fast decay)

    Args:
        d_adapter: Input/output dimension (bottleneck dimension)
        state_dim: Hidden state dimension N
        delta_init: Initial timescale value (0.001 for slow, 0.1 for fast)
    """

    def __init__(self, d_adapter: int, state_dim: int, delta_init: float = 0.01):
        super().__init__()
        self.d_adapter = d_adapter
        self.state_dim = state_dim
        self.delta_init = delta_init

        # A matrix: diagonal, negative for stability
        # Initialize with diverse values from 0.1 to 10
        # Store as log for unconstrained optimization
        log_A_init = torch.log(torch.linspace(0.1, 10.0, state_dim))
        self.log_A = nn.Parameter(log_A_init)

        # B: input to state projection
        self.B_proj = nn.Linear(d_adapter, state_dim, bias=False)

        # C: state to output projection
        self.C_proj = nn.Linear(state_dim, d_adapter, bias=False)

        # Delta: discretization step size (controls timescale)
        # Initialize to inverse_softplus(delta_init) for correct initial value
        log_delta_init = torch.full((d_adapter,), inverse_softplus(delta_init))
        self.log_delta = nn.Parameter(log_delta_init)

        # Initialize projections with small values
        nn.init.xavier_uniform_(self.B_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.C_proj.weight, gain=0.1)

    def get_A(self):
        """Get the A matrix (negative, for stability)."""
        return -torch.exp(self.log_A)

    def get_delta(self):
        """Get the discretization step size."""
        return F.softplus(self.log_delta)

    def get_effective_decay(self):
        """
        Get the effective decay rate A_bar = exp(delta * A).
        Returns the mean decay rate across state dimensions.
        """
        A = self.get_A()
        delta = self.get_delta().mean()
        A_bar = torch.exp(delta * A)
        return A_bar.mean().item()

    def forward(self, x):
        """
        Forward pass with sequential scan.

        Args:
            x: (batch, seq_len, d_adapter)

        Returns:
            y: (batch, seq_len, d_adapter)
        """
        batch, seq_len, d = x.shape
        assert d == self.d_adapter

        # Get SSM parameters
        A = self.get_A()  # (state_dim,)
        delta = self.get_delta().mean()  # Scalar (average across input dims)
        A_bar = torch.exp(delta * A)  # (state_dim,) - decay rates in (0, 1)

        # Initialize hidden state
        h = torch.zeros(batch, self.state_dim, device=x.device, dtype=x.dtype)

        # Sequential scan over time
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, d_adapter)

            # Update state: h_t = A_bar * h_{t-1} + B * x_t * delta
            B_x_t = self.B_proj(x_t) * delta  # (batch, state_dim)
            h = A_bar * h + B_x_t  # (batch, state_dim)

            # Output: y_t = C * h_t
            y_t = self.C_proj(h)  # (batch, d_adapter)
            outputs.append(y_t)

        # Stack outputs
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_adapter)
        return y


if __name__ == "__main__":
    # Standalone test
    print("Testing SSMChannel...")

    # Test different timescales
    for delta_init, label in [(0.001, "slow"), (0.01, "medium"), (0.1, "fast")]:
        print(f"\n{label.upper()} channel (delta_init={delta_init}):")
        ch = SSMChannel(d_adapter=128, state_dim=64, delta_init=delta_init)

        x = torch.randn(2, 50, 128)
        y = ch(x)

        assert y.shape == (2, 50, 128), f"Expected shape (2, 50, 128), got {y.shape}"
        print(f"  ✓ Output shape: {y.shape}")
        print(f"  ✓ Effective decay rate: {ch.get_effective_decay():.6f}")
        print(f"  ✓ Mean delta: {ch.get_delta().mean().item():.6f}")

    print("\n✅ All tests passed!")
