"""
Simplified SRA Module using Mamba (selective SSM).

Replaces custom multi-channel SSM + router with a single Mamba block.
Much simpler, uses proven architecture with parallel scan.

Architecture:
    Input h → LayerNorm → Down-project → MambaBlock → Up-project → Gate → Output
    with residual: h' = h + gate * adapter_output
"""

import torch
import torch.nn as nn
from mambapy.mamba import MambaBlock, MambaConfig


class SRAMambaModule(nn.Module):
    """
    SRA adapter using Mamba's selective SSM.

    Args:
        hidden_dim: Model hidden dimension D (e.g., 3584 for Qwen-7B)
        d_adapter: Bottleneck dimension (e.g., 128)
        d_state: Mamba SSM state dimension (default: 16)
        d_conv: Mamba convolution width (default: 4)
        expand_factor: Mamba expansion factor (default: 2)
        gate_bias_init: Initial gate bias (default: 0.0, sigmoid(0)=0.5)
    """

    def __init__(
        self,
        hidden_dim: int,
        d_adapter: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        gate_bias_init: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_adapter = d_adapter

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

        # Down-projection: D → d
        self.down_proj = nn.Linear(hidden_dim, d_adapter, bias=False)

        # Mamba block (selective SSM with parallel scan)
        mamba_config = MambaConfig(
            d_model=d_adapter,
            n_layers=1,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            dt_min=0.001,
            dt_max=0.1,
            pscan=True,       # Parallel scan (fast!)
            use_cuda=False,    # Pure PyTorch
        )
        self.mamba = MambaBlock(mamba_config)

        # Up-projection: d → D
        self.up_proj = nn.Linear(d_adapter, hidden_dim, bias=False)

        # Gate: controls how much adapter output is mixed in
        self.gate_proj = nn.Linear(hidden_dim, 1, bias=True)

        # Initialize
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, gate_bias_init)

    def to(self, *args, **kwargs):
        """Override to keep Mamba block in float32 (pscan requires it)."""
        result = super().to(*args, **kwargs)
        # Always keep mamba in float32
        self.mamba = self.mamba.float()
        return result

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        residual = hidden_states
        input_dtype = hidden_states.dtype

        # Cast to module dtype if needed (generate() may pass float32)
        module_dtype = self.norm.weight.dtype
        if hidden_states.dtype != module_dtype:
            hidden_states = hidden_states.to(module_dtype)
            residual = residual.to(module_dtype)

        # Normalize and down-project
        x = self.norm(hidden_states)
        x = self.down_proj(x)           # (batch, seq_len, d_adapter)

        # Mamba selective SSM (mambapy uses float32 internally)
        x = x.float()
        x = self.mamba(x)               # (batch, seq_len, d_adapter)
        x = x.to(input_dtype)

        # Up-project
        adapter_output = self.up_proj(x) # (batch, seq_len, hidden_dim)

        # Gating
        gate = torch.sigmoid(self.gate_proj(adapter_output))
        output = residual + gate * adapter_output

        return output


if __name__ == "__main__":
    print("Testing SRAMambaModule...")

    hidden_dim = 3584  # Qwen-7B
    d_adapter = 128

    sra = SRAMambaModule(
        hidden_dim=hidden_dim,
        d_adapter=d_adapter,
        d_state=16,
        gate_bias_init=-2.0,
    )

    # Test forward
    x = torch.randn(2, 100, hidden_dim)
    out = sra(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    assert out.shape == x.shape

    # Parameter count
    n_params = sum(p.numel() for p in sra.parameters())
    print(f"Parameters: {n_params:,}")

    # Gate value at init
    gate_val = torch.sigmoid(torch.tensor(-2.0)).item()
    print(f"Gate init: sigmoid(-2.0) = {gate_val:.4f} (vs old sigmoid(-5.0) = {torch.sigmoid(torch.tensor(-5.0)).item():.4f})")

    print("\nAll tests passed!")
