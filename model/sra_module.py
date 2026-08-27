"""
Stateful Reasoning Adapter (SRA) Module

Combines multiple SSM channels with different timescales, a learned router,
and gating to inject recurrent state into frozen transformer layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssm_channel import SSMChannel


class SRAModule(nn.Module):
    """
    Full SRA adapter module.

    Architecture:
        Input h → LayerNorm → Down-project → SSM channels (routed) → Up-project → Gate → Output
        with residual connection: h' = h + gate * adapter_output

    Args:
        hidden_dim: Model hidden dimension D (e.g., 2048 for Llama-1B)
        d_adapter: Bottleneck dimension (e.g., 128)
        state_dim: SSM state dimension N (e.g., 64)
        n_channels: Number of SSM channels K (e.g., 3)
        delta_inits: List of delta values for each channel (e.g., [0.001, 0.01, 0.1])
        gate_bias_init: Initial bias for gate (e.g., -5.0 for near-zero gate at start)
    """

    def __init__(
        self,
        hidden_dim: int,
        d_adapter: int = 128,
        state_dim: int = 64,
        n_channels: int = 3,
        delta_inits: list = None,
        gate_bias_init: float = -5.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_adapter = d_adapter
        self.state_dim = state_dim
        self.n_channels = n_channels

        if delta_inits is None:
            delta_inits = [0.001, 0.01, 0.1]  # slow, medium, fast
        assert len(delta_inits) == n_channels, f"Need {n_channels} delta values"

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

        # Down-projection: D → d
        self.down_proj = nn.Linear(hidden_dim, d_adapter, bias=False)

        # Router: assigns tokens to channels
        self.router = nn.Linear(d_adapter, n_channels, bias=True)

        # SSM channels with different timescales
        self.channels = nn.ModuleList([
            SSMChannel(d_adapter, state_dim, delta_init=delta_inits[i])
            for i in range(n_channels)
        ])

        # Up-projection: d → D
        self.up_proj = nn.Linear(d_adapter, hidden_dim, bias=False)

        # Gate: controls how much adapter output is mixed in
        self.gate_proj = nn.Linear(hidden_dim, 1, bias=True)

        # Initialize projections with small values
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        # Initialize gate to near-zero (start as identity)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, gate_bias_init)

    def forward(self, hidden_states, return_router_weights=False):
        """
        Forward pass.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            return_router_weights: If True, return router weights for visualization

        Returns:
            output: (batch, seq_len, hidden_dim)
            router_weights: (batch, seq_len, n_channels) if return_router_weights=True
        """
        residual = hidden_states

        # Normalize and down-project
        x = self.norm(hidden_states)  # (batch, seq_len, hidden_dim)
        x = self.down_proj(x)  # (batch, seq_len, d_adapter)

        # Router: compute channel weights
        router_logits = self.router(x)  # (batch, seq_len, n_channels)
        router_weights = F.softmax(router_logits, dim=-1)  # (batch, seq_len, n_channels)

        # Process through each SSM channel
        channel_outputs = []
        for i, channel in enumerate(self.channels):
            y_i = channel(x)  # (batch, seq_len, d_adapter)
            channel_outputs.append(y_i)

        # Stack and weight by router
        channel_outputs = torch.stack(channel_outputs, dim=-1)  # (batch, seq_len, d_adapter, n_channels)
        router_weights_expanded = router_weights.unsqueeze(2)  # (batch, seq_len, 1, n_channels)
        weighted_output = (channel_outputs * router_weights_expanded).sum(dim=-1)  # (batch, seq_len, d_adapter)

        # Up-project
        adapter_output = self.up_proj(weighted_output)  # (batch, seq_len, hidden_dim)

        # Gating
        gate = torch.sigmoid(self.gate_proj(adapter_output))  # (batch, seq_len, 1)
        gated_output = gate * adapter_output  # (batch, seq_len, hidden_dim)

        # Residual connection
        output = residual + gated_output

        if return_router_weights:
            return output, router_weights
        return output

    def get_stats(self):
        """
        Get statistics for logging.

        Returns:
            dict with:
                - gate_mean: average gate value
                - router_entropy: average router entropy (higher = more uniform)
                - channel_deltas: list of effective delta for each channel
                - channel_decays: list of effective decay rate for each channel
        """
        stats = {}

        # Gate statistics (requires a forward pass, so we'll compute this during training)
        # For now, just return channel statistics

        # Channel statistics
        channel_deltas = []
        channel_decays = []
        for i, channel in enumerate(self.channels):
            delta = channel.get_delta().mean().item()
            decay = channel.get_effective_decay()
            channel_deltas.append(delta)
            channel_decays.append(decay)

        stats['channel_deltas'] = channel_deltas
        stats['channel_decays'] = channel_decays

        return stats


if __name__ == "__main__":
    # Standalone test
    print("Testing SRAModule...")

    # Test with Llama-1B dimensions
    hidden_dim = 2048
    d_adapter = 128
    state_dim = 64
    n_channels = 3
    delta_inits = [0.001, 0.01, 0.1]

    sra = SRAModule(
        hidden_dim=hidden_dim,
        d_adapter=d_adapter,
        state_dim=state_dim,
        n_channels=n_channels,
        delta_inits=delta_inits,
        gate_bias_init=-5.0,
    )

    # Test forward pass
    batch, seq_len = 2, 50
    h = torch.randn(batch, seq_len, hidden_dim)

    # Forward without router weights
    output = sra(h)
    assert output.shape == (batch, seq_len, hidden_dim)
    print(f"✓ Output shape: {output.shape}")

    # Forward with router weights
    output, router_weights = sra(h, return_router_weights=True)
    assert output.shape == (batch, seq_len, hidden_dim)
    assert router_weights.shape == (batch, seq_len, n_channels)
    print(f"✓ Router weights shape: {router_weights.shape}")

    # Check router weights sum to 1
    router_sum = router_weights.sum(dim=-1)
    assert torch.allclose(router_sum, torch.ones_like(router_sum), atol=1e-6)
    print(f"✓ Router weights sum to 1.0")

    # Check gate is near zero at initialization
    gate_proj_output = sra.gate_proj(output)
    gate_values = torch.sigmoid(gate_proj_output)
    mean_gate = gate_values.mean().item()
    print(f"✓ Mean gate value at init: {mean_gate:.6f} (should be ~0.007)")
    assert mean_gate < 0.05, f"Gate too large: {mean_gate}"

    # Get statistics
    stats = sra.get_stats()
    print(f"\n Channel statistics:")
    for i in range(n_channels):
        print(f"  Channel {i}: delta={stats['channel_deltas'][i]:.6f}, decay={stats['channel_decays'][i]:.6f}")

    # Count parameters
    n_params = sum(p.numel() for p in sra.parameters())
    print(f"\n✓ Total parameters: {n_params:,}")

    print("\n✅ All tests passed!")
