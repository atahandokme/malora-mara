"""
MaDoRA: Mamba-Gated DoRA (Weight-Decomposed Low-Rank Adaptation with Token-Dependent Gating)

Combines DoRA's magnitude-direction decomposition with MaLoRA's token-dependent gating:

    h_t = m * (W₀ x_t + λ_t · B A x_t) / ||W₀ + λ_t · BA||_col

Where:
    - m: learnable magnitude vector (per output neuron), from DoRA
    - λ_t: scalar gate from Mamba SSM, per token
    - A, B: LoRA low-rank matrices
    - ||·||_col: column-wise L2 norm

The scalar gate λ_t allows efficient norm computation:
    ||W₀ + λ_t·BA||² = ||W₀||² + 2λ_t·<W₀, BA> + λ_t²·||BA||²
    (three precomputed scalars, trivial per-token cost)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


SOFTPLUS_BIAS_FOR_INIT_ONE = math.log(math.exp(1.0) - 1.0)  # ≈ 0.5414


class MaDoRAScalarGate(nn.Module):
    """
    Mamba SSM gate producing a scalar λ_t per token.
    Uses hproj mode: x_t → W_down → Mamba → scalar output.
    """

    def __init__(self, input_dim, rank, d_state=16, d_conv=4, expand_factor=2,
                 gate_activation="softplus"):
        super().__init__()
        from mambapy.mamba import MambaBlock, MambaConfig

        self.gate_activation = gate_activation

        # Project hidden state to rank space for Mamba input
        self.down = nn.Linear(input_dim, rank, bias=False)
        nn.init.xavier_uniform_(self.down.weight, gain=0.1)

        # Scalar output projection
        self.out = nn.Linear(rank, 1, bias=False)
        nn.init.xavier_uniform_(self.out.weight, gain=0.1)

        # Mamba SSM
        mamba_config = MambaConfig(
            d_model=rank,
            n_layers=1,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            dt_min=0.001,
            dt_max=0.1,
            pscan=True,
            use_cuda=False,
        )
        self.mamba = MambaBlock(mamba_config)

        # Gate bias
        self.gate_bias = nn.Parameter(torch.zeros(1))
        if gate_activation == "softplus":
            nn.init.constant_(self.gate_bias, SOFTPLUS_BIAS_FOR_INIT_ONE)

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self.mamba = self.mamba.float()
        return result

    def forward(self, x):
        """
        Args:
            x: (batch, seq, d) hidden state
        Returns:
            λ: (batch, seq, 1) scalar gate per token
        """
        input_dtype = x.dtype
        h = self.down(x)                          # (batch, seq, rank)
        h = self.mamba(h.float()).to(input_dtype)  # (batch, seq, rank)
        z = self.out(h) + self.gate_bias           # (batch, seq, 1)

        if self.gate_activation == "softplus":
            return F.softplus(z).to(input_dtype)
        elif self.gate_activation == "tanh":
            return (1.0 + torch.tanh(z)).to(input_dtype)
        else:
            return (2.0 * torch.sigmoid(z)).to(input_dtype)


class MaDoRALinear(nn.Module):
    """
    MaDoRA: DoRA + token-dependent scalar gating.

    h_t = m * (W₀ x_t + λ_t · B A x_t) / ||W₀ + λ_t · BA||_col

    Efficient norm: precompute ||W₀||², <W₀, BA>, ||BA||² per column.
    Per-token cost: 1 scalar multiply + sqrt.
    """

    def __init__(self, original_linear, rank, alpha, gate_module, dropout=0.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # DoRA magnitude vector (per output neuron)
        # Initialize to column norms of W₀
        with torch.no_grad():
            w = original_linear.weight.data  # (out, in)
            self.magnitude = nn.Parameter(w.norm(dim=1))  # (out,)

        # Dropout
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Gate
        self.gate = gate_module

        # Freeze original
        for param in self.original.parameters():
            param.requires_grad = False

        # Precomputed norm components (will be set in _update_norm_cache)
        self.register_buffer('_w0_norm_sq', None)
        self.register_buffer('_w0_dw_dot', None)
        self.register_buffer('_dw_norm_sq', None)
        self._norm_cache_valid = False

    def _update_norm_cache(self):
        """Precompute norm components for efficient per-token normalization."""
        W0 = self.original.weight.data  # (out, in)
        DW = self.lora_B.weight @ self.lora_A.weight * self.scaling  # (out, in)

        # Column norms (over output dim = dim 0)
        self._w0_norm_sq = (W0 ** 2).sum(dim=1)  # (out,)
        self._w0_dw_dot = (W0 * DW).sum(dim=1)   # (out,)
        self._dw_norm_sq = (DW ** 2).sum(dim=1)   # (out,)
        self._norm_cache_valid = True

    def forward(self, x):
        # Base output
        base_out = self.original(x)  # W₀ x_t

        # LoRA
        Ax = self.lora_A(self.lora_dropout(x))  # (batch, seq, r)
        lora_out = self.lora_B(Ax) * self.scaling  # (batch, seq, out)

        # Gate (scalar per token)
        lam = self.gate(x)  # (batch, seq, 1)

        # Gated output: W₀ x_t + λ_t · BA x_t
        adapted_out = base_out + lam * lora_out  # (batch, seq, out)

        # DoRA normalization: m * adapted / ||W₀ + λ_t·BA||
        # Update norm cache if needed (during training, B changes)
        if self.training or not self._norm_cache_valid:
            self._update_norm_cache()

        # Per-token column norm: ||W₀ + λ_t·ΔW||² = ||W₀||² + 2λ·<W₀,ΔW> + λ²·||ΔW||²
        lam_sq = lam.squeeze(-1)  # (batch, seq)
        # Broadcast: (batch, seq, 1) * (out,) → need to expand
        norm_sq = (self._w0_norm_sq.unsqueeze(0).unsqueeze(0)
                   + 2 * lam * self._w0_dw_dot.unsqueeze(0).unsqueeze(0)
                   + lam ** 2 * self._dw_norm_sq.unsqueeze(0).unsqueeze(0))
        # norm_sq: (batch, seq, out)
        norm = torch.sqrt(norm_sq.clamp(min=1e-8))

        # Apply DoRA: m * adapted / norm
        h = self.magnitude.unsqueeze(0).unsqueeze(0) * adapted_out / norm

        return h


def inject_madora(model, config):
    """
    Inject MaDoRA into a pretrained model.

    Args:
        model: pretrained transformer model
        config: dict with rank, alpha, gate params, target_modules, etc.

    Returns:
        model, gate_modules list
    """
    rank = config['rank']
    hidden_dim = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    target_modules = config.get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'up_proj', 'down_proj'])

    print(f"Injecting MaDoRA (scalar gate, rank={rank})...")

    # Freeze all base parameters
    for param in model.parameters():
        param.requires_grad = False

    # Get transformer layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        raise ValueError("Could not find transformer layers")

    gate_modules = []
    replaced = 0

    for layer_idx in range(n_layers):
        layer = layers[layer_idx]

        # Create gate (shared per layer or per matrix)
        gate_sharing = config.get('gate_sharing', 'per_matrix')

        if gate_sharing == 'per_layer':
            shared_gate = MaDoRAScalarGate(
                input_dim=hidden_dim, rank=rank,
                d_state=config.get('d_state', 16),
                d_conv=config.get('d_conv', 4),
                expand_factor=config.get('expand_factor', 2),
                gate_activation=config.get('gate_activation', 'softplus'),
            )
            gate_modules.append(shared_gate)

        for name in target_modules:
            attn = layer.self_attn if hasattr(layer, 'self_attn') else getattr(layer, 'attention', None)
            mlp = getattr(layer, 'mlp', None)

            if attn is not None and hasattr(attn, name):
                parent = attn
            elif mlp is not None and hasattr(mlp, name):
                parent = mlp
            else:
                continue

            original = getattr(parent, name)

            if gate_sharing == 'per_matrix':
                gate = MaDoRAScalarGate(
                    input_dim=original.in_features, rank=rank,
                    d_state=config.get('d_state', 16),
                    d_conv=config.get('d_conv', 4),
                    expand_factor=config.get('expand_factor', 2),
                    gate_activation=config.get('gate_activation', 'softplus'),
                )
                gate_modules.append(gate)
            else:
                gate = shared_gate

            madora = MaDoRALinear(
                original_linear=original,
                rank=rank,
                alpha=config.get('alpha', 32),
                gate_module=gate,
                dropout=config.get('dropout', 0.0),
            )

            device = next(original.parameters()).device
            madora = madora.to(device)
            setattr(parent, name, madora)
            replaced += 1

        print(f"  Layer {layer_idx}: replaced {len(target_modules)} modules")

    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total: {total:,}, Trainable: {trainable:,} ({100*trainable/total:.3f}%)")
    print(f"  Replaced {replaced} modules")

    return model, gate_modules
