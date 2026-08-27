"""
Gated LoRA: SSM/LSTM/MLP-gated Low-Rank Adaptation.

The gate modulates LoRA's contribution at each token position based on
sequence context. For recurrent gates (SSM, LSTM), the gate value depends
on the full history [1..t], making LoRA context-aware.

Architecture per target module (e.g., q_proj):
    h = W_frozen @ x + gate(x) * (alpha/r) * (B @ A @ x)
    where gate(x_t) = gate_scale * sigmoid(GateNetwork(x_1, ..., x_t))

Gate variants:
    - "mamba": Selective SSM gate (recurrent, input-adaptive)
    - "s4":    S4 SSM gate (recurrent, fixed dynamics — ablation for selectivity)
    - "lstm":  LSTM gate (recurrent, gated memory)
    - "gru":   GRU gate (recurrent, lighter gated memory)
    - "mlp":   MLP gate (per-token, no recurrence — ablation baseline)

Gate sharing modes:
    - "per_layer":  One gate shared across Q/K/V/O in each layer
    - "per_matrix": Separate gate per weight matrix (4x gates)
    - "multi_head": Shared Mamba + separate readout heads per matrix

Gate scaling:
    - gate_scale=1.0: g ∈ [0, 1], standard (suppression only)
    - gate_scale=2.0: g ∈ [0, 2], bidirectional (g=1 = standard LoRA at init)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Softplus bias that gives softplus(bias) ≈ 1.0 (so gate starts at standard LoRA)
SOFTPLUS_BIAS_FOR_INIT_ONE = math.log(math.exp(1.0) - 1.0)  # ≈ 0.5414


class FixedGate(nn.Module):
    """Fixed gate that always returns 1.0. Used for layers that get LoRA without gating."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.ones(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)


class MLPGate(nn.Module):
    """Per-token gate with no recurrence. Ablation baseline."""

    def __init__(self, input_dim, bottleneck_dim=32, bias_init=-2.0, gate_scale=1.0):
        super().__init__()
        self.gate_scale = gate_scale
        self.down = nn.Linear(input_dim, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, 1, bias=True)
        nn.init.xavier_uniform_(self.down.weight, gain=0.1)
        nn.init.xavier_uniform_(self.up.weight, gain=0.1)
        nn.init.constant_(self.up.bias, bias_init)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        h = torch.relu(self.down(x))
        return self.gate_scale * torch.sigmoid(self.up(h))  # (batch, seq_len, 1)


class LSTMGate(nn.Module):
    """LSTM-based recurrent gate. Classical recurrence baseline."""

    def __init__(self, input_dim, bottleneck_dim=32, bias_init=-2.0, gate_scale=1.0):
        super().__init__()
        self.gate_scale = gate_scale
        self.down = nn.Linear(input_dim, bottleneck_dim, bias=False)
        self.lstm = nn.LSTM(
            input_size=bottleneck_dim,
            hidden_size=bottleneck_dim,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(bottleneck_dim, 1, bias=True)
        nn.init.xavier_uniform_(self.down.weight, gain=0.1)
        nn.init.xavier_uniform_(self.out.weight, gain=0.1)
        nn.init.constant_(self.out.bias, bias_init)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        h = self.down(x)  # (batch, seq_len, bottleneck_dim)
        h, _ = self.lstm(h)  # (batch, seq_len, bottleneck_dim)
        return self.gate_scale * torch.sigmoid(self.out(h))  # (batch, seq_len, 1)


class GRUGate(nn.Module):
    """GRU-based recurrent gate. Recurrence without selectivity."""

    def __init__(self, input_dim, bottleneck_dim=32, bias_init=-2.0, gate_scale=1.0):
        super().__init__()
        self.gate_scale = gate_scale
        self.down = nn.Linear(input_dim, bottleneck_dim, bias=False)
        self.gru = nn.GRU(
            input_size=bottleneck_dim,
            hidden_size=bottleneck_dim,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(bottleneck_dim, 1, bias=True)
        nn.init.xavier_uniform_(self.down.weight, gain=0.1)
        nn.init.xavier_uniform_(self.out.weight, gain=0.1)
        nn.init.constant_(self.out.bias, bias_init)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        h = self.down(x)          # (batch, seq_len, bottleneck_dim)
        h, _ = self.gru(h)        # (batch, seq_len, bottleneck_dim)
        return self.gate_scale * torch.sigmoid(self.out(h))  # (batch, seq_len, 1)


class S4Gate(nn.Module):
    """S4-style SSM gate. Fixed (data-independent) state transitions with parallel scan.

    Unlike Mamba, B/C/delta are learned parameters, NOT input-dependent.
    This isolates the contribution of input-selectivity in the ablation.
    Uses HiPPO-style A initialization and the same pscan as Mamba.
    """

    def __init__(self, input_dim, bottleneck_dim=32, d_state=16, d_conv=4,
                 expand_factor=2, bias_init=-2.0, gate_scale=1.0):
        super().__init__()
        from mambapy.pscan import pscan
        self.pscan = pscan

        self.gate_scale = gate_scale
        self.d_state = d_state
        d_inner = bottleneck_dim * expand_factor

        self.down = nn.Linear(input_dim, bottleneck_dim, bias=False)

        # Input projection (no gating branch for simplicity, just expand)
        self.in_proj = nn.Linear(bottleneck_dim, d_inner, bias=False)

        # Short conv (same as Mamba, local context)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv, bias=True,
            groups=d_inner, padding=d_conv - 1,
        )

        # Fixed (non-input-dependent) SSM parameters
        # A: HiPPO-style diagonal initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # (d_inner, N)

        # B, C: fixed learned parameters (NOT input-dependent like Mamba)
        self.B = nn.Parameter(torch.randn(d_inner, d_state) * 0.01)  # (d_inner, N)
        self.C = nn.Parameter(torch.randn(d_inner, d_state) * 0.01)  # (d_inner, N)

        # Fixed delta (discretization step) — learned but not input-dependent
        dt_init = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        self.log_dt = nn.Parameter(torch.log(dt_init))  # (d_inner,)

        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection back to bottleneck_dim, then to scalar gate
        self.out_proj = nn.Linear(d_inner, bottleneck_dim, bias=False)
        self.gate_out = nn.Linear(bottleneck_dim, 1, bias=True)

        nn.init.xavier_uniform_(self.down.weight, gain=0.1)
        nn.init.xavier_uniform_(self.gate_out.weight, gain=0.1)
        nn.init.constant_(self.gate_out.bias, bias_init)

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        # Keep SSM params in float32 for pscan stability
        self.A_log.data = self.A_log.data.float()
        self.B.data = self.B.data.float()
        self.C.data = self.C.data.float()
        self.log_dt.data = self.log_dt.data.float()
        self.D.data = self.D.data.float()
        return result

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        input_dtype = x.dtype
        B_size, L, _ = x.shape

        # S4 linear layers may remain float32 when model is bfloat16
        # (injection only calls .to(device), not .to(dtype))
        x = x.float()
        h = self.down(x)  # (B, L, bottleneck_dim)
        h = self.in_proj(h)  # (B, L, d_inner)

        # Conv1d
        h = h.transpose(1, 2)  # (B, d_inner, L)
        h = self.conv1d(h)[:, :, :L]
        h = h.transpose(1, 2)  # (B, L, d_inner)
        h = torch.nn.functional.silu(h)

        # SSM with FIXED parameters
        h_float = h.float()
        A = -torch.exp(self.A_log.float())  # (d_inner, N)
        delta = torch.nn.functional.softplus(self.log_dt.float())  # (d_inner,)

        # Broadcast fixed params across batch and sequence
        # deltaA: (B, L, d_inner, N)
        deltaA = torch.exp(delta.unsqueeze(0).unsqueeze(0).unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaA = deltaA.expand(B_size, L, -1, -1)

        # deltaB * x: (B, L, d_inner, N)
        deltaB = delta.unsqueeze(-1) * self.B.float()  # (d_inner, N)
        BX = deltaB.unsqueeze(0).unsqueeze(0) * h_float.unsqueeze(-1)  # (B, L, d_inner, N)

        # Parallel scan
        hs = self.pscan(deltaA, BX)  # (B, L, d_inner, N)

        # Readout with fixed C
        y = (hs * self.C.float().unsqueeze(0).unsqueeze(0)).sum(-1)  # (B, L, d_inner)
        y = y + self.D.float() * h_float  # skip connection

        # Keep in float32 through output projections (weights may be float32)
        y = self.out_proj(y)  # (B, L, bottleneck_dim)
        return (self.gate_scale * torch.sigmoid(self.gate_out(y))).to(input_dtype)  # (B, L, 1)


class MambaGate(nn.Module):
    """Mamba SSM-based recurrent gate. Input-adaptive with parallel scan."""

    def __init__(self, input_dim, bottleneck_dim=32, d_state=16, d_conv=4,
                 expand_factor=2, bias_init=-2.0, gate_scale=1.0,
                 gate_activation="sigmoid"):
        super().__init__()
        from mambapy.mamba import MambaBlock, MambaConfig

        self.gate_scale = gate_scale
        self.gate_activation = gate_activation
        self.down = nn.Linear(input_dim, bottleneck_dim, bias=False)

        mamba_config = MambaConfig(
            d_model=bottleneck_dim,
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

        self.out = nn.Linear(bottleneck_dim, 1, bias=True)
        nn.init.xavier_uniform_(self.down.weight, gain=0.1)
        nn.init.xavier_uniform_(self.out.weight, gain=0.1)

        # Set bias so gate starts at ~1.0 (standard LoRA) for both activations
        if gate_activation == "softplus":
            nn.init.constant_(self.out.bias, SOFTPLUS_BIAS_FOR_INIT_ONE)
        else:
            nn.init.constant_(self.out.bias, bias_init)

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self.mamba = self.mamba.float()
        return result

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        input_dtype = x.dtype
        h = self.down(x)  # (batch, seq_len, bottleneck_dim)
        h = self.mamba(h.float()).to(input_dtype)  # (batch, seq_len, bottleneck_dim)
        z = self.out(h)  # (batch, seq_len, 1)

        if self.gate_activation == "softplus":
            # Range: (0, ∞), init ≈ 1.0, no ceiling
            return F.softplus(z).to(input_dtype)
        else:
            # Range: [0, gate_scale], init = gate_scale * sigmoid(bias_init)
            return (self.gate_scale * torch.sigmoid(z)).to(input_dtype)


class MultiHeadMambaGate(nn.Module):
    """
    Shared Mamba SSM with per-matrix readout heads (Option 3).

    One Mamba block processes the sequence once, but each weight matrix
    (q_proj, k_proj, v_proj, o_proj) gets its own learned readout head.
    This allows per-matrix gating decisions from a shared recurrent memory.
    """

    def __init__(self, input_dim, n_heads=4, bottleneck_dim=32, d_state=16,
                 d_conv=4, expand_factor=2, bias_init=0.0, gate_scale=2.0):
        super().__init__()
        from mambapy.mamba import MambaBlock, MambaConfig

        self.gate_scale = gate_scale
        self.n_heads = n_heads
        self.down = nn.Linear(input_dim, bottleneck_dim, bias=False)

        mamba_config = MambaConfig(
            d_model=bottleneck_dim,
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

        # Separate readout head per target matrix
        self.heads = nn.ModuleList([
            nn.Linear(bottleneck_dim, 1, bias=True) for _ in range(n_heads)
        ])

        # Initialize
        nn.init.xavier_uniform_(self.down.weight, gain=0.1)
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight, gain=0.1)
            nn.init.constant_(head.bias, bias_init)

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self.mamba = self.mamba.float()
        return result

    def forward(self, x, head_idx=0):
        # x: (batch, seq_len, input_dim)
        input_dtype = x.dtype
        h = self.down(x)  # (batch, seq_len, bottleneck_dim)
        h = self.mamba(h.float()).to(input_dtype)  # (batch, seq_len, bottleneck_dim)
        return self.gate_scale * torch.sigmoid(self.heads[head_idx](h))  # (batch, seq_len, 1)


class GateHeadWrapper(nn.Module):
    """
    Wraps a MultiHeadMambaGate with a specific head index.

    Allows GatedLoRALinear to call self.gate(x) without knowing
    about multi-head internals. Each wrapper targets a different head.
    """

    def __init__(self, multi_head_gate, head_idx):
        super().__init__()
        self.multi_head_gate = multi_head_gate
        self.head_idx = head_idx

    def forward(self, x):
        return self.multi_head_gate(x, head_idx=self.head_idx)


def create_gate(gate_type, input_dim, bottleneck_dim=32, bias_init=-2.0, gate_scale=1.0, **kwargs):
    """Factory function to create a gate module."""
    gate_activation = kwargs.get("gate_activation", "sigmoid")
    if gate_type == "mlp":
        return MLPGate(input_dim, bottleneck_dim, bias_init, gate_scale=gate_scale)
    elif gate_type == "lstm":
        return LSTMGate(input_dim, bottleneck_dim, bias_init, gate_scale=gate_scale)
    elif gate_type == "gru":
        return GRUGate(input_dim, bottleneck_dim, bias_init, gate_scale=gate_scale)
    elif gate_type == "s4":
        return S4Gate(
            input_dim, bottleneck_dim,
            d_state=kwargs.get("d_state", 16),
            d_conv=kwargs.get("d_conv", 4),
            expand_factor=kwargs.get("expand_factor", 2),
            bias_init=bias_init,
            gate_scale=gate_scale,
        )
    elif gate_type == "mamba":
        return MambaGate(
            input_dim, bottleneck_dim,
            d_state=kwargs.get("d_state", 16),
            d_conv=kwargs.get("d_conv", 4),
            expand_factor=kwargs.get("expand_factor", 2),
            bias_init=bias_init,
            gate_scale=gate_scale,
            gate_activation=gate_activation,
        )
    elif gate_type == "none":
        return FixedGate()
    else:
        raise ValueError(f"Unknown gate type: {gate_type}")


class GatedLoRALinear(nn.Module):
    """
    A linear layer with gated LoRA.

    Replaces a frozen linear layer with:
        output = W_frozen @ x + gate(x) * alpha/r * (B @ A @ x)

    The gate modulates LoRA's contribution per token position.
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

        # Initialize: A with Kaiming, B with zeros (standard LoRA init)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Dropout (same as PEFT LoRA)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Gate module (shared or per-module)
        self.gate = gate_module

        # Freeze the original weights
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Frozen base output
        base_out = self.original(x)

        # LoRA output (with dropout on input, same as PEFT)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling

        # Gate: (batch, seq_len, 1)
        g = self.gate(x)

        return base_out + g * lora_out


def inject_gated_lora(model, config):
    """
    Inject Gated LoRA into a pretrained model.

    Args:
        model: Pretrained transformer model
        config: Dict with:
            - gate_type: "mlp", "lstm", or "mamba"
            - rank: LoRA rank
            - alpha: LoRA alpha
            - gate_bottleneck: gate bottleneck dimension
            - gate_bias_init: initial gate bias
            - gate_scale: gate output multiplier (1.0=standard, 2.0=bidirectional)
            - target_modules: list of module names to apply to
            - inject_layers: list of layer indices (if None, apply to all)
            - gate_sharing: "per_layer", "per_matrix", or "multi_head"
            - share_gate_per_layer: (legacy) if True, equivalent to gate_sharing="per_layer"
            - d_state, d_conv, expand_factor: Mamba-specific params

    Returns:
        model: Modified model
        gate_modules: List of gate modules for tracking
    """
    gate_scale = config.get('gate_scale', 1.0)
    gate_activation = config.get('gate_activation', 'sigmoid')
    print(f"Injecting Gated LoRA (gate={config['gate_type']}, scale={gate_scale}, activation={gate_activation})...")

    hidden_dim = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    target_modules = config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    inject_layers = config.get("inject_layers") or list(range(n_layers))

    # Two-tier support: layers that get LoRA but no gate (fixed g=1.0)
    lora_only_layers = set(config.get("lora_only_layers") or [])
    if lora_only_layers:
        print(f"  LoRA-only layers (no gate): {sorted(lora_only_layers)}")

    # Resolve gate sharing mode (support legacy config key)
    gate_sharing = config.get("gate_sharing", None)
    if gate_sharing is None:
        share_gate = config.get("share_gate_per_layer", True)
        gate_sharing = "per_layer" if share_gate else "per_matrix"

    print(f"  Gate type: {config['gate_type']}")
    print(f"  Gate scale: {gate_scale} (range [0, {gate_scale}])")
    print(f"  Gate sharing: {gate_sharing}")
    print(f"  LoRA rank: {config['rank']}, alpha: {config['alpha']}")
    print(f"  Target modules: {target_modules}")
    print(f"  Inject at layers: {inject_layers}")

    # Get transformer layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        raise ValueError("Could not find transformer layers in model")

    # Freeze all base parameters
    for param in model.parameters():
        param.requires_grad = False

    gate_modules = []
    replaced_count = 0

    # Common gate kwargs
    gate_activation = config.get('gate_activation', 'sigmoid')
    gate_kwargs = dict(
        bottleneck_dim=config.get('gate_bottleneck', 32),
        bias_init=config.get('gate_bias_init', -2.0),
        gate_scale=gate_scale,
        gate_activation=gate_activation,
        d_state=config.get('d_state', 16),
        d_conv=config.get('d_conv', 4),
        expand_factor=config.get('expand_factor', 2),
    )

    for layer_idx in inject_layers:
        if layer_idx >= n_layers:
            raise ValueError(f"Layer {layer_idx} out of range (model has {n_layers} layers)")

        layer = layers[layer_idx]

        # Create gate(s) for this layer based on sharing mode
        is_lora_only = layer_idx in lora_only_layers

        if is_lora_only:
            # Fixed gate = 1.0, no trainable gate parameters
            shared_gate = FixedGate()
        elif gate_sharing == "per_layer":
            shared_gate = create_gate(
                config['gate_type'], input_dim=hidden_dim, **gate_kwargs
            )
            gate_modules.append(shared_gate)
        elif gate_sharing == "multi_head":
            multi_head_gate = MultiHeadMambaGate(
                input_dim=hidden_dim,
                n_heads=len(target_modules),
                bottleneck_dim=gate_kwargs['bottleneck_dim'],
                d_state=gate_kwargs['d_state'],
                d_conv=gate_kwargs['d_conv'],
                expand_factor=gate_kwargs['expand_factor'],
                bias_init=gate_kwargs['bias_init'],
                gate_scale=gate_scale,
            )
            gate_modules.append(multi_head_gate)

        # Find and replace target modules in this layer
        for head_idx, name in enumerate(target_modules):
            # Navigate to the module (e.g., layer.self_attn.q_proj or layer.mlp.up_proj)
            parts = name.split(".")
            if len(parts) > 1:
                parent = layer
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                attr_name = parts[-1]
            else:
                # Auto-detect: check self_attn/attention first, then mlp
                attn = layer.self_attn if hasattr(layer, 'self_attn') else getattr(layer, 'attention', None)
                mlp = getattr(layer, 'mlp', None)
                if attn is not None and hasattr(attn, name):
                    parent = attn
                elif mlp is not None and hasattr(mlp, name):
                    parent = mlp
                else:
                    raise AttributeError(f"Cannot find '{name}' in layer {layer_idx} (checked self_attn and mlp)")
                attr_name = name

            original_linear = getattr(parent, attr_name)

            # Select gate based on sharing mode
            if is_lora_only:
                gate = FixedGate()
            elif gate_sharing == "per_layer":
                gate = shared_gate
            elif gate_sharing == "per_matrix":
                gate = create_gate(
                    config['gate_type'], input_dim=original_linear.in_features, **gate_kwargs
                )
                gate_modules.append(gate)
            elif gate_sharing == "multi_head":
                gate = GateHeadWrapper(multi_head_gate, head_idx)
            else:
                raise ValueError(f"Unknown gate_sharing: {gate_sharing}")

            # Create gated LoRA replacement
            gated = GatedLoRALinear(
                original_linear=original_linear,
                rank=config['rank'],
                alpha=config['alpha'],
                gate_module=gate,
                dropout=config.get('dropout', 0.0),
            )

            # Move to same device/dtype
            device = next(original_linear.parameters()).device
            gated = gated.to(device)

            setattr(parent, attr_name, gated)
            replaced_count += 1

        print(f"  ✓ Layer {layer_idx}: replaced {len(target_modules)} modules")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameter summary:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.3f}%")
    print(f"  Replaced {replaced_count} modules across {len(inject_layers)} layers")

    return model, gate_modules


def load_model_with_gated_lora(model_name, checkpoint_path, config_path=None, device='cuda', torch_dtype=torch.bfloat16):
    """
    Load a pretrained model with Gated LoRA from a checkpoint.

    Args:
        model_name: HuggingFace model name
        checkpoint_path: Path to gated_lora_best.pt
        config_path: Path to config YAML (if None, uses config from checkpoint)
        device: Device to load on
        torch_dtype: Model dtype

    Returns:
        model, tokenizer
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device,
    )

    load_device = 'cuda' if device in ('auto', 'cuda') else device
    checkpoint = torch.load(checkpoint_path, map_location=load_device)
    gated_lora_config = checkpoint['config']

    # Inject gated LoRA
    model, gate_modules = inject_gated_lora(model, gated_lora_config)

    # Load states
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'GatedLoRALinear':
            if name in checkpoint['gated_lora_states']:
                states = checkpoint['gated_lora_states'][name]
                module.lora_A.load_state_dict(states['lora_A'])
                module.lora_B.load_state_dict(states['lora_B'])
                module.gate.load_state_dict(states['gate'])

    # Fix dtypes
    model_dtype = torch_dtype
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(dtype=model_dtype)

    for gate in gate_modules:
        if isinstance(gate, MambaGate):
            gate.mamba = gate.mamba.float()
        elif isinstance(gate, MultiHeadMambaGate):
            gate.mamba = gate.mamba.float()
        elif isinstance(gate, S4Gate):
            gate.float()  # S4Gate needs float32 everywhere (pscan + linear layers)

    epoch = checkpoint.get('epoch', '?')
    val_loss = checkpoint.get('val_loss', '?')
    print(f"  Loaded Gated LoRA checkpoint (epoch {epoch}, val_loss {val_loss})")
    print(f"  Gate type: {gated_lora_config['gate_type']}")

    return model, tokenizer


if __name__ == "__main__":
    print("Testing gate modules...")

    batch, seq_len, dim = 2, 100, 3584

    # Test standard gates (scale=1.0)
    for gate_type in ["mlp", "gru", "lstm", "mamba"]:
        print(f"\n--- {gate_type} gate (scale=1.0) ---")
        gate = create_gate(gate_type, input_dim=dim, bottleneck_dim=32)
        x = torch.randn(batch, seq_len, dim)
        g = gate(x)
        print(f"  Input: {x.shape}")
        print(f"  Gate output: {g.shape}")
        print(f"  Gate range: [{g.min().item():.4f}, {g.max().item():.4f}]")
        n_params = sum(p.numel() for p in gate.parameters())
        print(f"  Parameters: {n_params:,}")

    # Test 2x scaling
    print(f"\n--- mamba gate (scale=2.0, bias=0.0) ---")
    gate = create_gate("mamba", input_dim=dim, bottleneck_dim=32,
                       bias_init=0.0, gate_scale=2.0)
    x = torch.randn(batch, seq_len, dim)
    g = gate(x)
    print(f"  Gate range: [{g.min().item():.4f}, {g.max().item():.4f}]")
    print(f"  Gate mean: {g.mean().item():.4f} (should be ~1.0 at init)")

    # Test multi-head gate
    print(f"\n--- MultiHeadMambaGate (4 heads, scale=2.0) ---")
    mh_gate = MultiHeadMambaGate(
        input_dim=dim, n_heads=4, bottleneck_dim=32,
        bias_init=0.0, gate_scale=2.0
    )
    x = torch.randn(batch, seq_len, dim)
    for h in range(4):
        g = mh_gate(x, head_idx=h)
        print(f"  Head {h}: range=[{g.min().item():.4f}, {g.max().item():.4f}], "
              f"mean={g.mean().item():.4f}")
    n_params = sum(p.numel() for p in mh_gate.parameters())
    print(f"  Parameters: {n_params:,}")

    # Test GateHeadWrapper
    print(f"\n--- GateHeadWrapper ---")
    wrapper = GateHeadWrapper(mh_gate, head_idx=2)
    g = wrapper(x)
    print(f"  Wrapper output shape: {g.shape}")
    print(f"  Wrapper output mean: {g.mean().item():.4f}")

    print("\nAll tests passed!")
