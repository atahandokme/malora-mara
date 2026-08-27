"""
Residual-Gated LoRA (MaLoRA v2)

Instead of gating LoRA's output, we add a residual correction:
    h_t = W₀x_t + (α/r) * B * ((λ_t + 1) ⊙ Ax_t)
        = W₀x_t + (α/r) * BAx_t + (α/r) * B(λ_t ⊙ Ax_t)
        = Standard LoRA   +   Mamba-driven correction

Key properties:
    - LoRA path is always active (no gate can kill it)
    - Mamba learns a correction (additive, not multiplicative)
    - λ_t inits at 0 → starts as pure LoRA
    - No activation needed (raw Mamba output)
    - B gets gradients from step 1 through the LoRA path
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMambaGate(nn.Module):
    """
    Mamba-based residual correction for LoRA.

    Operates in rank space (direct mode): takes Ax_t as input,
    produces λ_t ∈ R^r as additive correction per rank direction.

    Output: λ_t (raw, no activation) — used as (λ_t + 1) ⊙ Ax_t
    """

    def __init__(self, rank, d_state=16, d_conv=4, expand_factor=2):
        super().__init__()
        from mambapy.mamba import MambaBlock, MambaConfig

        self.rank = rank

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

        # Output projection to rank dimensions
        # Init with small weights so λ starts near 0
        self.out = nn.Linear(rank, rank, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)  # λ = 0 at init → pure LoRA

    def forward(self, Ax):
        """
        Args:
            Ax: (batch, seq_len, rank) — LoRA's down-projection output
        Returns:
            lambda_t: (batch, seq_len, rank) — additive correction, init ~0
        """
        input_dtype = Ax.dtype
        h = self.mamba(Ax.float())
        lam = self.out(h.to(input_dtype))
        return lam


class ResidualGatedLoRALinear(nn.Module):
    """
    Linear layer with residual-gated LoRA.

    Forward:
        h = W₀x + (α/r) * B * ((λ + 1) ⊙ Ax)
          = W₀x + (α/r) * BAx + (α/r) * B(λ ⊙ Ax)

    When λ=0 (init), this is exactly standard LoRA.
    """

    def __init__(self, original_linear, rank, alpha, gate_module, dropout=0.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # LoRA matrices (standard init)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Dropout
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Residual gate
        self.gate = gate_module

        # Freeze original
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Frozen base
        base_out = self.original(x)

        # LoRA down-projection
        Ax = self.lora_A(self.lora_dropout(x))  # (batch, seq, rank)

        # Residual correction from Mamba
        lam = self.gate(Ax)  # (batch, seq, rank), init ~0

        # (λ + 1) ⊙ Ax = Ax + λ ⊙ Ax
        corrected_Ax = Ax + lam * Ax

        # LoRA up-projection
        lora_out = self.lora_B(corrected_Ax) * self.scaling

        return base_out + lora_out


def inject_residual_gated_lora(model, config):
    """
    Inject Residual-Gated LoRA into a pretrained model.
    """
    from transformers import AutoConfig

    rank = config['rank']
    alpha = config['alpha']
    dropout = config.get('dropout', 0.0)
    target_modules = config.get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'up_proj', 'down_proj'])
    d_state = config.get('d_state', 16)
    d_conv = config.get('d_conv', 4)
    expand_factor = config.get('expand_factor', 2)
    gate_sharing = config.get('gate_sharing', 'per_matrix')

    # Find transformer layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        raise ValueError("Cannot find transformer layers")

    n_layers = len(layers)
    hidden_dim = model.config.hidden_size

    inject_layers = config.get('inject_layers', list(range(n_layers)))

    gate_modules = []
    replaced_count = 0

    print(f"Injecting Residual-Gated LoRA (rank={rank}, d_state={d_state})...")
    print(f"  Target modules: {target_modules}")
    print(f"  Gate sharing: {gate_sharing}")
    print(f"  Inject at layers: {inject_layers}")

    for layer_idx in inject_layers:
        if layer_idx >= n_layers:
            continue

        layer = layers[layer_idx]

        # Create shared gate for this layer if per_layer sharing
        if gate_sharing == 'per_layer':
            shared_gate = ResidualMambaGate(rank, d_state, d_conv, expand_factor)
            gate_modules.append(shared_gate)

        for name in target_modules:
            # Find the module
            attn = layer.self_attn if hasattr(layer, 'self_attn') else getattr(layer, 'attention', None)
            mlp = getattr(layer, 'mlp', None)

            if attn is not None and hasattr(attn, name):
                parent = attn
            elif mlp is not None and hasattr(mlp, name):
                parent = mlp
            else:
                print(f"  Warning: {name} not found in layer {layer_idx}, skipping")
                continue

            original_linear = getattr(parent, name)

            # Select gate
            if gate_sharing == 'per_layer':
                gate = shared_gate
            elif gate_sharing == 'per_matrix':
                gate = ResidualMambaGate(rank, d_state, d_conv, expand_factor)
                gate_modules.append(gate)
            else:
                raise ValueError(f"Unknown gate_sharing: {gate_sharing}")

            # Replace
            gated = ResidualGatedLoRALinear(
                original_linear=original_linear,
                rank=rank,
                alpha=alpha,
                gate_module=gate,
                dropout=dropout,
            )

            device = next(original_linear.parameters()).device
            gated = gated.to(device)
            setattr(parent, name, gated)
            replaced_count += 1

        print(f"  ✓ Layer {layer_idx}: replaced {len(target_modules)} modules")

    # Freeze base model, ensure LoRA + gate are trainable
    for param in model.parameters():
        param.requires_grad = False

    for name, module in model.named_modules():
        if isinstance(module, ResidualGatedLoRALinear):
            for param in module.lora_A.parameters():
                param.requires_grad = True
            for param in module.lora_B.parameters():
                param.requires_grad = True
            for param in module.gate.parameters():
                param.requires_grad = True

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.3f}%")
    print(f"  Replaced {replaced_count} modules across {len(inject_layers)} layers")
    print(f"  All layers gated (residual)")

    return model, gate_modules


def load_model_with_residual_gated_lora(model_name, checkpoint_path, device='cuda', torch_dtype=torch.bfloat16):
    """Load a model with Residual-Gated LoRA from checkpoint."""
    import yaml
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt_dir = Path(checkpoint_path).parent
    config = yaml.safe_load(open(ckpt_dir / "config.yaml"))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device,
        attn_implementation="sdpa",
    )

    gated_lora_config = {
        'rank': config['gated_lora']['rank'],
        'alpha': config['gated_lora']['alpha'],
        'target_modules': config['gated_lora'].get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'up_proj', 'down_proj']),
        'gate_sharing': config['gated_lora'].get('gate_sharing', 'per_matrix'),
        'd_state': config['gated_lora'].get('d_state', 16),
        'd_conv': config['gated_lora'].get('d_conv', 4),
        'expand_factor': config['gated_lora'].get('expand_factor', 2),
        'dropout': config['gated_lora'].get('dropout', 0.0),
    }

    model, gate_modules = inject_residual_gated_lora(model, gated_lora_config)

    # Fix dtypes
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(dtype=torch_dtype)
    for gate in gate_modules:
        gate.mamba = gate.mamba.float()

    # Load checkpoint
    load_device = 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=load_device, weights_only=False)

    for name, module in model.named_modules():
        if isinstance(module, ResidualGatedLoRALinear):
            if name in checkpoint['gated_lora_states']:
                states = checkpoint['gated_lora_states'][name]
                module.lora_A.load_state_dict(states['lora_A'])
                module.lora_B.load_state_dict(states['lora_B'])
                module.gate.load_state_dict(states['gate'])

    return model, tokenizer
