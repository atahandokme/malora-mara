"""RecurrentLoRA: LoRA with stateful SSM correction inside the rank bottleneck.

Architecture (the "Design 2.5" we converged on after seeing chunk_residual leak):

    z       = A·x                       (rank-r down-projection)
    h       = LN(Mamba(z))               (SSM correction in rank space)
    α       = sigmoid(g)                 (learnable LayerScale-style mix)
    Δh      = B · (z + α · h)
            = B·A·x  +  α · B·h          # = standard LoRA + α·SSM correction

α=0  → exactly LoRA (h added with weight 0). Safe init.
α>0  → LoRA delta has a stateful SSM correction baked in.

Single shared down-projection (lora_A) feeds both:
  - the LoRA bottleneck z
  - the SSM input

vs. our chunk_residual gate that put h in the GATE signal multiplicatively,
this puts h in the LoRA DELTA additively. Closer to SSMLoRA's pattern, but with
an explicit α mixing weight for safety.

Per-token Mamba (over T tokens) → granularity preserved → exposure-bias risk
should be similar to standard LoRA (no chunk pooling pathology).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentLoRALinear(nn.Module):
    def __init__(self, original_linear, rank, alpha=16, dropout=0.05,
                 d_state=16, d_conv=4, expand_factor=2, n_layers=1,
                 alpha_init=0.0, sigmoid_alpha=True,
                 use_linear_gate=False,
                 use_separate_P=False):
        super().__init__()
        from mambapy.mamba import MambaBlock, Mamba, MambaConfig

        self.original = original_linear
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Standard LoRA bottleneck
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)  # so initial Δh is zero (LoRA convention)

        # Optional separate P projection for SSM input (hproj-style decoupling).
        # When False (direct mode): SSM reads z = A·x — same as LoRA bottleneck.
        # When True (hproj mode): SSM reads P·x via a dedicated d→r matrix, so
        # A specializes in the LoRA delta and P specializes in the SSM input.
        self.use_separate_P = use_separate_P
        if use_separate_P:
            self.lora_P = nn.Linear(in_features, rank, bias=False)
            nn.init.kaiming_uniform_(self.lora_P.weight, a=math.sqrt(5))

        # SSM operating in rank space
        cfg = MambaConfig(
            d_model=rank, n_layers=n_layers,
            d_state=d_state, d_conv=d_conv, expand_factor=expand_factor,
            dt_min=0.001, dt_max=0.1, pscan=True, use_cuda=False,
        )
        self.mamba = Mamba(cfg) if n_layers > 1 else MambaBlock(cfg)
        self.outer_ln = nn.LayerNorm(rank)

        # Mixing weight (LayerScale convention)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.sigmoid_alpha = sigmoid_alpha

        # Optional Linear gate (Design A hybrid):
        #   λ = softplus(W_lin · x + b_lin)  ∈ R^{B,T,1}  (scalar per token)
        # Multiplied onto the LoRA delta. SSM correction is added separately.
        # gate_bias init = inv_softplus(1) = log(e-1) ≈ 0.5413 → λ ≈ 1 at init.
        self.use_linear_gate = use_linear_gate
        if use_linear_gate:
            self.linear_gate = nn.Linear(in_features, 1, bias=False)
            nn.init.xavier_uniform_(self.linear_gate.weight, gain=0.1)
            self.gate_bias = nn.Parameter(torch.tensor(0.5413))

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        # Mamba pscan needs fp32
        self.mamba = self.mamba.float()
        return result

    def forward(self, x):
        out = self.original(x)
        x_drop = self.dropout(x)
        z = self.lora_A(x_drop)                          # (B, T, r) — LoRA bottleneck
        # SSM input: separate P projection if enabled, else share lora_A's z
        if self.use_separate_P:
            z_ssm = self.lora_P(x_drop)                  # (B, T, r) — dedicated SSM input
        else:
            z_ssm = z
        # SSM correction
        h = self.mamba(z_ssm.float())                    # fp32 for pscan
        ln_dtype = self.outer_ln.weight.dtype
        h = self.outer_ln(h.to(ln_dtype)).to(z.dtype)
        # Mixing weight
        alpha_val = torch.sigmoid(self.alpha) if self.sigmoid_alpha else self.alpha

        if self.use_linear_gate:
            # Design A hybrid: Δh = λ ⊙ (B·z) + α · (B·h)
            #   λ : scalar gate per token from input — captures "preserve base" property
            #   α · (B·h) : additive SSM correction — captures stateful long-context property
            lambda_gate = F.softplus(self.linear_gate(x).squeeze(-1) + self.gate_bias)  # (B, T)
            lambda_gate = lambda_gate.unsqueeze(-1).to(z.dtype)                          # (B, T, 1) for broadcasting
            gated_lora = lambda_gate * self.lora_B(z)
            ssm_correction = alpha_val.to(z.dtype) * self.lora_B(h)
            delta = (gated_lora + ssm_correction) * self.scaling
        else:
            # Original RecurrentLoRA: Δh = B·(z + α·h)
            z_corrected = z + alpha_val.to(z.dtype) * h
            delta = self.lora_B(z_corrected) * self.scaling

        return out + delta


def inject_recurrent_lora(model, config):
    """Replace target nn.Linear modules in transformer layers with RecurrentLoRALinear.

    config keys (all under 'recurrent_lora' section of the YAML):
        rank, alpha, dropout, d_state, d_conv, expand_factor, n_layers,
        alpha_init, sigmoid_alpha, target_modules
    """
    target_modules = config['target_modules']
    rank = config['rank']
    alpha = config.get('alpha', 32)
    dropout = config.get('dropout', 0.05)
    d_state = config.get('d_state', 16)
    d_conv = config.get('d_conv', 4)
    expand_factor = config.get('expand_factor', 2)
    n_layers = config.get('n_layers', 1)
    alpha_init = config.get('alpha_init', 0.0)
    sigmoid_alpha = config.get('sigmoid_alpha', True)
    use_linear_gate = config.get('use_linear_gate', False)
    use_separate_P = config.get('use_separate_P', False)

    # Freeze base parameters
    for p in model.parameters():
        p.requires_grad = False

    # Locate transformer layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        raise ValueError("Could not find transformer layers")

    print(f"Injecting RecurrentLoRA on {len(layers)} layers, target_modules={target_modules}")
    print(f"  rank={rank}, alpha={alpha}, n_layers={n_layers}, d_state={d_state}")
    print(f"  alpha_init={alpha_init}, sigmoid_alpha={sigmoid_alpha}")

    replaced = []
    for layer_idx, layer in enumerate(layers):
        for tname in target_modules:
            parent = None
            for parent_attr in ['self_attn', 'mlp']:
                if hasattr(layer, parent_attr) and hasattr(getattr(layer, parent_attr), tname):
                    parent = getattr(layer, parent_attr)
                    break
            if parent is None:
                continue
            original = getattr(parent, tname)
            if not isinstance(original, nn.Linear):
                continue
            new_module = RecurrentLoRALinear(
                original, rank=rank, alpha=alpha, dropout=dropout,
                d_state=d_state, d_conv=d_conv, expand_factor=expand_factor,
                n_layers=n_layers, alpha_init=alpha_init, sigmoid_alpha=sigmoid_alpha,
                use_linear_gate=use_linear_gate, use_separate_P=use_separate_P,
            )
            # Move to same device + dtype as the original linear we replaced
            target_device = original.weight.device
            target_dtype = original.weight.dtype
            new_module = new_module.to(device=target_device, dtype=target_dtype)
            # ...except keep Mamba/LN/alpha in fp32 (already enforced inside .to())
            setattr(parent, tname, new_module)
            replaced.append(new_module)
        if (layer_idx + 1) % 7 == 0:
            print(f"  ✓ Layer {layer_idx}: {len(replaced)} modules so far")

    print(f"Replaced {len(replaced)} modules total")
    return model, replaced


def save_recurrent_lora_checkpoint(replaced_modules, output_path, epoch=None, val_loss=None):
    """Save lora_A, lora_B, mamba, outer_ln, alpha for each RecurrentLoRALinear."""
    states = {}
    for i, module in enumerate(replaced_modules):
        m_state = {
            'lora_A': module.lora_A.state_dict(),
            'lora_B': module.lora_B.state_dict(),
            'mamba': module.mamba.state_dict(),
            'outer_ln': module.outer_ln.state_dict(),
            'alpha': module.alpha.data,
        }
        if getattr(module, 'use_linear_gate', False):
            m_state['linear_gate'] = module.linear_gate.state_dict()
            m_state['gate_bias'] = module.gate_bias.data
        if getattr(module, 'use_separate_P', False):
            m_state['lora_P'] = module.lora_P.state_dict()
        states[f"module_{i}"] = m_state
    ckpt = {'recurrent_lora_states': states}
    if epoch is not None:
        ckpt['epoch'] = epoch
    if val_loss is not None:
        ckpt['val_loss'] = val_loss
    torch.save(ckpt, output_path)


def load_model_with_recurrent_lora(model_name, checkpoint_path, device='auto',
                                    torch_dtype=torch.bfloat16, config=None):
    """Load base model + inject RecurrentLoRA + restore states from checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import yaml as _yaml

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device,
        attn_implementation="sdpa",
    )

    # Load config from checkpoint dir if not given
    if config is None:
        from pathlib import Path as _Path
        cfg_path = _Path(checkpoint_path).parent / "config.yaml"
        with open(cfg_path) as f:
            full_cfg = _yaml.safe_load(f)
        config = full_cfg.get('recurrent_lora', full_cfg)

    print(f"Injecting RecurrentLoRA from checkpoint config (target={config.get('target_modules')})")
    model, replaced = inject_recurrent_lora(model, config)

    # Load states
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    states = ckpt.get('recurrent_lora_states', {})
    for i, module in enumerate(replaced):
        key = f"module_{i}"
        if key not in states:
            continue
        s = states[key]
        module.lora_A.load_state_dict(s['lora_A'])
        module.lora_B.load_state_dict(s['lora_B'])
        module.mamba.load_state_dict(s['mamba'])
        module.outer_ln.load_state_dict(s['outer_ln'])
        module.alpha.data = s['alpha']
        if getattr(module, 'use_linear_gate', False) and 'linear_gate' in s:
            module.linear_gate.load_state_dict(s['linear_gate'])
            module.gate_bias.data = s['gate_bias']
        if getattr(module, 'use_separate_P', False) and 'lora_P' in s:
            module.lora_P.load_state_dict(s['lora_P'])

    # Match model dtype, except keep Mamba/LN/alpha in fp32
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch_dtype)
    for module in replaced:
        module.mamba = module.mamba.float()
        module.outer_ln = module.outer_ln.float()
        module.alpha.data = module.alpha.data.float()

    print(f"  Loaded RecurrentLoRA checkpoint (epoch {ckpt.get('epoch', '?')}, "
          f"val_loss {ckpt.get('val_loss', '?')})")
    return model, tokenizer
