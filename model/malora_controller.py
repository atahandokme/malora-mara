"""
MaLoRA-Controller: Two-stage LoRA with a deep-layer Mamba controller.

Motivation:
    In current MaLoRA/hybrid designs the gate is a per-module component that
    reads its own local input (x at layer i). On short-context QA this gate
    ends up near-trivial — it can't see enough context to selectively activate
    LoRA. The controller here reverses the structure:

        Stage 1 — train LoRA normally (frozen base, train A and B only).
                  This gives well-conditioned (A, B) pairs per module.
        Stage 2 — freeze (A, B). Add ONE big Mamba that reads the hidden
                  state at a late transformer layer k (e.g. layer 26 of 28).
                  Its output is a (B, T, N_modules) tensor: per-token gates
                  shared across all LoRA adapters in the network.

Causality:
    The controller is causally-shifted: the gate at position t is computed
    from Mamba applied to layer-k hiddens at positions 1..t-1. This avoids
    circular dependency within a single forward pass AND matches the "use
    accumulated state up to now" semantics we want for multi-turn contexts.

    For *layers past k*, the single forward pass is enough — we already have
    layer-k output for all tokens when layers k+1..L need gate values.
    For *layers before k*, only single-forward-pass option is to treat the
    gate as 1.0 (identity — pure Stage-1 LoRA behavior). If you want gates
    on layers 0..k, use two-pass inference (future work).

Output cardinality:
    One shared Mamba, one softplus, 140-dim readout (for Qwen 28-layer × 5
    target modules). Each MaLoRALinear subscribes to its slot.

Initialization:
    softplus(bias) ≈ 1 per module → at training start the gates act as
    identity, preserving Stage-1 LoRA behavior. Training drifts gates
    selectively.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


SOFTPLUS_BIAS_FOR_INIT_ONE = math.log(math.exp(1.0) - 1.0)  # ≈ 0.5414


class DeepLayerController(nn.Module):
    """
    Single big Mamba + linear readout. Reads d_hidden features from a chosen
    transformer layer; outputs n_modules gate values per token.

    Args:
        d_hidden:   input hidden dim (e.g. 3584 for Qwen, 4096 for Llama).
        n_modules:  number of LoRA modules to gate. Each reads its own slot.
        d_mamba:    Mamba d_model. Smaller than d_hidden is fine — Mamba's
                    job is state accumulation + decision, not feature
                    extraction (that's already done by layer k).
        n_layers:   Mamba stack depth. 2 is usually enough because inputs
                    are already attended/processed.
        d_state, d_conv, expand_factor: standard Mamba config.
        causal_shift: whether to shift layer-k hiddens right by 1 before
                    feeding Mamba. With shift, gate at position t depends
                    only on hiddens from positions 1..t-1. Keeps the
                    "no peeking at current token" invariant.
        per_rank:   if True, output n_modules * rank values (per-rank gates).
                    if False, scalar gate per module per token.
    """

    def __init__(self, d_hidden, n_modules,
                 d_mamba=512, n_layers=2,
                 d_state=16, d_conv=4, expand_factor=2,
                 causal_shift=True, per_rank=False, rank=16):
        super().__init__()
        from mambapy.mamba import MambaConfig, Mamba

        self.d_hidden = d_hidden
        self.n_modules = n_modules
        self.d_mamba = d_mamba
        self.n_layers = n_layers
        self.causal_shift = causal_shift
        self.per_rank = per_rank
        self.rank = rank

        out_dim = n_modules * (rank if per_rank else 1)

        # in_proj: d_hidden → d_mamba (bottleneck if d_mamba < d_hidden)
        self.in_proj = nn.Linear(d_hidden, d_mamba, bias=False)
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.1)

        # Mamba stack at d_mamba
        cfg = MambaConfig(
            d_model=d_mamba,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            dt_min=0.001,
            dt_max=0.1,
            pscan=True,
            use_cuda=False,
        )
        self.mamba = Mamba(cfg)

        # Readout: d_mamba → out_dim, with per-module bias init so softplus ≈ 1
        self.out_proj = nn.Linear(d_mamba, out_dim, bias=True)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.constant_(self.out_proj.bias, SOFTPLUS_BIAS_FOR_INIT_ONE)

    def to(self, *args, **kwargs):
        # Keep Mamba in fp32 regardless of outer dtype cast (pscan numerics).
        result = super().to(*args, **kwargs)
        self.mamba = self.mamba.float()
        return result

    def forward(self, x_hidden):
        """
        Args:
            x_hidden: (B, T, d_hidden) — hidden state at controller's layer.
        Returns:
            gates: (B, T, n_modules) if per_rank=False,
                   (B, T, n_modules, rank) if per_rank=True.
        """
        input_dtype = x_hidden.dtype
        B, T, _ = x_hidden.shape

        if self.causal_shift:
            # Shift right by 1: position 0 gets zeros, position t gets x[t-1].
            zero = torch.zeros_like(x_hidden[:, :1])
            shifted = torch.cat([zero, x_hidden[:, :-1]], dim=1)
        else:
            shifted = x_hidden

        h = self.in_proj(shifted)                # (B, T, d_mamba)
        h = self.mamba(h.float()).to(input_dtype)  # causal SSM scan
        # Manual linear: weight in bf16, bias kept in fp32 (softplus-bias safety).
        # Do weight matmul in bf16, upcast result, add fp32 bias → fp32 output.
        z = F.linear(h, self.out_proj.weight, None)          # (B, T, out_dim) bf16
        z = z.float() + self.out_proj.bias                   # fp32 bias add → fp32

        if self.per_rank:
            z = z.view(B, T, self.n_modules, self.rank)

        return F.softplus(z).to(input_dtype)


class ControlledGate(nn.Module):
    """
    A gate module that returns pre-computed controller output for its slot.

    Used as the `gate` attribute of an existing MaLoRALinear
    so we can reuse all the LoRA / forward / freeze machinery from the
    main malora codebase.

    The controller output is pushed into every ControlledGate before the
    forward pass via `set_controller_output()`. Each gate stores its own
    `module_idx` and pulls its slot.
    """

    def __init__(self, rank, module_idx, per_rank=False):
        super().__init__()
        self.rank = rank
        self.module_idx = module_idx
        self.per_rank = per_rank
        self._controller_output = None  # set externally per forward

    def set_controller_output(self, ctrl_output):
        self._controller_output = ctrl_output

    def forward(self, x):
        """Returns (B, T, rank) gate.

        Pre-controller layers (0..k, where k = controller probe layer) fire
        BEFORE the controller has produced output for this batch. They get
        gate=1.0 → behaves like pure Stage-1 LoRA (no modulation).

        Post-controller layers (k+1..L) get gates from controller output.
        """
        B, T = x.shape[:2]
        if self._controller_output is None:
            # Pre-controller layers: identity gate (pure Stage-1 LoRA behavior)
            return torch.ones(B, T, self.rank, device=x.device, dtype=x.dtype)
        if self.per_rank:
            lam = self._controller_output[:, :, self.module_idx, :]  # (B,T,rank)
        else:
            lam_scalar = self._controller_output[:, :, self.module_idx]  # (B,T)
            lam = lam_scalar.unsqueeze(-1).expand(-1, -1, self.rank)
        return lam.to(x.dtype)


def attach_controller(model, controller, controller_layer_idx,
                      gate_modules_list=None):
    """
    Install the controller on a model whose LoRA modules use ControlledGate.

    Adds a forward hook to the transformer layer at `controller_layer_idx`
    that captures that layer's output, runs the controller, and pushes the
    resulting gate tensor into every ControlledGate in the model.

    Args:
        model: HF transformer with MaLoRALinear modules that use
               ControlledGate as their `gate`.
        controller: DeepLayerController instance.
        controller_layer_idx: index of the transformer layer whose output
                              the controller reads (e.g. 26 for Qwen 28).
        gate_modules_list: optional list of ControlledGate instances to
                           update. If None, walks `model` and finds them.

    Returns:
        hook_handle: the registered forward hook (caller may want to remove
                     it later if swapping controllers).
    """
    # Find ControlledGate modules if not provided
    if gate_modules_list is None:
        gate_modules_list = [m for m in model.modules()
                             if isinstance(m, ControlledGate)]
    assert gate_modules_list, (
        "No ControlledGate modules found. Did you inject LoRA with this "
        "gate class? See `inject_controller_malora` helper.")

    # Locate transformer layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise ValueError("Couldn't find transformer layer stack in model.")

    target_layer = layers[controller_layer_idx]

    def hook(_module, _inputs, outputs):
        # outputs may be a tuple (hidden, ...) — take hidden_state
        h = outputs[0] if isinstance(outputs, tuple) else outputs
        ctrl_out = controller(h)                   # (B, T, n_modules) or per-rank
        for g in gate_modules_list:
            g.set_controller_output(ctrl_out)
        # Don't modify the layer output itself

    handle = target_layer.register_forward_hook(hook)

    # ALSO: clear controller state at the start of each forward pass (pre-hook
    # on layer 0) so batch N's layers 0..k don't see batch N-1's stale gates.
    first_layer = layers[0]
    def pre_hook(_module, _inputs):
        for g in gate_modules_list:
            g._controller_output = None
    pre_handle = first_layer.register_forward_pre_hook(pre_hook)

    return (handle, pre_handle)


def clear_controller_state(model):
    """Reset controller outputs on all ControlledGate modules (call before
    each forward pass if you want to fail loudly on missing hook fires)."""
    for m in model.modules():
        if isinstance(m, ControlledGate):
            m._controller_output = None
