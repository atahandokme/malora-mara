"""
MaLoRA: SSM-modulated Low-Rank Adaptation.

A Mamba recurrence over the token sequence produces a modulation factor that
scales the low-rank update at each position:

    h_t = W_frozen @ x_t + (α/r) * B @ (λ_t ⊙ A @ x_t)

where λ_t = softplus(φ(Mamba(P x_1, ..., P x_t)_t)).

The configuration reported in the paper is `gate_type: mamba` with
`scalar_output: true`, giving λ_t ∈ R^1 broadcast over all r rank directions.
Setting `scalar_output: false` gives λ_t ∈ R^r, one factor per rank direction;
that form appears only as an appendix ablation.

Gate variants:
    - "mamba":   Selective SSM modulator (recurrent, input-adaptive). Paper method.
    - "linear":  Single-linear softplus modulator, no recurrence (ablation baseline)
    - "toplora": Faithful TopLoRA (li2025beyond), σ(x) = exp(RMSNorm(x @ W))
    - "mlp":     MLP modulator, no recurrence (ablation baseline)

Layer gating modes:
    - "all":     All layers get modulators
    - "late":    Only layers >= gate_start_layer get gates, earlier layers use LoRA with g=1
    - "custom":  Specify gated_layers list explicitly
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Softplus bias that gives softplus(bias) ≈ 1.0 (so each λ_i starts at ~1.0 = standard LoRA)
SOFTPLUS_BIAS_FOR_INIT_ONE = math.log(math.exp(1.0) - 1.0)  # ≈ 0.5414

# For "amplify" mode: λ = 1 + softplus(z) - ln(2), bias starts at 0 → λ=1.0
AMPLIFY_BIAS_INIT = 0.0


class MLPModulator(nn.Module):
    """Per-token diagonal gate with no recurrence. Ablation baseline."""

    def __init__(self, input_dim, rank, bottleneck_dim=32, gate_activation="softplus"):
        super().__init__()
        self.gate_activation = gate_activation
        self.down = nn.Linear(input_dim, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, rank, bias=True)
        nn.init.xavier_uniform_(self.down.weight, gain=0.1)
        nn.init.xavier_uniform_(self.up.weight, gain=0.1)

        if gate_activation == "amplify":
            nn.init.constant_(self.up.bias, AMPLIFY_BIAS_INIT)
        elif gate_activation == "softplus":
            nn.init.constant_(self.up.bias, SOFTPLUS_BIAS_FOR_INIT_ONE)
        else:
            nn.init.constant_(self.up.bias, 0.0)  # sigmoid(0) = 0.5, * 2.0 = 1.0

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        h = torch.relu(self.down(x))
        z = self.up(h)  # (batch, seq_len, rank)

        if self.gate_activation == "amplify":
            LN2 = 0.6931471805599453
            return 1.0 + F.softplus(z) - LN2  # range [0.31, ∞), init = 1.0
        elif self.gate_activation == "softplus":
            return F.softplus(z)  # (batch, seq_len, rank), each ≈ 1.0 at init
        else:
            return 2.0 * torch.sigmoid(z)  # range [0, 2], init ≈ 1.0


class LinearModulator(nn.Module):
    """HydraLoRA-style single-linear gate (ablation baseline).

    λ_t = softplus(W · h_t) — no hidden layer, no intermediate activation.
    Matches the router architecture used in HydraLoRA (NeurIPS 2024) minus
    the softmax (we produce per-rank scaling, not routing probabilities).
    """

    def __init__(self, input_dim, rank, gate_activation="softplus", scalar_output=False,
                 freeze_bias=False, gate_clamp_max=None,
                 direct_mode=False):
        super().__init__()
        self.gate_activation = gate_activation
        self.scalar_output = scalar_output
        self.rank = rank
        self.gate_clamp_max = gate_clamp_max
        # Direct mode: gate sees Ax (rank-r) instead of x (d-model). Drops the
        # implicit W_proj from d→r; gate becomes r→r linear (or r→1 for scalar).
        # Strict subset of hproj — useful for "is the d→r projection needed?" ablation.
        self.direct_mode = direct_mode
        gate_in = rank if direct_mode else input_dim
        out_dim = 1 if scalar_output else rank
        self.linear = nn.Linear(gate_in, out_dim, bias=True)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)

        if gate_activation == "amplify":
            nn.init.constant_(self.linear.bias, AMPLIFY_BIAS_INIT)
        elif gate_activation == "softplus":
            nn.init.constant_(self.linear.bias, SOFTPLUS_BIAS_FOR_INIT_ONE)
        else:
            nn.init.constant_(self.linear.bias, 0.0)

        if freeze_bias:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        input_dtype = x.dtype
        # Matmul in x's dtype (weight is typically same as x after model cast).
        # Handle fp32 bias + bf16 weight/input case: do matmul without bias,
        # then add bias with dtype promotion.
        z = F.linear(x, self.linear.weight, None)
        if self.linear.bias is not None:
            # If bias is fp32 and z is bf16, upcast z so add produces fp32
            z = z.to(self.linear.bias.dtype) + self.linear.bias
        if self.gate_activation == "amplify":
            LN2 = 0.6931471805599453
            lam = 1.0 + F.softplus(z) - LN2
        elif self.gate_activation == "softplus":
            lam = F.softplus(z)
        elif self.gate_activation == "sigmoid":
            lam = torch.sigmoid(z)            # GateRA-style bounded gate in [0,1]
            self._last_gate = lam             # stash for entropy regularizer
        else:
            lam = 2.0 * torch.sigmoid(z)
        if self.gate_clamp_max is not None:
            lam = torch.clamp(lam, max=self.gate_clamp_max)
        return lam.to(input_dtype)


class TopLoRASingularGate(nn.Module):
    """Faithful TopLoRA (NeurIPS 2025) token-wise singular-value modulation.

    sigma(x) = exp(RMSNorm(x @ W)),  W in R^{in_features x r}, kaiming-init.
    Applied diagonally to the rank-r LoRA intermediate:
      h = W0 x + (alpha/r) * B (sigma(x) ⊙ A x).
    Reimplements Leopold1423/toplora-neurips25 mypeft/toplora.py:TopSingularValue
    (linear x->r, no bias; RMSNorm over r; exp). Diagonal (per-rank), gate sees x.
    """

    def __init__(self, input_dim, rank, **kwargs):
        super().__init__()
        self.rank = rank
        self.scalar_output = False   # TopLoRA is per-rank (diagonal)
        self.direct_mode = False     # gate sees x (in_features), matching TopLoRA's W
        self.weight = nn.Parameter(torch.empty(input_dim, rank))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5), mode="fan_out")
        try:
            self.rms_norm = nn.RMSNorm([rank])
        except (AttributeError, TypeError):
            self.rms_norm = None
            self.rms_eps = 1e-6
            self.rms_weight = nn.Parameter(torch.ones(rank))

    def forward(self, x):
        in_dtype = x.dtype
        w = x.float() @ self.weight.float()              # (b, s, r), fp32 for stability
        if self.rms_norm is not None:
            w = self.rms_norm(w)
        else:
            w = w * torch.rsqrt(w.pow(2).mean(-1, keepdim=True) + self.rms_eps) * self.rms_weight
        return torch.exp(w).to(in_dtype)


class MambaModulator(nn.Module):
    """
    Mamba SSM gate that produces a per-rank gate vector λ_t ∈ R^r.

    Pipeline (hproj mode):
        x_t ∈ R^d  →  W_proj  →  R^r  →  Mamba  →  R^r = λ_t  (optionally reduced to scalar)

    Pipeline (direct mode — no projection, Mamba operates in rank space):
        Ax_t ∈ R^r  →  Mamba(s_{t-1}, Ax_t)  →  λ_t ∈ R^r (or scalar)

    The Mamba recurrence gives the gate causal memory: λ_t depends on x_1,...,x_t.
    Each component λ_i^(t) independently controls rank direction i at token t.

    Bottleneck mode (W_down → R^b → Mamba → R^b → W_up → R^r) was removed: it
    ignored scalar_output and its separately-allocated gate_bias was a dead
    parameter (forward used self.out.bias instead). Use hproj_mode or direct_mode.
    """

    def __init__(self, input_dim, rank,
                 d_state=16, d_conv=4, expand_factor=2,
                 gate_activation="softplus", direct_mode=False, scalar_output=False,
                 hproj_mode=False, gate_scale=2.0,
                 n_layers=1, residual_outer=False, alpha_init=0.1,
                 sigmoid_alpha=False,
                 zero_init_down=False, freeze_bias=False,
                 binary_mode=False, binary_threshold=0.5,
                 gate_clamp_max=None,
                 zero_init_a_log=False):
        super().__init__()
        from mambapy.mamba import MambaBlock, Mamba, MambaConfig

        if not (direct_mode or hproj_mode):
            raise ValueError(
                "MambaModulator requires direct_mode=True or hproj_mode=True. "
                "Bottleneck mode was removed (it had two bugs: scalar_output ignored, "
                "gate_bias unused). Set hproj_mode: true in config."
            )
        if direct_mode and hproj_mode:
            raise ValueError("direct_mode and hproj_mode are mutually exclusive.")

        self.gate_activation = gate_activation
        self.gate_scale = gate_scale
        self.rank = rank
        self.direct_mode = direct_mode
        self.scalar_output = scalar_output
        self.hproj_mode = hproj_mode
        self.n_layers = n_layers
        self.residual_outer = residual_outer
        self.zero_init_down = zero_init_down
        self.binary_mode = binary_mode
        self.binary_threshold = binary_threshold
        self.gate_clamp_max = gate_clamp_max

        if direct_mode:
            # No projections — Mamba operates directly in R^r
            # Ax_t ∈ R^r → Mamba → λ_t ∈ R^r (or scalar)
            mamba_dim = rank
            self.down = None
            if scalar_output:
                self.out = nn.Linear(rank, 1, bias=False)
                nn.init.xavier_uniform_(self.out.weight, gain=0.1)
            else:
                self.out = None
        else:  # hproj_mode
            # h_t ∈ R^d → W_proj → R^r → Mamba → R^r = λ_t
            # Single projection to rank space
            mamba_dim = rank
            self.down = nn.Linear(input_dim, rank, bias=False)
            if zero_init_down:
                nn.init.zeros_(self.down.weight)   # forces h_base=0 at init
            else:
                nn.init.xavier_uniform_(self.down.weight, gain=0.1)
            if scalar_output:
                # Reduce rank-dim Mamba output to scalar gate per token
                self.out = nn.Linear(rank, 1, bias=False)
                nn.init.xavier_uniform_(self.out.weight, gain=0.1)
            else:
                self.out = None

        # Mamba SSM recurrence
        mamba_config = MambaConfig(
            d_model=mamba_dim,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            dt_min=0.001,
            dt_max=0.1,
            pscan=True,
            use_cuda=False,
        )
        # Use stacked Mamba (RMSNorm + residual per block) when n_layers > 1 or
        # residual_outer; otherwise single bare MambaBlock for backward compat.
        if n_layers > 1 or residual_outer:
            self.mamba = Mamba(mamba_config)
        else:
            self.mamba = MambaBlock(mamba_config)

        # Optionally override A_log = 0 (so A = -exp(0) = -1 uniformly across all
        # state channels, removing the default channel-differentiation prior from
        # log(1..d_state) init). Forces the optimizer to learn channel-wise A_log
        # values rather than relying on the prior. Test for "is A_log untrainable?"
        # vs "does the prior just happen to be near-optimal for our task?"
        if zero_init_a_log:
            if isinstance(self.mamba, MambaBlock):
                self.mamba.A_log.data.zero_()
            else:  # stacked Mamba (n_layers > 1)
                for layer in self.mamba.layers:
                    layer.mixer.A_log.data.zero_()

        # Regularizer support: snapshot A_log at init so we can reward drift
        # later (state-engagement penalty: loss -= drift_weight * ||A_log - A_log_init||^2).
        # If A_log NEVER moves under this pressure, the SSM is structurally
        # untrainable in this PEFT regime (paper-pivot evidence).
        self._a_log_inits = []
        if isinstance(self.mamba, MambaBlock):
            self.register_buffer("_a_log_init_0", self.mamba.A_log.data.clone(), persistent=False)
            self._a_log_inits.append(("_a_log_init_0", self.mamba))
        else:
            for li, layer in enumerate(self.mamba.layers):
                buf_name = f"_a_log_init_{li}"
                self.register_buffer(buf_name, layer.mixer.A_log.data.clone(), persistent=False)
                self._a_log_inits.append((buf_name, layer.mixer))

        # Outer residual around Mamba (paper-style block residual + LN)
        # When sigmoid_alpha=True, self.alpha stores the logit g; effective mixing
        # weight is σ(g) ∈ (0, 1). alpha_init should be the logit value (e.g. -3
        # for σ(-3) ≈ 0.047, a LayerScale-style near-off start).
        self.sigmoid_alpha = sigmoid_alpha
        if residual_outer:
            self.outer_ln = nn.LayerNorm(mamba_dim)
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.outer_ln = None
            self.alpha = None

        # Gate bias (applied after Mamba output)
        self.gate_bias = nn.Parameter(torch.zeros(rank))

        # Init gate bias so each λ_i starts at ~1.0
        if gate_activation == "amplify":
            # λ = 1 + softplus(z), want softplus(bias) ≈ 0 at init
            bias_init = AMPLIFY_BIAS_INIT * torch.ones(rank)
            bias_init += 0.1 * torch.randn(rank)  # break symmetry
            self.gate_bias.data.copy_(bias_init)
        elif gate_activation == "softplus":
            bias_init = SOFTPLUS_BIAS_FOR_INIT_ONE * torch.ones(rank)
            bias_init += 0.1 * torch.randn(rank)  # break symmetry
            self.gate_bias.data.copy_(bias_init)
        elif gate_activation == "sigmoid" and gate_scale > 1.0 and gate_scale != 2.0:
            # For sigmoid×S (S>1): S * sigmoid(bias) = 1.0 → bias = log(1/(S-1))
            import math
            sigmoid_bias = math.log(1.0 / (gate_scale - 1.0))
            bias_init = sigmoid_bias * torch.ones(rank)
            bias_init += 0.1 * torch.randn(rank)
            self.gate_bias.data.copy_(bias_init)
        # else: sigmoid×2(0) = 1.0, scale=1 handled in binary_mode block below

        # Binary-mode init override: push gate above threshold at init so binary=1
        # everywhere at start (safe "all-on" init). Training then learns to turn off.
        if binary_mode and gate_activation == "sigmoid":
            # σ(2.2) ≈ 0.9, well above binary_threshold (default 0.5 for scale=1,
            # or need > 1.0 if scale=2). Use bias that gives gate output ≈ 1.5×threshold.
            import math
            target_continuous = 1.5 * binary_threshold
            if gate_scale == 1.0:
                # σ(bias) = target_continuous → bias = logit(target)
                # For target=0.75: bias ≈ 1.1
                t = min(max(target_continuous, 0.55), 0.95)
                bias_init_val = math.log(t / (1.0 - t))
            else:
                # S·σ(bias) = target_continuous → σ(bias) = target/S
                t = min(max(target_continuous / gate_scale, 0.55), 0.95)
                bias_init_val = math.log(t / (1.0 - t))
            bias_init = bias_init_val * torch.ones(rank)
            bias_init += 0.1 * torch.randn(rank)
            self.gate_bias.data.copy_(bias_init)

        # Freeze bias at init (intentional architectural choice)
        if freeze_bias:
            self.gate_bias.requires_grad = False

    def state_drift_loss(self) -> torch.Tensor:
        """Returns mean squared drift of A_log from its init value.
        BIGGER = more SSM dynamics learned. Used as a NEGATIVE term in the loss
        (i.e., REWARD drift) to force Mamba's recurrence to engage.
        """
        drifts = []
        for buf_name, mod in self._a_log_inits:
            init = getattr(self, buf_name)
            drifts.append(((mod.A_log - init) ** 2).mean())
        if not drifts:
            return torch.zeros(1, device=next(self.parameters()).device).squeeze()
        return torch.stack(drifts).mean()

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self.mamba = self.mamba.float()
        return result

    def forward(self, x, Ax=None):
        """
        Args:
            x: hidden state (batch, seq, d) — used in hproj mode
            Ax: LoRA A output (batch, seq, r) — used in direct mode
        """
        input_dtype = x.dtype

        if self.direct_mode:
            # Direct: Mamba operates in rank space on Ax
            assert Ax is not None, "direct_mode requires Ax input"
            h = self.mamba(Ax.float()).to(input_dtype)    # (batch, seq, r)
            if self.scalar_output:
                z = self.out(h)                            # (batch, seq, 1)
                z = z + self.gate_bias[:1]                 # use first bias only
            else:
                z = h + self.gate_bias                     # (batch, seq, r)
        else:  # hproj_mode
            h_base = self.down(x)                          # (batch, seq, r)
            h_rec = self.mamba(h_base.float()).to(input_dtype)
            if self.residual_outer:
                # Match LayerNorm param dtype (LN params follow model dtype cast;
                # h_rec follows x.dtype which can diverge under autocast).
                ln_dtype = self.outer_ln.weight.dtype
                h_rec = self.outer_ln(h_rec.to(ln_dtype)).to(h_base.dtype)
                alpha_val = torch.sigmoid(self.alpha) if self.sigmoid_alpha else self.alpha
                z = h_base + alpha_val * h_rec            # (batch, seq, r)
            else:
                z = h_rec                                  # (batch, seq, r)
            if self.scalar_output:
                # Linear reduction rank → 1, single-value bias
                z = self.out(z) + self.gate_bias[:1]       # (batch, seq, 1)
            else:
                z = z + self.gate_bias                     # (batch, seq, r)

        if self.gate_activation == "amplify":
            LN2 = 0.6931471805599453
            lam = 1.0 + F.softplus(z) - LN2
        elif self.gate_activation == "softplus":
            lam = F.softplus(z)
        elif self.gate_activation == "tanh":
            lam = 1.0 + torch.tanh(z)
        elif self.gate_activation == "none":
            lam = z + 1.0
        else:
            scale = self.gate_scale if hasattr(self, 'gate_scale') else 2.0
            lam = scale * torch.sigmoid(z)

        if self.gate_clamp_max is not None:
            lam = torch.clamp(lam, max=self.gate_clamp_max)

        if self.binary_mode:
            # Straight-through estimator: forward uses binary, backward uses continuous.
            # Init: λ ≈ 1 (since bias chosen so softplus(bias)=1), so λ_bin=1 everywhere
            # at init. Training learns to drive specific (token, module, rank) to 0.
            lam_bin = (lam > self.binary_threshold).to(lam.dtype)
            lam = lam_bin + (lam - lam.detach())  # forward=binary, backward=continuous
        return lam.to(input_dtype)


class RNNModulator(nn.Module):
    """
    GRU/LSTM recurrent gate — classical-recurrence baseline for MambaModulator.

    Same hproj pipeline as MambaModulator:
        x_t ∈ R^d  →  W_proj  →  R^r  →  GRU/LSTM  →  head  →  λ_t
    Only the sequence mixer differs (gated RNN instead of a selective SSM),
    isolating whether MaLoRA's gains come from *any* recurrence or from the
    selective state-space formulation specifically.

    Default hidden sizes match the canonical Mamba gate's parameter count
    (MambaBlock d_model=16, d_state=16, expand=2 ≈ 3.4k params/gate):
    gru → hidden 26, lstm → hidden 22 (both within ~5%; exact counts printed
    by the training script's param summary).
    """

    def __init__(self, input_dim, rank, rnn_type="gru", rnn_hidden=None,
                 gate_activation="softplus", scalar_output=False,
                 hproj_mode=False, zero_init_down=False,
                 freeze_bias=False, gate_clamp_max=None):
        super().__init__()
        if not hproj_mode:
            raise ValueError(
                "RNNModulator supports hproj_mode only (the canonical MaLoRA "
                "pipeline). Set hproj_mode: true in config."
            )
        if rnn_type not in ("gru", "lstm"):
            raise ValueError(f"rnn_type must be 'gru' or 'lstm', got {rnn_type}")

        self.gate_activation = gate_activation
        self.rank = rank
        self.rnn_type = rnn_type
        self.scalar_output = scalar_output
        self.hproj_mode = hproj_mode
        self.gate_clamp_max = gate_clamp_max

        if rnn_hidden is None:
            rnn_hidden = 26 if rnn_type == "gru" else 22
        self.rnn_hidden = rnn_hidden

        # h_t ∈ R^d → W_proj → R^r (same projection as the Mamba gate)
        self.down = nn.Linear(input_dim, rank, bias=False)
        if zero_init_down:
            nn.init.zeros_(self.down.weight)
        else:
            nn.init.xavier_uniform_(self.down.weight, gain=0.1)

        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(rank, rnn_hidden, num_layers=1, batch_first=True)

        # Head: hidden → 1 (scalar) or hidden → rank (diag)
        out_dim = 1 if scalar_output else rank
        self.out = nn.Linear(rnn_hidden, out_dim, bias=False)
        nn.init.xavier_uniform_(self.out.weight, gain=0.1)

        # Gate bias (same init-to-λ≈1 scheme as MambaModulator)
        self.gate_bias = nn.Parameter(torch.zeros(rank))
        if gate_activation == "amplify":
            bias_init = AMPLIFY_BIAS_INIT * torch.ones(rank)
        elif gate_activation == "softplus":
            bias_init = SOFTPLUS_BIAS_FOR_INIT_ONE * torch.ones(rank)
        else:
            bias_init = torch.zeros(rank)
        bias_init += 0.1 * torch.randn(rank)  # break symmetry
        self.gate_bias.data.copy_(bias_init)
        if freeze_bias:
            self.gate_bias.requires_grad = False

    def _apply(self, fn, recurse=True):
        # Keep the RNN in float32 regardless of model-wide dtype casts (which
        # recurse through _apply, not .to()). Mirrors the Mamba gate's fp32
        # recurrence; cuDNN RNNs reject mixed input/weight dtypes.
        result = super()._apply(fn, recurse)
        self.rnn = self.rnn.float()
        return result

    def forward(self, x, Ax=None):
        input_dtype = x.dtype
        h_base = self.down(x)                              # (batch, seq, r)
        # fp32 recurrence, autocast disabled: cuDNN RNNs are strict about
        # dtype agreement, and the Mamba gate's recurrence is fp32 too.
        if h_base.is_cuda:
            with torch.autocast(device_type='cuda', enabled=False):
                h_rec, _ = self.rnn(h_base.float())
        else:
            h_rec, _ = self.rnn(h_base.float())
        h_rec = h_rec.to(input_dtype)
        if self.scalar_output:
            z = self.out(h_rec) + self.gate_bias[:1]       # (batch, seq, 1)
        else:
            z = self.out(h_rec) + self.gate_bias           # (batch, seq, r)

        if self.gate_activation == "amplify":
            LN2 = 0.6931471805599453
            lam = 1.0 + F.softplus(z) - LN2
        elif self.gate_activation == "softplus":
            lam = F.softplus(z)
        elif self.gate_activation == "tanh":
            lam = 1.0 + torch.tanh(z)
        elif self.gate_activation == "none":
            lam = z + 1.0
        else:
            lam = 2.0 * torch.sigmoid(z)

        if self.gate_clamp_max is not None:
            lam = torch.clamp(lam, max=self.gate_clamp_max)
        return lam.to(input_dtype)


class FixedModulator(nn.Module):
    """Fixed gate that returns 1.0 for all rank directions. Used for ungated layers."""

    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def forward(self, x):
        return torch.ones(x.shape[0], x.shape[1], self.rank, device=x.device, dtype=x.dtype)


class MaLoRALinear(nn.Module):
    """
    A linear layer with MaLoRA.

    h_t = W_frozen @ x_t + (α/r) * B @ (λ_t ⊙ A @ x_t)

    λ_t is produced by the gate module. Under the paper configuration it is a
    scalar in R^1, broadcast over the r rank directions; under the diagonal
    ablation it is a vector in R^r, one factor per direction.
    """

    def __init__(self, original_linear, rank, alpha, gate_module, dropout=0.0,
                 lora_b_init_std=0.0):
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

        # Initialize: A with Kaiming
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        # B init: zeros (standard LoRA convention) OR small nonzero (breaks the
        # chicken-and-egg pathology where gate gradients are 0 at init because
        # ∂L/∂gate ∝ B). Set lora_b_init_std=1e-4 to give the gate immediate
        # gradient signal from step 0.
        if lora_b_init_std > 0.0:
            nn.init.normal_(self.lora_B.weight, mean=0.0, std=lora_b_init_std)
        else:
            nn.init.zeros_(self.lora_B.weight)

        # Dropout (same as PEFT LoRA)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Modulator module
        self.gate = gate_module

        # Answer mask: set externally before forward pass during training.
        # True at answer token positions → gate forced to 1.0 there.
        self._answer_mask = None

        # Freeze the original weights
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Frozen base output
        base_out = self.original(x)

        # Compute Ax first — needed for both gating and LoRA
        Ax = self.lora_A(self.lora_dropout(x))  # (batch, seq, r)

        # Diagonal gating: h = W0·x + (α/r) * B·(λ ⊙ A·x)
        # lam shape is (batch, seq, rank) for diag or (batch, seq, 1) for scalar_output
        # (broadcasts across rank dim when multiplied with Ax).
        if hasattr(self.gate, 'direct_mode') and self.gate.direct_mode:
            # Direct mode: gate sees Ax (rank-r) instead of x (d-model)
            if isinstance(self.gate, LinearModulator):
                lam = self.gate(Ax)              # linear gate gets Ax directly
            else:
                lam = self.gate(x, Ax=Ax)        # Mamba gate uses both
        else:
            lam = self.gate(x)         # hproj / linear / MLP: gate sees x (hidden space)

        # Mask answer tokens: force λ=1.0 at answer positions during training
        # so the gate only learns to modulate context/question tokens.
        if self._answer_mask is not None:
            # _answer_mask: (batch, seq) bool, True at answer positions
            mask = self._answer_mask.unsqueeze(-1)  # (batch, seq, 1) — broadcasts over rank
            lam = torch.where(mask, torch.ones_like(lam), lam)

        gated_Ax = lam * Ax            # element-wise per-rank gating
        lora_out = self.lora_B(gated_Ax) * self.scaling

        return base_out + lora_out


def set_answer_mask(model, labels):
    """Set answer mask on all MaLoRALinear modules before forward pass.

    Args:
        model: The full model containing MaLoRALinear modules.
        labels: (batch, seq) tensor where labels != -100 marks answer tokens.
                Pass None to clear the mask (inference mode).
    """
    if labels is not None:
        answer_mask = (labels != -100)  # (batch, seq), True at answer positions
    else:
        answer_mask = None

    for module in model.modules():
        if isinstance(module, MaLoRALinear):
            module._answer_mask = answer_mask


def create_modulator(gate_type, input_dim, rank, bottleneck_dim=32,
                         gate_activation="softplus", **kwargs):
    """Factory function to create a modulator module."""
    direct_mode = kwargs.get("direct_mode", False)
    scalar_output = kwargs.get("scalar_output", False)
    hproj_mode = kwargs.get("hproj_mode", False)
    if gate_type == "mlp":
        return MLPModulator(input_dim, rank, bottleneck_dim, gate_activation)
    elif gate_type == "toplora":
        # Faithful TopLoRA (NeurIPS 2025): sigma(x)=exp(RMSNorm(x@W)), diagonal.
        return TopLoRASingularGate(input_dim, rank)
    elif gate_type == "controller":
        # Gate output is sourced from a shared DeepLayerController (set externally).
        # module_idx is assigned post-injection by train_malora_controller.py.
        from model.malora_controller import ControlledGate
        return ControlledGate(rank=rank, module_idx=-1,
                              per_rank=kwargs.get("per_rank", False))
    elif gate_type == "chunk_mamba":
        # Per-module MaLoRA that runs Mamba over CHUNK summaries instead of
        # per-token. Chunk boundaries are set externally on every module
        # before forward via set_chunk_ends_on_model().
        from model.chunk_mamba_gate import ChunkMambaGate
        return ChunkMambaGate(
            input_dim, rank,
            d_state=kwargs.get("d_state", 16),
            d_conv=kwargs.get("d_conv", 4),
            expand_factor=kwargs.get("expand_factor", 2),
            n_layers=kwargs.get("n_layers", 1),
            gate_activation=gate_activation,
            scalar_output=scalar_output,
            freeze_bias=kwargs.get("freeze_bias", False),
            gate_clamp_max=kwargs.get("gate_clamp_max", None),
        )
    elif gate_type == "chunk_token_mamba":
        # Hybrid: Mamba over [chunk_reps, h_base[answer_tokens]]. Chunk gates
        # broadcast to prompt tokens (as chunk_mamba), but per-token gates
        # for answer-region tokens — the Mamba state evolves during answer
        # generation, restoring per-token gate evolution that pure chunk_mamba
        # lacks. Fixes the "can't terminate" failure mode.
        from model.chunk_mamba_gate import ChunkTokenMambaGate
        return ChunkTokenMambaGate(
            input_dim, rank,
            d_state=kwargs.get("d_state", 16),
            d_conv=kwargs.get("d_conv", 4),
            expand_factor=kwargs.get("expand_factor", 2),
            n_layers=kwargs.get("n_layers", 1),
            gate_activation=gate_activation,
            scalar_output=scalar_output,
            freeze_bias=kwargs.get("freeze_bias", False),
            gate_clamp_max=kwargs.get("gate_clamp_max", None),
        )
    elif gate_type == "chunk_residual_mamba":
        # L2 (per-token linear) + L3 (per-chunk Mamba state) hybrid:
        #   z = h_base + α · h_chunk_per_token
        # h_base is the per-token linear signal (shared `down` projection).
        # h_chunk_per_token is the per-chunk Mamba state, broadcast to tokens
        # in that chunk. α=sigmoid(g) is a learnable scalar mixing weight.
        # α=0 collapses to LinearModulator (safe baseline).
        from model.chunk_mamba_gate import ChunkResidualMambaGate
        return ChunkResidualMambaGate(
            input_dim, rank,
            d_state=kwargs.get("d_state", 16),
            d_conv=kwargs.get("d_conv", 4),
            expand_factor=kwargs.get("expand_factor", 2),
            n_layers=kwargs.get("n_layers", 1),
            gate_activation=gate_activation,
            scalar_output=scalar_output,
            alpha_init=kwargs.get("alpha_init", 0.0),
            sigmoid_alpha=kwargs.get("sigmoid_alpha", True),
            freeze_bias=kwargs.get("freeze_bias", False),
            gate_clamp_max=kwargs.get("gate_clamp_max", None),
        )
    elif gate_type == "linear":
        return LinearModulator(
            input_dim, rank,
            gate_activation=gate_activation,
            scalar_output=kwargs.get("scalar_output", False),
            freeze_bias=kwargs.get("freeze_bias", False),
            gate_clamp_max=kwargs.get("gate_clamp_max", None),
            direct_mode=direct_mode,
        )
    elif gate_type == "mamba":
        return MambaModulator(
            input_dim, rank,
            d_state=kwargs.get("d_state", 16),
            d_conv=kwargs.get("d_conv", 4),
            expand_factor=kwargs.get("expand_factor", 2),
            gate_activation=gate_activation,
            direct_mode=direct_mode,
            scalar_output=scalar_output,
            hproj_mode=hproj_mode,
            gate_scale=kwargs.get("gate_scale", 2.0),
            n_layers=kwargs.get("n_layers", 1),
            residual_outer=kwargs.get("residual_outer", False),
            alpha_init=kwargs.get("alpha_init", 0.1),
            sigmoid_alpha=kwargs.get("sigmoid_alpha", False),
            zero_init_down=kwargs.get("zero_init_down", False),
            freeze_bias=kwargs.get("freeze_bias", False),
            binary_mode=kwargs.get("binary_mode", False),
            binary_threshold=kwargs.get("binary_threshold", 0.5),
            gate_clamp_max=kwargs.get("gate_clamp_max", None),
            zero_init_a_log=kwargs.get("zero_init_a_log", False),
        )
    elif gate_type in ("gru", "lstm"):
        return RNNModulator(
            input_dim, rank,
            rnn_type=gate_type,
            rnn_hidden=kwargs.get("rnn_hidden", None),
            gate_activation=gate_activation,
            scalar_output=scalar_output,
            hproj_mode=hproj_mode,
            zero_init_down=kwargs.get("zero_init_down", False),
            freeze_bias=kwargs.get("freeze_bias", False),
            gate_clamp_max=kwargs.get("gate_clamp_max", None),
        )
    else:
        raise ValueError(f"Unknown modulator type: {gate_type}")


def inject_malora(model, config):
    """
    Inject MaLoRA into a pretrained model.

    Args:
        model: Pretrained transformer model
        config: Dict with:
            - gate_type: "mlp" or "mamba"
            - rank: LoRA rank
            - alpha: LoRA alpha
            - gate_bottleneck: gate bottleneck dimension
            - gate_activation: "softplus" or "sigmoid"
            - target_modules: list of module names to apply to
            - gate_sharing: "per_layer" or "per_matrix"
            - gate_start_layer: layers before this get LoRA with fixed g=1 (no gate)
            - d_state, d_conv, expand_factor: Mamba-specific params

    Returns:
        model: Modified model
        gate_modules: List of gate modules for tracking
    """
    rank = config['rank']
    gate_activation = config.get('gate_activation', 'softplus')
    gate_sharing = config.get('gate_sharing', 'per_matrix')
    gate_start_layer = config.get('gate_start_layer', 0)

    print(f"Injecting MaLoRA (gate={config['gate_type']}, activation={gate_activation})...")

    hidden_dim = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    target_modules = config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    inject_layers = config.get("inject_layers") or list(range(n_layers))

    print(f"  Gate type: {config['gate_type']}")
    _scalar_out = config.get("scalar_output", False)
    print(f"  Gate output: {'scalar (R^1)' if _scalar_out else f'diagonal (λ ∈ R^{rank})'}")
    print(f"  Gate sharing: {gate_sharing}")
    print(f"  Gate activation: {gate_activation}")
    print(f"  LoRA rank: {rank}, alpha: {config['alpha']}")
    print(f"  Target modules: {target_modules}")
    print(f"  Gate start layer: {gate_start_layer} (layers 0-{gate_start_layer-1} ungated)" if gate_start_layer > 0 else f"  All layers gated")

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

    gate_kwargs = dict(
        bottleneck_dim=config.get('gate_bottleneck', 32),
        gate_activation=gate_activation,
        d_state=config.get('d_state', 16),
        d_conv=config.get('d_conv', 4),
        expand_factor=config.get('expand_factor', 2),
        direct_mode=config.get('direct_mode', False),
        scalar_output=config.get('scalar_output', False),
        hproj_mode=config.get('hproj_mode', False),
        n_layers=config.get('n_layers', 1),
        residual_outer=config.get('residual_outer', False),
        alpha_init=config.get('alpha_init', 0.1),
        sigmoid_alpha=config.get('sigmoid_alpha', False),
        zero_init_down=config.get('zero_init_down', False),
        zero_init_a_log=config.get('zero_init_a_log', False),
        freeze_bias=config.get('freeze_bias', False),
        binary_mode=config.get('binary_mode', False),
        binary_threshold=config.get('binary_threshold', 0.5),
        gate_clamp_max=config.get('gate_clamp_max', None),
        rnn_hidden=config.get('rnn_hidden', None),
    )

    for layer_idx in inject_layers:
        if layer_idx >= n_layers:
            raise ValueError(f"Layer {layer_idx} out of range (model has {n_layers} layers)")

        layer = layers[layer_idx]
        is_gated = layer_idx >= gate_start_layer

        # Create shared gate for per_layer mode
        if is_gated and gate_sharing == "per_layer":
            shared_gate = create_modulator(
                config['gate_type'], input_dim=hidden_dim, rank=rank, **gate_kwargs
            )
            gate_modules.append(shared_gate)

        for name in target_modules:
            parts = name.split(".")
            attr_name = parts[-1] if len(parts) > 1 else name

            # Find parent: check self_attn first, then mlp
            attn = layer.self_attn if hasattr(layer, 'self_attn') else getattr(layer, 'attention', None)
            mlp = getattr(layer, 'mlp', None)

            if attn is not None and hasattr(attn, attr_name):
                parent = attn
            elif mlp is not None and hasattr(mlp, attr_name):
                parent = mlp
            else:
                print(f"  Warning: {attr_name} not found in layer {layer_idx}, skipping")
                continue

            for part in parts[:-1]:
                parent = getattr(parent, part)

            original_linear = getattr(parent, attr_name)

            # Select gate
            if not is_gated:
                gate = FixedModulator(rank)
            elif gate_sharing == "per_layer":
                gate = shared_gate
            elif gate_sharing == "per_matrix":
                gate = create_modulator(
                    config['gate_type'], input_dim=original_linear.in_features,
                    rank=rank, **gate_kwargs
                )
                gate_modules.append(gate)
            else:
                raise ValueError(f"Unknown gate_sharing: {gate_sharing}")

            # Create MaLoRA replacement
            gated = MaLoRALinear(
                original_linear=original_linear,
                rank=rank,
                alpha=config['alpha'],
                gate_module=gate,
                dropout=config.get('dropout', 0.0),
                lora_b_init_std=config.get('lora_b_init_std', 0.0),
            )

            device = next(original_linear.parameters()).device
            gated = gated.to(device)

            setattr(parent, attr_name, gated)
            replaced_count += 1

        label = "gated" if is_gated else "fixed (g=1)"
        print(f"  ✓ Layer {layer_idx}: replaced {len(target_modules)} modules ({label})")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameter summary:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.3f}%")
    print(f"  Replaced {replaced_count} modules across {len(inject_layers)} layers")
    print(f"  Gated layers: {gate_start_layer}-{n_layers-1} ({n_layers - gate_start_layer} layers)")
    if gate_start_layer > 0:
        print(f"  Ungated layers: 0-{gate_start_layer-1} ({gate_start_layer} layers, LoRA with fixed g=1)")

    return model, gate_modules


def load_model_with_malora(model_name, checkpoint_path, config_path=None,
                                         device='cuda', torch_dtype=torch.bfloat16):
    """
    Load a pretrained model with MaLoRA from a checkpoint.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device,
    )

    load_device = 'cuda' if device in ('auto', 'cuda') else device
    checkpoint = torch.load(checkpoint_path, map_location=load_device, weights_only=False)
    malora_config = dict(checkpoint['config'])

    # Backward-compat: old pre-fix hproj_scalar checkpoints have no `out.weight`
    # (the scalar_output flag was silently ignored in hproj mode). They are
    # architecturally diag. Override scalar_output=False so inject creates the
    # matching architecture.
    if (malora_config.get('gate_type') == 'mamba'
            and malora_config.get('hproj_mode')
            and malora_config.get('scalar_output')):
        gate_states_list = checkpoint.get('gate_modules', [])
        if isinstance(gate_states_list, list) and gate_states_list:
            first_keys = set(gate_states_list[0].keys())
            if 'out.weight' not in first_keys:
                print("  [compat] Old hproj_scalar checkpoint (no out.weight); "
                      "loading as architecturally-diag (scalar_output=False).")
                malora_config['scalar_output'] = False

    # Inject MaLoRA
    model, gate_modules = inject_malora(model, malora_config)

    # Load states
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'MaLoRALinear':
            if name in checkpoint['gated_lora_states']:
                states = checkpoint['gated_lora_states'][name]
                module.lora_A.load_state_dict(states['lora_A'])
                module.lora_B.load_state_dict(states['lora_B'])
                module.gate.load_state_dict(states['gate'])

    # Fix dtypes
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(dtype=torch_dtype)

    for gate in gate_modules:
        if isinstance(gate, MambaModulator):
            gate.mamba = gate.mamba.float()
        elif isinstance(gate, RNNModulator):
            # Same treatment as training: RNN fp32, gate_bias fp32.
            gate.rnn = gate.rnn.float()
            if hasattr(gate, 'gate_bias') and gate.gate_bias is not None:
                gate.gate_bias.data = gate.gate_bias.data.float()
        else:
            # Late-import to avoid circular import
            try:
                from model.chunk_mamba_gate import (
                    ChunkMambaGate, ChunkTokenMambaGate, ChunkResidualMambaGate,
                )
                if isinstance(gate, (ChunkMambaGate, ChunkTokenMambaGate, ChunkResidualMambaGate)):
                    gate.mamba = gate.mamba.float()
                    if hasattr(gate, 'gate_bias') and gate.gate_bias is not None:
                        gate.gate_bias.data = gate.gate_bias.data.float()
            except ImportError:
                pass

    epoch = checkpoint.get('epoch', '?')
    val_loss = checkpoint.get('val_loss', '?')
    print(f"  Loaded MaLoRA checkpoint (epoch {epoch}, val_loss {val_loss})")
    _scalar = malora_config.get('scalar_output', False)
    _out = "scalar (R^1)" if _scalar else f"diagonal (R^{malora_config['rank']})"
    print(f"  Gate type: {malora_config['gate_type']}, output: {_out}")

    return model, tokenizer
