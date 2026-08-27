"""
ChunkMambaGate: MaLoRA gate that operates on CHUNKS instead of tokens.

    x (B, T, d)
      ↓ down: d → r       (per-module, same as current MaLoRA)
    h_base (B, T, r)
      ↓ last-token pool at chunk_ends[b]
    chunk_reps (B, K, r)
      ↓ Mamba (K recurrent steps, K << T)
    h_rec (B, K, r)
      ↓ + gate_bias, softplus
    λ_chunks (B, K, r)
      ↓ expand back: tokens in chunk k → λ_chunks[k]
    λ (B, T, r)           ← per-chunk gate broadcast to tokens

Chunk boundaries are set externally before each forward pass via
`set_chunk_ends_on_model(model, batch_chunk_ends)`.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


SOFTPLUS_BIAS_FOR_INIT_ONE = math.log(math.exp(1.0) - 1.0)   # ≈ 0.5414


class ChunkMambaGate(nn.Module):
    def __init__(self, input_dim, rank,
                 d_state=16, d_conv=4, expand_factor=2, n_layers=1,
                 gate_activation="softplus", scalar_output=False,
                 freeze_bias=False, gate_clamp_max=None):
        super().__init__()
        self.gate_clamp_max = gate_clamp_max
        self.freeze_bias_flag = freeze_bias
        from mambapy.mamba import MambaBlock, Mamba, MambaConfig

        self.rank = rank
        self.gate_activation = gate_activation
        self.scalar_output = scalar_output

        # Down-projection (per-module, d_hidden → rank), like standard MaLoRA
        self.down = nn.Linear(input_dim, rank, bias=False)
        nn.init.xavier_uniform_(self.down.weight, gain=0.1)

        # Mamba over chunk sequence (operates at d_model=rank)
        cfg = MambaConfig(
            d_model=rank, n_layers=n_layers,
            d_state=d_state, d_conv=d_conv, expand_factor=expand_factor,
            dt_min=0.001, dt_max=0.1, pscan=True, use_cuda=False,
        )
        self.mamba = Mamba(cfg) if n_layers > 1 else MambaBlock(cfg)

        # Scalar-output readout: reduce rank → 1 after Mamba (matches the scalar
        # variant of MambaModulator). If False, gate is per-rank (diag).
        if scalar_output:
            self.out = nn.Linear(rank, 1, bias=False)
            nn.init.xavier_uniform_(self.out.weight, gain=0.1)
        else:
            self.out = None

        # Bias: rank-long; if scalar_output, only slot [0] is used
        self.gate_bias = nn.Parameter(torch.zeros(rank))
        with torch.no_grad():
            if gate_activation == "softplus":
                self.gate_bias.data.fill_(SOFTPLUS_BIAS_FOR_INIT_ONE)
                self.gate_bias.data.add_(0.1 * torch.randn(rank))

        # Per-forward chunk info: list-of-lists, one inner list per batch item
        # Each inner list: token indices of last-token-per-chunk (ascending, ≤ T-1).
        self._chunk_ends = None
        # Diagnostic: last forward's per-chunk gate values (detached, cpu) for
        # tracking. Shape: (B, K_max, r) if diag, (B, K_max, 1) if scalar.
        self._last_chunk_gates = None
        # KV-cache speedup: cache lam_chunks (B, K, r_or_1) computed on the
        # full-prompt forward, then reuse for subsequent T==1 forwards
        # (autoregressive generation). Invalidated when set_chunk_ends() is
        # called for a new sample.
        self._cached_lam_chunks = None

        # Optional tracker head: predicts which paragraphs are supporting
        # (gold-supervised externally). Reads h_rec at the LAST chunk position
        # — which has integrated all context including Q and A through Mamba's
        # recurrence — and emits n_paragraphs logits. Stays None unless the
        # caller invokes attach_tracker_head().
        self.tracker_head = None
        self._last_tracker_logits = None  # (B, n_paragraphs) fp32, with grad

        # Safety: freeze bias so it stays at softplus-init (= 1.0) baseline.
        # This prevents bias from drifting under high LR; Mamba alone drives
        # per-chunk differentiation.
        if freeze_bias:
            self.gate_bias.requires_grad = False

    def attach_tracker_head(self, n_paragraphs):
        """Attach a tracker head for coverage supervision.

        After every full-prompt forward pass, the head reads h_rec at the LAST
        chunk position (per sample) and emits `n_paragraphs` logits, stored in
        `_last_tracker_logits`. The training loop reads them and applies BCE
        against gold supporting labels.

        The head lives in fp32 (matching Mamba's internal precision).
        """
        head = nn.Linear(self.rank, n_paragraphs, bias=True)
        nn.init.zeros_(head.bias)
        nn.init.xavier_uniform_(head.weight, gain=0.1)
        head = head.float()
        self.tracker_head = head
        return self

    def to(self, *args, **kwargs):
        # Keep Mamba in fp32 (pscan numerics); same for tracker_head if present.
        result = super().to(*args, **kwargs)
        self.mamba = self.mamba.float()
        if self.tracker_head is not None:
            self.tracker_head = self.tracker_head.float()
        return result

    def forward(self, x):
        """
        x: (B, T, d_hidden) — same as standard MaLoRA gate input.
        Returns: (B, T, rank) — per-rank gate, broadcast from per-chunk Mamba output.
        If no chunk info set → returns ones (gate = 1, identity behavior).
        """
        B, T, _ = x.shape
        input_dtype = x.dtype

        if self._chunk_ends is None:
            # No chunk info → identity gate (equivalent to pure LoRA)
            out_dim = 1 if self.scalar_output else self.rank
            return torch.ones(B, T, out_dim, device=x.device, dtype=input_dtype)

        # KV-cache fast path: during autoregressive generation, T==1 and the
        # prompt's chunk_reps don't change. Reuse cached lam_chunks → broadcast
        # the LAST chunk's gate to the single new token. Avoids re-running
        # Mamba over 12 chunks at every generation step.
        if T == 1 and self._cached_lam_chunks is not None:
            cached = self._cached_lam_chunks
            if cached.shape[0] == B:
                out_dim = 1 if self.scalar_output else self.rank
                lam = torch.empty(B, 1, out_dim, device=x.device, dtype=input_dtype)
                for b in range(B):
                    last_k = cached.shape[1] - 1
                    lam[b, 0] = cached[b, last_k].to(input_dtype)
                return lam

        # Replicate chunk_ends if batch dim expanded (e.g. beam search makes
        # B=num_beams from B=1). Repeat each sample's chunk_ends accordingly.
        raw_ce = self._chunk_ends
        if len(raw_ce) != B:
            if B % len(raw_ce) == 0:
                rep = B // len(raw_ce)
                raw_ce = [ce for ce in raw_ce for _ in range(rep)]
            else:
                # Fallback: pad by replicating the last sample
                raw_ce = list(raw_ce) + [raw_ce[-1]] * (B - len(raw_ce))

        # Clamp chunk_ends to valid range [0, T-1]. Eval/val samples may have been
        # truncated to max_length; their pre-computed chunk_ends could index past T.
        chunk_ends = [
            [min(e, T - 1) for e in ce] for ce in raw_ce
        ]
        K_max = max(len(ce) for ce in chunk_ends)

        # 1. Down-project tokens into rank space (same as standard MaLoRA)
        h_base = self.down(x)                     # (B, T, r)

        # 2. Last-token pool at chunk boundaries → (B, K_max, r)
        chunk_reps = torch.zeros(
            B, K_max, self.rank, device=x.device, dtype=h_base.dtype,
        )
        for b in range(B):
            ends_b = chunk_ends[b]
            if len(ends_b) > 0:
                idx = torch.as_tensor(ends_b, device=x.device, dtype=torch.long)
                chunk_reps[b, :len(ends_b)] = h_base[b, idx, :]

        # 3. Mamba over chunks (K recurrent steps)
        h_rec_fp32 = self.mamba(chunk_reps.float())  # (B, K_max, r) fp32
        h_rec = h_rec_fp32.to(input_dtype)

        # 3b. Tracker head (optional, coverage supervision):
        # Reads Mamba state at the LAST chunk position per sample (which has
        # integrated all paragraphs + Q + A via recurrence) and emits
        # n_paragraphs logits. fp32 throughout for numerical stability.
        if self.tracker_head is not None:
            last_idx = torch.tensor(
                [max(0, len(ce) - 1) for ce in chunk_ends],
                device=x.device, dtype=torch.long,
            )
            final_state = h_rec_fp32[torch.arange(B, device=x.device), last_idx]  # (B, r) fp32
            self._last_tracker_logits = self.tracker_head(final_state)  # (B, n_paragraphs) fp32

        # 4. Gate bias + (optional scalar readout) + softplus
        if self.scalar_output:
            z = self.out(h_rec) + self.gate_bias[:1]   # (B, K_max, 1)
        else:
            z = h_rec + self.gate_bias                  # (B, K_max, r)

        if self.gate_activation == "softplus":
            lam_chunks = F.softplus(z)
        elif self.gate_activation == "sigmoid":
            lam_chunks = 2.0 * torch.sigmoid(z)
        else:
            lam_chunks = F.softplus(z)

        # Safety clamp (softplus is unbounded above; high LR can blow it up)
        if self.gate_clamp_max is not None:
            lam_chunks = torch.clamp(lam_chunks, max=self.gate_clamp_max)

        # Cache for diagnostics (no grad, no memory cost in backward)
        self._last_chunk_gates = lam_chunks.detach()   # (B, K_max, r or 1)

        # KV-cache: store lam_chunks for autoregressive generation. During the
        # next T==1 forward call, reuse this instead of re-running Mamba.
        # Detached so we don't leak grad through cache (eval is no_grad anyway).
        self._cached_lam_chunks = lam_chunks.detach()

        # 5. Expand chunk gates back to token positions
        #    Gate output shape: (B, T, r) if diag, (B, T, 1) if scalar.
        #    Scalar broadcasts across rank in the downstream element-wise mul.
        #
        #    Tokens AFTER the last chunk_end inherit the LAST chunk's gate.
        #    This matters at generation time: prompts have chunk_ends, but new
        #    autoregressively-generated tokens extend past the last chunk_end.
        #    Without this, generated tokens would use gate=1.0 (identity),
        #    creating a train/eval mismatch since training had answer tokens
        #    inside the last chunk.
        out_dim = 1 if self.scalar_output else self.rank
        lam = torch.ones(B, T, out_dim, device=x.device, dtype=input_dtype)
        for b in range(B):
            ends_b = chunk_ends[b]
            if len(ends_b) == 0:
                continue
            prev_end = -1
            for k, end_tok in enumerate(ends_b):
                start_tok = prev_end + 1
                lam[b, start_tok:end_tok + 1] = lam_chunks[b, k]
                prev_end = end_tok
            # Tokens after the last chunk_end get the last chunk's gate
            last_end = ends_b[-1]
            if last_end < T - 1:
                lam[b, last_end + 1:] = lam_chunks[b, len(ends_b) - 1]

        return lam


class ChunkTokenMambaGate(nn.Module):
    """Hybrid gate: Mamba processes [chunk_reps, h_base[answer-region tokens]].

    Same chunk pooling as ChunkMambaGate for prompt tokens, but the Mamba state
    continues evolving over each answer-region token. This makes the gate at
    answer-position t depend on tokens 0..t (not just the prompt's chunks),
    so the model can carry "I've said enough" / per-token info via the gate
    during answer generation. Restores the per-token gate evolution that pure
    chunk_mamba lacks.

    Combined input to the Mamba:
        [chunk_rep_0, ..., chunk_rep_K-1, h_base[A_end+1], ..., h_base[T-1]]
        shape (B, K + T_ans, r)

    Output mapping:
        First K outputs   → chunk gates → broadcast to prompt chunks (as today)
        Remaining outputs → per-token gates → applied directly to answer region
    """

    def __init__(self, input_dim, rank,
                 d_state=16, d_conv=4, expand_factor=2, n_layers=1,
                 gate_activation="softplus", scalar_output=False,
                 freeze_bias=False, gate_clamp_max=None):
        super().__init__()
        self.gate_clamp_max = gate_clamp_max
        self.freeze_bias_flag = freeze_bias
        from mambapy.mamba import MambaBlock, Mamba, MambaConfig

        self.rank = rank
        self.gate_activation = gate_activation
        self.scalar_output = scalar_output

        self.down = nn.Linear(input_dim, rank, bias=False)
        nn.init.xavier_uniform_(self.down.weight, gain=0.1)

        cfg = MambaConfig(
            d_model=rank, n_layers=n_layers,
            d_state=d_state, d_conv=d_conv, expand_factor=expand_factor,
            dt_min=0.001, dt_max=0.1, pscan=True, use_cuda=False,
        )
        self.mamba = Mamba(cfg) if n_layers > 1 else MambaBlock(cfg)

        if scalar_output:
            self.out = nn.Linear(rank, 1, bias=False)
            nn.init.xavier_uniform_(self.out.weight, gain=0.1)
        else:
            self.out = None

        self.gate_bias = nn.Parameter(torch.zeros(rank))
        with torch.no_grad():
            if gate_activation == "softplus":
                self.gate_bias.data.fill_(SOFTPLUS_BIAS_FOR_INIT_ONE)
                self.gate_bias.data.add_(0.1 * torch.randn(rank))

        self._chunk_ends = None
        self._last_chunk_gates = None

        # Eval-time caches:
        # _cached_chunk_reps: (B, K, r) — set once on full-prompt forward
        # _cached_answer_h_base: (B, T_so_far, r) — grows as generation proceeds
        # Both reset on set_chunk_ends_on_model() (= new sample).
        self._cached_chunk_reps = None
        self._cached_answer_h_base = None

        if freeze_bias:
            self.gate_bias.requires_grad = False

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self.mamba = self.mamba.float()
        return result

    def _activate(self, z):
        if self.gate_activation == "softplus":
            return F.softplus(z)
        elif self.gate_activation == "sigmoid":
            return 2.0 * torch.sigmoid(z)
        return F.softplus(z)

    def forward(self, x):
        B, T, _ = x.shape
        input_dtype = x.dtype

        if self._chunk_ends is None:
            out_dim = 1 if self.scalar_output else self.rank
            return torch.ones(B, T, out_dim, device=x.device, dtype=input_dtype)

        # Replicate chunk_ends if batch dim expanded (beam search etc.)
        raw_ce = self._chunk_ends
        if len(raw_ce) != B:
            if B % len(raw_ce) == 0:
                rep = B // len(raw_ce)
                raw_ce = [ce for ce in raw_ce for _ in range(rep)]
            else:
                raw_ce = list(raw_ce) + [raw_ce[-1]] * (B - len(raw_ce))

        # Down-project (always — needed for new tokens at gen time).
        h_base = self.down(x)  # (B, T, r)

        # ============= GENERATION FAST PATH (T==1) =============
        # During autoregressive generation, x is just the new token. Append its
        # h_base to the cache, then run Mamba over [cached_chunks, all_answer_h].
        # Output's last position is the gate for this new token.
        if T == 1 and self._cached_chunk_reps is not None and self._cached_chunk_reps.shape[0] == B:
            new_h = h_base  # (B, 1, r)
            if self._cached_answer_h_base is None:
                self._cached_answer_h_base = new_h.detach()
            else:
                self._cached_answer_h_base = torch.cat(
                    [self._cached_answer_h_base, new_h.detach()], dim=1
                )

            combined = torch.cat([self._cached_chunk_reps, self._cached_answer_h_base], dim=1)
            mamba_out = self.mamba(combined.float()).to(input_dtype)
            new_token_state = mamba_out[:, -1:, :]  # (B, 1, r)

            if self.scalar_output:
                z = self.out(new_token_state) + self.gate_bias[:1]
            else:
                z = new_token_state + self.gate_bias
            lam = self._activate(z)
            if self.gate_clamp_max is not None:
                lam = torch.clamp(lam, max=self.gate_clamp_max)
            return lam.to(input_dtype)

        # ============= FULL FORWARD =============
        # Used at training (full prompt+answer) and eval first prompt forward.
        chunk_ends = [
            [min(e, T - 1) for e in ce] for ce in raw_ce
        ]
        K_max = max(len(ce) for ce in chunk_ends)

        # Pool chunk reps from prompt positions
        chunk_reps = torch.zeros(
            B, K_max, self.rank, device=x.device, dtype=h_base.dtype,
        )
        for b in range(B):
            ends_b = chunk_ends[b]
            if len(ends_b) > 0:
                idx = torch.as_tensor(ends_b, device=x.device, dtype=torch.long)
                chunk_reps[b, :len(ends_b)] = h_base[b, idx, :]

        # Determine answer-region: tokens after the last chunk_end (per sample).
        # For batched processing we use the SAME slice index = max(last_end)+1
        # so all samples have the same combined-input length. Samples whose
        # last_end is smaller will have extra "post-A" tokens included in their
        # answer segment too — for HotpotQA that's OK because right-pad never
        # comes between chunks (chunks span the whole prompt).
        last_ends = [ce[-1] if len(ce) > 0 else -1 for ce in chunk_ends]
        ans_start = max(last_ends) + 1  # global start of answer region

        if ans_start < T:
            answer_h = h_base[:, ans_start:, :]  # (B, T_ans, r)
            combined = torch.cat([chunk_reps, answer_h], dim=1)
        else:
            combined = chunk_reps  # eval first forward: prompt only, no answer tokens

        mamba_out = self.mamba(combined.float()).to(input_dtype)  # (B, K + T_ans, r)
        chunk_states = mamba_out[:, :K_max, :]                    # (B, K, r)
        answer_states = mamba_out[:, K_max:, :]                   # (B, T_ans, r) — possibly empty

        # Gate from each Mamba output (with bias + activation + clamp)
        if self.scalar_output:
            z_chunks = self.out(chunk_states) + self.gate_bias[:1]
            z_answer = self.out(answer_states) + self.gate_bias[:1] if answer_states.shape[1] > 0 else None
        else:
            z_chunks = chunk_states + self.gate_bias
            z_answer = answer_states + self.gate_bias if answer_states.shape[1] > 0 else None

        lam_chunks = self._activate(z_chunks)
        if self.gate_clamp_max is not None:
            lam_chunks = torch.clamp(lam_chunks, max=self.gate_clamp_max)
        if z_answer is not None:
            lam_answer = self._activate(z_answer)
            if self.gate_clamp_max is not None:
                lam_answer = torch.clamp(lam_answer, max=self.gate_clamp_max)
        else:
            lam_answer = None

        self._last_chunk_gates = lam_chunks.detach()

        # Cache for generation: chunk_reps and any answer h_base seen so far
        self._cached_chunk_reps = chunk_reps.detach()
        # Pre-fill answer cache with whatever answer-region tokens this forward saw
        # (training: full answer; eval first forward: empty)
        if ans_start < T:
            self._cached_answer_h_base = h_base[:, ans_start:, :].detach()
        else:
            self._cached_answer_h_base = None

        # Build per-position lam: prompt → chunk gates, answer → per-token gates
        out_dim = 1 if self.scalar_output else self.rank
        lam = torch.ones(B, T, out_dim, device=x.device, dtype=input_dtype)
        for b in range(B):
            ends_b = chunk_ends[b]
            if len(ends_b) == 0:
                continue
            prev_end = -1
            for k, end_tok in enumerate(ends_b):
                start_tok = prev_end + 1
                lam[b, start_tok:end_tok + 1] = lam_chunks[b, k]
                prev_end = end_tok
            # Answer region: per-token gates from Mamba's continuation
            if lam_answer is not None and ans_start < T:
                lam[b, ans_start:] = lam_answer[b]

        return lam


class ChunkResidualMambaGate(nn.Module):
    """L2 (per-token linear) + L3 (per-chunk Mamba state) hybrid gate.

    Design (additive residual, LayerScale-style mixing):

        Step 1: h_base = down(x)                            (B, T, r) — L2 per-token
        Step 2: chunk_reps = h_base[chunk_ends]             (B, K, r) — pool at boundaries
        Step 3: h_rec_chunks = LN(Mamba(chunk_reps))        (B, K, r) — L3 per-chunk state
        Step 4: h_chunk_per_token = broadcast(h_rec_chunks) (B, T, r) — same value per chunk
        Step 5: α = sigmoid(g)
                z = h_base + α · h_chunk_per_token          (B, T, r)
        Step 6: λ = softplus(z + bias)                      (B, T, r)

    Properties:
      - α=0 → pure linear gate (= LinearModulator, the safe baseline).
      - α>0 → linear + per-chunk modulation (hierarchical signal).
      - Mamba runs over K (~12) chunks instead of T (~1500) tokens — 100× faster.
      - Tokens in same chunk share chunk-Mamba contribution; tokens in different
        chunks see different contributions.
      - Tokens past last chunk_end inherit last chunk's state (covers answer-region
        in training and generated tokens at eval).
    """

    def __init__(self, input_dim, rank,
                 d_state=16, d_conv=4, expand_factor=2, n_layers=1,
                 gate_activation="softplus", scalar_output=False,
                 alpha_init=0.0, sigmoid_alpha=True,
                 freeze_bias=False, gate_clamp_max=None):
        super().__init__()
        self.gate_clamp_max = gate_clamp_max
        from mambapy.mamba import MambaBlock, Mamba, MambaConfig

        self.rank = rank
        self.gate_activation = gate_activation
        self.scalar_output = scalar_output
        self.sigmoid_alpha = sigmoid_alpha

        # Single shared down-projection used by BOTH L2 (per-token signal) and
        # L3 (chunk Mamba input).
        self.down = nn.Linear(input_dim, rank, bias=False)
        nn.init.xavier_uniform_(self.down.weight, gain=0.1)

        # L3: Mamba over chunks (operates at d_model=rank, sees K~12 chunks)
        cfg = MambaConfig(
            d_model=rank, n_layers=n_layers,
            d_state=d_state, d_conv=d_conv, expand_factor=expand_factor,
            dt_min=0.001, dt_max=0.1, pscan=True, use_cuda=False,
        )
        self.mamba = Mamba(cfg) if n_layers > 1 else MambaBlock(cfg)

        # LayerNorm on Mamba output (LayerScale convention — pins ‖h_rec‖ to a
        # known scale so α has interpretable meaning).
        self.outer_ln = nn.LayerNorm(rank)

        # α: scalar mixing weight, in (0, 1) when sigmoid_alpha=True.
        # alpha_init=0.0 + sigmoid_alpha=True → α=σ(0)=0.5 at init.
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        # Scalar reduction (rank → 1) at the end if scalar_output.
        if scalar_output:
            self.out = nn.Linear(rank, 1, bias=False)
            nn.init.xavier_uniform_(self.out.weight, gain=0.1)
        else:
            self.out = None

        # Bias on z (rank-long; only [:1] used if scalar_output)
        self.gate_bias = nn.Parameter(torch.zeros(rank))
        with torch.no_grad():
            if gate_activation == "softplus":
                self.gate_bias.data.fill_(SOFTPLUS_BIAS_FOR_INIT_ONE)
                self.gate_bias.data.add_(0.1 * torch.randn(rank))

        if freeze_bias:
            self.gate_bias.requires_grad = False

        # Per-forward chunk info (set externally by set_chunk_ends_on_model)
        self._chunk_ends = None
        # Diagnostic: last forward's per-chunk Mamba states (for analysis).
        self._last_chunk_states = None
        # KV-cache for autoregressive generation: at T==1, reuse the last
        # forward's broadcast value at the last chunk position (= h_rec_chunks[-1]).
        # Invalidated when set_chunk_ends_on_model is called for a new sample.
        self._cached_last_chunk_state = None  # (B, r) — Mamba state at last chunk

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        # Mamba pscan needs fp32; LN params follow normal cast (handled outside).
        self.mamba = self.mamba.float()
        return result

    def _activate(self, z):
        if self.gate_activation == "softplus":
            return F.softplus(z)
        elif self.gate_activation == "sigmoid":
            return 2.0 * torch.sigmoid(z)
        return F.softplus(z)

    def forward(self, x):
        B, T, _ = x.shape
        input_dtype = x.dtype

        if self._chunk_ends is None:
            # No chunk info → identity gate (= pure LoRA, no L2/L3 modulation).
            out_dim = 1 if self.scalar_output else self.rank
            return torch.ones(B, T, out_dim, device=x.device, dtype=input_dtype)

        # --- L2: per-token linear signal (always computed, including T==1 gen) ---
        h_base = self.down(x)  # (B, T, r)

        # --- KV-cache fast path (T==1, generation) ---
        # Reuse last chunk's Mamba state (chunk_reps don't change during gen).
        if T == 1 and self._cached_last_chunk_state is not None:
            cached = self._cached_last_chunk_state
            if cached.shape[0] == B:
                # h_chunk_per_token at this single new token = last chunk's state
                h_chunk_per_tok = cached.unsqueeze(1).to(h_base.dtype)  # (B, 1, r)
                alpha_val = (torch.sigmoid(self.alpha) if self.sigmoid_alpha
                             else self.alpha).to(h_base.dtype)
                z = h_base + alpha_val * h_chunk_per_tok
                if self.scalar_output:
                    z = self.out(z) + self.gate_bias[:1]
                else:
                    z = z + self.gate_bias
                lam = self._activate(z)
                if self.gate_clamp_max is not None:
                    lam = torch.clamp(lam, max=self.gate_clamp_max)
                return lam.to(input_dtype)

        # --- Full forward (training, eval first prompt forward) ---
        raw_ce = self._chunk_ends
        if len(raw_ce) != B:
            if B % len(raw_ce) == 0:
                rep = B // len(raw_ce)
                raw_ce = [ce for ce in raw_ce for _ in range(rep)]
            else:
                raw_ce = list(raw_ce) + [raw_ce[-1]] * (B - len(raw_ce))

        chunk_ends = [
            [min(e, T - 1) for e in ce] for ce in raw_ce
        ]
        K_max = max(len(ce) for ce in chunk_ends)

        # Pool chunk reps from h_base at chunk-end positions
        chunk_reps = torch.zeros(
            B, K_max, self.rank, device=x.device, dtype=h_base.dtype,
        )
        for b in range(B):
            ends_b = chunk_ends[b]
            if len(ends_b) > 0:
                idx = torch.as_tensor(ends_b, device=x.device, dtype=torch.long)
                chunk_reps[b, :len(ends_b)] = h_base[b, idx, :]

        # L3: Mamba over chunks (K_max recurrent steps, K << T)
        h_rec_chunks = self.mamba(chunk_reps.float())  # fp32
        # LayerNorm on Mamba output — LN params follow model dtype, h_rec follows fp32
        ln_dtype = self.outer_ln.weight.dtype
        h_rec_chunks = self.outer_ln(h_rec_chunks.to(ln_dtype)).to(h_base.dtype)
        # (B, K_max, r)
        self._last_chunk_states = h_rec_chunks.detach()

        # Cache last chunk's state for generation
        last_idx = torch.tensor(
            [max(0, len(ce) - 1) for ce in chunk_ends],
            device=x.device, dtype=torch.long,
        )
        # Gather per-batch: last chunk index for each sample
        last_states = h_rec_chunks[torch.arange(B, device=x.device), last_idx]  # (B, r)
        self._cached_last_chunk_state = last_states.detach()

        # Broadcast each chunk's state to its tokens
        h_chunk_per_tok = torch.zeros(B, T, self.rank,
                                      device=x.device, dtype=h_base.dtype)
        for b in range(B):
            ends_b = chunk_ends[b]
            if len(ends_b) == 0:
                continue
            prev_end = -1
            for k, end_tok in enumerate(ends_b):
                start_tok = prev_end + 1
                h_chunk_per_tok[b, start_tok:end_tok + 1] = h_rec_chunks[b, k]
                prev_end = end_tok
            # Tokens past last chunk_end inherit last chunk's state
            last_end = ends_b[-1]
            if last_end < T - 1:
                h_chunk_per_tok[b, last_end + 1:] = h_rec_chunks[b, len(ends_b) - 1]

        # Residual combination: linear + α · chunk-Mamba correction
        alpha_val = (torch.sigmoid(self.alpha) if self.sigmoid_alpha
                     else self.alpha).to(h_base.dtype)
        z = h_base + alpha_val * h_chunk_per_tok  # (B, T, r)

        # Scalar reduction (if requested) and bias
        if self.scalar_output:
            z = self.out(z) + self.gate_bias[:1]  # (B, T, 1)
        else:
            z = z + self.gate_bias                # (B, T, r)

        lam = self._activate(z)
        if self.gate_clamp_max is not None:
            lam = torch.clamp(lam, max=self.gate_clamp_max)
        return lam.to(input_dtype)


def set_chunk_ends_on_model(model, batch_chunk_ends):
    """Broadcast chunk-ends list-of-lists to every chunk gate variant in model.

    batch_chunk_ends: List[B] of List[int] (last-token indices per chunk).
    Also invalidates per-sample caches (new sample → new chunk reps + state).
    """
    for m in model.modules():
        if isinstance(m, ChunkMambaGate):
            m._chunk_ends = batch_chunk_ends
            m._cached_lam_chunks = None
        elif isinstance(m, ChunkTokenMambaGate):
            m._chunk_ends = batch_chunk_ends
            m._cached_chunk_reps = None
            m._cached_answer_h_base = None
        elif isinstance(m, ChunkResidualMambaGate):
            m._chunk_ends = batch_chunk_ends
            m._cached_last_chunk_state = None


def clear_chunk_ends_on_model(model):
    for m in model.modules():
        if isinstance(m, ChunkMambaGate):
            m._chunk_ends = None
            m._cached_lam_chunks = None
        elif isinstance(m, ChunkTokenMambaGate):
            m._chunk_ends = None
            m._cached_chunk_reps = None
            m._cached_answer_h_base = None
        elif isinstance(m, ChunkResidualMambaGate):
            m._chunk_ends = None
            m._cached_last_chunk_state = None


def snapshot_chunk_gates(model, reduce="mean_rank"):
    """Collect the last-forward per-chunk gate values from every ChunkMambaGate.

    Args:
        model: model containing ChunkMambaGate modules
        reduce: "mean_rank" (avg over rank dim), "none" (keep full tensor),
                "keep_rank" (return rank-wise values).

    Returns:
        dict {module_name: gate_tensor}
          If reduce="mean_rank": tensor shape (B, K_max) — compact for logging.
          Else: full (B, K_max, r_or_1).
    """
    snap = {}
    for name, m in model.named_modules():
        if isinstance(m, ChunkMambaGate):
            g = m._last_chunk_gates
            if g is None:
                continue
            if reduce == "mean_rank":
                snap[name] = g.float().mean(dim=-1).cpu()    # (B, K)
            elif reduce == "keep_rank":
                snap[name] = g.float().cpu()                 # (B, K, r)
            else:
                snap[name] = g.cpu()
    return snap


def summarize_chunk_gates(snap, chunk_names_per_batch=None):
    """Pretty-print a snapshot dict (from snapshot_chunk_gates).

    Returns a list of dicts suitable for JSONL logging:
        {module, batch_idx, chunk_names, gate_means, gate_std_across_chunks}
    """
    records = []
    for mod_name, g in snap.items():
        # g: (B, K)
        for b in range(g.shape[0]):
            means = g[b].tolist()
            names = (chunk_names_per_batch[b] if chunk_names_per_batch else
                     [f"C{k}" for k in range(len(means))])
            std_across = float(g[b].std().item()) if g[b].numel() > 1 else 0.0
            records.append({
                "module": mod_name,
                "batch_idx": b,
                "chunk_names": names,
                "gate_means": means,
                "std_across_chunks": std_across,
            })
    return records
