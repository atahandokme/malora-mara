"""
EvidenceRouter: per-segment state tracker backing MaRA, the retrieval adapter.

Pass 1 (router-only, no LM loss):
    For each paragraph P_t (t = 1..N):
        1. Embed [Q ; P_t] with frozen base model (embed + first K layers, no_grad)
        2. Attention-pool over P_t-region positions   ->  e_t   (d_model,)
        3. Linear-project e_t down to mamba_dim       ->  z_t   (mamba_dim,)
    Run chunk-Mamba over (z_1, ..., z_N)              ->  s_t   (mamba_dim,)

Scoring:
    local head  over s_t   -> per-segment relevance logit
    global head over s_N   -> aggregate auxiliary logit

The `router_kind` and `h_mlp_hidden` arguments are accepted for backward
compatibility with older checkpoints but are no longer used. The simplified
adapter scores directly from the mixer output; see the note in __init__.

Frozen vs trainable:
    Frozen:    base_model.embed_tokens, base_model.layers[:K], rotary_emb
    Trainable: attn_pool_W, attn_pool_q, linear_in, mamba, local_head, global_head

This module adapts the input context rather than the model weights. It selects
which segments reach the generator and splices nothing into the LM input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceRouter(nn.Module):
    def __init__(
        self,
        base_model,
        encoder_K=4,
        mamba_dim=64,
        n_paragraphs=10,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_n_layers=1,
        attn_pool_dim=256,
        router_kind="global_mlp",       # {"causal", "global_mlp"}
        h_mlp_hidden=None,              # default = mamba_dim
        joint_encoding=False,           # if True, encode Q+all_paragraphs jointly (one LM pass per question)
        iterative=False,                # if True, run 2-pass routing (pass 2 conditions on pass-1's top-1 paragraph)
        use_mean_pool=False,            # if True, replace attn-pool with mean over paragraph tokens (ablation)
        scoring_head="linear",          # {"linear", "mlp"} local head architecture (appendix ablation)
    ):
        super().__init__()
        from mambapy.mamba import MambaBlock, Mamba, MambaConfig

        assert router_kind in ("causal", "global_mlp"), router_kind
        self.router_kind = router_kind
        self.encoder_K = encoder_K
        self.mamba_dim = mamba_dim
        self.n_paragraphs = n_paragraphs
        self.attn_pool_dim = attn_pool_dim
        self.joint_encoding = joint_encoding
        self.iterative = iterative
        self.use_mean_pool = use_mean_pool

        d_model = base_model.config.hidden_size
        self.d_model = d_model

        # Hold a non-registered ref to the frozen base model (avoid double-registering its params).
        object.__setattr__(self, "_base_model", base_model)

        # Attention-pool over paragraph tokens (replaces mean-pool).
        self.attn_pool_W = nn.Linear(d_model, attn_pool_dim, bias=False)
        nn.init.xavier_uniform_(self.attn_pool_W.weight, gain=0.1)
        self.attn_pool_q = nn.Linear(attn_pool_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.attn_pool_q.weight, gain=0.1)

        # d_model -> mamba_dim
        self.linear_in = nn.Linear(d_model, mamba_dim, bias=False)
        nn.init.xavier_uniform_(self.linear_in.weight, gain=0.1)

        # Chunk-Mamba over paragraph reps
        cfg = MambaConfig(
            d_model=mamba_dim, n_layers=mamba_n_layers,
            d_state=mamba_d_state, d_conv=mamba_d_conv,
            expand_factor=mamba_expand,
            dt_min=0.001, dt_max=0.1, pscan=True, use_cuda=False,
        )
        self.mamba = (Mamba(cfg) if mamba_n_layers > 1 else MambaBlock(cfg))

        # NOTE: `router_kind` and `h_mlp_hidden` are accepted for backward
        # compatibility but no longer used. The simplified router scores
        # directly from the mixer output `s` (local head) and the aggregate
        # `g` (global head). The old h_mlp combiner produced an `h` that no
        # scoring head consumed, so it received zero gradient and is removed.

        # Heads (BCE supervision in stage 1)
        assert scoring_head in ("linear", "mlp"), scoring_head
        self.scoring_head = scoring_head
        if scoring_head == "mlp":
            h = h_mlp_hidden or mamba_dim
            self.local_head = nn.Sequential(
                nn.Linear(mamba_dim, h, bias=True),
                nn.GELU(),
                nn.Linear(h, 1, bias=True),
            )
            for m in self.local_head:
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.bias)
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
        else:
            self.local_head = nn.Linear(mamba_dim, 1, bias=True)
            nn.init.zeros_(self.local_head.bias)
            nn.init.xavier_uniform_(self.local_head.weight, gain=0.1)

        self.global_head = nn.Linear(mamba_dim, n_paragraphs, bias=True)
        nn.init.zeros_(self.global_head.bias)
        nn.init.xavier_uniform_(self.global_head.weight, gain=0.1)

        # Diagnostics (populated in forward)
        self._last_local_logits = None     # (B, N)
        self._last_global_logits = None    # (B, N)
        self._last_attn_entropy = None     # (B, N)
        self._last_attn_max = None         # (B, N)
        self._last_h = None                # (B, N, mamba_dim)
        self._last_g = None                # (B, mamba_dim)
        self._last_z = None                # (B, N, mamba_dim)

    @property
    def base_model(self):
        return self._base_model

    def to(self, *args, **kwargs):
        # Keep Mamba (and h_mlp downstream of it) numerically stable in fp32.
        result = super().to(*args, **kwargs)
        self.mamba = self.mamba.float()
        return result

    @torch.no_grad()
    def _run_k_layers(self, input_ids):
        """Embed + first-K Qwen layers, frozen, no_grad."""
        bm = self._base_model
        device = input_ids.device
        seq_len = input_ids.size(1)

        hidden = bm.model.embed_tokens(input_ids)
        # Gemma scales embeddings by sqrt(hidden_size); apply it for Gemma/Gemma2.
        # Qwen/Llama don't do this scaling.
        if getattr(bm.config, "model_type", "") in ("gemma", "gemma2"):
            hidden = hidden * (bm.config.hidden_size ** 0.5)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(input_ids)
        position_embeddings = bm.model.rotary_emb(hidden, position_ids)

        for layer in bm.model.layers[: self.encoder_K]:
            out = layer(
                hidden,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden = out[0] if isinstance(out, tuple) else out
        return hidden  # (B*N, T_enc, d_model)

    def _encode_and_score(self, input_ids, p_chunk_spans, q_span, pad_token_id,
                          top1_idx=None, anchor_idxs=None):
        """Encode all paragraphs (optionally with anchor texts prepended) then run Mamba + heads.

        Args:
            top1_idx: None for pass-1 (rows = [Q + P_t]); (B,) tensor of paragraph indices
                      for pass-2 (rows = [Q + P_top1 + P_t]). Single-anchor compat path.
            anchor_idxs: None or list-of-list of paragraph indices (length-B). For each batch
                      item b, the row for candidate t is [Q + P_a1 + P_a2 + ... + P_an + P_t]
                      where (a1..an) = anchor_idxs[b]. Used by N-pass chain routing.
                      If both top1_idx and anchor_idxs are None, behaves as Pass 1.

        Returns: (h, g, local_logits, global_logits, attn_entropy, attn_max, z, e_t)
        """
        B = input_ids.size(0)
        # Per-call N: each batch element's actual paragraph count (from p_chunk_spans).
        # For mixed-pool training (MQ has 20, TW/HP have 10), N varies per call. We assume B=1
        # for the variable-N path (router training always uses batch_size=1). For B>1, all
        # batch items must share the same N (legacy single-dataset case).
        N = len(p_chunk_spans[0])
        assert B == 1 or all(len(s) == N for s in p_chunk_spans), \
            f"variable-N requires B=1; got B={B} with mixed N={[len(s) for s in p_chunk_spans]}"
        device = input_ids.device

        if self.joint_encoding:
            # Single LM forward pass over the full prompt (Q + all paragraphs concatenated as in input_ids).
            # Each paragraph's hidden states now causally attend to all earlier paragraphs (and the question).
            encoder_hidden = self._run_k_layers(input_ids)  # (B, T, d_model)
            e_list = []
            attn_entropy = torch.zeros(B * N, dtype=torch.float32, device=device)
            attn_max = torch.zeros(B * N, dtype=torch.float32, device=device)
            for b in range(B):
                for t in range(N):
                    ps, pe = p_chunk_spans[b][t]
                    chunk_h = encoder_hidden[b, ps : pe + 1, :]            # (pl, d_model)
                    if chunk_h.size(0) == 0:
                        e_list.append(torch.zeros(self.d_model, dtype=chunk_h.dtype, device=device))
                        continue
                    if self.use_mean_pool:
                        e_list.append(chunk_h.mean(dim=0))
                        attn_w = torch.full((chunk_h.size(0),), 1.0 / chunk_h.size(0),
                                            device=chunk_h.device, dtype=torch.float32)
                    else:
                        scored = self.attn_pool_q(
                            torch.tanh(self.attn_pool_W(chunk_h.float()))
                        ).squeeze(-1)
                        attn_w = F.softmax(scored, dim=0)
                        e_list.append((attn_w.to(chunk_h.dtype).unsqueeze(-1) * chunk_h).sum(dim=0))
                    with torch.no_grad():
                        i = b * N + t
                        p = attn_w.float().clamp_min(1e-12)
                        attn_entropy[i] = -(p * p.log()).sum()
                        attn_max[i] = attn_w.float().max()
            e_t = torch.stack(e_list, dim=0).view(B, N, self.d_model)       # (B, N, d_model)
        else:
            # Original independent encoding: B*N separate rows.
            #   pass-1 (top1_idx=None):     row = [Q + P_t]
            #   pass-2 (top1_idx provided): row = [Q + P_top1(b) + P_t]
            enc_inputs, p_lens, q_lens, prefix_lens = [], [], [], []
            max_enc_len = 0
            # Resolve anchors per batch item: prefer anchor_idxs (list-of-list), else top1_idx (single).
            if anchor_idxs is None and top1_idx is not None:
                anchors_per_b = [[int(top1_idx[b].item())] for b in range(B)]
            elif anchor_idxs is not None:
                anchors_per_b = [list(anchor_idxs[b]) for b in range(B)]
            else:
                anchors_per_b = [[] for _ in range(B)]

            for b in range(B):
                qs, qe = q_span[b]
                q_ids = input_ids[b, qs : qe + 1]
                ql = q_ids.size(0)
                # Build prefix = Q + P_a1 + P_a2 + ... + P_an
                anchor_ids_list = []
                for a in anchors_per_b[b]:
                    aps, ape = p_chunk_spans[b][a]
                    anchor_ids_list.append(input_ids[b, aps : ape + 1])
                anchor_total_len = sum(t.size(0) for t in anchor_ids_list)
                pre_len = ql + anchor_total_len
                for t in range(N):
                    ps, pe = p_chunk_spans[b][t]
                    p_ids = input_ids[b, ps : pe + 1]
                    pl = p_ids.size(0)
                    if anchor_ids_list:
                        row = torch.cat([q_ids] + anchor_ids_list + [p_ids], dim=0)
                    else:
                        row = torch.cat([q_ids, p_ids], dim=0)
                    enc_inputs.append(row)
                    p_lens.append(pl)
                    q_lens.append(ql)
                    prefix_lens.append(pre_len)
                    max_enc_len = max(max_enc_len, row.size(0))

            enc_input_ids = torch.full(
                (B * N, max_enc_len), pad_token_id, dtype=input_ids.dtype, device=device,
            )
            for i, t in enumerate(enc_inputs):
                enc_input_ids[i, : t.size(0)] = t

            encoder_hidden = self._run_k_layers(enc_input_ids)   # (B*N, T_enc, d_model)

            # Attention-pool over the P_t portion (skip Q+top1 prefix)
            e_list = []
            attn_entropy = torch.zeros(B * N, dtype=torch.float32, device=device)
            attn_max = torch.zeros(B * N, dtype=torch.float32, device=device)
            for i in range(B * N):
                pre = prefix_lens[i]
                pl = p_lens[i]
                chunk_h = encoder_hidden[i, pre : pre + pl, :]                  # (pl, d_model)
                if chunk_h.size(0) == 0:
                    e_list.append(torch.zeros(self.d_model, dtype=encoder_hidden.dtype, device=device))
                    continue
                if self.use_mean_pool:
                    e_list.append(chunk_h.mean(dim=0))
                    attn_w = torch.full((chunk_h.size(0),), 1.0 / chunk_h.size(0),
                                        device=chunk_h.device, dtype=torch.float32)
                else:
                    scored = self.attn_pool_q(
                        torch.tanh(self.attn_pool_W(chunk_h.float()))
                    ).squeeze(-1)                                                   # (pl,)
                    attn_w = F.softmax(scored, dim=0)                               # (pl,)
                    e_list.append((attn_w.to(chunk_h.dtype).unsqueeze(-1) * chunk_h).sum(dim=0))
                with torch.no_grad():
                    p = attn_w.float().clamp_min(1e-12)
                    attn_entropy[i] = -(p * p.log()).sum()
                    attn_max[i] = attn_w.float().max()
            e_t = torch.stack(e_list, dim=0).view(B, N, self.d_model)           # (B, N, d_model)

        # Project to mamba_dim and run chunk-Mamba (fp32)
        z = self.linear_in(e_t.float())                                     # (B, N, r) fp32
        s = self.mamba(z)                                                   # (B, N, r) fp32
        g = s[:, -1, :]                                                     # (B, r)

        # Simplified router: score directly from the mixer output.
        #   local head : per-paragraph causal score from s_t
        #   global head: all-N scores from the final state g = s_N
        h = s  # kept for return-signature compatibility (no h_mlp combiner)

        local_logits = self.local_head(s).squeeze(-1)                       # (B, N)
        # global_head outputs fixed self.n_paragraphs dim (set to max N at construction).
        # Slice to actual N so loss shapes match local_logits and gold.
        global_logits = self.global_head(g)[:, :N]                          # (B, N)

        return h, g, local_logits, global_logits, attn_entropy, attn_max, z, e_t

    def forward(self, input_ids, p_chunk_spans, q_span, pad_token_id):
        """Run routing — single-pass or 2-pass iterative depending on self.iterative.

        Single-pass returns:  (h, g, local_logits, global_logits)
        Iterative returns:    (h, g, local_logits, global_logits, aux_dict)
                              where aux_dict has 'pass1_local' and 'pass1_global'
        """
        B = input_ids.size(0)
        # Per-call N from actual chunk spans (supports variable-N for mixed-pool training).
        N = len(p_chunk_spans[0])

        # Pass 1
        h1, g1, local_logits_1, global_logits_1, attn_entropy_1, attn_max_1, z1, e_t_1 = (
            self._encode_and_score(input_ids, p_chunk_spans, q_span, pad_token_id, top1_idx=None)
        )

        if not self.iterative:
            # Diagnostics
            self._last_attn_entropy = attn_entropy_1.view(B, N).detach().cpu()
            self._last_attn_max = attn_max_1.view(B, N).detach().cpu()
            self._last_local_logits = local_logits_1
            self._last_global_logits = global_logits_1
            self._last_h = h1
            self._last_g = g1
            self._last_z = z1
            return h1, g1, local_logits_1, global_logits_1

        # Pass 2: condition each paragraph row on pass-1's predicted top-1.
        # argmax is non-differentiable; gradient flow into pass-1 logits comes
        # from the auxiliary loss applied externally on local_logits_1.
        with torch.no_grad():
            top1_idx = local_logits_1.argmax(dim=-1)  # (B,)

        h2, g2, local_logits_2, global_logits_2, attn_entropy_2, attn_max_2, z2, e_t_2 = (
            self._encode_and_score(input_ids, p_chunk_spans, q_span, pad_token_id, top1_idx=top1_idx)
        )

        # Diagnostics (use pass-2 as primary)
        self._last_attn_entropy = attn_entropy_2.view(B, N).detach().cpu()
        self._last_attn_max = attn_max_2.view(B, N).detach().cpu()
        self._last_local_logits = local_logits_2
        self._last_global_logits = global_logits_2
        self._last_h = h2
        self._last_g = g2
        self._last_z = z2

        aux = {
            "pass1_local": local_logits_1,
            "pass1_global": global_logits_1,
            "top1_idx": top1_idx,
        }
        return h2, g2, local_logits_2, global_logits_2, aux

    def forward_chain_train(self, input_ids, p_chunk_spans, q_span, pad_token_id,
                            n_passes=4):
        """Training-time N-pass chain routing.

        Step k (k=1..N):
          - Encode each candidate paragraph with prefix [Q + P_a1 + ... + P_a_{k-1}]
          - Score all N paragraphs
          - Mask already-selected anchors, argmax to pick next anchor
          - Append to anchor list

        Returns dict with:
          - logits_per_step: list[N] of (B, n_paragraphs) local logits
          - globals_per_step: list[N] of (B, n_paragraphs) global logits
          - anchor_mask_per_step: list[N] of (B, n_paragraphs) bool (True = already selected)
          - anchors: List[List[int]] final picks per batch (length n_passes)
        Caller computes per-step BCE with masked positions excluded from the loss.
        """
        B = input_ids.size(0)
        N = self.n_paragraphs

        saved_iterative = self.iterative
        self.iterative = False
        try:
            anchors_per_b = [[] for _ in range(B)]
            logits_per_step = []
            globals_per_step = []
            masks_per_step = []

            for step in range(n_passes):
                # Mask: True = already selected (don't pick again, exclude from loss)
                anchor_mask = torch.zeros(B, N, dtype=torch.bool, device=input_ids.device)
                for b in range(B):
                    for a in anchors_per_b[b]:
                        anchor_mask[b, a] = True

                h_s, g_s, local_logits_s, global_logits_s, _, _, _, _ = (
                    self._encode_and_score(
                        input_ids, p_chunk_spans, q_span, pad_token_id,
                        anchor_idxs=anchors_per_b,
                    )
                )
                logits_per_step.append(local_logits_s)
                globals_per_step.append(global_logits_s)
                masks_per_step.append(anchor_mask)

                # Pick next anchor (argmax over non-anchor positions)
                with torch.no_grad():
                    masked_logits = local_logits_s.masked_fill(anchor_mask, float('-inf'))
                    next_idx = masked_logits.argmax(dim=-1)  # (B,)
                for b in range(B):
                    anchors_per_b[b].append(int(next_idx[b].item()))
        finally:
            self.iterative = saved_iterative

        return {
            "logits_per_step": logits_per_step,
            "globals_per_step": globals_per_step,
            "anchor_mask_per_step": masks_per_step,
            "anchors": anchors_per_b,
        }

    def forward_beam_train(self, input_ids, p_chunk_spans, q_span, pad_token_id,
                           beam_k=2):
        """Training-time beam: Pass 1 -> top-K anchors -> Pass 2 once per anchor.

        Returns dict with:
          - pass1_local, pass1_global: Pass-1 logits (B, N)
          - pass2_logits_list: list[K] of Pass-2 local_logits (B, N)
          - pass2_globals_list: list[K] of Pass-2 global_logits (B, N)
          - beam_idx: (B, K) indices of anchors used
        Caller computes per-pass losses and averages across the K anchors.
        """
        B = input_ids.size(0)
        N = self.n_paragraphs

        saved_iterative = self.iterative
        self.iterative = False
        try:
            # Pass 1
            h1, g1, local_logits_1, global_logits_1, attn_entropy_1, attn_max_1, z1, e_t_1 = (
                self._encode_and_score(input_ids, p_chunk_spans, q_span, pad_token_id, top1_idx=None)
            )

            # Top-K anchors (no gradient through argsort/topk)
            beam_k_eff = min(beam_k, N)
            with torch.no_grad():
                topk_idx = local_logits_1.topk(beam_k_eff, dim=-1).indices   # (B, K)

            # Pass 2 once per anchor (with gradient)
            pass2_locals = []
            pass2_globals = []
            for i in range(beam_k_eff):
                anchor_idx = topk_idx[:, i]
                h2, g2, ll_i, gl_i, _, _, _, _ = self._encode_and_score(
                    input_ids, p_chunk_spans, q_span, pad_token_id, top1_idx=anchor_idx
                )
                pass2_locals.append(ll_i)
                pass2_globals.append(gl_i)
        finally:
            self.iterative = saved_iterative

        return {
            'pass1_local': local_logits_1,
            'pass1_global': global_logits_1,
            'pass2_locals': pass2_locals,
            'pass2_globals': pass2_globals,
            'beam_idx': topk_idx,
        }


    @torch.no_grad()
    def forward_beam(self, input_ids, p_chunk_spans, q_span, pad_token_id,
                     beam_k=3, aggregate='mean'):
        """Beam-iter inference: extend 2-pass iter with K-hypothesis anchors.

        Pass 1: score all N paragraphs (same as iter).
        Take top-K of Pass 1 as candidate anchors.
        Pass 2: for EACH of K anchors, re-encode each paragraph conditioned on
                that anchor, get a local_logits vector. Total K pass-2 forwards.
        Final: aggregate the K pass-2 logits (mean or max) -> (B, N) scores.

        Returns (h1, g1, final_local_logits, global_logits_1, aux_dict).
        Drop-in compatible with .forward(iterative=True) return shape.
        """
        B = input_ids.size(0)
        N = self.n_paragraphs

        saved_iterative = self.iterative
        self.iterative = False
        try:
            h1, g1, local_logits_1, global_logits_1, attn_entropy_1, attn_max_1, z1, e_t_1 = (
                self._encode_and_score(input_ids, p_chunk_spans, q_span, pad_token_id, top1_idx=None)
            )

            # Top-K anchor indices
            beam_k_eff = min(beam_k, N)
            topk = local_logits_1.topk(beam_k_eff, dim=-1).indices   # (B, K)

            pass2_logits = []
            for i in range(beam_k_eff):
                anchor_idx = topk[:, i]
                h2, g2, ll_i, gl_i, _, _, _, _ = self._encode_and_score(
                    input_ids, p_chunk_spans, q_span, pad_token_id, top1_idx=anchor_idx
                )
                pass2_logits.append(ll_i)
        finally:
            self.iterative = saved_iterative

        stacked = torch.stack(pass2_logits, dim=0)   # (K, B, N)
        if aggregate == 'mean':
            final_local = stacked.mean(dim=0)
        elif aggregate == 'max':
            final_local = stacked.max(dim=0).values
        elif aggregate == 'logsumexp':
            final_local = torch.logsumexp(stacked, dim=0) - torch.log(torch.tensor(float(beam_k_eff)))
        else:
            raise ValueError(f"unknown aggregate: {aggregate}")

        aux = {
            'pass1_local': local_logits_1,
            'pass1_global': global_logits_1,
            'beam_anchor_idx': topk,
            'pass2_logits_per_anchor': stacked,
        }
        return h1, g1, final_local, global_logits_1, aux


    def forward_cascade(self, input_ids, p_chunk_spans, q_span, pad_token_id,
                        n_passes=4, drop_per_pass=4, use_anchor=False,
                        gold=None):
        """Multi-pass cascade scoring (B=1 only).

        At each pass:
          1. Build new input as [Q + (anchor paragraphs if use_anchor) + (survivor paragraphs)].
          2. Run _encode_and_score on the trimmed input with n_paragraphs=K.
          3. Score → (if use_anchor) pick new anchor (top scorer not yet anchored).
          4. Drop worst `drop_per_pass` survivors. Anchors are protected.
             If `gold` provided (training): teacher-force, always keep all gold.
             If not (inference): pure model-driven drop.

        Returns: list of per-pass dicts:
          [{'survivors': list[K] orig indices,
            'local_logits': tensor(1, K),
            'anchors_so_far': list orig indices},
            ...]
        """
        B = input_ids.size(0)
        assert B == 1, "forward_cascade only supports B=1"

        p_spans = p_chunk_spans[0]
        qs, qe = q_span[0]
        q_ids = input_ids[0, qs:qe + 1]
        N_orig = len(p_spans)

        survivors = list(range(N_orig))
        anchors = []
        pass_outputs = []

        # During cascade, each internal pass is a single forward.
        saved_iterative = self.iterative
        saved_n = self.n_paragraphs
        self.iterative = False

        try:
            for pass_idx in range(n_passes):
                K = len(survivors)

                # Build new input_ids: [Q + (anchors_tokens if use_anchor) + (survivor_tokens)]
                parts = [q_ids]
                cursor = q_ids.size(0)
                new_q_span = (0, q_ids.size(0) - 1)

                if use_anchor and anchors:
                    for a in anchors:
                        aps, ape = p_spans[a]
                        parts.append(input_ids[0, aps:ape + 1])
                        cursor += parts[-1].size(0)

                new_p_spans = []
                for s in survivors:
                    ps, pe = p_spans[s]
                    parts.append(input_ids[0, ps:pe + 1])
                    new_p_spans.append((cursor, cursor + parts[-1].size(0) - 1))
                    cursor += parts[-1].size(0)

                new_input_ids = torch.cat(parts, dim=0).unsqueeze(0)

                self.n_paragraphs = K
                h, g, local_logits, global_logits, _, _, _, _ = self._encode_and_score(
                    new_input_ids, [new_p_spans], [new_q_span], pad_token_id, top1_idx=None
                )
                # local_logits: (1, K)

                pass_outputs.append({
                    'survivors': list(survivors),
                    'local_logits': local_logits,
                    'anchors_so_far': list(anchors),
                })

                # Anchor pick: top scorer not yet an anchor
                if use_anchor:
                    with torch.no_grad():
                        sorted_K = local_logits[0].argsort(descending=True).cpu().tolist()
                    for K_pos in sorted_K:
                        orig = survivors[K_pos]
                        if orig not in anchors:
                            anchors.append(orig)
                            break

                # Drop worst drop_per_pass (anchors and, if training, gold are protected)
                if pass_idx < n_passes - 1:
                    keep_count = K - drop_per_pass
                    kept_set = set(anchors)

                    if gold is not None:
                        # Teacher-forced: always keep gold
                        gold_b = gold[0].detach().cpu().tolist() if torch.is_tensor(gold) else gold[0]
                        for s in survivors:
                            if gold_b[s] > 0.5:
                                kept_set.add(s)

                    # Fill rest by score (descending)
                    with torch.no_grad():
                        sorted_K = local_logits[0].argsort(descending=True).cpu().tolist()
                    for K_pos in sorted_K:
                        if len(kept_set) >= keep_count:
                            break
                        orig = survivors[K_pos]
                        if orig not in kept_set:
                            kept_set.add(orig)
                    survivors = sorted(kept_set)
        finally:
            self.iterative = saved_iterative
            self.n_paragraphs = saved_n

        return pass_outputs


def derive_p_chunk_spans(chunk_ends, chunk_names):
    """Return (p_spans, q_spans) from per-batch chunk_ends + chunk_names.

    p_spans[b]: List[(s, e_inclusive)] for each P paragraph
    q_spans[b]: (s, e_inclusive) for the question chunk
    """
    p_all, q_all = [], []
    for ends, names in zip(chunk_ends, chunk_names):
        starts = [0] + [e + 1 for e in ends[:-1]]
        p_spans = []
        q_span = None
        for i, name in enumerate(names):
            if name.startswith("P"):
                p_spans.append((starts[i], ends[i]))
            elif name == "Q":
                q_span = (starts[i], ends[i])
        p_all.append(p_spans)
        q_all.append(q_span)
    return p_all, q_all


class LastTokenRouter(nn.Module):
    """
    Baseline router: no attn-pool, no mamba, no iteration.
    Per paragraph i: take hidden_states[last_token_of_paragraph_i] and pass through
    a single linear layer -> score. That's it.

    Same encoder K layers as EvidenceRouter (frozen base LM up to layer K).
    Returns (h, g, local_logits, global_logits) in the same shape so the trainer
    code (BCE + pairwise + topk eval) works without changes. global_logits are
    just zeros (no global head).
    """
    def __init__(
        self,
        base_model,
        encoder_K=16,
        n_paragraphs=10,
        **_unused,  # accept and ignore mamba_*, attn_pool_*, joint_encoding, iterative, etc.
    ):
        super().__init__()
        self.encoder_K = encoder_K
        self.n_paragraphs = n_paragraphs
        self.iterative = False  # baseline is single-pass

        d_model = base_model.config.hidden_size
        self.d_model = d_model
        object.__setattr__(self, "_base_model", base_model)

        # The only learnable head: 1 linear from hidden_state -> 1 score.
        self.local_head = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(self.local_head.bias)
        nn.init.xavier_uniform_(self.local_head.weight, gain=0.1)

    @torch.no_grad()
    def _run_k_layers(self, input_ids):
        """Same encoder as EvidenceRouter: embed + first-K LM layers, frozen."""
        bm = self._base_model
        device = input_ids.device
        seq_len = input_ids.size(1)
        hidden = bm.model.embed_tokens(input_ids)
        if getattr(bm.config, "model_type", "") in ("gemma", "gemma2"):
            hidden = hidden * (bm.config.hidden_size ** 0.5)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(input_ids)
        position_embeddings = bm.model.rotary_emb(hidden, position_ids)
        for layer in bm.model.layers[: self.encoder_K]:
            out = layer(
                hidden, attention_mask=None, position_ids=position_ids,
                past_key_value=None, use_cache=False, cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden = out[0] if isinstance(out, tuple) else out
        return hidden  # (B, T, d_model)

    def forward(self, input_ids, p_chunk_spans, q_span, pad_token_id):
        """Match EvidenceRouter signature so the trainer can use this drop-in.

        Returns: (h_dummy, g_dummy, local_logits, global_logits)
          local_logits  : (B, N_p) per-paragraph score
          global_logits : (B, N_p) zeros (no global head in baseline)
          h, g          : dummy zeros (kept for sig compat; tracker not trained here)
        """
        B, T = input_ids.shape
        hidden = self._run_k_layers(input_ids)  # (B, T, d_model)

        # For each paragraph in each example, take its last token's hidden state.
        # p_chunk_spans[b] is a list of (s, e_inclusive) tuples.
        all_local = []
        for b in range(B):
            spans = p_chunk_spans[b]
            # Take last-token state per paragraph (e_inclusive)
            last_states = []
            for s, e in spans:
                # Guard against truncation pushing e beyond T-1
                e_clamped = min(e, T - 1)
                last_states.append(hidden[b, e_clamped])  # (d_model,)
            # Pad to n_paragraphs with zeros if fewer (shouldn't happen for our data)
            while len(last_states) < self.n_paragraphs:
                last_states.append(torch.zeros(self.d_model, device=hidden.device, dtype=hidden.dtype))
            feats = torch.stack(last_states[: self.n_paragraphs], dim=0)  # (N_p, d_model)
            feats = feats.to(self.local_head.weight.dtype)  # match head dtype (head is fp32, feats from bf16 LM)
            scores = self.local_head(feats).squeeze(-1)  # (N_p,)
            all_local.append(scores)
        local_logits = torch.stack(all_local, dim=0)  # (B, N_p)

        # Dummies for signature compat
        h_dummy = torch.zeros(B, self.n_paragraphs, self.d_model, device=hidden.device, dtype=hidden.dtype)
        g_dummy = torch.zeros(B, self.d_model, device=hidden.device, dtype=hidden.dtype)
        global_logits = torch.zeros(B, self.n_paragraphs, device=hidden.device, dtype=hidden.dtype)
        return h_dummy, g_dummy, local_logits, global_logits


class MLPRouter(nn.Module):
    """
    MLP baseline (no Mamba, no iteration).
    Keeps the same encoder K layers + attn-pool as EvidenceRouter, but replaces
    the cross-paragraph Mamba SSM with a 2-layer MLP that operates on the
    concat'd paragraph vectors. Has cross-paragraph info (via concat) but
    NO state-space transitions, NO recurrence, NO iterative refinement.

    Drop-in for EvidenceRouter (returns 4-tuple, same eval pipeline).
    """
    def __init__(
        self,
        base_model,
        encoder_K=16,
        mamba_dim=256,           # reused as the per-paragraph hidden dim
        n_paragraphs=10,
        attn_pool_dim=256,
        mlp_hidden=None,         # default 4 * mamba_dim
        **_unused,
    ):
        super().__init__()
        self.encoder_K = encoder_K
        self.mamba_dim = mamba_dim
        self.n_paragraphs = n_paragraphs
        self.attn_pool_dim = attn_pool_dim
        self.iterative = False

        d_model = base_model.config.hidden_size
        self.d_model = d_model
        object.__setattr__(self, "_base_model", base_model)

        # Same attn-pool as EvidenceRouter
        self.attn_pool_W = nn.Linear(d_model, attn_pool_dim, bias=False)
        nn.init.xavier_uniform_(self.attn_pool_W.weight, gain=0.1)
        self.attn_pool_q = nn.Linear(attn_pool_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.attn_pool_q.weight, gain=0.1)

        self.linear_in = nn.Linear(d_model, mamba_dim, bias=False)
        nn.init.xavier_uniform_(self.linear_in.weight, gain=0.1)

        # Cross-paragraph MLP (replaces Mamba). Input/output: (N_p * mamba_dim).
        h = mlp_hidden if mlp_hidden is not None else 4 * mamba_dim
        self.cross_mlp = nn.Sequential(
            nn.Linear(n_paragraphs * mamba_dim, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, n_paragraphs * mamba_dim),
        )
        for m in self.cross_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Heads (same simplified skeleton as EvidenceRouter)
        self.local_head = nn.Linear(mamba_dim, 1, bias=True)
        nn.init.zeros_(self.local_head.bias)
        nn.init.xavier_uniform_(self.local_head.weight, gain=0.1)

        self.global_head = nn.Linear(mamba_dim, n_paragraphs, bias=True)
        nn.init.zeros_(self.global_head.bias)
        nn.init.xavier_uniform_(self.global_head.weight, gain=0.1)

    @torch.no_grad()
    def _run_k_layers(self, input_ids):
        bm = self._base_model
        device = input_ids.device
        seq_len = input_ids.size(1)
        hidden = bm.model.embed_tokens(input_ids)
        if getattr(bm.config, "model_type", "") in ("gemma", "gemma2"):
            hidden = hidden * (bm.config.hidden_size ** 0.5)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(input_ids)
        position_embeddings = bm.model.rotary_emb(hidden, position_ids)
        for layer in bm.model.layers[: self.encoder_K]:
            out = layer(
                hidden, attention_mask=None, position_ids=position_ids,
                past_key_value=None, use_cache=False, cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden = out[0] if isinstance(out, tuple) else out
        return hidden  # (B, T, d_model)

    def _attn_pool_per_paragraph(self, hidden, p_chunk_spans):
        """Attn-pool over each paragraph's tokens, mirroring EvidenceRouter."""
        B = hidden.size(0)
        all_p_vecs = []
        for b in range(B):
            spans = p_chunk_spans[b]
            p_vecs = []
            for s, e in spans:
                e_clamped = min(e, hidden.size(1) - 1)
                chunk_h = hidden[b, s:e_clamped + 1]  # (L, d_model)
                if chunk_h.size(0) == 0:
                    p_vecs.append(torch.zeros(self.d_model, device=hidden.device, dtype=hidden.dtype))
                    continue
                kv = self.attn_pool_W(chunk_h.to(self.attn_pool_W.weight.dtype))  # (L, attn_pool_dim)
                attn_logits = self.attn_pool_q(kv).squeeze(-1)  # (L,)
                attn_w = torch.softmax(attn_logits.float(), dim=0).to(chunk_h.dtype)  # (L,)
                pooled = (attn_w.unsqueeze(-1) * chunk_h).sum(dim=0)  # (d_model,)
                p_vecs.append(pooled)
            while len(p_vecs) < self.n_paragraphs:
                p_vecs.append(torch.zeros(self.d_model, device=hidden.device, dtype=hidden.dtype))
            all_p_vecs.append(torch.stack(p_vecs[: self.n_paragraphs], dim=0))
        return torch.stack(all_p_vecs, dim=0)  # (B, N_p, d_model)

    def forward(self, input_ids, p_chunk_spans, q_span, pad_token_id):
        B = input_ids.size(0)
        hidden = self._run_k_layers(input_ids)  # (B, T, d_model)

        # Per-paragraph attn-pool
        p_vecs = self._attn_pool_per_paragraph(hidden, p_chunk_spans)  # (B, N_p, d_model)
        p_vecs = p_vecs.to(self.linear_in.weight.dtype)
        z = self.linear_in(p_vecs)  # (B, N_p, mamba_dim)  -- z_t

        # Cross-paragraph MLP (the mixer; replaces Mamba)
        flat = z.reshape(B, -1)
        s = self.cross_mlp(flat).reshape(B, self.n_paragraphs, self.mamba_dim)  # s_t

        # Simplified skeleton: score directly from the mixer output.
        #   local head : per-paragraph score from s_t
        #   global head: all-N scores from the aggregate g = mean over paragraphs
        g = s.mean(dim=1)  # (B, mamba_dim)
        local_logits = self.local_head(s).squeeze(-1)
        global_logits = self.global_head(g)

        h_dummy = torch.zeros(B, self.n_paragraphs, self.mamba_dim, device=s.device, dtype=s.dtype)
        g_dummy = torch.zeros(B, self.mamba_dim, device=s.device, dtype=s.dtype)
        return h_dummy, g_dummy, local_logits, global_logits


class TransformerRouter(nn.Module):
    """
    Transformer baseline (2-layer encoder over paragraph sequence, no Mamba, no iter).
    Keeps the same encoder K layers + attn-pool as EvidenceRouter, replaces the Mamba SSM
    with a 2-layer Transformer encoder over the (N_p, dim) paragraph sequence.
    Cross-paragraph via attention instead of state-space transitions.

    Drop-in for EvidenceRouter (returns 4-tuple).
    """
    def __init__(
        self,
        base_model,
        encoder_K=16,
        mamba_dim=256,           # used as transformer d_model (kept name for arg compat)
        n_paragraphs=10,
        attn_pool_dim=256,
        n_layers=2,
        n_heads=4,
        ffn_dim=None,            # default 4 * dim
        **_unused,
    ):
        super().__init__()
        self.encoder_K = encoder_K
        self.mamba_dim = mamba_dim
        self.n_paragraphs = n_paragraphs
        self.attn_pool_dim = attn_pool_dim
        self.iterative = False

        d_model = base_model.config.hidden_size
        self.d_model = d_model
        object.__setattr__(self, "_base_model", base_model)

        # Same attn-pool as EvidenceRouter
        self.attn_pool_W = nn.Linear(d_model, attn_pool_dim, bias=False)
        nn.init.xavier_uniform_(self.attn_pool_W.weight, gain=0.1)
        self.attn_pool_q = nn.Linear(attn_pool_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.attn_pool_q.weight, gain=0.1)

        self.linear_in = nn.Linear(d_model, mamba_dim, bias=False)
        nn.init.xavier_uniform_(self.linear_in.weight, gain=0.1)

        # Cross-paragraph Transformer (replaces Mamba)
        ffn = ffn_dim if ffn_dim is not None else 4 * mamba_dim
        enc_layer = nn.TransformerEncoderLayer(
            d_model=mamba_dim, nhead=n_heads, dim_feedforward=ffn,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.cross_transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Heads (simplified skeleton: score directly from the mixer output)
        self.local_head = nn.Linear(mamba_dim, 1, bias=True)
        nn.init.zeros_(self.local_head.bias)
        nn.init.xavier_uniform_(self.local_head.weight, gain=0.1)

        self.global_head = nn.Linear(mamba_dim, n_paragraphs, bias=True)
        nn.init.zeros_(self.global_head.bias)
        nn.init.xavier_uniform_(self.global_head.weight, gain=0.1)

    @torch.no_grad()
    def _run_k_layers(self, input_ids):
        bm = self._base_model
        device = input_ids.device
        seq_len = input_ids.size(1)
        hidden = bm.model.embed_tokens(input_ids)
        if getattr(bm.config, "model_type", "") in ("gemma", "gemma2"):
            hidden = hidden * (bm.config.hidden_size ** 0.5)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(input_ids)
        position_embeddings = bm.model.rotary_emb(hidden, position_ids)
        for layer in bm.model.layers[: self.encoder_K]:
            out = layer(
                hidden, attention_mask=None, position_ids=position_ids,
                past_key_value=None, use_cache=False, cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden = out[0] if isinstance(out, tuple) else out
        return hidden

    def _attn_pool_per_paragraph(self, hidden, p_chunk_spans):
        B = hidden.size(0)
        all_p_vecs = []
        for b in range(B):
            spans = p_chunk_spans[b]
            p_vecs = []
            for s, e in spans:
                e_clamped = min(e, hidden.size(1) - 1)
                chunk_h = hidden[b, s:e_clamped + 1]
                if chunk_h.size(0) == 0:
                    p_vecs.append(torch.zeros(self.d_model, device=hidden.device, dtype=hidden.dtype))
                    continue
                kv = self.attn_pool_W(chunk_h.to(self.attn_pool_W.weight.dtype))
                attn_logits = self.attn_pool_q(kv).squeeze(-1)
                attn_w = torch.softmax(attn_logits.float(), dim=0).to(chunk_h.dtype)
                pooled = (attn_w.unsqueeze(-1) * chunk_h).sum(dim=0)
                p_vecs.append(pooled)
            while len(p_vecs) < self.n_paragraphs:
                p_vecs.append(torch.zeros(self.d_model, device=hidden.device, dtype=hidden.dtype))
            all_p_vecs.append(torch.stack(p_vecs[: self.n_paragraphs], dim=0))
        return torch.stack(all_p_vecs, dim=0)

    def forward(self, input_ids, p_chunk_spans, q_span, pad_token_id):
        B = input_ids.size(0)
        hidden = self._run_k_layers(input_ids)
        p_vecs = self._attn_pool_per_paragraph(hidden, p_chunk_spans)
        p_vecs = p_vecs.to(self.linear_in.weight.dtype)
        z = self.linear_in(p_vecs)                       # z_t: (B, N_p, dim)
        s = self.cross_transformer(z)                    # s_t: (B, N_p, dim) — bidirectional attention

        # Simplified skeleton: score directly from the mixer output.
        #   local head : per-paragraph score from s_t
        #   global head: all-N scores from the aggregate g = mean over paragraphs
        g = s.mean(dim=1)
        local_logits = self.local_head(s).squeeze(-1)
        global_logits = self.global_head(g)

        h_dummy = torch.zeros(B, self.n_paragraphs, self.mamba_dim, device=s.device, dtype=s.dtype)
        g_dummy = torch.zeros(B, self.mamba_dim, device=s.device, dtype=s.dtype)
        return h_dummy, g_dummy, local_logits, global_logits


class RNNRouter(nn.Module):
    """
    GRU/LSTM baseline (classical recurrence over the paragraph sequence).
    Keeps the same encoder K layers + attn-pool as EvidenceRouter, replaces the
    2-layer Mamba mixer with a 2-layer GRU or LSTM over the (N_p, dim) paragraph
    sequence. Causal, like the Mamba mixer — isolates whether MaRA's gains come
    from any recurrence or from the selective state-space formulation.

    Drop-in for EvidenceRouter (returns 4-tuple). Global readout = last
    recurrent state (mirrors EvidenceRouter's g = s_N).
    """
    def __init__(
        self,
        base_model,
        encoder_K=16,
        mamba_dim=256,           # used as RNN hidden dim (kept name for arg compat)
        n_paragraphs=10,
        attn_pool_dim=256,
        rnn_type="gru",
        n_layers=2,
        **_unused,
    ):
        super().__init__()
        if rnn_type not in ("gru", "lstm"):
            raise ValueError(f"rnn_type must be 'gru' or 'lstm', got {rnn_type}")
        self.encoder_K = encoder_K
        self.mamba_dim = mamba_dim
        self.n_paragraphs = n_paragraphs
        self.attn_pool_dim = attn_pool_dim
        self.rnn_type = rnn_type
        self.iterative = False

        d_model = base_model.config.hidden_size
        self.d_model = d_model
        object.__setattr__(self, "_base_model", base_model)

        # Same attn-pool as EvidenceRouter
        self.attn_pool_W = nn.Linear(d_model, attn_pool_dim, bias=False)
        nn.init.xavier_uniform_(self.attn_pool_W.weight, gain=0.1)
        self.attn_pool_q = nn.Linear(attn_pool_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.attn_pool_q.weight, gain=0.1)

        self.linear_in = nn.Linear(d_model, mamba_dim, bias=False)
        nn.init.xavier_uniform_(self.linear_in.weight, gain=0.1)

        # Cross-paragraph RNN (replaces Mamba); fp32 like the Mamba mixer
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(mamba_dim, mamba_dim, num_layers=n_layers, batch_first=True)

        # Heads (simplified skeleton: score directly from the mixer output)
        self.local_head = nn.Linear(mamba_dim, 1, bias=True)
        nn.init.zeros_(self.local_head.bias)
        nn.init.xavier_uniform_(self.local_head.weight, gain=0.1)

        self.global_head = nn.Linear(mamba_dim, n_paragraphs, bias=True)
        nn.init.zeros_(self.global_head.bias)
        nn.init.xavier_uniform_(self.global_head.weight, gain=0.1)

    def _apply(self, fn, recurse=True):
        # Keep the RNN in fp32 regardless of model-wide casts (mirrors the
        # fp32 Mamba mixer; cuDNN RNNs reject mixed input/weight dtypes).
        result = super()._apply(fn, recurse)
        self.rnn = self.rnn.float()
        return result

    _run_k_layers = TransformerRouter._run_k_layers
    _attn_pool_per_paragraph = TransformerRouter._attn_pool_per_paragraph

    def forward(self, input_ids, p_chunk_spans, q_span, pad_token_id):
        B = input_ids.size(0)
        hidden = self._run_k_layers(input_ids)
        p_vecs = self._attn_pool_per_paragraph(hidden, p_chunk_spans)
        p_vecs = p_vecs.to(self.linear_in.weight.dtype)
        z = self.linear_in(p_vecs)                       # z_t: (B, N_p, dim)
        if z.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                s, _ = self.rnn(z.float())               # s_t: (B, N_p, dim) fp32, causal
        else:
            s, _ = self.rnn(z.float())
        s = s.to(z.dtype)

        # Causal recurrence: global readout from the LAST state (g = s_N),
        # mirroring EvidenceRouter.
        g = s[:, -1]
        local_logits = self.local_head(s).squeeze(-1)
        global_logits = self.global_head(g)

        h_dummy = torch.zeros(B, self.n_paragraphs, self.mamba_dim, device=s.device, dtype=s.dtype)
        g_dummy = torch.zeros(B, self.mamba_dim, device=s.device, dtype=s.dtype)
        return h_dummy, g_dummy, local_logits, global_logits


class TokenMambaRouter(nn.Module):
    """
    Per-token Mamba router (granularity ablation, global-readout variant).

    The SSM is unrolled over RAW TOKENS of the full context (no per-paragraph
    structure given to the router). At the end of the sequence the recurrent
    state is read out, and a single multi-output head produces all N_p
    paragraph relevance scores at once. The router never extracts
    per-paragraph representations explicitly. It must encode "which
    paragraphs are relevant" into one global summary.

    Pipeline:
        1. Frozen K-layer encoder over the full input sequence (Q + all P_i).
        2. linear_in projects every token: d_model -> mamba_dim    -> z_t.
        3. Mamba over the full T-token sequence                   -> s_t.
        4. Take the recurrence output at the LAST non-pad position -> s_final.
        5. Single multi-output head: s_final -> R^{N_p}             -> scores.

    Returns 4-tuple matching EvidenceRouter signature.
    """
    def __init__(
        self,
        base_model,
        encoder_K=16,
        mamba_dim=256,
        n_paragraphs=10,
        mamba_d_state=64,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_n_layers=2,
        attn_pool_dim=256,         # unused, kept for arg-compat with EvidenceRouter
        **_unused,
    ):
        super().__init__()
        from mambapy.mamba import MambaBlock, Mamba, MambaConfig

        self.encoder_K = encoder_K
        self.mamba_dim = mamba_dim
        self.n_paragraphs = n_paragraphs
        self.iterative = False

        d_model = base_model.config.hidden_size
        self.d_model = d_model
        object.__setattr__(self, "_base_model", base_model)

        # Linear: d_model -> mamba_dim. No attn-pool; recurrence sees every token.
        self.linear_in = nn.Linear(d_model, mamba_dim, bias=False)
        nn.init.xavier_uniform_(self.linear_in.weight, gain=0.1)

        # Mamba over the token sequence (same hyperparameters as chunk-Mamba).
        cfg = MambaConfig(
            d_model=mamba_dim, n_layers=mamba_n_layers,
            d_state=mamba_d_state, d_conv=mamba_d_conv,
            expand_factor=mamba_expand,
            dt_min=0.001, dt_max=0.1, pscan=True, use_cuda=False,
        )
        self.mamba = (Mamba(cfg) if mamba_n_layers > 1 else MambaBlock(cfg))

        # Multi-output head: s_final (mamba_dim) -> N_p paragraph scores.
        # No per-paragraph extraction; one summary -> all scores.
        self.score_head = nn.Sequential(
            nn.Linear(mamba_dim, mamba_dim),
            nn.GELU(),
            nn.Linear(mamba_dim, n_paragraphs),
        )
        for m in self.score_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def to(self, *args, **kwargs):
        # Keep Mamba (and h_mlp downstream of it) in fp32 for numerical stability.
        result = super().to(*args, **kwargs)
        self.mamba = self.mamba.float()
        return result

    @torch.no_grad()
    def _run_k_layers(self, input_ids):
        bm = self._base_model
        device = input_ids.device
        seq_len = input_ids.size(1)
        hidden = bm.model.embed_tokens(input_ids)
        if getattr(bm.config, "model_type", "") in ("gemma", "gemma2"):
            hidden = hidden * (bm.config.hidden_size ** 0.5)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(input_ids)
        position_embeddings = bm.model.rotary_emb(hidden, position_ids)
        for layer in bm.model.layers[: self.encoder_K]:
            out = layer(
                hidden, attention_mask=None, position_ids=position_ids,
                past_key_value=None, use_cache=False, cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden = out[0] if isinstance(out, tuple) else out
        return hidden

    def forward(self, input_ids, p_chunk_spans, q_span, pad_token_id):
        B, T = input_ids.shape

        # Encoder + project to mamba_dim per token.
        hidden = self._run_k_layers(input_ids)                              # (B, T, d_model)
        z_tok = self.linear_in(hidden.to(self.linear_in.weight.dtype))      # (B, T, mamba_dim)

        # Mamba over the token sequence (causal recurrence on raw tokens).
        s_tok = self.mamba(z_tok.float()).to(z_tok.dtype)                   # (B, T, mamba_dim)

        # Locate the last non-pad token per example (== Mamba's full-context summary).
        # If pad_token_id is None (no padding), fall back to the literal last position.
        if pad_token_id is None:
            last_idx = torch.full((B,), T - 1, device=input_ids.device, dtype=torch.long)
        else:
            non_pad = (input_ids != pad_token_id)                           # (B, T) bool
            # If a row is all pad (shouldn't happen), this returns 0; clamp to avoid -1.
            last_idx = non_pad.sum(dim=1).clamp(min=1) - 1                  # (B,)

        # Gather s_tok at the last non-pad position per row.
        idx_exp = last_idx.view(B, 1, 1).expand(-1, 1, s_tok.size(-1))
        s_final = s_tok.gather(1, idx_exp).squeeze(1)                       # (B, mamba_dim)

        # Single multi-output head -> all N_p paragraph scores at once.
        s_final_in = s_final.to(self.score_head[0].weight.dtype)
        local_logits = self.score_head(s_final_in)                          # (B, N_p)

        # Dummies for signature compatibility (no per-paragraph latent state).
        h_dummy = torch.zeros(B, self.n_paragraphs, self.mamba_dim,
                              device=s_final.device, dtype=s_final.dtype)
        g_dummy = torch.zeros(B, self.mamba_dim, device=s_final.device, dtype=s_final.dtype)
        global_logits = torch.zeros(B, self.n_paragraphs,
                                    device=s_final.device, dtype=local_logits.dtype)
        return h_dummy, g_dummy, local_logits, global_logits


class PoolOnlyRouter(nn.Module):
    """
    Attention-pool-only baseline (no cross-paragraph mixer).

    Isolates the contribution of the pooling method from the mixer. Same
    encoder + attention-pool as EvidenceRouter, but the mixer is the
    identity: each paragraph is scored from its own pooled vector with no
    cross-paragraph interaction.

    Ladder position (2-D decomposition of the architecture ablation):
        LastTok        : last-token pool   + no mixer
        PoolOnly (this): attention pool    + no mixer   <- isolates pooling
        MLP            : attention pool    + cross-paragraph MLP
        Transformer    : attention pool    + self-attention
        MeanPool       : mean pool         + Mamba
        EvidenceRouter : attention pool    + Mamba       (ours)

    Returns 4-tuple matching EvidenceRouter signature.
    """
    def __init__(
        self,
        base_model,
        encoder_K=16,
        mamba_dim=256,           # reused as the per-paragraph hidden dim
        n_paragraphs=10,
        attn_pool_dim=256,
        **_unused,
    ):
        super().__init__()
        self.encoder_K = encoder_K
        self.mamba_dim = mamba_dim
        self.n_paragraphs = n_paragraphs
        self.attn_pool_dim = attn_pool_dim
        self.iterative = False

        d_model = base_model.config.hidden_size
        self.d_model = d_model
        object.__setattr__(self, "_base_model", base_model)

        # Same attn-pool as EvidenceRouter / MLPRouter.
        self.attn_pool_W = nn.Linear(d_model, attn_pool_dim, bias=False)
        nn.init.xavier_uniform_(self.attn_pool_W.weight, gain=0.1)
        self.attn_pool_q = nn.Linear(attn_pool_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.attn_pool_q.weight, gain=0.1)

        self.linear_in = nn.Linear(d_model, mamba_dim, bias=False)
        nn.init.xavier_uniform_(self.linear_in.weight, gain=0.1)

        # No mixer. Heads only (simplified skeleton).
        self.local_head = nn.Linear(mamba_dim, 1, bias=True)
        nn.init.zeros_(self.local_head.bias)
        nn.init.xavier_uniform_(self.local_head.weight, gain=0.1)
        self.global_head = nn.Linear(mamba_dim, n_paragraphs, bias=True)
        nn.init.zeros_(self.global_head.bias)
        nn.init.xavier_uniform_(self.global_head.weight, gain=0.1)

    @torch.no_grad()
    def _run_k_layers(self, input_ids):
        bm = self._base_model
        device = input_ids.device
        seq_len = input_ids.size(1)
        hidden = bm.model.embed_tokens(input_ids)
        if getattr(bm.config, "model_type", "") in ("gemma", "gemma2"):
            hidden = hidden * (bm.config.hidden_size ** 0.5)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(input_ids)
        position_embeddings = bm.model.rotary_emb(hidden, position_ids)
        for layer in bm.model.layers[: self.encoder_K]:
            out = layer(
                hidden, attention_mask=None, position_ids=position_ids,
                past_key_value=None, use_cache=False, cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden = out[0] if isinstance(out, tuple) else out
        return hidden  # (B, T, d_model)

    def _attn_pool_per_paragraph(self, hidden, p_chunk_spans):
        """Attn-pool over each paragraph's tokens, mirroring EvidenceRouter."""
        B = hidden.size(0)
        all_p_vecs = []
        for b in range(B):
            spans = p_chunk_spans[b]
            p_vecs = []
            for s, e in spans:
                e_clamped = min(e, hidden.size(1) - 1)
                chunk_h = hidden[b, s:e_clamped + 1]
                if chunk_h.size(0) == 0:
                    p_vecs.append(torch.zeros(self.d_model, device=hidden.device, dtype=hidden.dtype))
                    continue
                kv = self.attn_pool_W(chunk_h.to(self.attn_pool_W.weight.dtype))
                attn_logits = self.attn_pool_q(kv).squeeze(-1)
                attn_w = torch.softmax(attn_logits.float(), dim=0).to(chunk_h.dtype)
                pooled = (attn_w.unsqueeze(-1) * chunk_h).sum(dim=0)
                p_vecs.append(pooled)
            while len(p_vecs) < self.n_paragraphs:
                p_vecs.append(torch.zeros(self.d_model, device=hidden.device, dtype=hidden.dtype))
            all_p_vecs.append(torch.stack(p_vecs[: self.n_paragraphs], dim=0))
        return torch.stack(all_p_vecs, dim=0)  # (B, N_p, d_model)

    def forward(self, input_ids, p_chunk_spans, q_span, pad_token_id):
        B = input_ids.size(0)
        hidden = self._run_k_layers(input_ids)
        p_vecs = self._attn_pool_per_paragraph(hidden, p_chunk_spans)        # (B, N_p, d_model)
        p_vecs = p_vecs.to(self.linear_in.weight.dtype)
        z = self.linear_in(p_vecs)                                          # (B, N_p, mamba_dim)

        # No mixer: s == z. Score directly.
        s = z
        g = s.mean(dim=1)                                                   # (B, mamba_dim)
        local_logits = self.local_head(s).squeeze(-1)
        global_logits = self.global_head(g)

        h_dummy = torch.zeros(B, self.n_paragraphs, self.mamba_dim, device=s.device, dtype=s.dtype)
        g_dummy = torch.zeros(B, self.mamba_dim, device=s.device, dtype=s.dtype)
        return h_dummy, g_dummy, local_logits, global_logits
