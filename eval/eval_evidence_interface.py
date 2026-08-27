"""
eval_evidence_interface.py — inference-only test of router-score signal applied
at PROMPT-LEVEL (reorder / annotate / top-k filter), with no model retraining.

Pipeline per sample:
  1. Build the standard `format_hotpotqa_prompt` text (training-time format) so the
     frozen K=16 EvidenceRouter can score paragraphs.
  2. Tokenize, detect chunks, run router → 10 paragraph scores (sigmoid of local
     logits) + gold support flags.
  3. Reorder / annotate / filter the 10 paragraphs per --mode.
  4. Build a NEW prompt (Question:\n... Evidence:\n... Instruction:\n...).
  5. Generate with beam=4, max_new_tokens=32, do_sample=False on the loaded
     MaLoRA checkpoint.
  6. Compute F1, EM, support_recall@k, log per-example diagnostics.

The MaLoRA checkpoint is unchanged across modes; only the prompt
text changes.

Outputs (under <out-dir>/<mode>/):
  predictions.jsonl
  per_example_scores.jsonl
  metrics.json
  formatted_prompt_samples.txt   — first 5 prompts
  run_config.yaml
"""

# --- repo root on sys.path (entry points live one level below it) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "chunking", _ROOT / "mara", _ROOT / "train", _ROOT / "eval"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------


import argparse, json, os, re, sys, yaml, time, warnings, logging
from pathlib import Path
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
import torch
from datasets import load_dataset

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "chunking"))
sys.path.insert(0, str(ROOT / "mara"))

from data.musique import format_musique_prompt
from data.twowikimultihopqa import format_twowikimultihopqa_prompt, _load_2wiki
from mara import EvidenceRouter, derive_p_chunk_spans
from model.malora import load_model_with_malora


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="diag-gated LoRA checkpoint (.pt). Required unless --no-lora or --lora-ckpt.")
    p.add_argument("--lora-ckpt", default=None,
                   help="Pure PEFT LoRA dir (e.g., outputs/.../lora_epoch_2). Loaded with PeftModel + merge_and_unload.")
    p.add_argument("--router-ckpt", default=None,
                   help="Required for routed modes (topk_score etc.). Skipped for all_original/oracle_topk/random_topk.")
    p.add_argument("--dataset", choices=["hotpotqa", "musique", "twowikimultihopqa"], default="hotpotqa")
    p.add_argument("--no-lora", action="store_true",
                   help="Skip LoRA loading; eval the bare base model. Useful for base/instruct comparisons.")
    p.add_argument("--chat-template", action="store_true",
                   help="Wrap the prompt with the tokenizer's chat template (for Instruct models).")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--mode", required=True,
                   choices=["all_original", "all_score_sorted", "all_score_annotated",
                            "topk_score", "shuffled_score_sorted", "shuffled_score_annotated",
                            "shuffled_score_topk", "random_topk",
                            "oracle_topk", "gold_only", "topk_bm25"])
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    # Router architecture (defaults match the K=16 router used in SCORE_ONLY)
    p.add_argument("--router-kind", default="global_mlp")
    p.add_argument("--mamba-dim", type=int, default=256)
    p.add_argument("--mamba-n-layers", type=int, default=2)
    p.add_argument("--mamba-d-state", type=int, default=64)
    p.add_argument("--mamba-d-conv", type=int, default=4)
    p.add_argument("--mamba-expand", type=int, default=2)
    p.add_argument("--no-cache", action="store_true",
                   help="Disable the KV cache (use_cache=False). Forces a full re-prefill at every "
                        "decode step, which restores the correct Mamba recurrent state across "
                        "generated tokens for SSM-based adapters. Slow; diagnostic use only.")
    p.add_argument("--encoder-K", type=int, default=16)
    p.add_argument("--attn-pool-dim", type=int, default=256)
    p.add_argument("--iterative", action="store_true",
                   help="Use iterative (2-pass) routing — match how router was trained.")
    p.add_argument("--joint-encoding", action="store_true",
                   help="Joint LM forward over Q+all paragraphs. Must match how router was trained.")
    # --- Cascade scoring (multi-pass elimination, inference-time only) ---
    p.add_argument("--cascade-passes", type=int, default=0,
                   help="If >0, use N-pass cascade scoring: each pass drops drop-per-pass worst survivors.")
    p.add_argument("--cascade-drop-per-pass", type=int, default=4,
                   help="Number of survivors to drop per cascade pass.")
    p.add_argument("--cascade-anchor", action="store_true",
                   help="Cumulative-anchor cascade: each pass conditions surviving P_t encoding on prior pass top-1 anchors.")
    # --- Chain (no-drop N-pass) inference: matches forward_chain_train (chain-router training) ---
    p.add_argument("--chain-passes", type=int, default=0,
                   help="If >0, use no-drop N-pass chain routing (cumulative anchors), matching chain-router training.")
    # --- Beam-iter inference (top-K hypotheses extension of 2-pass iter) ---
    p.add_argument("--beam-k", type=int, default=0,
                   help="If >0, use beam-iter inference with K anchor hypotheses.")
    p.add_argument("--beam-aggregate", default='mean', choices=['mean', 'max', 'logsumexp'],
                   help="How to combine K pass-2 logits into final scores.")
    # --- V9 score cache (pre-computed per-paragraph scores from external-encoder router) ---
    p.add_argument("--v9-scores-cache", default=None,
                   help="Path to pre-computed V9 scores .pt {scores: (N,20), ids: [N]}. "
                        "When set, skips router instantiation and uses cached scores directly.")
    return p.parse_args()


# --- F1 / EM scoring (matches eval_hotpotqa.py exactly) ---
def normalize_answer(s):
    import re, string
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if len(pred_tokens) == 0 and len(gold_tokens) == 0: return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0: return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common: return 0.0
    n_same = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
    if n_same == 0: return 0.0
    p = n_same / len(pred_tokens)
    r = n_same / len(gold_tokens)
    return 2 * p * r / (p + r)


# --- Score categorization ---
def score_label(s):
    if s >= 0.75: return "high"
    if s >= 0.40: return "medium"
    return "low"


# --- Prompt builders ---
INSTR_DEFAULT = "Answer the question using the evidence above. Be concise."
INSTR_ANNOT = ("Answer the question using the evidence above. "
               "Give more attention to high-relevance evidence, "
               "but check the full evidence for contradictions. Be concise.")
INSTR_TOPK = "Answer the question using only the selected evidence. Be concise."


def fmt_chunks(paragraphs, annotate=False, scores=None):
    """paragraphs: list of (title, text). If annotate, prepend [Evidence relevance: X]."""
    parts = []
    for i, (title, text) in enumerate(paragraphs):
        if annotate:
            lab = score_label(scores[i])
            parts.append(f"[Evidence relevance: {lab}]\n{title}: {text}")
        else:
            parts.append(f"{title}: {text}")
    return "\n\n".join(parts)


def build_prompt(question, paragraphs, mode, scores_aligned=None):
    """Returns the inference prompt string (model will generate after the trailing 'Answer:').

    Uses the TRAINING-format wrapper exactly:
        Context:
        {paragraphs}

        Question: {q}

        Answer:

    The mode only varies the BODY of the Context block:
      all_original: paragraphs in original order, no annotations.
      all_score_sorted: paragraphs sorted by router score desc.
      all_score_annotated: paragraphs in original order with [Evidence relevance: X] tags.
      topk_score: only the top-k paragraphs by router score, sorted desc.
      oracle_topk: top-k by gold support (padded by router score), sorted gold-first.
      gold_only: gold supporting paragraphs only, original order, no padding to k.
      shuffled_score_*: same as score-sorted/annotated but with permuted scores
                       (control to break score-paragraph alignment).
    """
    if mode in ("all_score_annotated", "shuffled_score_annotated"):
        body = fmt_chunks(paragraphs, annotate=True, scores=scores_aligned)
    else:
        body = fmt_chunks(paragraphs, annotate=False)
    return f"Context:\n{body}\n\nQuestion: {question}"


def wrap_for_generation(user_text, tok, use_chat_template):
    """Wrap user_text into the final prompt that the model generates after."""
    if use_chat_template:
        msgs = [{"role": "user", "content": user_text + "\n\nAnswer the question concisely."}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return user_text + "\n\nAnswer:"


def truncate_pred(s):
    s = s.strip()
    for sep in ['\n', '. ', ', who ', ', which ', ', the ']:
        if sep in s:
            s = s.split(sep)[0]; break
    return s.strip()


@torch.no_grad()
def cascade_score(router, input_ids, p_spans_all, q_spans, pad_id,
                  n_passes, drop_per_pass, use_anchor, n_paragraphs_orig):
    """Multi-pass cascade scoring (batch_size=1 only).

    Each pass:
      1. Build a new [Q + (anchors if anchor mode) + survivors_in_original_order] input.
      2. Run router with n_paragraphs temporarily overridden to current K = #survivors.
      3. Get local_logits (K-dim) → sort descending.
      4. (anchor mode) Identify top scorer not-yet-anchor → new anchor.
      5. Drop worst drop_per_pass survivors (anchors are protected).

    Returns: (ranked_orig_indices, final_scores_aligned_to_ranked_orig)
      - ranked_orig_indices: list of paragraph indices in ORIGINAL numbering, ranked best-first
      - final_scores_aligned: float list, score per ranked paragraph (from final pass; -inf for anchors picked earlier)
    """
    p_spans = p_spans_all[0]   # (B=1) → list of (start,end) per orig paragraph
    qs, qe = q_spans[0]
    q_ids = input_ids[0, qs:qe + 1]

    survivors = list(range(n_paragraphs_orig))   # ORIGINAL indices, ascending
    anchors = []                                  # ORIGINAL indices, in selection order
    anchor_scores = {}                            # orig_idx -> score at pick time
    last_local_logits = None                      # tensor (K_final,) from last pass

    # Disable router's internal iterative path during cascade — each pass is a single forward.
    saved_iterative = router.iterative
    router.iterative = False
    saved_n_paragraphs = router.n_paragraphs

    try:
        for pass_idx in range(n_passes):
            K = len(survivors)

            # Build new input_ids: [Q + (anchor_paragraph_tokens) + (survivor_paragraph_tokens in survivor order)]
            parts = [q_ids]
            cursor = q_ids.size(0)
            new_q_span = (0, q_ids.size(0) - 1)

            if use_anchor and anchors:
                for a_idx in anchors:
                    aps, ape = p_spans[a_idx]
                    parts.append(input_ids[0, aps:ape + 1])
                    cursor += parts[-1].size(0)

            new_p_spans = []
            for s_idx in survivors:
                ps, pe = p_spans[s_idx]
                p_ids = input_ids[0, ps:pe + 1]
                new_p_spans.append((cursor, cursor + p_ids.size(0) - 1))
                parts.append(p_ids)
                cursor += p_ids.size(0)

            new_input_ids = torch.cat(parts, dim=0).unsqueeze(0)   # (1, T_new)

            router.n_paragraphs = K
            try:
                out = router(new_input_ids, [new_p_spans], [new_q_span], pad_id)
            except RuntimeError:
                # If router fails (e.g. global_head dim mismatch), bail out of cascade
                router.n_paragraphs = saved_n_paragraphs
                return None, None
            local_logits = out[2][0]   # (K,)
            last_local_logits = local_logits

            sorted_K_idx = local_logits.argsort(descending=True).cpu().tolist()

            # Anchor pick: top scorer that's not already an anchor
            if use_anchor:
                for K_pos in sorted_K_idx:
                    orig_idx = survivors[K_pos]
                    if orig_idx not in anchors:
                        anchors.append(orig_idx)
                        anchor_scores[orig_idx] = float(local_logits[K_pos].item())
                        break

            # Drop worst drop_per_pass (but never drop an anchor)
            if pass_idx < n_passes - 1:
                keep_count = K - drop_per_pass
                # Always keep anchors. Among non-anchors, keep by score (top first).
                kept_set = set(anchors) if use_anchor else set()
                for K_pos in sorted_K_idx:
                    if len(kept_set) >= keep_count:
                        break
                    orig_idx = survivors[K_pos]
                    if orig_idx not in kept_set:
                        kept_set.add(orig_idx)
                # Maintain ascending original order for next pass's span building
                survivors = sorted(kept_set)
    finally:
        router.n_paragraphs = saved_n_paragraphs
        router.iterative = saved_iterative

    # Build final ranked list
    if use_anchor:
        # anchors[0..n_passes-1] are ranked in pick order
        # Any remaining survivors (not in anchors) get appended sorted by last-pass scores
        rest = [s for s in survivors if s not in anchors]
        # Rank rest by their last-pass scores (descending)
        rest_scores = []
        sorted_K_idx_final = last_local_logits.argsort(descending=True).cpu().tolist()
        for K_pos in sorted_K_idx_final:
            orig_idx = survivors[K_pos]
            if orig_idx in rest:
                rest_scores.append(orig_idx)
        ranked_orig = list(anchors) + rest_scores
        score_aligned = []
        for orig in ranked_orig:
            if orig in anchor_scores:
                score_aligned.append(anchor_scores[orig])
            else:
                K_pos = survivors.index(orig)
                score_aligned.append(float(last_local_logits[K_pos].item()))
    else:
        # No-anchor mode: rank final survivors purely by last-pass scores
        sorted_K_idx_final = last_local_logits.argsort(descending=True).cpu().tolist()
        ranked_orig = [survivors[i] for i in sorted_K_idx_final]
        score_aligned = [float(last_local_logits[i].item()) for i in sorted_K_idx_final]

    return ranked_orig, score_aligned


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[evid] mode={args.mode} → {out_dir}")

    # --- Load model (+ LoRA ckpt unless --no-lora) ---
    if args.no_lora:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"[evid] loading bare base model {args.model} (no LoRA)")
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        )
    elif args.lora_ckpt is not None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        print(f"[evid] loading {args.model} + PEFT LoRA from {args.lora_ckpt}")
        tok = AutoTokenizer.from_pretrained(args.model)
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        )
        model = PeftModel.from_pretrained(base, args.lora_ckpt)
        model = model.merge_and_unload()
    else:
        if args.ckpt is None:
            raise SystemExit("--ckpt or --lora-ckpt required unless --no-lora")
        print(f"[evid] loading {args.model} + diag-gated LoRA from {args.ckpt}")
        model, tok = load_model_with_malora(args.model, args.ckpt, device="cuda")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model.eval()

    # --- Load V9 score cache (if specified, replaces router) ---
    v9_cache = None
    v9_id_to_idx = None
    if args.v9_scores_cache is not None:
        print(f"[evid] loading V9 score cache from {args.v9_scores_cache}")
        v9_cache = torch.load(args.v9_scores_cache, map_location='cpu', weights_only=False)
        v9_id_to_idx = {str(qid): i for i, qid in enumerate(v9_cache['ids'])}
        print(f"[evid] V9 cache: {v9_cache['scores'].shape[0]} samples, {v9_cache['scores'].shape[1]} paragraphs each")

    # --- Load router (frozen) ---
    # Skip router entirely for modes that do not consume router scores.
    _router_free_modes = {"all_original", "oracle_topk", "gold_only", "random_topk", "topk_bm25"}
    if args.mode in _router_free_modes:
        print(f"[evid] mode={args.mode} does not use the router — skipping router load")
        router = None
    elif v9_cache is not None:
        print(f"[evid] using V9 score cache — skipping router load")
        router = None
    else:
        print(f"[evid] loading router from {args.router_ckpt}")
        base_for_router = model  # router uses base model only for embedding (frozen)
        _ckpt = torch.load(args.router_ckpt, map_location="cuda", weights_only=False)
        _sd = _ckpt["router_state_dict"]
        # Auto-detect scoring-head architecture from the checkpoint keys:
        # MLP head -> local_head.0.weight (Sequential); linear head -> local_head.weight.
        _scoring_head = "mlp" if "local_head.0.weight" in _sd else "linear"
        router = EvidenceRouter(
            base_model=base_for_router,
            encoder_K=args.encoder_K, mamba_dim=args.mamba_dim,
            mamba_n_layers=args.mamba_n_layers, mamba_d_state=args.mamba_d_state,
            mamba_d_conv=args.mamba_d_conv, mamba_expand=args.mamba_expand,
            attn_pool_dim=args.attn_pool_dim,
            n_paragraphs=20 if args.dataset == "musique" else 10, router_kind=args.router_kind,  # 2Wiki + HP both have 10 paragraphs
            joint_encoding=args.joint_encoding,
            iterative=args.iterative,
            scoring_head=_scoring_head,
        ).to("cuda")
        # strict=False: older router checkpoints carry unused h_mlp.* keys
        # (the h_mlp combiner was dropped from the simplified router; it never
        # fed any scoring head, so ignoring those keys is exact, not lossy).
        _missing, _unexpected = router.load_state_dict(_sd, strict=False)
        _unexpected = [k for k in _unexpected if not k.startswith("h_mlp.")]
        if _missing or _unexpected:
            raise RuntimeError(
                f"router load mismatch beyond h_mlp: missing={_missing} unexpected={_unexpected}")
        router.eval()
        for p in router.parameters():
            p.requires_grad = False

    # --- Dataset ---
    if args.dataset == "musique":
        N_PARAGRAPHS = 20
        ds_full = load_dataset("dgslibisey/MuSiQue", split="validation")
        n = min(args.n_samples, len(ds_full))
        print(f"[evid] loading MuSiQue val (first {n} of {len(ds_full)})")
        ds = ds_full.select(range(n))
    elif args.dataset == "twowikimultihopqa":
        N_PARAGRAPHS = 10
        ds_full = _load_2wiki("dev")
        n = min(args.n_samples, len(ds_full))
        print(f"[evid] loading 2WikiMultihopQA dev (first {n} of {len(ds_full)})")
        ds = ds_full[:n]  # plain Python list, not HF Dataset
    else:
        N_PARAGRAPHS = 10
        ds_full = load_dataset("hotpot_qa", "distractor", split="validation")
        n = min(args.n_samples, len(ds_full))
        print(f"[evid] loading HotpotQA distractor val (first {n} of {len(ds_full)})")
        ds = ds_full.select(range(n))

    # --- Output files ---
    pred_f = open(out_dir / "predictions.jsonl", "w")
    score_f = open(out_dir / "per_example_scores.jsonl", "w")
    sample_f = open(out_dir / "formatted_prompt_samples.txt", "w")
    n_prompt_samples_saved = 0

    f1s, ems, supp_recalls, in_tokens, n_chunks = [], [], [], [], []
    score_dists_selected = []  # mean score of selected chunks

    rng = torch.Generator(device="cpu").manual_seed(args.seed)

    t0 = time.time()
    for i, ex in enumerate(ds):
        q = ex["question"]
        gold = ex["answer"]
        if args.dataset == "musique":
            if not ex.get("answerable", True): continue
            mq_paras = ex["paragraphs"]
            ctx_titles = [p["title"] for p in mq_paras]
            ctx_text   = [p["paragraph_text"] for p in mq_paras]
            is_supporting = [bool(p["is_supporting"]) for p in mq_paras]
            paragraphs = list(zip(ctx_titles, ctx_text))
            if len(paragraphs) != N_PARAGRAPHS:
                continue
            std_prompt = format_musique_prompt(q, mq_paras, answer=None)
            ctx_for_router = list(zip(ctx_titles, [[t] for t in ctx_text]))  # router uses (title, [sentences]) for chunk detection
        elif args.dataset == "twowikimultihopqa":
            ctx = ex["context"]
            if isinstance(ctx, dict) and "title" in ctx:
                ctx_titles = ctx["title"]
                ctx_sents = ctx["sentences"]
            else:
                ctx_titles = [c[0] for c in ctx]
                ctx_sents = [c[1] for c in ctx]
            sf = ex.get("supporting_facts", [])
            if sf and isinstance(sf[0], dict):
                supp_titles = {s["title"] for s in sf}
            else:
                supp_titles = {s[0] for s in sf}
            is_supporting = [t in supp_titles for t in ctx_titles]
            paragraphs = [(t, " ".join(s) if isinstance(s, list) else s) for t, s in zip(ctx_titles, ctx_sents)]
            if len(paragraphs) != N_PARAGRAPHS:
                continue
            std_prompt = format_twowikimultihopqa_prompt(q, list(zip(ctx_titles, ctx_sents)), answer=None)
        else:
            ctx_titles = ex["context"]["title"]
            ctx_sents = ex["context"]["sentences"]
            supp_titles = set(ex["supporting_facts"]["title"])
            is_supporting = [t in supp_titles for t in ctx_titles]
            paragraphs = [(t, " ".join(s)) for t, s in zip(ctx_titles, ctx_sents)]
            if len(paragraphs) != N_PARAGRAPHS:
                continue
            from data.hotpotqa import format_hotpotqa_prompt
            std_prompt = format_hotpotqa_prompt(q, list(zip(ctx_titles, ctx_sents)), answer=None)
        gold_supp_ids = [k for k, v in enumerate(is_supporting) if v]

        # --- Step 1+2: router scoring on standard prompt format ---
        enc = tok(std_prompt, return_tensors="pt", return_offsets_mapping=True,
                  truncation=True, max_length=args.max_length)
        input_ids = enc["input_ids"].to("cuda")
        offsets = enc["offset_mapping"][0].tolist()
        try:
            from chunk_detector import detect_hotpotqa_chunks
            chunks = detect_hotpotqa_chunks(std_prompt, offsets)
        except AssertionError:
            chunks = []
        chunk_ends = [c["end_tok"] for c in chunks]
        chunk_names = [c["name"] for c in chunks]
        if sum(1 for n in chunk_names if n.startswith("P")) != N_PARAGRAPHS:
            continue
        p_spans, q_spans = derive_p_chunk_spans([chunk_ends], [chunk_names])
        # Skip samples where any paragraph chunk has zero tokens after truncation
        # (router's attn-pool can't softmax over an empty sequence).
        if any(pe < ps for ps, pe in p_spans[0]):
            continue
        # --- V9 cached scores (skip router, use precomputed scores by sample ID) ---
        v9_order = None
        v9_scores_aligned = None
        if v9_cache is not None and args.mode == "topk_score":
            sample_id = str(ex.get('id', i))
            if sample_id not in v9_id_to_idx:
                continue   # sample not in V9 precompute (skipped during precompute)
            cached_logits = v9_cache['scores'][v9_id_to_idx[sample_id]]   # (20,)
            scores = torch.sigmoid(cached_logits).float().tolist()
            sorted_idx = sorted(range(N_PARAGRAPHS), key=lambda k: -scores[k])
            v9_order = sorted_idx[:args.top_k]
            v9_scores_aligned = [scores[k] for k in v9_order]

        # --- Beam-iter scoring path (top-K anchor hypotheses) ---
        beam_order = None
        beam_scores = None
        if v9_order is None and router is not None and args.beam_k > 0 and args.mode == "topk_score":
            try:
                with torch.no_grad():
                    out = router.forward_beam(
                        input_ids, p_spans, q_spans, tok.pad_token_id,
                        beam_k=args.beam_k, aggregate=args.beam_aggregate,
                    )
                local_logits_beam = out[2]   # (B, N)
            except RuntimeError:
                continue
            scores = torch.sigmoid(local_logits_beam[0]).float().cpu().tolist()
            sorted_idx = sorted(range(N_PARAGRAPHS), key=lambda k: -scores[k])
            beam_order = sorted_idx[:args.top_k]
            beam_scores = [scores[k] for k in beam_order]

        # --- Chain (no-drop N-pass) scoring path: cumulative anchors, matches chain-router training ---
        chain_order = None
        chain_scores = None
        if v9_order is None and beam_order is None and router is not None and args.chain_passes > 0 and args.mode == "topk_score":
            try:
                with torch.no_grad():
                    out_chain = router.forward_chain_train(
                        input_ids, p_spans, q_spans, tok.pad_token_id, n_passes=args.chain_passes,
                    )
            except RuntimeError:
                continue
            anchors = list(out_chain["anchors"][0])              # pick order, length n_passes
            final_logits = out_chain["logits_per_step"][-1][0]   # (N,)
            sorted_idx = final_logits.argsort(descending=True).cpu().tolist()
            rest = [k for k in sorted_idx if k not in anchors]
            ranked = anchors + rest
            scores = torch.sigmoid(final_logits).float().cpu().tolist()  # aligned to N
            chain_order = ranked[:args.top_k]
            chain_scores = [scores[k] for k in chain_order]

        # --- Cascade scoring path (multi-pass elimination, replaces single forward + sort) ---
        cascade_order = None
        cascade_scores = None
        if chain_order is None and beam_order is None and router is not None and args.cascade_passes > 0 and args.mode == "topk_score":
            ranked_orig, ranked_scores = cascade_score(
                router, input_ids, p_spans, q_spans, tok.pad_token_id,
                n_passes=args.cascade_passes,
                drop_per_pass=args.cascade_drop_per_pass,
                use_anchor=args.cascade_anchor,
                n_paragraphs_orig=N_PARAGRAPHS,
            )
            if ranked_orig is None:
                continue   # router failed on this sample; skip
            cascade_order = ranked_orig[:args.top_k]
            # build a "scores" vector aligned to original paragraph indices (for downstream annotation)
            scores = [0.0] * N_PARAGRAPHS
            for orig, sc in zip(ranked_orig, ranked_scores):
                scores[orig] = float(torch.sigmoid(torch.tensor(sc)).item())
            cascade_scores = [scores[k] for k in cascade_order]
        elif router is None:
            # Router-free modes (all_original / oracle_topk / random_topk) don't consume scores.
            # Provide a constant score vector so downstream branches still work.
            scores = [0.0] * N_PARAGRAPHS
        elif chain_order is None:
            try:
                with torch.no_grad():
                    out = router(input_ids, p_spans, q_spans, tok.pad_token_id)
                # iterative router returns 5-tuple (h, g, local_logits, global_logits, aux_dict);
                # non-iterative returns 4-tuple (h, g, local_logits, global_logits).
                local_logits = out[2]
            except RuntimeError:
                continue
            scores = torch.sigmoid(local_logits[0]).float().cpu().tolist()  # list of N_PARAGRAPHS

        # --- Step 3: reorder / annotate / filter per mode ---
        order = list(range(N_PARAGRAPHS))  # original order
        scores_aligned = list(scores)  # scores aligned to current order
        # Chain short-circuit: use the chain's ranked top-k directly
        if chain_order is not None:
            order = chain_order
            scores_aligned = chain_scores
        # Cascade short-circuit: use the cascade's ranked top-k directly
        if cascade_order is not None:
            order = cascade_order
            scores_aligned = cascade_scores
        # Beam-iter short-circuit
        if beam_order is not None:
            order = beam_order
            scores_aligned = beam_scores
        # V9 cached scores short-circuit
        if v9_order is not None:
            order = v9_order
            scores_aligned = v9_scores_aligned
        if args.mode == "all_original":
            pass
        elif args.mode == "all_score_sorted":
            order = sorted(order, key=lambda k: -scores[k])
            scores_aligned = [scores[k] for k in order]
        elif args.mode == "all_score_annotated":
            # original order, annotated with relevance labels
            pass
        elif args.mode == "shuffled_score_sorted":
            perm = torch.randperm(N_PARAGRAPHS, generator=rng).tolist()
            shuffled_scores = [scores[perm[k]] for k in range(N_PARAGRAPHS)]
            order = sorted(order, key=lambda k: -shuffled_scores[k])
            scores_aligned = [shuffled_scores[k] for k in order]
        elif args.mode == "shuffled_score_annotated":
            perm = torch.randperm(N_PARAGRAPHS, generator=rng).tolist()
            shuffled_scores = [scores[perm[k]] for k in range(N_PARAGRAPHS)]
            scores_aligned = shuffled_scores
        elif args.mode == "topk_score":
            order = sorted(order, key=lambda k: -scores[k])[:args.top_k]
            scores_aligned = [scores[k] for k in order]
        elif args.mode == "shuffled_score_topk":
            perm = torch.randperm(N_PARAGRAPHS, generator=rng).tolist()
            shuffled = [scores[perm[k]] for k in range(N_PARAGRAPHS)]
            order = sorted(order, key=lambda k: -shuffled[k])[:args.top_k]
            scores_aligned = [shuffled[k] for k in order]
        elif args.mode == "random_topk":
            perm = torch.randperm(N_PARAGRAPHS, generator=rng).tolist()
            order = sorted(perm[:args.top_k])
            scores_aligned = [scores[k] for k in order]
        elif args.mode == "topk_bm25":
            from rank_bm25 import BM25Okapi
            corpus = [(t + " " + x).lower().split() for (t, x) in paragraphs]
            bm25 = BM25Okapi(corpus)
            bm_scores = bm25.get_scores(q.lower().split())
            order = sorted(range(N_PARAGRAPHS), key=lambda k: -bm_scores[k])[:args.top_k]
            scores_aligned = [float(bm_scores[k]) for k in order]
        elif args.mode == "gold_only":
            # Gold supporting paragraphs ONLY, in original order, with no padding to k.
            order = [k for k in range(N_PARAGRAPHS) if is_supporting[k]]
            scores_aligned = [scores[k] for k in order]
        elif args.mode == "oracle_topk":
            gold_first = [k for k in range(N_PARAGRAPHS) if is_supporting[k]]
            non_gold_by_score = sorted([k for k in range(N_PARAGRAPHS) if not is_supporting[k]],
                                       key=lambda k: -scores[k])
            sel = (gold_first + non_gold_by_score)[:args.top_k]
            order = sel
            scores_aligned = [scores[k] for k in order]

        chosen_paragraphs = [paragraphs[k] for k in order]

        # --- Step 4: build new prompt ---
        user_text = build_prompt(q, chosen_paragraphs, args.mode, scores_aligned)
        new_prompt = wrap_for_generation(user_text, tok, args.chat_template)

        if n_prompt_samples_saved < 5:
            sample_f.write(f"=== sample {i}  mode={args.mode}  k_selected={len(order)} ===\n")
            sample_f.write(f"order={order}\nscores_aligned={[round(s,3) for s in scores_aligned]}\n")
            sample_f.write(new_prompt + "\n\n")
            sample_f.flush()
            n_prompt_samples_saved += 1

        # --- Step 5: generate ---
        new_enc = tok(new_prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
        new_input_ids = new_enc["input_ids"].to("cuda")
        new_attn = new_enc["attention_mask"].to("cuda")
        with torch.no_grad():
            out_ids = model.generate(
                input_ids=new_input_ids, attention_mask=new_attn,
                max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
                do_sample=False, pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=(not args.no_cache),
            )
        gen_only = out_ids[0, new_input_ids.shape[1]:]
        pred = truncate_pred(tok.decode(gen_only, skip_special_tokens=True))

        # --- Step 6: metrics ---
        em = exact_match_score(pred, gold)
        f1 = f1_score(pred, gold)
        f1s.append(f1); ems.append(em)
        in_tokens.append(new_input_ids.shape[1])
        n_chunks.append(len(order))
        if order:
            score_dists_selected.append(sum(scores[k] for k in order) / len(order))

        # support_recall@k
        sup_in_sel = sum(1 for k in order if is_supporting[k])
        n_gold = max(1, sum(is_supporting))
        sr = sup_in_sel / n_gold
        supp_recalls.append(sr)

        pred_f.write(json.dumps({
            "i": i, "question": q, "gold": gold, "pred": pred,
            "f1": f1, "em": em, "mode": args.mode,
        }) + "\n")
        score_f.write(json.dumps({
            "i": i, "chunk_scores": [round(s, 4) for s in scores],
            "selected_chunk_ids": order,
            "gold_support_chunk_ids": gold_supp_ids,
            "support_recall_at_k": sr,
            "input_tokens": int(new_input_ids.shape[1]),
            "n_selected": len(order),
        }) + "\n")

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"[evid] {args.mode} {i+1}/{len(ds)}  "
                  f"F1={sum(f1s)/len(f1s):.4f}  EM={sum(ems)/len(ems):.4f}  "
                  f"supp_recall={sum(supp_recalls)/len(supp_recalls):.3f}  "
                  f"avg_tok={int(sum(in_tokens)/len(in_tokens))}  "
                  f"elapsed={elapsed:.0f}s")

    pred_f.close(); score_f.close(); sample_f.close()

    metrics = {
        "mode": args.mode,
        "ckpt": args.ckpt,
        "n": len(f1s),
        "f1": sum(f1s) / max(len(f1s), 1),
        "em": sum(ems) / max(len(ems), 1),
        "soft_acc": sum(1 for x in f1s if x >= 0.5) / max(len(f1s), 1),
        "support_recall_at_k": sum(supp_recalls) / max(len(supp_recalls), 1),
        "avg_input_tokens": sum(in_tokens) / max(len(in_tokens), 1),
        "avg_n_chunks": sum(n_chunks) / max(len(n_chunks), 1),
        "avg_selected_score": sum(score_dists_selected) / max(len(score_dists_selected), 1),
        "top_k": args.top_k if args.mode in ("topk_score", "oracle_topk", "topk_bm25") else None,
        "elapsed_sec": time.time() - t0,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    cfg = {**vars(args), "ts": datetime.now().isoformat()}
    with open(out_dir / "run_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    print(f"\n[evid] DONE  mode={args.mode}  n={len(f1s)}")
    print(f"  F1={metrics['f1']:.4f}  EM={metrics['em']:.4f}  "
          f"soft_acc={metrics['soft_acc']:.3f}  "
          f"supp_recall={metrics['support_recall_at_k']:.3f}")
    print(f"  avg_tokens={metrics['avg_input_tokens']:.0f}  avg_chunks={metrics['avg_n_chunks']:.1f}")


if __name__ == "__main__":
    main()
