"""
MaRA training: retrieval adapter only.

    frozen backbone (first K layers), no adapter attached
    no LM loss
    train: attn_pool_W/q, linear_in, mamba, local_head, global_head
    objective: weighted local BCE + intra-example pairwise margin + auxiliary global BCE

The paper configuration is --pos-weight 8.0 --pairwise-weight 0.5
--pairwise-margin 1.0 --chain-passes 4 --batch-size 2 --grad-accum 2.
Several argparse defaults differ from it; see the README.

Selects the best checkpoint by held-out margin + 2 * recall@4.
Saved as: <out>/router_best.pt  +  config.yaml.
"""

# --- repo root on sys.path (entry points live one level below it) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "chunking", _ROOT / "mara", _ROOT / "train", _ROOT / "eval"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------


import argparse, json, os, sys, yaml
from pathlib import Path
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "chunking"))
sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root

from collate import collate_chunked
from chunked_musique import ChunkedMuSiQueDataset
from chunked_twowikimultihopqa import ChunkedTwoWikiMultihopQADataset
from mara import EvidenceRouter, LastTokenRouter, MLPRouter, TransformerRouter, TokenMambaRouter, PoolOnlyRouter, RNNRouter, derive_p_chunk_spans


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--router-kind", choices=["causal", "global_mlp"], default="global_mlp")
    p.add_argument("--mamba-dim", type=int, default=256)
    p.add_argument("--mamba-n-layers", type=int, default=2)
    p.add_argument("--mamba-d-state", type=int, default=64)
    p.add_argument("--mamba-d-conv", type=int, default=4)
    p.add_argument("--mamba-expand", type=int, default=2)
    p.add_argument("--encoder-K", type=int, default=16)
    p.add_argument("--attn-pool-dim", type=int, default=256)

    p.add_argument("--max-train-samples", type=int, default=10000)
    p.add_argument("--max-val-samples", type=int, default=500)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--shuffle-paragraphs", action="store_true")
    p.add_argument("--dynamic-shuffle", action="store_true")

    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    p.add_argument("--pos-weight", type=float, default=4.0)
    p.add_argument("--global-bce-weight", type=float, default=0.5)
    p.add_argument("--pairwise-weight", type=float, default=0.0,
                   help="optional pairwise margin loss between gold and distractor logits")
    p.add_argument("--pairwise-margin", type=float, default=1.0)

    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", choices=["hotpotqa", "musique", "twowikimultihopqa", "mixed"], default="musique")
    p.add_argument("--mixed-pool-paths", default=None, type=str,
                   help="Comma-separated 3 paths PATH_MU,PATH_TW,PATH_HP for --dataset mixed.")
    p.add_argument("--joint-encoding", action="store_true",
                   help="Encode Q+all_paragraphs in a single LM forward pass (cross-paragraph context).")
    p.add_argument("--iterative", action="store_true",
                   help="2-pass routing: pass-2 conditions each paragraph encoding on pass-1's top-1 paragraph.")
    p.add_argument("--aux-pass1-weight", type=float, default=0.3,
                   help="Loss weight on pass-1 logits when --iterative is set.")
    p.add_argument("--chain-passes", type=int, default=0,
                   help="If >0, train as N-pass chain routing (extends --iterative to N=4 etc). "
                        "Each step conditions on all previously selected anchors. Overrides --iterative.")
    p.add_argument("--chain-step-weight-aux", type=float, default=0.5,
                   help="Loss weight on intermediate chain steps (1..N-1); final step (N) gets weight 1.0.")
    # --- Cascade training (N-pass elimination cascade) ---
    p.add_argument("--cascade-passes", type=int, default=0,
                   help="If >0, train as cascade with N passes (overrides --iterative).")
    p.add_argument("--cascade-drop-per-pass", type=int, default=4,
                   help="Survivors to drop per cascade pass.")
    p.add_argument("--cascade-anchor", action="store_true",
                   help="Cumulative-anchor cascade: each pass conditions on prior pass anchors.")
    p.add_argument("--cascade-weights", type=str, default="0.2,0.3,0.4,1.0",
                   help="Comma-separated per-pass loss weights (must match --cascade-passes).")
    # --- Beam training (K-anchor extension of iter, paper-validated optimum K=2) ---
    p.add_argument("--beam-train-k", type=int, default=0,
                   help="If >0, train with K-anchor beam: Pass 1 -> top-K anchors -> K Pass-2 forwards. "
                        "Overrides --iterative when set.")
    p.add_argument("--lora-pool-path", default=None,
                   help="Path to canonical 10k JSON; if set, dataset loads from this instead of HF + filter.")
    p.add_argument("--baseline-last-token", action="store_true",
                   help="Use the LastTokenRouter baseline (no attn-pool, no mamba, no iteration). "
                        "Drop-in replacement; same loss/eval/data pipeline.")
    p.add_argument("--baseline-mlp", action="store_true",
                   help="Use the MLPRouter baseline (attn-pool + cross-paragraph MLP, no mamba, no iteration).")
    p.add_argument("--baseline-transformer", action="store_true",
                   help="Use the TransformerRouter baseline (attn-pool + 2-layer Transformer encoder, no mamba).")
    p.add_argument("--baseline-token-mamba", action="store_true",
                   help="Use the TokenMambaRouter baseline: Mamba over RAW TOKENS (no chunk pooling before SSM); "
                        "span-pool the Mamba outputs into per-paragraph scores. Granularity ablation.")
    p.add_argument("--baseline-gru", action="store_true",
                   help="Use the RNNRouter baseline with a 2-layer GRU mixer (attn-pool kept, "
                        "Mamba replaced by classical gated recurrence over paragraphs).")
    p.add_argument("--baseline-lstm", action="store_true",
                   help="Use the RNNRouter baseline with a 2-layer LSTM mixer (attn-pool kept, "
                        "Mamba replaced by classical gated recurrence over paragraphs).")
    p.add_argument("--baseline-pool-only", action="store_true",
                   help="Use the PoolOnlyRouter baseline: attention-pool per chunk, then score directly "
                        "with NO cross-paragraph mixer. Isolates the pooling method from the mixer.")
    p.add_argument("--mean-pool", action="store_true",
                   help="Ablation: replace attn-pool with mean-pool over paragraph tokens in the Mamba router.")
    p.add_argument("--scoring-head", choices=["linear", "mlp"], default="linear",
                   help="Local scoring-head architecture (appendix ablation). mlp = Linear->GELU->Linear.")
    p.add_argument("--base-lora-ckpt", default=None,
                   help="Optional path to a trained LoRA/gated-LoRA checkpoint. If set, applied to the base "
                        "model BEFORE the router's K-layer encoder. Gives the router task-tuned features. "
                        "Accepts PEFT LoRA dir (with adapter_config.json) or gated_lora .pt file.")
    return p


def pairwise_ranking_loss(logits, gold, margin=1.0):
    """Hinge loss on (positive - negative) pairs per sample."""
    B, N = logits.shape
    loss = logits.new_zeros(())
    n_pairs = 0
    for b in range(B):
        pos_idx = (gold[b] > 0.5).nonzero(as_tuple=True)[0]
        neg_idx = (gold[b] < 0.5).nonzero(as_tuple=True)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        pos_l = logits[b, pos_idx]                       # (P,)
        neg_l = logits[b, neg_idx]                       # (Q,)
        # pairwise margin
        diff = neg_l.unsqueeze(0) - pos_l.unsqueeze(1) + margin   # (P, Q)
        loss = loss + F.relu(diff).mean()
        n_pairs += 1
    if n_pairs == 0:
        return logits.new_zeros(())
    return loss / n_pairs


@torch.no_grad()
def evaluate_cascade(router, loader, pad_id, device,
                     n_passes, drop_per_pass, use_anchor):
    """Eval cascade router. Reports R@4 / R@8 / top1 from the final ranked top-K."""
    router.eval()
    n = 0
    n_top1 = 0
    r2_sum = r3_sum = r4_sum = r8_sum = 0.0
    both3 = both4 = 0
    mean_gold_rank_sum = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        gold = batch["is_supporting"].to(device)
        p_spans, q_spans = derive_p_chunk_spans(batch["chunk_ends"], batch["chunk_names"])
        if input_ids.size(0) != 1:
            # cascade only supports B=1
            continue
        try:
            pass_outputs = router.forward_cascade(
                input_ids, p_spans, q_spans, pad_id,
                n_passes=n_passes, drop_per_pass=drop_per_pass,
                use_anchor=use_anchor, gold=None,
            )
        except RuntimeError:
            continue

        last = pass_outputs[-1]
        survivors = last['survivors']
        anchors = last['anchors_so_far']
        local_logits = last['local_logits'][0]

        sorted_K = local_logits.argsort(descending=True).cpu().tolist()
        if use_anchor:
            rest = [survivors[i] for i in sorted_K if survivors[i] not in anchors]
            ranked = list(anchors) + rest
        else:
            ranked = [survivors[i] for i in sorted_K]

        g = gold[0]
        gold_idxs = set((g > 0.5).nonzero(as_tuple=True)[0].cpu().tolist())
        n_gold = max(len(gold_idxs), 1)

        if len(ranked) > 0 and ranked[0] in gold_idxs:
            n_top1 += 1

        # mean rank of gold paragraphs in the ranked list (rank 0 = best)
        gold_ranks = [ranked.index(gi) if gi in ranked else len(ranked) for gi in gold_idxs]
        mean_gold_rank_sum += sum(gold_ranks) / max(len(gold_ranks), 1)

        for k in (2, 3, 4, 8):
            top_k = ranked[:k]
            hits = sum(1 for idx in top_k if idx in gold_idxs)
            rec = hits / n_gold
            if k == 2: r2_sum += rec
            elif k == 3:
                r3_sum += rec
                if hits == n_gold: both3 += 1
            elif k == 4:
                r4_sum += rec
                if hits == n_gold: both4 += 1
            else:  # k == 8
                r8_sum += rec

        n += 1

    router.train()
    if n == 0:
        return {"margin": 0.0, "top1": 0.0, "recall_at_2": 0.0, "recall_at_3": 0.0,
                "recall_at_4": 0.0, "recall_at_8": 0.0, "both_gold_top3": 0.0, "both_gold_top4": 0.0,
                "mean_gold_rank": 0.0, "position_prior_corr": 0.0, "n_eval": 0}
    return {
        "margin": 0.0,   # not computed in cascade eval
        "top1": n_top1 / n,
        "recall_at_2": r2_sum / n,
        "recall_at_3": r3_sum / n,
        "recall_at_4": r4_sum / n,
        "recall_at_8": r8_sum / n,
        "both_gold_top3": both3 / n,
        "both_gold_top4": both4 / n,
        "mean_gold_rank": mean_gold_rank_sum / n,
        "position_prior_corr": 0.0,
        "n_eval": n,
    }


@torch.no_grad()
def evaluate_chain(router, loader, pad_id, device, n_passes):
    """Eval no-drop CHAIN router (matches forward_chain_train inference).

    Each pass conditions every candidate on the accumulated anchors via per-paragraph
    rows [Q + P_a1 + ... + P_a_{k-1} + P_t] (short rows, no sequence explosion), then
    picks the next anchor (argmax over non-anchors). Final ranking = anchors in pick
    order, then remaining paragraphs by the last pass's logits.
    """
    router.eval()
    n = 0
    n_top1 = 0
    r2_sum = r3_sum = r4_sum = r8_sum = 0.0
    both3 = both4 = 0
    mean_gold_rank_sum = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        gold = batch["is_supporting"].to(device)
        p_spans, q_spans = derive_p_chunk_spans(batch["chunk_ends"], batch["chunk_names"])
        try:
            out = router.forward_chain_train(input_ids, p_spans, q_spans, pad_id, n_passes=n_passes)
        except RuntimeError:
            continue
        B = input_ids.size(0)
        final_logits = out["logits_per_step"][-1]       # (B, N)
        for b in range(B):
            anchors = list(out["anchors"][b])           # length n_passes, pick order
            fl = final_logits[b]                        # (N,)
            sorted_idx = fl.argsort(descending=True).cpu().tolist()
            rest = [i for i in sorted_idx if i not in anchors]
            ranked = anchors + rest

            g = gold[b]
            gold_idxs = set((g > 0.5).nonzero(as_tuple=True)[0].cpu().tolist())
            if not gold_idxs:
                continue
            n_gold = len(gold_idxs)
            if ranked and ranked[0] in gold_idxs:
                n_top1 += 1
            gold_ranks = [ranked.index(gi) if gi in ranked else len(ranked) for gi in gold_idxs]
            mean_gold_rank_sum += sum(gold_ranks) / len(gold_ranks)
            for k in (2, 3, 4, 8):
                hits = sum(1 for idx in ranked[:k] if idx in gold_idxs)
                rec = hits / n_gold
                if k == 2: r2_sum += rec
                elif k == 3:
                    r3_sum += rec
                    if hits == n_gold: both3 += 1
                elif k == 4:
                    r4_sum += rec
                    if hits == n_gold: both4 += 1
                else: r8_sum += rec
            n += 1

    router.train()
    if n == 0:
        return {"margin": 0.0, "top1": 0.0, "recall_at_2": 0.0, "recall_at_3": 0.0,
                "recall_at_4": 0.0, "recall_at_8": 0.0, "both_gold_top3": 0.0, "both_gold_top4": 0.0,
                "mean_gold_rank": 0.0, "position_prior_corr": 0.0, "n_eval": 0}
    return {
        "margin": 0.0,
        "top1": n_top1 / n,
        "recall_at_2": r2_sum / n,
        "recall_at_3": r3_sum / n,
        "recall_at_4": r4_sum / n,
        "recall_at_8": r8_sum / n,
        "both_gold_top3": both3 / n,
        "both_gold_top4": both4 / n,
        "mean_gold_rank": mean_gold_rank_sum / n,
        "position_prior_corr": 0.0,
        "n_eval": n,
    }


@torch.no_grad()
def evaluate(router, loader, pad_id, device):
    router.eval()
    n = 0
    n_top1 = 0
    r2_sum = r3_sum = r4_sum = r8_sum = 0.0
    both3 = both4 = 0
    margin_sum = 0.0
    mean_gold_rank_sum = 0.0
    pos_corr_sum, pos_corr_n = 0.0, 0
    n_pos = 10  # HotpotQA distractor: always 10 paragraphs

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        gold = batch["is_supporting"].to(device)
        p_spans, q_spans = derive_p_chunk_spans(batch["chunk_ends"], batch["chunk_names"])
        out = router(input_ids, p_spans, q_spans, pad_id)
        local_logits = out[2]  # tuple is (h, g, local_logits, global_logits, [aux])

        for b in range(input_ids.size(0)):
            ll = local_logits[b]                                   # (10,)
            g = gold[b]                                            # (10,) {0,1}
            pos = ll[g > 0.5]
            neg = ll[g < 0.5]
            if pos.numel() == 0 or neg.numel() == 0:
                continue

            margin_sum += (pos.mean() - neg.mean()).item()

            # ranks: argsort descending; rank[i] = position of paragraph i in sorted order
            sorted_idx = ll.argsort(descending=True)
            rank_of = torch.empty_like(sorted_idx)
            rank_of[sorted_idx] = torch.arange(ll.numel(), device=ll.device)
            gold_idx = (g > 0.5).nonzero(as_tuple=True)[0]
            mean_gold_rank_sum += rank_of[gold_idx].float().mean().item()

            # top1
            if g[sorted_idx[0]] > 0.5:
                n_top1 += 1

            n_gold = max(int(g.sum().item()), 1)
            for k, sumref in [(2, "r2"), (3, "r3"), (4, "r4"), (8, "r8")]:
                topk_idx = sorted_idx[:k]
                hits = (g[topk_idx] > 0.5).sum().item()
                rec = hits / n_gold
                if k == 2: r2_sum += rec
                elif k == 3:
                    r3_sum += rec
                    if hits == n_gold:  # all gold in top-3
                        both3 += 1
                elif k == 4:
                    r4_sum += rec
                    if hits == n_gold:
                        both4 += 1
                else:  # k == 8
                    r8_sum += rec

            # position-prior correlation: pearson(rank, paragraph_index)
            # Detects whether router just predicts "earlier paragraphs more likely".
            positions = torch.arange(ll.numel(), device=ll.device, dtype=torch.float32)
            ranks = rank_of.float()
            ll_centered = positions - positions.mean()
            r_centered = ranks - ranks.mean()
            denom = (ll_centered.norm() * r_centered.norm()).clamp_min(1e-6)
            pos_corr_sum += float((ll_centered * r_centered).sum() / denom)
            pos_corr_n += 1

            n += 1

    router.train()
    if n == 0:
        return {"margin": 0.0, "top1": 0.0, "recall_at_2": 0.0, "recall_at_3": 0.0,
                "recall_at_4": 0.0, "recall_at_8": 0.0, "both_gold_top3": 0.0, "both_gold_top4": 0.0,
                "mean_gold_rank": 0.0, "position_prior_corr": 0.0, "n_eval": 0}
    return {
        "margin": margin_sum / n,
        "top1": n_top1 / n,
        "recall_at_2": r2_sum / n,
        "recall_at_3": r3_sum / n,
        "recall_at_4": r4_sum / n,
        "recall_at_8": r8_sum / n,
        "both_gold_top3": both3 / n,
        "both_gold_top4": both4 / n,
        "mean_gold_rank": mean_gold_rank_sum / n,
        "position_prior_corr": pos_corr_sum / max(pos_corr_n, 1),
        "n_eval": n,
    }


def main():
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(vars(args), f)
    log_f = open(out_dir / "log.jsonl", "w")

    print(f"[stage1] loading model {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.base_lora_ckpt is not None:
        ckpt = args.base_lora_ckpt
        print(f"[stage1] applying base-LoRA adapter from {ckpt}")
        if ckpt.endswith(".pt"):
            # Gated LoRA (HPSC / HPDI / MaLoRA)
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from model.malora import load_model_with_malora
            base, _tok = load_model_with_malora(
                args.model, ckpt, device="cuda", torch_dtype=torch.bfloat16,
            )
        else:
            # PEFT LoRA / DoRA adapter dir — merge into base so router sees LoRA'd features
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            ).to("cuda")
            base = PeftModel.from_pretrained(base, ckpt)
            base = base.merge_and_unload()  # permanently fold LoRA into base weights
    else:
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to("cuda")
    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    if args.dataset == "musique":
        n_paragraphs = 20
        print(f"[stage1] dataset: MuSiQue train ({args.max_train_samples}) + val ({args.max_val_samples})")
        manifest_path = str(Path(args.out_dir) / "lora_sample_ids.json")
        train_ds = ChunkedMuSiQueDataset(
            tokenizer=tok, max_length=args.max_length, split="train",
            max_samples=args.max_train_samples,
            shuffle_paragraphs=args.shuffle_paragraphs,
            manifest_path=manifest_path,
            lora_pool_path=args.lora_pool_path,
        )
        val_ds = ChunkedMuSiQueDataset(
            tokenizer=tok, max_length=args.max_length, split="validation",
            max_samples=args.max_val_samples,
            shuffle_paragraphs=False,
        )
    elif args.dataset == "twowikimultihopqa":
        n_paragraphs = 10
        print(f"[stage1] dataset: 2WikiMultihopQA train ({args.max_train_samples}) + val ({args.max_val_samples})")
        manifest_path = str(Path(args.out_dir) / "lora_sample_ids.json")
        train_ds = ChunkedTwoWikiMultihopQADataset(
            tokenizer=tok, max_length=args.max_length, split="train",
            max_samples=args.max_train_samples,
            shuffle_paragraphs=args.shuffle_paragraphs,
            manifest_path=manifest_path,
            lora_pool_path=args.lora_pool_path,
        )
        val_ds = ChunkedTwoWikiMultihopQADataset(
            tokenizer=tok, max_length=args.max_length, split="validation",
            max_samples=args.max_val_samples,
            shuffle_paragraphs=False,
        )
    elif args.dataset == "mixed":
        # MIXED 30k pool: 10k MQ + 10k TW + 10k HP, single router per backbone.
        # Each dataset's __getitem__ returns a self-contained dict; ConcatDataset routes by index.
        # max_length must accommodate MQ (4096); paragraph counts vary (20 MQ vs 10 TW/HP).
        from torch.utils.data import ConcatDataset
        assert args.mixed_pool_paths, "--mixed-pool-paths PATH_MU,PATH_TW,PATH_HP required for mixed"
        pool_mu, pool_tw, pool_hp = args.mixed_pool_paths.split(",")
        n_paragraphs = 20  # max over the three
        print(f"[stage1] dataset: MIXED (MQ+TW+HP) train pool from {pool_mu} | {pool_tw} | {pool_hp}")
        per_ds_train = args.max_train_samples // 3
        per_ds_val = args.max_val_samples // 3
        train_mu = ChunkedMuSiQueDataset(tokenizer=tok, max_length=args.max_length, split="train",
            max_samples=per_ds_train, shuffle_paragraphs=args.shuffle_paragraphs, lora_pool_path=pool_mu)
        train_tw = ChunkedTwoWikiMultihopQADataset(tokenizer=tok, max_length=args.max_length, split="train",
            max_samples=per_ds_train, shuffle_paragraphs=args.shuffle_paragraphs, lora_pool_path=pool_tw)
        from chunked_hotpotqa import ChunkedHotpotQADataset
        train_hp = ChunkedHotpotQADataset(tokenizer=tok, max_length=args.max_length, split="train",
            max_samples=per_ds_train, shuffle_paragraphs=args.shuffle_paragraphs, lora_pool_path=pool_hp)
        train_ds = ConcatDataset([train_mu, train_tw, train_hp])
        val_mu = ChunkedMuSiQueDataset(tokenizer=tok, max_length=args.max_length, split="validation",
            max_samples=per_ds_val, shuffle_paragraphs=False)
        val_tw = ChunkedTwoWikiMultihopQADataset(tokenizer=tok, max_length=args.max_length, split="validation",
            max_samples=per_ds_val, shuffle_paragraphs=False)
        val_hp = ChunkedHotpotQADataset(tokenizer=tok, max_length=args.max_length, split="validation",
            max_samples=per_ds_val, shuffle_paragraphs=False)
        val_ds = ConcatDataset([val_mu, val_tw, val_hp])
        print(f"[stage1]   train={len(train_ds)} val={len(val_ds)}")
    else:
        n_paragraphs = 10
        print(f"[stage1] dataset: HotpotQA train ({args.max_train_samples}) + val ({args.max_val_samples})")
        train_ds = ChunkedHotpotQADataset(
            tokenizer=tok, max_length=args.max_length, split="train",
            max_samples=args.max_train_samples,
            shuffle_paragraphs=args.shuffle_paragraphs,
            dynamic_shuffle=args.dynamic_shuffle,
            lora_pool_path=args.lora_pool_path,
        )
        val_ds = ChunkedHotpotQADataset(
            tokenizer=tok, max_length=args.max_length, split="validation",
            max_samples=args.max_val_samples,
            shuffle_paragraphs=False,
        )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_chunked(b, tok.pad_token_id), num_workers=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_chunked(b, tok.pad_token_id), num_workers=2,
    )

    if args.baseline_last_token:
        print("[stage1] using LastTokenRouter baseline (no attn-pool, no mamba, no iteration)")
        router = LastTokenRouter(
            base_model=base,
            encoder_K=args.encoder_K,
            n_paragraphs=n_paragraphs,
        ).to("cuda")
    elif args.baseline_mlp:
        print("[stage1] using MLPRouter baseline (attn-pool + cross-paragraph MLP, no mamba, no iteration)")
        router = MLPRouter(
            base_model=base,
            encoder_K=args.encoder_K,
            mamba_dim=args.mamba_dim,
            n_paragraphs=n_paragraphs,
            attn_pool_dim=args.attn_pool_dim,
        ).to("cuda")
    elif args.baseline_transformer:
        print("[stage1] using TransformerRouter baseline (attn-pool + 2-layer Transformer over paragraphs, no mamba)")
        router = TransformerRouter(
            base_model=base,
            encoder_K=args.encoder_K,
            mamba_dim=args.mamba_dim,
            n_paragraphs=n_paragraphs,
            attn_pool_dim=args.attn_pool_dim,
            n_layers=2,
            n_heads=4,
        ).to("cuda")
    elif args.baseline_gru or args.baseline_lstm:
        rnn_type = "gru" if args.baseline_gru else "lstm"
        print(f"[stage1] using RNNRouter baseline (attn-pool + 2-layer {rnn_type.upper()} over paragraphs, no mamba)")
        router = RNNRouter(
            base_model=base,
            encoder_K=args.encoder_K,
            mamba_dim=args.mamba_dim,
            n_paragraphs=n_paragraphs,
            attn_pool_dim=args.attn_pool_dim,
            rnn_type=rnn_type,
            n_layers=2,
        ).to("cuda")
    elif args.baseline_token_mamba:
        print("[stage1] using TokenMambaRouter baseline (Mamba over raw tokens; span-pool after recurrence)")
        router = TokenMambaRouter(
            base_model=base,
            encoder_K=args.encoder_K,
            mamba_dim=args.mamba_dim,
            mamba_n_layers=args.mamba_n_layers,
            mamba_d_state=args.mamba_d_state,
            mamba_d_conv=args.mamba_d_conv,
            mamba_expand=args.mamba_expand,
            n_paragraphs=n_paragraphs,
        ).to("cuda")
    elif args.baseline_pool_only:
        print("[stage1] using PoolOnlyRouter baseline (attn-pool + NO mixer; isolates pooling from mixer)")
        router = PoolOnlyRouter(
            base_model=base,
            encoder_K=args.encoder_K,
            mamba_dim=args.mamba_dim,
            n_paragraphs=n_paragraphs,
            attn_pool_dim=args.attn_pool_dim,
        ).to("cuda")
    else:
        router = EvidenceRouter(
            base_model=base,
            encoder_K=args.encoder_K,
            mamba_dim=args.mamba_dim,
            mamba_n_layers=args.mamba_n_layers,
            mamba_d_state=args.mamba_d_state,
            mamba_d_conv=args.mamba_d_conv,
            mamba_expand=args.mamba_expand,
            attn_pool_dim=args.attn_pool_dim,
            n_paragraphs=n_paragraphs,
            router_kind=args.router_kind,
            joint_encoding=args.joint_encoding,
            iterative=args.iterative,
            use_mean_pool=args.mean_pool,
            scoring_head=args.scoring_head,
        ).to("cuda")

    n_trainable = sum(p.numel() for p in router.parameters() if p.requires_grad)
    print(f"[stage1] router trainable params: {n_trainable:,}")

    opt = torch.optim.AdamW(
        [p for p in router.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    total_steps = (len(train_loader) // max(args.grad_accum, 1)) * args.epochs
    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * step / max(args.warmup_steps, 1)
        # cosine decay to 10% of base
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        import math
        return args.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(min(progress, 1.0) * math.pi)))

    pos_weight_t = torch.tensor([args.pos_weight], device="cuda")
    best_metric, best_path = -1e9, out_dir / "router_best.pt"

    # Parse cascade weights if cascade mode
    cascade_weights = None
    if args.cascade_passes > 0:
        cascade_weights = [float(w) for w in args.cascade_weights.split(',')]
        assert len(cascade_weights) == args.cascade_passes, (
            f"--cascade-weights ({len(cascade_weights)}) must match --cascade-passes ({args.cascade_passes})"
        )
        print(f"[stage1] CASCADE: passes={args.cascade_passes}, drop={args.cascade_drop_per_pass}, "
              f"anchor={args.cascade_anchor}, weights={cascade_weights}")

    step = 0
    micro = 0
    for epoch in range(args.epochs):
        for batch in train_loader:
            input_ids = batch["input_ids"].to("cuda")
            gold = batch["is_supporting"].to("cuda")
            p_spans, q_spans = derive_p_chunk_spans(batch["chunk_ends"], batch["chunk_names"])

            # ----- Cascade training path -----
            if args.cascade_passes > 0:
                try:
                    pass_outputs = router.forward_cascade(
                        input_ids, p_spans, q_spans, tok.pad_token_id,
                        n_passes=args.cascade_passes,
                        drop_per_pass=args.cascade_drop_per_pass,
                        use_anchor=args.cascade_anchor,
                        gold=gold,
                    )
                except RuntimeError as e:
                    print(f"[stage1] cascade RuntimeError on batch (skipped): {e}")
                    continue

                loss = input_ids.new_zeros((), dtype=torch.float32)
                total_w = sum(cascade_weights)
                last_pass_bce = None
                for w, po in zip(cascade_weights, pass_outputs):
                    survivors = po['survivors']
                    logits = po['local_logits'][0]   # (K,)
                    survivor_t = torch.tensor(survivors, device=gold.device, dtype=torch.long)
                    targets = gold[0].index_select(0, survivor_t).float()   # (K,)
                    pass_bce = F.binary_cross_entropy_with_logits(
                        logits, targets, pos_weight=pos_weight_t,
                    )
                    if args.pairwise_weight > 0.0:
                        pass_pair = pairwise_ranking_loss(
                            logits.unsqueeze(0), targets.unsqueeze(0),
                            margin=args.pairwise_margin,
                        )
                        pass_bce = pass_bce + args.pairwise_weight * pass_pair
                    loss = loss + w * pass_bce
                    last_pass_bce = pass_bce
                loss = loss / total_w
                # Stand-ins for logging
                local_bce = last_pass_bce.detach() if last_pass_bce is not None else loss.detach()
                global_bce = torch.tensor(0.0, device="cuda")
            elif args.chain_passes > 0:
                # ---- N-pass chain routing (extends iter to N steps) ----
                out = router.forward_chain_train(
                    input_ids, p_spans, q_spans, tok.pad_token_id,
                    n_passes=args.chain_passes,
                )
                # Per-step BCE with already-selected anchors masked out of loss
                step_local_bces = []
                step_global_bces = []
                step_pair_terms = []
                for k, (ll_k, gl_k, mask_k) in enumerate(zip(
                    out['logits_per_step'], out['globals_per_step'], out['anchor_mask_per_step'])):
                    # Build a weighted-mask gold: positions in mask_k contribute 0 (excluded)
                    keep = (~mask_k).float()  # (B, N), 1 if not yet selected
                    bce_local_per = F.binary_cross_entropy_with_logits(
                        ll_k, gold, pos_weight=pos_weight_t, reduction='none')
                    bce_global_per = F.binary_cross_entropy_with_logits(
                        gl_k, gold, pos_weight=pos_weight_t, reduction='none')
                    denom = keep.sum().clamp_min(1.0)
                    step_local_bces.append((bce_local_per * keep).sum() / denom)
                    step_global_bces.append((bce_global_per * keep).sum() / denom)
                    if args.pairwise_weight > 0.0:
                        step_pair_terms.append(
                            pairwise_ranking_loss(ll_k, gold, margin=args.pairwise_margin))
                # Final step (N) gets weight 1.0; intermediate steps get aux weight
                N = len(step_local_bces)
                weights = [args.chain_step_weight_aux] * (N - 1) + [1.0]
                total_w = sum(weights)
                local_bce = sum(w * b for w, b in zip(weights, step_local_bces)) / total_w
                global_bce = sum(w * b for w, b in zip(weights, step_global_bces)) / total_w
                loss = local_bce + args.global_bce_weight * global_bce
                if step_pair_terms:
                    pair_avg = sum(w * p for w, p in zip(weights, step_pair_terms)) / total_w
                    loss = loss + args.pairwise_weight * pair_avg
            elif args.beam_train_k > 0:
                # ---- Beam training (K-anchor extension of iter) ----
                out = router.forward_beam_train(
                    input_ids, p_spans, q_spans, tok.pad_token_id,
                    beam_k=args.beam_train_k,
                )
                # Pass-1 aux loss (same as iter)
                aux_local_bce = F.binary_cross_entropy_with_logits(
                    out['pass1_local'], gold, pos_weight=pos_weight_t,
                )
                aux_global_bce = F.binary_cross_entropy_with_logits(
                    out['pass1_global'], gold, pos_weight=pos_weight_t,
                )
                pass1_aux = aux_local_bce + args.global_bce_weight * aux_global_bce

                # Pass-2 loss averaged across K anchors
                pass2_local_terms = []
                pass2_global_terms = []
                pass2_pair_terms = []
                for ll_i, gl_i in zip(out['pass2_locals'], out['pass2_globals']):
                    pass2_local_terms.append(
                        F.binary_cross_entropy_with_logits(ll_i, gold, pos_weight=pos_weight_t)
                    )
                    pass2_global_terms.append(
                        F.binary_cross_entropy_with_logits(gl_i, gold, pos_weight=pos_weight_t)
                    )
                    if args.pairwise_weight > 0.0:
                        pass2_pair_terms.append(
                            pairwise_ranking_loss(ll_i, gold, margin=args.pairwise_margin)
                        )
                local_bce = torch.stack(pass2_local_terms).mean()
                global_bce = torch.stack(pass2_global_terms).mean()
                loss = local_bce + args.global_bce_weight * global_bce
                if pass2_pair_terms:
                    loss = loss + args.pairwise_weight * torch.stack(pass2_pair_terms).mean()
                loss = loss + args.aux_pass1_weight * pass1_aux
            else:
                out = router(input_ids, p_spans, q_spans, tok.pad_token_id)
                if router.iterative:
                    _, _, local_logits, global_logits, aux = out
                else:
                    _, _, local_logits, global_logits = out
                local_bce = F.binary_cross_entropy_with_logits(
                    local_logits, gold, pos_weight=pos_weight_t,
                )
                global_bce = F.binary_cross_entropy_with_logits(
                    global_logits, gold, pos_weight=pos_weight_t,
                )
                loss = local_bce + args.global_bce_weight * global_bce
                if args.pairwise_weight > 0.0:
                    loss = loss + args.pairwise_weight * pairwise_ranking_loss(
                        local_logits, gold, margin=args.pairwise_margin,
                    )
                # Auxiliary loss on pass-1 logits to ensure iterative router actually
                # learns to identify hop-1 in the first pass (otherwise top1_idx is junk).
                if router.iterative:
                    aux_local_bce = F.binary_cross_entropy_with_logits(
                        aux["pass1_local"], gold, pos_weight=pos_weight_t,
                    )
                    aux_global_bce = F.binary_cross_entropy_with_logits(
                        aux["pass1_global"], gold, pos_weight=pos_weight_t,
                    )
                    loss = loss + args.aux_pass1_weight * (aux_local_bce + args.global_bce_weight * aux_global_bce)

            (loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in router.parameters() if p.requires_grad],
                    args.max_grad_norm,
                )
                for pg in opt.param_groups:
                    pg["lr"] = lr_at(step)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0:
                    rec = {
                        "step": step, "epoch": epoch,
                        "loss": float(loss.item()),
                        "local_bce": float(local_bce.item()),
                        "global_bce": float(global_bce.item()),
                        "lr": float(opt.param_groups[0]["lr"]),
                    }
                    print(f"[stage1] {rec}")
                    log_f.write(json.dumps(rec) + "\n")
                    log_f.flush()

                if step % args.eval_every == 0:
                    if args.cascade_passes > 0:
                        m = evaluate_cascade(
                            router, val_loader, tok.pad_token_id, "cuda",
                            n_passes=args.cascade_passes,
                            drop_per_pass=args.cascade_drop_per_pass,
                            use_anchor=args.cascade_anchor,
                        )
                    elif args.chain_passes > 0:
                        m = evaluate_chain(
                            router, val_loader, tok.pad_token_id, "cuda",
                            n_passes=args.chain_passes,
                        )
                    else:
                        m = evaluate(router, val_loader, tok.pad_token_id, "cuda")
                    rec = {"step": step, "eval": m}
                    print(f"[stage1] EVAL {rec}")
                    log_f.write(json.dumps(rec) + "\n")
                    log_f.flush()
                    score = m["margin"] + 2.0 * m["recall_at_4"]
                    if score > best_metric:
                        best_metric = score
                        torch.save({
                            "router_state_dict": router.state_dict(),
                            "config": vars(args),
                            "metric": m,
                            "step": step,
                        }, best_path)
                        print(f"[stage1] saved best -> {best_path} (score={score:.4f})")

    # Final eval and save
    if args.cascade_passes > 0:
        m = evaluate_cascade(
            router, val_loader, tok.pad_token_id, "cuda",
            n_passes=args.cascade_passes,
            drop_per_pass=args.cascade_drop_per_pass,
            use_anchor=args.cascade_anchor,
        )
    elif args.chain_passes > 0:
        m = evaluate_chain(
            router, val_loader, tok.pad_token_id, "cuda",
            n_passes=args.chain_passes,
        )
    else:
        m = evaluate(router, val_loader, tok.pad_token_id, "cuda")
    score = m["margin"] + 2.0 * m["recall_at_4"]
    if score > best_metric:
        best_metric = score
        torch.save({
            "router_state_dict": router.state_dict(),
            "config": vars(args), "metric": m, "step": step,
        }, best_path)
    torch.save({
        "router_state_dict": router.state_dict(),
        "config": vars(args), "metric": m, "step": step,
    }, out_dir / "router_last.pt")
    log_f.write(json.dumps({"step": step, "eval_final": m}) + "\n")
    log_f.close()
    print(f"[stage1] done. best score={best_metric:.4f}, final eval={m}")


if __name__ == "__main__":
    main()
