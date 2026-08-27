"""
Evaluate model on MuSiQue validation set.

Usage:
  # Baseline:
  python eval_musique.py --model Qwen/Qwen2.5-7B-Instruct

  # Gated LoRA:
  python eval_musique.py --model Qwen/Qwen2.5-7B-Instruct --gated-lora-checkpoint outputs/<dir>/gated_lora_best.pt

  # LoRA:
  python eval_musique.py --model Qwen/Qwen2.5-7B-Instruct --lora-checkpoint outputs/<dir>
"""

# --- repo root on sys.path (entry points live one level below it) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "chunking", _ROOT / "mara", _ROOT / "train", _ROOT / "eval"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------


import torch
import argparse
import re
import string
import yaml
from pathlib import Path

# Enable TF32 for A100
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from data.musique import format_musique_prompt


def normalize_answer(s):
    """Normalize answer for comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common = set(pred_tokens) & set(truth_tokens)
    num_same = len(common)

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def best_score(prediction, answer, answer_aliases, metric_fn):
    """Return best score across answer and all aliases."""
    all_answers = [answer] + (answer_aliases or [])
    return max(metric_fn(prediction, a) for a in all_answers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B')
    parser.add_argument('--gated-lora-checkpoint', type=str, default=None)
    parser.add_argument('--malora-checkpoint', type=str, default=None)
    parser.add_argument('--recurrent-lora-checkpoint', type=str, default=None,
                        help='Path to RecurrentLoRA checkpoint (.pt) — additive SSM correction in rank space')
    parser.add_argument('--lora-checkpoint', type=str, default=None)
    parser.add_argument('--madora-checkpoint', type=str, default=None,
                        help='Path to MaDoRA checkpoint (.pt)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Generation batch size (prompts per forward pass)')
    parser.add_argument('--num-beams', type=int, default=4,
                        help='Beam search width; set to 1 for greedy decoding')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Number of samples (default: all)')
    parser.add_argument('--max-length', type=int, default=32768)
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable KV cache (use_cache=False). Forces full re-prefilling each decode step. '
                             'Slow but restores correct Mamba recurrent state across decode steps for SSM-based adapters.')
    args = parser.parse_args()

    print(f"Loading MuSiQue validation set...")
    dataset = load_dataset('dgslibisey/MuSiQue', split='validation')

    # Filter to answerable only
    dataset = dataset.filter(lambda x: x.get('answerable', True))

    if args.n_samples is not None:
        # Stratified sampling to maintain hop distribution
        import random
        rng = random.Random(42)
        indices_by_hop = {}
        for i, item in enumerate(dataset):
            qid = item.get('id', '')
            if qid.startswith('2hop'):
                hop = 2
            elif qid.startswith('3hop'):
                hop = 3
            elif qid.startswith('4hop'):
                hop = 4
            else:
                hop = 0
            indices_by_hop.setdefault(hop, []).append(i)

        # Proportional allocation
        total = len(dataset)
        selected = []
        for hop in sorted(indices_by_hop.keys()):
            idx = indices_by_hop[hop]
            rng.shuffle(idx)
            n_select = max(1, int(args.n_samples * len(idx) / total))
            selected.extend(idx[:n_select])
        rng.shuffle(selected)
        selected = selected[:args.n_samples]
        dataset = dataset.select(selected)
        n_samples = len(dataset)
    else:
        n_samples = len(dataset)
    print(f"Evaluating on {n_samples} answerable validation samples\n")

    # Load model
    if args.malora_checkpoint:
        from model.malora import load_model_with_malora
        print(f"Loading model with MaLoRA from {args.malora_checkpoint}...")
        model, tokenizer = load_model_with_malora(
            args.model, args.malora_checkpoint, device='auto',
        )
        model_name = "MaLoRA"

    elif args.recurrent_lora_checkpoint:
        from model.recurrent_lora import load_model_with_recurrent_lora
        print(f"Loading model with RecurrentLoRA from {args.recurrent_lora_checkpoint}...")
        model, tokenizer = load_model_with_recurrent_lora(
            args.model, args.recurrent_lora_checkpoint, device='auto',
        )
        model_name = "RecurrentLoRA"

    elif args.gated_lora_checkpoint:
        from model.gated_lora import load_model_with_gated_lora
        print(f"Loading model with Gated LoRA from {args.gated_lora_checkpoint}...")
        model, tokenizer = load_model_with_gated_lora(
            args.model, args.gated_lora_checkpoint, device='auto',
        )
        model_name = "Gated LoRA"

    elif args.lora_checkpoint:
        from peft import PeftModel
        print(f"Loading model with LoRA from {args.lora_checkpoint}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map='auto',
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(base_model, args.lora_checkpoint)
        model = model.merge_and_unload()
        model_name = "LoRA"

    elif args.madora_checkpoint:
        from model.madora import inject_madora, MaDoRALinear, MaDoRAScalarGate
        print(f"Loading model with MaDoRA from {args.madora_checkpoint}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map='auto',
            attn_implementation="sdpa",
        )
        ckpt = torch.load(args.madora_checkpoint, map_location='cpu', weights_only=False)
        madora_config = ckpt['config']
        base_model, gate_modules = inject_madora(base_model, madora_config)
        for p in base_model.parameters():
            if p.requires_grad:
                p.data = p.data.to(dtype=torch.bfloat16)
        for gate in gate_modules:
            if isinstance(gate, MaDoRAScalarGate):
                gate.mamba = gate.mamba.float()
        for name, module in base_model.named_modules():
            if isinstance(module, MaDoRALinear):
                if name in ckpt['madora_states']:
                    states = ckpt['madora_states'][name]
                    module.lora_A.load_state_dict(states['lora_A'])
                    module.lora_B.load_state_dict(states['lora_B'])
                    module.magnitude.data = states['magnitude'].to(module.magnitude.device)
                    module.gate.load_state_dict(states['gate'])
        model = base_model
        model_name = "MaDoRA"
        print(f"  Loaded MaDoRA (epoch {ckpt.get('epoch', '?')})")

    else:
        print(f"Loading baseline model: {args.model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map='auto',
            attn_implementation="sdpa",
        )
        model_name = "Baseline"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print(f"Model loaded on {model.device}\n")

    # Evaluate
    correct_em = 0
    total_f1 = 0.0
    total = 0

    # Track by hop count
    by_hops = {}

    results = []

    # Left-pad for batched causal-LM generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect chunk-MaLoRA variants: need to compute per-sample chunk_ends and
    # set them on the gates before each forward (otherwise gate falls back to
    # identity = pure LoRA). Reuses HP chunk detector since prompts share the
    # same Context: / Question: / Answer: structure.
    is_chunk_eval = False
    try:
        from model.chunk_mamba_gate import (
            ChunkMambaGate, ChunkTokenMambaGate, ChunkResidualMambaGate,
            set_chunk_ends_on_model,
        )
        from chunking.chunk_detector import detect_hotpotqa_chunks
        for m in model.modules():
            if isinstance(m, (ChunkMambaGate, ChunkTokenMambaGate, ChunkResidualMambaGate)):
                is_chunk_eval = True
                break
    except ImportError:
        pass
    if is_chunk_eval:
        print(f"[Chunk eval] Detected chunk-gate modules — eval will compute "
              f"per-sample chunk_ends with left-pad offset adjustment.")

    items = list(dataset)

    # --- Sequence-budget guard -------------------------------------------
    # Prompts longer than the budget are SKIPPED, never truncated: truncation
    # would silently drop candidate paragraphs (possibly supporting ones) and
    # confound retrieval quality with context loss. On the released splits this
    # guard removes nothing (longest MuSiQue prompt is well under the budget);
    # it is enforced so the guarantee holds for any split or tokenizer.
    _kept, _n_over = [], 0
    for _it in items:
        _p = format_musique_prompt(_it['question'], _it['paragraphs'], answer=None)
        if len(tokenizer(_p, add_special_tokens=True)['input_ids']) > args.max_length:
            _n_over += 1
            continue
        _kept.append(_it)
    if _n_over:
        print(f"[budget] skipped {_n_over} prompt(s) exceeding max_length="
              f"{args.max_length}; evaluating {len(_kept)} of {len(items)}")
    else:
        print(f"[budget] 0 prompts exceed max_length={args.max_length}; "
              f"evaluating all {len(items)}")
    items = _kept
    # ---------------------------------------------------------------------

    batch_size = max(1, int(args.batch_size))
    for start in tqdm(range(0, len(items), batch_size), desc="MuSiQue Eval"):
        batch = items[start:start + batch_size]

        prompts = [
            format_musique_prompt(item['question'], item['paragraphs'], answer=None)
            for item in batch
        ]

        if is_chunk_eval:
            inputs = tokenizer(
                prompts, return_tensors='pt',
                padding=True, truncation=True,
                max_length=args.max_length,
                return_offsets_mapping=True,
            )
            offsets_per_sample = inputs.pop("offset_mapping")
            batch_chunk_ends = []
            for b in range(len(batch)):
                offs = offsets_per_sample[b].tolist()
                attn = inputs['attention_mask'][b]
                first_real = int((attn != 0).nonzero()[0].item()) if attn.numel() > 0 else 0
                real_offs = offs[first_real:]
                try:
                    chunks = detect_hotpotqa_chunks(prompts[b], real_offs)
                    ce = [c["end_tok"] + first_real for c in chunks]
                except AssertionError:
                    ce = []
                batch_chunk_ends.append(ce)
            set_chunk_ends_on_model(model, batch_chunk_ends)
        else:
            inputs = tokenizer(
                prompts, return_tensors='pt',
                padding=True, truncation=True,
                max_length=args.max_length,
            )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=32, num_beams=args.num_beams,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
                use_cache=(not args.no_cache),
            )

        prompt_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)

        for item, prediction in zip(batch, decoded):
            question = item['question']
            answer = item['answer']
            answer_aliases = item.get('answer_aliases', [])

            # Determine hop count from question ID
            qid = item.get('id', '')
            if qid.startswith('2hop'):
                n_hops = '2-hop'
            elif qid.startswith('3hop'):
                n_hops = '3-hop'
            elif qid.startswith('4hop'):
                n_hops = '4-hop'
            else:
                n_hops = 'unknown'

            prediction = prediction.strip()
            # Post-process: take first sentence/line
            for sep in ['\n', '. ', ', who ', ', which ', ', the ']:
                if sep in prediction:
                    prediction = prediction.split(sep)[0]
                    break

            # Evaluate with best score across aliases
            em = best_score(prediction, answer, answer_aliases, exact_match_score)
            f1 = best_score(prediction, answer, answer_aliases, f1_score)

            if em:
                correct_em += 1
            total_f1 += f1
            total += 1

            # Track by hop count
            if n_hops not in by_hops:
                by_hops[n_hops] = {'correct': 0, 'total': 0, 'f1': 0.0}
            by_hops[n_hops]['total'] += 1
            if em:
                by_hops[n_hops]['correct'] += 1
            by_hops[n_hops]['f1'] += f1

            results.append({
                'question': question,
                'prediction': prediction,
                'answer': answer,
                'em': em,
                'f1': f1,
                'hops': n_hops,
            })

    # Compute metrics
    em_accuracy = correct_em / total
    avg_f1 = total_f1 / total
    soft_correct = sum(1 for r in results if r['f1'] >= 0.5)
    soft_accuracy = soft_correct / total

    print(f"\n{'='*70}")
    print(f"MuSiQue Results ({model_name})")
    print(f"{'='*70}")
    print(f"Samples: {total}")
    print(f"Exact Match: {em_accuracy:.1%} ({correct_em}/{total})")
    print(f"F1 Score: {avg_f1:.3f}")
    print(f"Soft Accuracy (F1≥0.5): {soft_accuracy:.1%} ({soft_correct}/{total})")
    print(f"{'='*70}")

    # By hop count
    print(f"\nBy Hop Count:")
    for hops in sorted(by_hops.keys()):
        stats = by_hops[hops]
        hop_em = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        hop_f1 = stats['f1'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {hops:10s}: EM={hop_em:5.1%} ({stats['correct']:3d}/{stats['total']:3d}), F1={hop_f1:.3f}")

    # Show examples
    print(f"\n{'='*70}")
    print("Sample Predictions:")
    print(f"{'='*70}")
    for i in range(min(5, len(results))):
        r = results[i]
        print(f"\n[{i+1}] Hops: {r['hops']}")
        print(f"Question: {r['question']}")
        print(f"Ground Truth: {r['answer']}")
        print(f"Prediction: {r['prediction']}")
        print(f"EM: {r['em']}, F1: {r['f1']:.3f}")

    print(f"\n{'='*70}")
    print("Evaluation complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
