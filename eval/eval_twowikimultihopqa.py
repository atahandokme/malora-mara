"""
Evaluate model on 2WikiMultihopQA validation (dev) set.

Usage:
  # LoRA epoch_2:
  python eval_twowikimultihopqa.py --model Qwen/Qwen2.5-7B-Instruct \
      --lora-checkpoint outputs/twowikimultihopqa_lora_r16_qwen2.5-7b_*/lora_epoch_2 \
      --n-samples 3000

  # MaLoRA (scalar or diagonal output) epoch_2:
  python eval_twowikimultihopqa.py --model Qwen/Qwen2.5-7B-Instruct \
      --malora-checkpoint outputs/twowikimultihopqa_hproj_*/gated_lora_epoch_2.pt \
      --n-samples 3000
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
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Enable TF32 for A100
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from data.twowikimultihopqa import format_twowikimultihopqa_prompt, _load_2wiki
from model.inject import load_model_with_sra


def normalize_answer(s):
    """Normalize answer for comparison."""
    import string

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
    """Check if prediction matches ground truth (exact match)."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
    """Compute F1 score between prediction and ground truth tokens."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B')
    parser.add_argument('--sra-checkpoint', type=str, default=None,
                        help='Path to SRA checkpoint (optional)')
    parser.add_argument('--lora-checkpoint', type=str, default=None,
                        help='Path to LoRA checkpoint directory (optional)')
    parser.add_argument('--gated-lora-checkpoint', type=str, default=None,
                        help='Path to Gated LoRA checkpoint (optional)')
    parser.add_argument('--malora-checkpoint', type=str, default=None,
                        help='Path to MaLoRA checkpoint (optional)')
    parser.add_argument('--residual-gated-lora-checkpoint', type=str, default=None,
                        help='Path to Residual-Gated LoRA checkpoint (optional)')
    parser.add_argument('--madora-checkpoint', type=str, default=None,
                        help='Path to MaDoRA checkpoint (optional)')
    parser.add_argument('--recurrent-lora-checkpoint', type=str, default=None,
                        help='Path to RecurrentLoRA checkpoint (Design 2.5)')
    parser.add_argument('--config', type=str, default='configs/llama.yaml',
                        help='Config file (needed if using SRA)')
    parser.add_argument('--n-samples', type=int, default=3000,
                        help='Number of samples to evaluate on')
    parser.add_argument('--max-length', type=int, default=32768,
                        help='Max sequence length for context')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Generation batch size (prompts per forward pass)')
    parser.add_argument('--num-beams', type=int, default=4,
                        help='Beam search width; set to 1 for greedy decoding')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable KV cache (use_cache=False). Forces full re-prefilling each decode step. '
                             'Slow but restores correct Mamba recurrent state across decode steps for SSM-based adapters.')
    args = parser.parse_args()

    print(f"Loading 2WikiMultihopQA dev set...")
    dataset = _load_2wiki('dev')

    # Limit to n_samples (dataset is a plain list)
    n_samples = min(args.n_samples, len(dataset))
    dataset = dataset[:n_samples]
    print(f"Evaluating on {n_samples} validation samples\n")

    # Load model
    if args.sra_checkpoint and args.lora_checkpoint:
        # Combined SRA+LoRA mode
        from peft import PeftModel
        from model.inject import inject_sra
        print(f"Loading model with SRA+LoRA...")
        print(f"  LoRA: {args.lora_checkpoint}")
        print(f"  SRA:  {args.sra_checkpoint}")

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map='auto',
            attn_implementation="sdpa",
        )

        # Step 1: Load and merge LoRA
        model = PeftModel.from_pretrained(base_model, args.lora_checkpoint)
        model = model.merge_and_unload()
        print(f"  Merged LoRA into base weights")

        # Step 2: Inject SRA on top
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        model, sra_modules = inject_sra(model, config['sra'])

        checkpoint = torch.load(args.sra_checkpoint, map_location='cuda')
        for sra, state_dict in zip(sra_modules, checkpoint['sra_modules']):
            sra.load_state_dict(state_dict)

        model_dtype = next(model.parameters()).dtype
        for sra in sra_modules:
            sra.to(dtype=model_dtype)

        print(f"  Loaded SRA checkpoint (epoch {checkpoint.get('epoch', '?')})")
        model_name = "SRA+LoRA"

    elif args.sra_checkpoint:
        print(f"Loading model with SRA from {args.sra_checkpoint}...")

        # Load config
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Load model with SRA
        model, tokenizer, sra_modules = load_model_with_sra(
            model_name=args.model,
            sra_config=config['sra'],
            device='cuda',
            torch_dtype=torch.bfloat16,
        )

        # Load SRA checkpoint
        checkpoint = torch.load(args.sra_checkpoint)
        for sra, state_dict in zip(sra_modules, checkpoint['sra_modules']):
            sra.load_state_dict(state_dict)

        # Convert SRA modules to model dtype
        model_dtype = next(model.parameters()).dtype
        for sra in sra_modules:
            sra.to(dtype=model_dtype)

        print(f"  Loaded SRA checkpoint (epoch {checkpoint.get('epoch', '?')})")
        print(f"  Converted SRA modules to {model_dtype}")
        model_name = f"SRA (epoch {checkpoint.get('epoch', '?')})"

    elif args.residual_gated_lora_checkpoint:
        from model.residual_gated_lora import load_model_with_residual_gated_lora
        print(f"Loading model with Residual-Gated LoRA from {args.residual_gated_lora_checkpoint}...")
        model, tokenizer = load_model_with_residual_gated_lora(
            args.model, args.residual_gated_lora_checkpoint, device='auto',
        )
        model_name = "Residual-Gated LoRA"

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

    elif args.malora_checkpoint:
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
            args.model,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(base_model, args.lora_checkpoint)
        model = model.merge_and_unload()
        model_name = "LoRA"
        print(f"  Loaded and merged LoRA adapter")

    else:
        print(f"Loading baseline model: {args.model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map='auto',
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

    # Track by question type
    by_type = {}

    results = []

    # Left-pad for batched causal-LM generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect chunk-MaLoRA: if the model has any ChunkMambaGate, eval needs
    # to compute per-sample chunk_ends and set them on the gates before each
    # forward (otherwise gates default to 1.0 — a train/eval mismatch that
    # destroys generation accuracy).
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
        print(f"[Chunk eval] Detected ChunkMambaGate modules — eval will compute "
              f"per-sample chunk_ends (with left-pad offset adjustment) before generation.")
        if not getattr(tokenizer, "is_fast", False):
            print(f"[Chunk eval] WARN: tokenizer is not fast; offset_mapping may be unavailable.")

    items = list(dataset)

    # --- Sequence-budget guard -------------------------------------------
    # Prompts longer than the budget are SKIPPED, never truncated: truncation
    # would silently drop candidate paragraphs (possibly supporting ones) and
    # confound retrieval quality with context loss. On the released splits this
    # guard removes nothing (longest 2Wiki prompt is well under the budget);
    # it is enforced so the guarantee holds for any split or tokenizer.
    _kept, _n_over = [], 0
    for _it in items:
        _p = format_twowikimultihopqa_prompt(_it['question'], _it['context'], answer=None)
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
    # Print partial running metrics every PARTIAL_EVERY samples processed.
    PARTIAL_EVERY = 400
    next_partial_at = PARTIAL_EVERY
    for start in tqdm(range(0, len(items), batch_size), desc="2WikiMultihopQA Eval"):
        batch = items[start:start + batch_size]

        # Format prompts for this batch. 2wiki context is List[[title, [sentences]]] already.
        prompts = []
        for item in batch:
            ctx = item['context']
            prompts.append(format_twowikimultihopqa_prompt(item['question'], ctx, answer=None))

        # Tokenize. For chunk eval we need offset_mapping to detect chunks.
        if is_chunk_eval:
            inputs = tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_offsets_mapping=True,
            )
            offsets_per_sample = inputs.pop("offset_mapping")
            # Compute chunk_ends for each sample
            batch_chunk_ends = []
            for b in range(len(batch)):
                # With left-pad and batch_size=1, prompt starts at position 0
                offs = offsets_per_sample[b].tolist()
                # Skip leading pad tokens (left-padding pushes content right)
                attn = inputs['attention_mask'][b]
                first_real = int((attn != 0).nonzero()[0].item()) if attn.numel() > 0 else 0
                # Convert offsets within real prompt tokens
                real_offs = offs[first_real:]
                try:
                    chunks = detect_hotpotqa_chunks(prompts[b], real_offs)
                    # Re-add the first_real offset
                    ce = [c["end_tok"] + first_real for c in chunks]
                except AssertionError:
                    ce = []
                batch_chunk_ends.append(ce)
            set_chunk_ends_on_model(model, batch_chunk_ends)
        else:
            inputs = tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Batched generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                num_beams=args.num_beams,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=(not args.no_cache),
            )

        # Slice off the prompt prefix (left-padded so all prompts end at same index)
        prompt_len = inputs["input_ids"].shape[1]
        gen_only = outputs[:, prompt_len:]
        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        for item, prediction in zip(batch, decoded):
            question = item['question']
            answer = item['answer']
            q_type = item.get('type', 'unknown')
            prediction = prediction.strip()

            # Post-process: take first sentence/line only (models often generate verbose answers)
            for sep in ['\n', '. ', ', who ', ', which ', ', the ']:
                if sep in prediction:
                    prediction = prediction.split(sep)[0]
                    break

            # Evaluate
            em = exact_match_score(prediction, answer)
            f1 = f1_score(prediction, answer)

            if em:
                correct_em += 1
            total_f1 += f1
            total += 1

            # Track by type
            if q_type not in by_type:
                by_type[q_type] = {'correct': 0, 'total': 0, 'f1': 0.0}
            by_type[q_type]['total'] += 1
            if em:
                by_type[q_type]['correct'] += 1
            by_type[q_type]['f1'] += f1

            results.append({
                'question': question,
                'prediction': prediction,
                'answer': answer,
                'em': em,
                'f1': f1,
                'type': q_type,
            })

        # Partial-results checkpoint every ~400 samples
        if total >= next_partial_at:
            partial_em = correct_em / total
            partial_f1 = total_f1 / total
            partial_soft = sum(1 for r in results if r['f1'] >= 0.5) / total
            print(
                f"\n[Partial @ {total}/{len(items)}] "
                f"EM={partial_em:.1%} ({correct_em}/{total})  "
                f"F1={partial_f1:.3f}  "
                f"Soft(F1>=0.5)={partial_soft:.1%}",
                flush=True,
            )
            next_partial_at += PARTIAL_EVERY

    # Compute metrics
    em_accuracy = correct_em / total
    avg_f1 = total_f1 / total

    soft_correct = sum(1 for r in results if r['f1'] >= 0.5)
    soft_accuracy = soft_correct / total

    print(f"\n{'='*70}")
    print(f"2WikiMultihopQA Results ({model_name})")
    print(f"{'='*70}")
    print(f"Samples: {total}")
    print(f"Exact Match: {em_accuracy:.1%} ({correct_em}/{total})")
    print(f"F1 Score: {avg_f1:.3f}")
    print(f"Soft Accuracy (F1≥0.5): {soft_accuracy:.1%} ({soft_correct}/{total})")
    print(f"{'='*70}")

    # By question type
    print(f"\nBy Question Type:")
    for q_type in sorted(by_type.keys()):
        stats = by_type[q_type]
        type_em = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        type_f1 = stats['f1'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {q_type:15s}: EM={type_em:5.1%} ({stats['correct']:3d}/{stats['total']:3d}), F1={type_f1:.3f}")

    # Show some examples
    print(f"\n{'='*70}")
    print("Sample Predictions:")
    print(f"{'='*70}")
    for i in range(min(5, len(results))):
        r = results[i]
        print(f"\n[{i+1}] Type: {r['type']}")
        print(f"Question: {r['question']}")
        print(f"Ground Truth: {r['answer']}")
        print(f"Prediction: {r['prediction']}")
        print(f"EM: {r['em']}, F1: {r['f1']:.3f}")

    print(f"\n{'='*70}")
    print("✅ Evaluation complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
