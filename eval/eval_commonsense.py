"""
Evaluate model on commonsense MC benchmarks (preservation / forgetting probe).

Loads HP/MQ-trained adapters and evaluates on 8 commonsense MC tasks using
single-token logit argmax over option letters (same approach as eval_quality.py).

Benchmarks (matches DoRA paper / LLM-Adapters):
  boolq, piqa, siqa, hellaswag, winogrande, arc_easy, arc_challenge, openbookqa

Usage:
  # Baseline (no adapter):
  python eval_commonsense.py --model Qwen/Qwen2.5-7B --benchmark all

  # With MQ-trained LoRA:
  python eval_commonsense.py --model Qwen/Qwen2.5-7B --benchmark all \\
      --lora-checkpoint outputs/musique_lora_r16_qwen2.5-7b_20260411_140702/lora_epoch_2

  # With MQ-trained MaLoRA:
  python eval_commonsense.py --model Qwen/Qwen2.5-7B --benchmark all \\
      --malora-checkpoint outputs/musique_hproj_scalar_r16_d16_softplus_qwen2.5-7b_20260415_114123/gated_lora_epoch_2.pt
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
import yaml
import csv
import os
import time
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.inject import load_model_with_sra


BENCHMARKS = [
    'boolq', 'piqa', 'siqa', 'hellaswag', 'winogrande',
    'arc_easy', 'arc_challenge', 'openbookqa',
]


# ---------- Benchmark loaders + prompt formatters ----------

def load_bench(name):
    """Return list of {prompt, options (list of letter strings), answer_idx}."""
    examples = []

    if name == 'boolq':
        d = load_dataset('boolq', split='validation')
        for x in d:
            prompt = (
                f"Passage: {x['passage']}\n"
                f"Question: {x['question']}\n"
                f"(A) Yes\n(B) No\nAnswer:"
            )
            examples.append({
                'prompt': prompt,
                'options': ['A', 'B'],
                'answer_idx': 0 if x['answer'] else 1,
            })

    elif name == 'piqa':
        d = load_dataset('baber/piqa', split='validation')
        for x in d:
            prompt = (
                f"Goal: {x['goal']}\n"
                f"(A) {x['sol1']}\n(B) {x['sol2']}\nAnswer:"
            )
            examples.append({
                'prompt': prompt,
                'options': ['A', 'B'],
                'answer_idx': int(x['label']),
            })

    elif name == 'siqa':
        d = load_dataset('lighteval/siqa', split='validation')
        for x in d:
            prompt = (
                f"Context: {x['context']}\n"
                f"Question: {x['question']}\n"
                f"(A) {x['answerA']}\n(B) {x['answerB']}\n(C) {x['answerC']}\nAnswer:"
            )
            examples.append({
                'prompt': prompt,
                'options': ['A', 'B', 'C'],
                'answer_idx': int(x['label']) - 1,
            })

    elif name == 'hellaswag':
        d = load_dataset('Rowan/hellaswag', split='validation')
        for x in d:
            ctx = (x['ctx_a'] + ' ' + x['ctx_b'].capitalize()).strip()
            if x['activity_label']:
                ctx = f"{x['activity_label']}: {ctx}"
            opts_text = '\n'.join(f"({chr(65+i)}) {c}" for i, c in enumerate(x['endings']))
            prompt = f"{ctx}\n{opts_text}\nAnswer:"
            examples.append({
                'prompt': prompt,
                'options': ['A', 'B', 'C', 'D'],
                'answer_idx': int(x['label']),
            })

    elif name == 'winogrande':
        d = load_dataset('allenai/winogrande', 'winogrande_xl', split='validation')
        for x in d:
            prompt = (
                f"Fill in the blank: {x['sentence']}\n"
                f"(A) {x['option1']}\n(B) {x['option2']}\nAnswer:"
            )
            examples.append({
                'prompt': prompt,
                'options': ['A', 'B'],
                'answer_idx': int(x['answer']) - 1,
            })

    elif name in ('arc_easy', 'arc_challenge'):
        cfg = 'ARC-Easy' if name == 'arc_easy' else 'ARC-Challenge'
        d = load_dataset('ai2_arc', cfg, split='test')
        for x in d:
            labels = x['choices']['label']
            texts = x['choices']['text']
            # ARC sometimes uses numeric labels (1/2/3/4) — remap to A/B/C/D
            if any(l.isdigit() for l in labels):
                mapped_labels = [chr(65 + i) for i in range(len(labels))]
            else:
                mapped_labels = labels
            opts_text = '\n'.join(f"({L}) {t}" for L, t in zip(mapped_labels, texts))
            prompt = f"Question: {x['question']}\n{opts_text}\nAnswer:"
            if x['answerKey'] not in labels:
                continue
            ans_idx = labels.index(x['answerKey'])
            examples.append({
                'prompt': prompt,
                'options': mapped_labels,
                'answer_idx': ans_idx,
            })

    elif name == 'openbookqa':
        d = load_dataset('openbookqa', 'main', split='test')
        for x in d:
            labels = x['choices']['label']
            texts = x['choices']['text']
            opts_text = '\n'.join(f"({L}) {t}" for L, t in zip(labels, texts))
            prompt = f"Question: {x['question_stem']}\n{opts_text}\nAnswer:"
            if x['answerKey'] not in labels:
                continue
            examples.append({
                'prompt': prompt,
                'options': labels,
                'answer_idx': labels.index(x['answerKey']),
            })

    else:
        raise ValueError(f"Unknown benchmark: {name}")

    return examples


# ---------- Model loading (switchboard copied from eval_quality.py) ----------

def load_model(args):
    if args.sra_checkpoint and args.lora_checkpoint:
        from peft import PeftModel
        from model.inject import inject_sra
        print(f"Loading model with SRA+LoRA...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map='auto',
        )
        model = PeftModel.from_pretrained(base_model, args.lora_checkpoint)
        model = model.merge_and_unload()
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        model, sra_modules = inject_sra(model, config['sra'])
        ckpt = torch.load(args.sra_checkpoint, map_location='cuda')
        for sra, sd in zip(sra_modules, ckpt['sra_modules']):
            sra.load_state_dict(sd)
        md = next(model.parameters()).dtype
        for sra in sra_modules:
            sra.to(dtype=md)
        return model, tokenizer, "SRA+LoRA"

    if args.sra_checkpoint:
        print(f"Loading model with SRA from {args.sra_checkpoint}...")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        model, tokenizer, sra_modules = load_model_with_sra(
            model_name=args.model, sra_config=config['sra'],
            device='cuda', torch_dtype=torch.bfloat16,
        )
        ckpt = torch.load(args.sra_checkpoint)
        for sra, sd in zip(sra_modules, ckpt['sra_modules']):
            sra.load_state_dict(sd)
        md = next(model.parameters()).dtype
        for sra in sra_modules:
            sra.to(dtype=md)
        return model, tokenizer, f"SRA(epoch {ckpt.get('epoch', '?')})"

    if args.madora_checkpoint:
        from model.madora import inject_madora, MaDoRALinear, MaDoRAScalarGate
        print(f"Loading MaDoRA from {args.madora_checkpoint}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map='auto',
        )
        ckpt = torch.load(args.madora_checkpoint, map_location='cpu', weights_only=False)
        base, gate_modules = inject_madora(base, ckpt['config'])
        for p in base.parameters():
            if p.requires_grad:
                p.data = p.data.to(dtype=torch.bfloat16)
        for gate in gate_modules:
            if isinstance(gate, MaDoRAScalarGate):
                gate.mamba = gate.mamba.float()
        for name, mod in base.named_modules():
            if isinstance(mod, MaDoRALinear) and name in ckpt['madora_states']:
                s = ckpt['madora_states'][name]
                mod.lora_A.load_state_dict(s['lora_A'])
                mod.lora_B.load_state_dict(s['lora_B'])
                mod.magnitude.data = s['magnitude'].to(mod.magnitude.device)
                mod.gate.load_state_dict(s['gate'])
        return base, tokenizer, "MaDoRA"

    if args.residual_gated_lora_checkpoint:
        from model.residual_gated_lora import load_model_with_residual_gated_lora
        m, t = load_model_with_residual_gated_lora(args.model, args.residual_gated_lora_checkpoint, device='auto')
        return m, t, "Residual-Gated LoRA"

    if args.malora_checkpoint:
        from model.malora import load_model_with_malora
        m, t = load_model_with_malora(args.model, args.malora_checkpoint, device='auto')
        return m, t, "MaLoRA"

    if args.gated_lora_checkpoint:
        from model.gated_lora import load_model_with_gated_lora
        m, t = load_model_with_gated_lora(args.model, args.gated_lora_checkpoint, device='auto')
        return m, t, "Gated LoRA"

    if args.lora_checkpoint:
        from peft import PeftModel
        print(f"Loading LoRA/DoRA from {args.lora_checkpoint}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map='auto',
        )
        model = PeftModel.from_pretrained(base, args.lora_checkpoint)
        model = model.merge_and_unload()
        return model, tokenizer, "LoRA/DoRA"

    print(f"Loading baseline: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map='auto',
    )
    return model, tokenizer, "Baseline"


# ---------- Core eval loop (logit argmax) ----------

def evaluate_benchmark(model, tokenizer, examples, max_length, batch_size, desc):
    all_letters = sorted({L for ex in examples for L in ex['options']})
    letter_token = {L: tokenizer.encode(L, add_special_tokens=False)[0] for L in all_letters}

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    correct = 0
    total = 0
    for start in tqdm(range(0, len(examples), batch_size), desc=desc):
        batch = examples[start:start + batch_size]
        prompts = [x['prompt'] for x in batch]
        inputs = tokenizer(
            prompts, return_tensors='pt', padding=True,
            truncation=True, max_length=max_length,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, num_logits_to_keep=1)
            last = out.logits[:, -1, :]
        for i, ex in enumerate(batch):
            valid_tok = [letter_token[L] for L in ex['options']]
            pred_idx = int(last[i, valid_tok].argmax())
            if pred_idx == ex['answer_idx']:
                correct += 1
            total += 1
    return correct / total if total else 0.0, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B')
    parser.add_argument('--sra-checkpoint', type=str, default=None)
    parser.add_argument('--lora-checkpoint', type=str, default=None)
    parser.add_argument('--gated-lora-checkpoint', type=str, default=None)
    parser.add_argument('--malora-checkpoint', type=str, default=None)
    parser.add_argument('--residual-gated-lora-checkpoint', type=str, default=None)
    parser.add_argument('--madora-checkpoint', type=str, default=None)
    parser.add_argument('--config', type=str, default=None,
                        help='Required for --sra-checkpoint only')
    parser.add_argument('--benchmark', type=str, default='all',
                        help=f"One of {BENCHMARKS} or 'all'")
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Cap samples per benchmark (default: all)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--out-csv', type=str, default='eval_results/commonsense.csv',
                        help='Append-mode CSV output')
    parser.add_argument('--run-tag', type=str, default='run',
                        help='Identifier written to CSV (e.g. MQ_Qwen_malora_scalar)')
    args = parser.parse_args()

    benches = BENCHMARKS if args.benchmark == 'all' else [args.benchmark]
    for b in benches:
        if b not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark {b}; choices: {BENCHMARKS}")

    model, tokenizer, model_tag = load_model(args)
    model.eval()
    print(f"Model loaded: {model_tag} on {model.device}\n")

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    new_file = not os.path.exists(args.out_csv)

    all_accs = {}
    for bench in benches:
        print(f"\n=== {bench} ===")
        t0 = time.time()
        examples = load_bench(bench)
        if args.n_samples:
            examples = examples[:args.n_samples]
        acc, c, n = evaluate_benchmark(
            model, tokenizer, examples,
            max_length=args.max_length,
            batch_size=args.batch_size,
            desc=bench,
        )
        elapsed = time.time() - t0
        all_accs[bench] = acc
        print(f"  {bench}: {acc:.3%} ({c}/{n})  runtime={elapsed:.0f}s")

        with open(args.out_csv, 'a', newline='') as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(['run_tag', 'model', 'benchmark', 'accuracy', 'correct', 'total', 'runtime_s'])
                new_file = False
            w.writerow([args.run_tag, args.model, bench, f"{acc:.4f}", c, n, int(elapsed)])

    print(f"\n{'='*60}")
    print(f"Summary ({model_tag})")
    print(f"{'='*60}")
    for b, a in all_accs.items():
        print(f"  {b:15s}  {a:.3%}")
    if len(all_accs) > 1:
        avg = sum(all_accs.values()) / len(all_accs)
        print(f"  {'average':15s}  {avg:.3%}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
