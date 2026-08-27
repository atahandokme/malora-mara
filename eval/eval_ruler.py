"""
Evaluate model on RULER validation set (rbiswasfc/ruler on HuggingFace).

4 sub-tasks × 2 lengths (4k, 8k) = 8 splits:
  - niah_multikey_1_{4k,8k}  — Needle-in-a-Haystack (multi-key retrieval)
  - qa_2_{4k,8k}             — HotpotQA-based long-context QA
  - vt_{4k,8k}               — Variable Tracking (chained assignments)
  - cwe_{4k,8k}              — Common Words Extraction

Usage:
  # Baseline:
  python eval_ruler.py --model Qwen/Qwen2.5-7B --task niah_multikey_1 --length 4k

  # Gated/diag LoRA:
  python eval_ruler.py --model Qwen/Qwen2.5-7B --task qa_2 --length 8k \
      --malora-checkpoint outputs/<dir>/gated_lora_epoch_2.pt

  # LoRA:
  python eval_ruler.py --model Qwen/Qwen2.5-7B --task vt --length 4k \
      --lora-checkpoint outputs/<dir>/lora_epoch_2
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
import json

# TF32 for speed
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


# RULER task-specific expected max output length + scoring hints
TASK_MAX_NEW = {
    "niah_multikey_1": 32,
    "qa_2": 48,
    "vt": 96,
    "cwe": 128,
}


def normalize(s):
    """Lowercase + strip punctuation for comparison."""
    s = str(s).lower().strip()
    s = re.sub(r"[%s]" % re.escape(string.punctuation), " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def score_sample(task, prediction, outputs):
    """
    Per-sample RULER score.

    - niah_multikey_1: 1.0 if any output-string appears in prediction (retrieval recall).
    - qa_2: F1-style token-overlap, taking max across output variants (like SQuAD).
    - vt / cwe: recall over the outputs set — fraction of expected strings present.

    Returns a float in [0, 1].
    """
    pred_norm = normalize(prediction)
    outs_norm = [normalize(o) for o in outputs]

    if task == "niah_multikey_1":
        return 1.0 if any(o in pred_norm for o in outs_norm) else 0.0

    if task == "qa_2":
        # F1 against best output variant (SQuAD-style)
        pred_tokens = pred_norm.split()
        if not pred_tokens:
            return 0.0
        best_f1 = 0.0
        for o in outs_norm:
            truth_tokens = o.split()
            if not truth_tokens:
                continue
            common = set(pred_tokens) & set(truth_tokens)
            n = len(common)
            if n == 0:
                continue
            p = n / len(pred_tokens)
            r = n / len(truth_tokens)
            best_f1 = max(best_f1, 2 * p * r / (p + r))
        return best_f1

    if task in ("vt", "cwe"):
        # Recall over expected set
        if not outs_norm:
            return 0.0
        hits = sum(1 for o in outs_norm if o in pred_norm)
        return hits / len(outs_norm)

    raise ValueError(f"Unknown task: {task}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--task", type=str, required=True,
                        choices=["niah_multikey_1", "qa_2", "vt", "cwe"])
    parser.add_argument("--length", type=str, required=True,
                        choices=["4k", "8k", "16k", "32k"])
    parser.add_argument("--gated-lora-checkpoint", type=str, default=None)
    parser.add_argument("--malora-checkpoint", type=str, default=None)
    parser.add_argument("--lora-checkpoint", type=str, default=None)
    parser.add_argument("--madora-checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Generation batch size (long context → keep small)")
    parser.add_argument("--num-beams", type=int, default=1,
                        help="Beam width; 1 = greedy (deterministic, fastest)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Limit number of samples (default: all 500)")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Max input tokens (safety truncation). Default auto-set "
                             "from --length: 4k→4608, 8k→9216, 16k→18432, 32k→36864.")
    parser.add_argument("--out-json", type=str, default=None,
                        help="Save per-sample results JSON (default: eval_results/ruler_<tag>.json)")
    args = parser.parse_args()
    if args.max_length is None:
        args.max_length = {"4k": 4608, "8k": 9216, "16k": 18432, "32k": 36864}[args.length]

    # 4k/8k → rbiswasfc/ruler (existing pre-tokenized prompts in `input` field).
    # 16k/32k → xAlg-AI/att-hub-ruler-{16k,32k} (split fields: context+question+answer_prefix).
    if args.length in ("4k", "8k"):
        split_name = f"{args.task}_{args.length}"
        print(f"Loading RULER split: rbiswasfc/ruler / {split_name}")
        dataset = load_dataset("rbiswasfc/ruler", split_name, split="validation")
    else:
        ds_name = f"xAlg-AI/att-hub-ruler-{args.length}"
        print(f"Loading RULER split: {ds_name} / {args.task}")
        d = load_dataset(ds_name, args.task)
        # Only one split; pick whichever it is
        split_key = list(d.keys())[0]
        dataset = d[split_key]
        # Adapt schema: build `input` from context+question+answer_prefix, map answer→outputs
        def _adapt(item):
            text = item['context']
            if not text.endswith('\n'):
                text += '\n\n'
            text += item['question'] + '\n\n' + item['answer_prefix']
            return {'input': text, 'outputs': list(item['answer'])}
        dataset = dataset.map(_adapt, remove_columns=[c for c in dataset.column_names
                                                       if c not in ('input', 'outputs')])
    if args.n_samples is not None:
        dataset = dataset.select(range(min(args.n_samples, len(dataset))))
    n_samples = len(dataset)
    print(f"Evaluating on {n_samples} samples (task={args.task}, length={args.length})\n")

    # --- Load model (same patterns as eval_musique.py) ---
    if args.malora_checkpoint:
        from model.malora import load_model_with_malora
        print(f"Loading MaLoRA from {args.malora_checkpoint}...")
        model, tokenizer = load_model_with_malora(
            args.model, args.malora_checkpoint, device="auto",
        )
        model_name = "MaLoRA"

    elif args.gated_lora_checkpoint:
        from model.gated_lora import load_model_with_gated_lora
        print(f"Loading Gated LoRA from {args.gated_lora_checkpoint}...")
        model, tokenizer = load_model_with_gated_lora(
            args.model, args.gated_lora_checkpoint, device="auto",
        )
        model_name = "Gated LoRA"

    elif args.lora_checkpoint:
        from peft import PeftModel
        print(f"Loading LoRA from {args.lora_checkpoint}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(base, args.lora_checkpoint).merge_and_unload()
        model_name = "LoRA"

    elif args.madora_checkpoint:
        from model.madora import inject_madora, MaDoRALinear, MaDoRAScalarGate
        print(f"Loading MaDoRA from {args.madora_checkpoint}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="sdpa",
        )
        ckpt = torch.load(args.madora_checkpoint, map_location="cpu", weights_only=False)
        base, gate_modules = inject_madora(base, ckpt["config"])
        for p in base.parameters():
            if p.requires_grad:
                p.data = p.data.to(dtype=torch.bfloat16)
        for g in gate_modules:
            if isinstance(g, MaDoRAScalarGate):
                g.mamba = g.mamba.float()
        for name, module in base.named_modules():
            if isinstance(module, MaDoRALinear):
                if name in ckpt["madora_states"]:
                    s = ckpt["madora_states"][name]
                    module.lora_A.load_state_dict(s["lora_A"])
                    module.lora_B.load_state_dict(s["lora_B"])
                    module.magnitude.data = s["magnitude"].to(module.magnitude.device)
                    module.gate.load_state_dict(s["gate"])
        model = base
        model_name = "MaDoRA"

    else:
        print(f"Loading baseline model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="sdpa",
        )
        model_name = "Baseline"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    print(f"Model loaded on {model.device}\n")

    # --- Evaluate ---
    max_new_tokens = TASK_MAX_NEW[args.task]
    results = []
    total_score = 0.0

    items = list(dataset)
    batch_size = max(1, int(args.batch_size))
    for start in tqdm(range(0, len(items), batch_size), desc=f"RULER {args.task}@{args.length}"):
        batch = items[start : start + batch_size]
        prompts = [it["input"] for it in batch]

        inputs = tokenizer(
            prompts, return_tensors="pt",
            padding=True, truncation=True,
            max_length=args.max_length,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                num_beams=args.num_beams, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)

        for it, prediction in zip(batch, decoded):
            prediction = prediction.strip()
            score = score_sample(args.task, prediction, it["outputs"])
            total_score += score
            results.append({
                "index": it.get("index", -1),
                "prediction": prediction[:300],
                "outputs": list(it["outputs"]),
                "score": score,
            })

    avg_score = total_score / len(results)

    print(f"\n{'='*70}")
    print(f"RULER {args.task}@{args.length} Results ({model_name})")
    print(f"{'='*70}")
    print(f"Samples: {len(results)}")
    print(f"Mean score: {avg_score:.4f}")
    if args.task == "niah_multikey_1":
        n_correct = sum(1 for r in results if r["score"] == 1.0)
        print(f"NIAH Retrieval Accuracy: {n_correct}/{len(results)} = {n_correct/len(results):.1%}")
    elif args.task == "qa_2":
        print(f"QA F1: {avg_score:.4f}")
    elif args.task in ("vt", "cwe"):
        print(f"Recall ({args.task}): {avg_score:.4f}")
    print(f"{'='*70}")

    # Show examples
    print(f"\nSample Predictions:")
    for i in range(min(3, len(results))):
        r = results[i]
        print(f"\n[{i+1}] score={r['score']:.3f}")
        print(f"  expected: {r['outputs']}")
        print(f"  predicted: {r['prediction'][:150]}")

    # Save JSON
    if args.out_json is None:
        import os
        os.makedirs("eval_results/ruler", exist_ok=True)
        # Tag from checkpoint path for identification
        ck = (args.malora_checkpoint or args.gated_lora_checkpoint
              or args.lora_checkpoint or args.madora_checkpoint or "baseline")
        tag = str(ck).split("/")[-2] if "/" in str(ck) else "baseline"
        args.out_json = f"eval_results/ruler/{args.task}_{args.length}__{model_name.replace(' ','').replace('/','-')}__{tag}.json"
    out = {
        "task": args.task,
        "length": args.length,
        "model": args.model,
        "method": model_name,
        "checkpoint": (args.malora_checkpoint or args.gated_lora_checkpoint
                       or args.lora_checkpoint or args.madora_checkpoint),
        "num_samples": len(results),
        "mean_score": avg_score,
        "per_sample": results,
    }
    import os
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.out_json}")
    print(f"\nEvaluation complete!\n")


if __name__ == "__main__":
    main()
