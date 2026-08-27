"""
Standalone 5-shot eval for gsm-hard and GSM-Symbolic variants.

Few-shot exemplars come from GSM8k train (with chain-of-thought).
Test set comes from gsm-hard or apple/GSM-Symbolic.
Generation: greedy, max_new_tokens=512.
Metric: strict-match exact_match on the final number after "####".

Usage:
  python eval_gsm_variants.py --model Qwen/Qwen2.5-7B --dataset gsm-hard --out logs/eval_gsm_hard_qwen.json
  python eval_gsm_variants.py --model meta-llama/Llama-3.1-8B --dataset gsm-symbolic-main --out ...
"""

# --- repo root on sys.path (entry points live one level below it) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "chunking", _ROOT / "mara", _ROOT / "train", _ROOT / "eval"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------

import argparse, json, os, re, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ANSWER_RE_STRICT = re.compile(r"#### *(-?[0-9][0-9,\.]*)")
ANSWER_RE_FLEX   = re.compile(r"(-?[0-9][0-9,\.]*)")

def normalize(s):
    return s.replace(",", "").rstrip(".").strip()

def extract_strict(text):
    m = ANSWER_RE_STRICT.search(text)
    return normalize(m.group(1)) if m else None

def extract_flex(text):
    m = list(ANSWER_RE_FLEX.finditer(text))
    return normalize(m[-1].group(1)) if m else None

def numeric_eq(pred, gold):
    if pred is None: return False
    try: return abs(float(pred) - float(gold)) < 1e-4
    except: return pred == gold

def load_test(name):
    """Return list of {question, gold_str}."""
    if name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        out = []
        for ex in ds:
            ans = ex["answer"]
            m = ANSWER_RE_STRICT.search(ans)
            gold = normalize(m.group(1)) if m else None
            if gold is not None:
                out.append({"question": ex["question"], "gold": gold})
        return out
    elif name == "gsm-hard":
        ds = load_dataset("reasoning-machines/gsm-hard", split="train")
        return [{"question": ex["input"], "gold": str(ex["target"])} for ex in ds]
    elif name.startswith("gsm-symbolic-"):
        sub = name.split("-", 2)[2]   # main | p1 | p2
        ds = load_dataset("apple/GSM-Symbolic", sub, split="test")
        out = []
        for ex in ds:
            ans = ex["answer"]
            m = ANSWER_RE_STRICT.search(ans)
            gold = normalize(m.group(1)) if m else None
            if gold is not None:
                out.append({"question": ex["question"], "gold": gold})
        return out
    raise ValueError(f"unknown dataset {name}")

def build_fewshot_prefix(n_shot=5, seed=0):
    """Take n_shot examples from gsm8k train as the few-shot prefix."""
    ds = load_dataset("gsm8k", "main", split="train")
    import random
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), n_shot)
    parts = []
    for i in idxs:
        ex = ds[i]
        parts.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
    return "\n\n".join(parts) + "\n\n"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True,
                   choices=["gsm8k", "gsm-hard", "gsm-symbolic-main", "gsm-symbolic-p1", "gsm-symbolic-p2"])
    p.add_argument("--lora-checkpoint", type=str, default=None,
                   help="Path to PEFT LoRA/DoRA checkpoint dir (e.g. .../lora_epoch_2)")
    p.add_argument("--malora-checkpoint", type=str, default=None,
                   help="Path to MaLoRA .pt (linear hproj or MaLoRA)")
    p.add_argument("--num-fewshot", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None, help="cap test size for debugging")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--out", required=True, help="output JSON path")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    if args.lora_checkpoint and args.malora_checkpoint:
        raise ValueError("Pass at most one of --lora-checkpoint / --malora-checkpoint")
    if args.malora_checkpoint:
        from model.malora import load_model_with_malora
        print(f"Loading model with MaLoRA from {args.malora_checkpoint}...", flush=True)
        model, tok = load_model_with_malora(
            args.model, args.malora_checkpoint, device='auto', torch_dtype=torch.bfloat16,
        )
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model.eval()
    elif args.lora_checkpoint:
        from peft import PeftModel
        print(f"Loading model with LoRA/DoRA adapter from {args.lora_checkpoint}...", flush=True)
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base, args.lora_checkpoint)
        model = model.merge_and_unload()
        model.eval()
    else:
        print(f"Loading base model {args.model}...", flush=True)
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()

    print(f"Loading test set {args.dataset}...", flush=True)
    test = load_test(args.dataset)
    if args.limit: test = test[:args.limit]
    print(f"  {len(test)} examples")

    print(f"Building {args.num_fewshot}-shot prefix from gsm8k train (seed={args.seed})...")
    prefix = build_fewshot_prefix(args.num_fewshot, args.seed)
    print(f"  prefix length: {len(tok(prefix).input_ids)} tokens")

    # Build all prompts (left-padded for generation)
    prompts = [prefix + f"Question: {ex['question']}\nAnswer:" for ex in test]
    results = []
    n_strict = 0; n_flex = 0
    bs = args.batch_size
    pbar = tqdm(range(0, len(prompts), bs), desc=args.dataset)
    for start in pbar:
        batch = prompts[start:start+bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
        gen = out[:, enc.input_ids.shape[1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        for i, txt in enumerate(texts):
            # Truncate at next "Question:" so we only score this answer
            txt = txt.split("Question:")[0]
            gold = test[start+i]["gold"]
            ps = extract_strict(txt); pf = extract_flex(txt)
            ok_s = numeric_eq(ps, gold); ok_f = numeric_eq(pf, gold)
            n_strict += int(ok_s); n_flex += int(ok_f)
            results.append({"question": test[start+i]["question"], "gold": gold,
                            "pred_strict": ps, "pred_flex": pf,
                            "em_strict": int(ok_s), "em_flex": int(ok_f),
                            "completion": txt})
        done = start + len(batch)
        pbar.set_postfix(strict=f"{n_strict/done:.3f}", flex=f"{n_flex/done:.3f}")

    elapsed = time.time() - t0
    summary = {"model": args.model, "dataset": args.dataset, "num_fewshot": args.num_fewshot,
               "n_examples": len(results), "em_strict": n_strict / len(results),
               "em_flex": n_flex / len(results), "elapsed_s": elapsed,
               "results_path": args.out + ".jsonl"}
    with open(args.out, "w") as f: json.dump(summary, f, indent=2)
    with open(args.out + ".jsonl", "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    print("\n=== DONE ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
