#!/usr/bin/env python3
"""Trained OUTSOURCED retriever baseline: fine-tune a dense bi-encoder (E5-large)
on the supporting-paragraph supervision. Fills the 'trained + outsourced' cell.

Bi-encoder: q = enc('query: '+Q), p_i = enc('passage: '+P_i), score = q.p_i.
Loss: multi-positive InfoNCE over the per-example candidates (positives = supporting,
negatives = that example's distractors = hard negatives). Same labels as the router.
Reports recall@{4,8,12} on the same val-500 + gold as eval_retrieval_baseline.py.
"""

# --- repo root on sys.path (entry points live one level below it) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "chunking", _ROOT / "mara", _ROOT / "train", _ROOT / "eval"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------

import argparse, time, torch, torch.nn.functional as F
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from eval_retrieval_baseline import load_ds, extract, mean_pool

def encode(enc, tok, texts, max_length, device):
    b = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    out = enc(**b)
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return F.normalize(mean_pool(h, b["attention_mask"]).float(), p=2, dim=1)

@torch.no_grad()
def evaluate(enc, tok, val, max_length, device):
    enc.eval()
    Ks = [4, 8, 12]; rec = {k: [] for k in Ks}
    for q, texts, gold in val:
        qv = encode(enc, tok, ["query: " + q], max_length, device)
        pv = encode(enc, tok, ["passage: " + t for t in texts], max_length, device)
        order = (qv @ pv.T)[0].argsort(descending=True).cpu().tolist()
        gs = set(gold)
        for k in Ks:
            rec[k].append(sum(1 for j in order[:k] if j in gs) / len(gs))
    enc.train()
    return {f"R@{k}": round(sum(rec[k]) / len(rec[k]), 4) for k in Ks}

def build(split_ds, name, npar, limit):
    out = []
    for i in range(len(split_ds)):
        if len(out) >= limit: break
        r = extract(split_ds[i], name, npar)
        if r: out.append(r)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["musique", "twowikimultihopqa", "hotpotqa"])
    ap.add_argument("--base-model", default="intfloat/e5-large-v2")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--train-samples", type=int, default=10000)
    ap.add_argument("--val-samples", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(a.base_model)
    enc = AutoModel.from_pretrained(a.base_model, torch_dtype=torch.float32).to(dev)
    opt = torch.optim.AdamW(enc.parameters(), lr=a.lr, weight_decay=0.01)

    # train split + val-500 (val uses the same loader/gold as eval_retrieval_baseline)
    if a.dataset == "musique":
        train_raw = load_dataset("dgslibisey/MuSiQue", split="train"); npar = 20
    elif a.dataset == "twowikimultihopqa":
        from data.twowikimultihopqa import _load_2wiki
        train_raw = _load_2wiki("train"); npar = 10
    else:
        train_raw = load_dataset("hotpot_qa", "distractor", split="train"); npar = 10
    val_raw, _ = load_ds(a.dataset)

    train = build(train_raw, a.dataset, npar, a.train_samples)
    val = build(val_raw, a.dataset, npar, a.val_samples)
    print(f"[dpr] {a.dataset}: train={len(train)} val={len(val)} npar={npar}")

    best = {"R@4": -1}; t0 = time.time()
    for ep in range(a.epochs):
        enc.train()
        for step, (q, texts, gold) in enumerate(train):
            qv = encode(enc, tok, ["query: " + q], a.max_length, dev)             # (1,d)
            pv = encode(enc, tok, ["passage: " + t for t in texts], a.max_length, dev)  # (N,d)
            scores = (qv @ pv.T)[0] / a.temp                                       # (N,)
            logp = F.log_softmax(scores, dim=0)
            loss = -logp[gold].mean()                                             # multi-positive InfoNCE
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 500 == 0:
                print(f"[dpr] ep{ep} step{step} loss={loss.item():.4f} ({time.time()-t0:.0f}s)")
        m = evaluate(enc, tok, val, a.max_length, dev)
        print(f"[dpr] ep{ep} val {m}")
        if m["R@4"] > best["R@4"]:
            best = {**m, "epoch": ep}
            enc.save_pretrained(a.out_dir); tok.save_pretrained(a.out_dir)
    best.update(dataset=a.dataset, base_model=a.base_model, n_val=len(val))
    print(f"[dpr] DONE {a.dataset}: BEST {best}")
    if a.out_json:
        import json; json.dump(best, open(a.out_json, "w"), indent=2)

if __name__ == "__main__":
    main()
