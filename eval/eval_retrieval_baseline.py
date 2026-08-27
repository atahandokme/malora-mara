#!/usr/bin/env python3
"""Zero-shot retrieval baselines -> recall@k (no training, no generation).

Presets:
  e5         intfloat/e5-large-v2        mean pool, 'query:'/'passage:' prefixes
  contriever facebook/contriever         mean pool, no prefix
  qwen3emb   Qwen/Qwen3-Embedding-8B     last-token pool, instruction prefix
  frozen-lm  <--backbone> (decoder LM)   last-token pool of the SAME backbone (model-attached cosine)

For each val example: encode query + each candidate paragraph, cosine-rank,
report mean recall@{4,8,12} of the gold supporting paragraphs.
"""

# --- repo root on sys.path (entry points live one level below it) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "chunking", _ROOT / "mara", _ROOT / "train", _ROOT / "eval"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------

import argparse, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

PRESETS = {
    "e5":         dict(model="intfloat/e5-large-v2",      pool="mean", qpre="query: ",   ppre="passage: "),
    "contriever": dict(model="facebook/contriever",        pool="mean", qpre="",          ppre=""),
    "qwen3emb":   dict(model="Qwen/Qwen3-Embedding-8B",    pool="last", qpre="Instruct: Given a question, retrieve passages that answer the question.\nQuery: ", ppre=""),
    "frozen-lm":  dict(model=None,                          pool="last", qpre="",          ppre=""),
    "bm25":       dict(model=None,                          pool=None,   qpre="",          ppre=""),  # sparse, no GPU
}

def mean_pool(h, mask):
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-6)

def last_pool(h, mask):
    # works for left or right padding
    left = (mask[:, -1].sum() == mask.size(0))
    if left:
        return h[:, -1]
    idx = mask.sum(1) - 1
    return h[torch.arange(h.size(0)), idx]

@torch.no_grad()
def encode(enc, tok, texts, pool, max_length, device):
    b = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    out = enc(**b)
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    e = mean_pool(h, b["attention_mask"]) if pool == "mean" else last_pool(h, b["attention_mask"])
    return F.normalize(e.float(), p=2, dim=1)

def load_ds(name):
    if name == "musique":
        return load_dataset("dgslibisey/MuSiQue", split="validation"), 20
    if name == "twowikimultihopqa":
        from data.twowikimultihopqa import _load_2wiki
        return _load_2wiki("validation"), 10
    return load_dataset("hotpot_qa", "distractor", split="validation"), 10

def extract(ex, name, npar):
    """-> (question, [paragraph_texts], [gold_support_idx]) or None to skip."""
    if name == "musique":
        if not ex.get("answerable", True): return None
        pp = ex["paragraphs"]
        if len(pp) != npar: return None
        texts = [f"{p['title']}: {p['paragraph_text']}" for p in pp]
        gold = [i for i, p in enumerate(pp) if p["is_supporting"]]
    elif name == "twowikimultihopqa":
        ctx = ex.get("context") or []
        if len(ctx) != npar: return None
        titles = [c[0] for c in ctx]; sents = [c[1] for c in ctx]
        texts = [f"{t}: {' '.join(s)}" for t, s in zip(titles, sents)]
        sf = ex.get("supporting_facts", [])
        supp = {s["title"] for s in sf} if (sf and isinstance(sf[0], dict)) else {s[0] for s in sf}
        gold = [i for i, t in enumerate(titles) if t in supp]
    else:
        c = ex["context"]
        titles = c["title"]; sents = c["sentences"]
        if len(titles) != npar: return None
        texts = [f"{titles[j]}: {' '.join(sents[j])}" for j in range(npar)]
        supp = set(ex["supporting_facts"]["title"])
        gold = [i for i, t in enumerate(titles) if t in supp]
    if not gold: return None
    return ex["question"], texts, gold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=list(PRESETS))
    ap.add_argument("--backbone", default=None, help="for frozen-lm: HF id of the decoder LM")
    ap.add_argument("--dataset", required=True, choices=["musique", "twowikimultihopqa", "hotpotqa"])
    ap.add_argument("--n-samples", type=int, default=3000)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    cfg = PRESETS[a.preset]
    is_bm25 = a.preset == "bm25"
    model_id = "bm25"
    if is_bm25:
        from rank_bm25 import BM25Okapi
        print(f"[ret] preset=bm25 (sparse, no GPU) ds={a.dataset}")
    else:
        model_id = a.backbone if a.preset == "frozen-lm" else cfg["model"]
        assert model_id, "frozen-lm requires --backbone"
        print(f"[ret] preset={a.preset} model={model_id} pool={cfg['pool']} ds={a.dataset}")
        tok = AutoTokenizer.from_pretrained(model_id, padding_side="left", trust_remote_code=True)
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
        enc = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()

    ds, npar = load_ds(a.dataset)
    n = min(a.n_samples, len(ds))
    Ks = [4, 8, 12]; rec = {k: [] for k in Ks}; done = 0; t0 = time.time()
    for i in range(n):
        r = extract(ds[i], a.dataset, npar)
        if r is None: continue
        q, texts, gold = r
        if is_bm25:
            bm = BM25Okapi([t.lower().split() for t in texts])
            scores = bm.get_scores(q.lower().split())
            order = sorted(range(len(texts)), key=lambda j: scores[j], reverse=True)
        else:
            qv = encode(enc, tok, [cfg["qpre"] + q], cfg["pool"], a.max_length, "cuda")
            pv = encode(enc, tok, [cfg["ppre"] + t for t in texts], cfg["pool"], a.max_length, "cuda")
            order = (qv @ pv.T)[0].argsort(descending=True).cpu().tolist()
        gs = set(gold)
        for k in Ks:
            rec[k].append(sum(1 for j in order[:k] if j in gs) / len(gs))
        done += 1
        if done % 200 == 0:
            print(f"[ret] {done} ({i+1}/{n}) elapsed={time.time()-t0:.0f}s "
                  + " ".join(f"R@{k}={sum(rec[k])/len(rec[k]):.4f}" for k in Ks))
    res = {f"R@{k}": round(sum(rec[k]) / len(rec[k]), 4) for k in Ks}
    res.update(n_eval=done, preset=a.preset, model=model_id, dataset=a.dataset)
    print(f"[ret] DONE {a.preset}/{a.dataset}: {res}")
    if a.out_json:
        import json; json.dump(res, open(a.out_json, "w"), indent=2)

if __name__ == "__main__":
    main()
