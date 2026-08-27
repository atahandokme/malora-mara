"""
Dump canonical 10k pools where every example passes:
  - max(qwen_tokens, llama_tokens) ≤ max_length         (both tokenizers ok)
  - chunk-detect succeeds (router-compatible)

Same 10k usable by Qwen LoRA, Llama LoRA, and the router → 100% overlap and
100% utilization. No router-side data loss.

Selection: stratified by num_hops (MQ) / type (2Wiki) + Random(42),
proportional n_select per bucket.

Outputs:
  data/preprocessed_10k/musique_canonical_n10000.json
  data/preprocessed_10k/twowikimultihopqa_canonical_n10000.json
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from data.musique import format_musique_prompt
from data.twowikimultihopqa import format_twowikimultihopqa_prompt, _load_2wiki
sys.path.insert(0, str(ROOT / "chunking"))
from chunk_detector import detect_hotpotqa_chunks


def _chunk_detect_ok(text, tokenizer, expected_n_p):
    """Return True iff tokenizer sees expected_n_p paragraph chunks + Q + A
    that the chunk_detector can resolve correctly."""
    try:
        enc = tokenizer(
            text, return_tensors="pt", return_offsets_mapping=True,
            truncation=False, add_special_tokens=True,
        )
        offsets = enc["offset_mapping"][0].tolist()
        chunks = detect_hotpotqa_chunks(text, offsets)
        names = [c["name"] for c in chunks]
        n_p = sum(1 for n in names if n.startswith("P"))
        return n_p == expected_n_p
    except Exception:
        return False

OUT_DIR = ROOT / "data" / "preprocessed_10k"
QWEN = "Qwen/Qwen2.5-7B"
LLAMA = "meta-llama/Llama-3.1-8B"
SEED = 42
N = 10000


def dump_musique_canonical(max_length=4096):
    print(f"\n=== MuSiQue canonical (max_length={max_length} for BOTH tokenizers) ===")
    tok_q = AutoTokenizer.from_pretrained(QWEN, use_fast=True)
    tok_l = AutoTokenizer.from_pretrained(LLAMA, use_fast=True)
    ds = load_dataset("dgslibisey/MuSiQue", split="train")

    hop_buckets = defaultdict(list)
    n_unanswerable = n_noanswer = n_qwen_only = n_llama_only = n_both_long = 0
    n_chunk_q = n_chunk_l = 0
    for item in ds:
        if not item.get("answerable", True):
            n_unanswerable += 1
            continue
        a = item.get("answer", "")
        if not a or not a.strip():
            n_noanswer += 1
            continue
        full_text = format_musique_prompt(item["question"], item["paragraphs"], a)
        q_len = len(tok_q.encode(full_text, add_special_tokens=True))
        l_len = len(tok_l.encode(full_text, add_special_tokens=True))
        if q_len > max_length and l_len > max_length:
            n_both_long += 1
            continue
        if q_len > max_length:
            n_qwen_only += 1
            continue
        if l_len > max_length:
            n_llama_only += 1
            continue
        # token-length OK; now check chunk-detect with both tokenizers
        if not _chunk_detect_ok(full_text, tok_q, expected_n_p=20):
            n_chunk_q += 1
            continue
        if not _chunk_detect_ok(full_text, tok_l, expected_n_p=20):
            n_chunk_l += 1
            continue
        # passes all filters
        num_hops = len(item.get("question_decomposition", []))
        hop_buckets[num_hops].append(item)

    total = sum(len(v) for v in hop_buckets.values())
    rng = random.Random(SEED)
    selected = []
    for hop_count in sorted(hop_buckets.keys()):
        bucket = hop_buckets[hop_count]
        rng.shuffle(bucket)
        n_select = int(N * len(bucket) / total)
        n_select = max(1, min(n_select, len(bucket)))
        selected.extend(bucket[:n_select])
    rng.shuffle(selected)

    print(f"  pool (passes both tokenizers + chunk-detect): {total}")
    print(f"  buckets: { {k: len(v) for k, v in hop_buckets.items()} }")
    print(f"  filter drops: {n_unanswerable}u/{n_noanswer}na/{n_both_long}both_long/"
          f"{n_qwen_only}qwen_only_long/{n_llama_only}llama_only_long/"
          f"{n_chunk_q}chunk_qwen/{n_chunk_l}chunk_llama")
    print(f"  selected: {len(selected)}")

    payload = {
        "dataset": "musique",
        "tokenizers": [QWEN, LLAMA],
        "max_length": max_length,
        "filter_rule": "max(qwen_tokens, llama_tokens) <= max_length AND chunk_detect_ok(P20+Q+A) for both tokenizers",
        "max_samples": N,
        "seed": SEED,
        "selection": "stratified_by_num_hops_random42_canonical",
        "n_selected": len(selected),
        "examples": [
            {
                "id": it.get("id", ""),
                "question": it["question"],
                "answer": it["answer"],
                "answer_aliases": it.get("answer_aliases", []),
                "paragraphs": it["paragraphs"],
                "num_hops": len(it.get("question_decomposition", [])),
                "question_decomposition": it.get("question_decomposition", []),
            }
            for it in selected
        ],
    }
    out = OUT_DIR / f"musique_canonical_n{N}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {out} ({out.stat().st_size:,} bytes)")
    return out


def dump_twowiki_canonical(max_length=2048):
    print(f"\n=== 2WikiMultihopQA canonical (max_length={max_length} for BOTH tokenizers) ===")
    tok_q = AutoTokenizer.from_pretrained(QWEN, use_fast=True)
    tok_l = AutoTokenizer.from_pretrained(LLAMA, use_fast=True)
    ds = _load_2wiki("train")

    type_buckets = defaultdict(list)
    n_noanswer = n_qwen_only = n_llama_only = n_both_long = 0
    n_chunk_q = n_chunk_l = 0
    for item in ds:
        a = item.get("answer", "")
        if not a or not a.strip():
            n_noanswer += 1
            continue
        ctx = item["context"]
        context = [(c[0], c[1]) for c in ctx] if isinstance(ctx, list) else list(zip(ctx["title"], ctx["sentences"]))
        text = format_twowikimultihopqa_prompt(item["question"], context, a)
        q_len = len(tok_q.encode(text, add_special_tokens=True))
        l_len = len(tok_l.encode(text, add_special_tokens=True))
        if q_len > max_length and l_len > max_length:
            n_both_long += 1
            continue
        if q_len > max_length:
            n_qwen_only += 1
            continue
        if l_len > max_length:
            n_llama_only += 1
            continue
        if not _chunk_detect_ok(text, tok_q, expected_n_p=10):
            n_chunk_q += 1
            continue
        if not _chunk_detect_ok(text, tok_l, expected_n_p=10):
            n_chunk_l += 1
            continue
        t = item.get("type", "unknown")
        type_buckets[t].append(item)

    total = sum(len(v) for v in type_buckets.values())
    rng = random.Random(SEED)
    selected = []
    for type_name in sorted(type_buckets.keys()):
        bucket = type_buckets[type_name]
        rng.shuffle(bucket)
        n_select = int(N * len(bucket) / total)
        n_select = max(1, min(n_select, len(bucket)))
        selected.extend(bucket[:n_select])
    rng.shuffle(selected)

    print(f"  pool (passes both tokenizers + chunk-detect): {total}")
    print(f"  buckets: { {k: len(v) for k, v in type_buckets.items()} }")
    print(f"  filter drops: {n_noanswer}na/{n_both_long}both_long/"
          f"{n_qwen_only}qwen_only_long/{n_llama_only}llama_only_long/"
          f"{n_chunk_q}chunk_qwen/{n_chunk_l}chunk_llama")
    print(f"  selected: {len(selected)}")

    payload = {
        "dataset": "twowikimultihopqa",
        "tokenizers": [QWEN, LLAMA],
        "max_length": max_length,
        "filter_rule": "max(qwen_tokens, llama_tokens) <= max_length AND chunk_detect_ok(P10+Q+A) for both tokenizers",
        "max_samples": N,
        "seed": SEED,
        "selection": "stratified_by_type_random42_canonical",
        "n_selected": len(selected),
        "examples": [
            {
                "id": it.get("_id", ""),
                "question": it["question"],
                "answer": it["answer"],
                "context": it["context"],
                "supporting_facts": it.get("supporting_facts", []),
                "evidences": it.get("evidences", []),
                "type": it.get("type", "unknown"),
            }
            for it in selected
        ],
    }
    out = OUT_DIR / f"twowikimultihopqa_canonical_n{N}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {out} ({out.stat().st_size:,} bytes)")
    return out


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dump_musique_canonical()
    dump_twowiki_canonical()
