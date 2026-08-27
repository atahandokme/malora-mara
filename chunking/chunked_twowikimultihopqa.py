"""
Chunked 2WikiMultihopQA dataset.

Mirrors ChunkedHotpotQADataset (10 paragraphs, supporting_facts → is_supporting
matched by title). 2Wiki context format is list of [title, [sentences]] tuples
and supporting_facts is list of [title, sent_id] pairs.

Sample selection mirrors what the *future stratified-by-type* 2Wiki LoRA
training will use (same scheme as MuSiQue but bucket by `type`):
  - linear scan: skip empty answer
  - filter token_len(full_text, add_special_tokens=True) > lora_max_length
  - bucket by `type` ∈ {compositional, comparison, inference, bridge_comparison}
  - Random(lora_seed)-shuffle each bucket, take proportional n_select per
    bucket, then Random(lora_seed)-shuffle the final list

Items that pass the LoRA filter but fail chunk-detect are dropped (router
cannot use them) but no NEW items are pulled in to fill — the router trains
on a strict subset of LoRA's 10k. Selected example IDs are saved to
<manifest_path> if provided.

To reproduce the OLD linear-first-N (legacy 2Wiki LoRA) behavior, pass
`stratify_by_type=False`.
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.twowikimultihopqa import format_twowikimultihopqa_prompt, _load_2wiki
sys.path.insert(0, str(Path(__file__).parent))
from chunk_detector import detect_hotpotqa_chunks


class ChunkedTwoWikiMultihopQADataset(Dataset):
    """2WikiMultihopQA with per-example chunk_ends + is_supporting precomputed."""

    def __init__(self, tokenizer, max_length=4096, split="train", max_samples=None,
                 shuffle_paragraphs=False, shuffle_seed=0,
                 lora_max_length=2048, lora_seed=42,
                 stratify_by_type=True, manifest_path=None,
                 lora_pool_path=None):
        assert tokenizer.is_fast, "Need a fast tokenizer (return_offsets_mapping)"
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_paragraphs = shuffle_paragraphs
        rng_p = random.Random(shuffle_seed)

        # Canonical pool: load from JSON instead of HF + filter
        if lora_pool_path is not None and split == "train":
            print(f"Loading 2WikiMultihopQA from canonical pool {lora_pool_path}")
            with open(lora_pool_path) as _f:
                _pool = json.load(_f)
            ds_iter = [
                {
                    "_id": ex.get("id", ""),
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "context": ex["context"],
                    "supporting_facts": ex.get("supporting_facts", []),
                    "type": ex.get("type", "unknown"),
                }
                for ex in _pool["examples"]
            ]
            early_stop_on_max = False
        elif stratify_by_type and max_samples is not None and split == "train":
            print(f"Loading 2WikiMultihopQA {split} split "
                  f"(stratify_by_type={stratify_by_type}, lora_max_length={lora_max_length}) ...")
            ds = _load_2wiki(split if split != "validation" else "dev")
            ds_iter = self._select_lora_aligned(
                ds, tokenizer, max_samples, lora_max_length, lora_seed,
                manifest_path,
            )
            early_stop_on_max = False
        else:
            print(f"Loading 2WikiMultihopQA {split} split (legacy linear, lora_max_length={lora_max_length}) ...")
            ds = _load_2wiki(split if split != "validation" else "dev")
            ds_iter = ds
            early_stop_on_max = True

        self.examples = []
        n_skipped_noanswer = 0
        n_skipped_chunk = 0
        n_skipped_toolong = 0
        n_skipped_toolong_lora = 0
        n_kept = 0

        for item in ds_iter:
            if early_stop_on_max and max_samples is not None and len(self.examples) >= max_samples:
                break
            q = item["question"]
            a = item.get("answer", "")
            if not a or not a.strip():
                n_skipped_noanswer += 1
                continue

            ctx = item["context"]
            if isinstance(ctx, dict) and "title" in ctx:
                context = list(zip(ctx["title"], ctx["sentences"]))
            else:
                context = [(c[0], c[1]) for c in ctx]

            paragraph_titles = [t for t, _ in context]
            sf = item.get("supporting_facts", [])
            if sf and isinstance(sf[0], dict):
                supporting_titles = {s["title"] for s in sf}
            else:
                supporting_titles = {s[0] for s in sf}
            is_supporting = [t in supporting_titles for t in paragraph_titles]

            text = format_twowikimultihopqa_prompt(q, context, answer=a)

            # In legacy linear mode, apply LoRA filter here. In stratified mode,
            # the filter was already applied by _select_lora_aligned, so we skip.
            if not stratify_by_type:
                lora_token_len = len(tokenizer.encode(text, add_special_tokens=True))
                if lora_token_len > lora_max_length:
                    n_skipped_toolong_lora += 1
                    continue

            n_kept += 1

            if shuffle_paragraphs and len(context) > 1:
                perm = list(range(len(context)))
                rng_p.shuffle(perm)
                context = [context[i] for i in perm]
                is_supporting = [is_supporting[i] for i in perm]
                text = format_twowikimultihopqa_prompt(q, context, answer=a)

            enc = tokenizer(
                text, return_tensors="pt", return_offsets_mapping=True,
                truncation=True, max_length=max_length,
            )
            input_ids = enc["input_ids"][0]
            offsets = enc["offset_mapping"][0].tolist()

            if offsets[-1][1] < len(text) - 2:
                n_skipped_toolong += 1
                continue

            try:
                chunks = detect_hotpotqa_chunks(text, offsets)
            except AssertionError:
                n_skipped_chunk += 1
                continue

            chunk_ends = [c["end_tok"] for c in chunks]
            chunk_names = [c["name"] for c in chunks]

            n_p_chunks = sum(1 for n in chunk_names if n.startswith("P"))
            if n_p_chunks != 10 or len(context) != 10:
                n_skipped_chunk += 1
                continue

            answer_match = re.search(r"\nAnswer:\s+", text)
            if answer_match is None:
                n_skipped_chunk += 1
                continue
            answer_start_char = answer_match.end()
            answer_start_tok = None
            # Use end > answer_start_char so leading-space tokens (e.g., " yes")
            # whose start is BEFORE the answer character are correctly captured.
            for i, (s, e) in enumerate(offsets):
                if e > answer_start_char:
                    answer_start_tok = i
                    break
            if answer_start_tok is None:
                n_skipped_chunk += 1
                continue

            eos_id = tokenizer.eos_token_id
            if eos_id is not None:
                input_ids = torch.cat([
                    input_ids, torch.tensor([eos_id], dtype=input_ids.dtype),
                ])
            answer_tokens_alone = tokenizer.encode(a, add_special_tokens=False)
            answer_plus_eos_len = len(answer_tokens_alone) + (1 if eos_id is not None else 0)
            answer_start_tok = max(0, len(input_ids) - answer_plus_eos_len)

            labels = torch.full_like(input_ids, -100)
            labels[answer_start_tok:] = input_ids[answer_start_tok:]

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": labels,
                "chunk_ends": chunk_ends,
                "chunk_names": chunk_names,
                "is_supporting": is_supporting,
                "answer_text": a,
                "question": q,
            })

        print(f"  Loaded {len(self.examples)} examples")
        if n_skipped_noanswer:        print(f"  Skipped (no answer): {n_skipped_noanswer}")
        if n_skipped_chunk:           print(f"  Skipped (chunk detect failed): {n_skipped_chunk}")
        if n_skipped_toolong:         print(f"  Skipped (truncated at chunked-tokenize): {n_skipped_toolong}")
        if n_skipped_toolong_lora:    print(f"  Skipped (LoRA filter too-long): {n_skipped_toolong_lora}")

    @staticmethod
    def _select_lora_aligned(ds, tokenizer, max_samples, lora_max_length, lora_seed,
                             manifest_path):
        """Stratified-by-type selection mirroring future 2Wiki LoRA training.

        2Wiki has 4 types: compositional, comparison, inference, bridge_comparison.
        Bucket by type, Random(seed) shuffle each bucket, take proportional
        n_select, then Random(seed) shuffle the final list. Same scheme as
        MuSiQue's stratified-by-hop logic.
        """
        print(f"  [LoRA-aligned 2Wiki] filtering token<={lora_max_length}, "
              f"stratified by type, Random({lora_seed}) shuffle ...")
        type_buckets = defaultdict(list)
        n_noanswer = 0
        n_toolong = 0
        for item in ds:
            q = item.get("question", "")
            a = item.get("answer", "")
            if not a or not a.strip():
                n_noanswer += 1
                continue
            ctx = item["context"]
            if isinstance(ctx, dict) and "title" in ctx:
                context = list(zip(ctx["title"], ctx["sentences"]))
            else:
                context = [(c[0], c[1]) for c in ctx]
            text = format_twowikimultihopqa_prompt(q, context, answer=a)
            token_len = len(tokenizer.encode(text, add_special_tokens=True))
            if token_len > lora_max_length:
                n_toolong += 1
                continue
            t = item.get("type", "unknown")
            type_buckets[t].append(item)

        total_available = sum(len(v) for v in type_buckets.values())
        rng = random.Random(lora_seed)
        selected = []
        for type_name in sorted(type_buckets.keys()):
            bucket = type_buckets[type_name]
            rng.shuffle(bucket)
            n_select = int(max_samples * len(bucket) / total_available)
            n_select = max(1, min(n_select, len(bucket)))
            selected.extend(bucket[:n_select])
        rng.shuffle(selected)

        print(f"  [LoRA-aligned 2Wiki] LoRA pool: {total_available} "
              f"(skipped {n_noanswer}na/{n_toolong}long)")
        print(f"  [LoRA-aligned 2Wiki] selected {len(selected)} from buckets: "
              f"{ {k: len(v) for k, v in type_buckets.items()} }")

        if manifest_path is not None:
            ids = [it.get("_id", "") for it in selected]
            Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w") as f:
                json.dump({
                    "dataset": "twowikimultihopqa",
                    "tokenizer": getattr(tokenizer, "name_or_path", "unknown"),
                    "lora_max_length": lora_max_length,
                    "lora_seed": lora_seed,
                    "stratify_by_type": True,
                    "n_selected": len(selected),
                    "ids": ids,
                }, f, indent=2)
            print(f"  [LoRA-aligned 2Wiki] wrote manifest to {manifest_path}")
        return selected

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
