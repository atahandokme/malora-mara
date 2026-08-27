"""
Chunked MuSiQue dataset.

Same shape as ChunkedHotpotQADataset, but for MuSiQue (20 paragraphs, native
is_supporting flag per paragraph).

Each item:
    input_ids       : (T,) long
    attention_mask  : (T,) long
    labels          : (T,) long, -100 except answer span
    chunk_ends      : List[int] — last-token index per chunk (P1..P20, Q, A)
    chunk_names     : List[str] — ['P1', ..., 'P20', 'Q', 'A']
    is_supporting   : List[bool] of length 20 (P chunks only, Q/A excluded)

When `match_lora_samples=True` (default), sample selection mirrors
data/musique.py used for the MuSiQue LoRA checkpoints:
  - linear scan: skip unanswerable, skip empty answer
  - filter token_len(format_musique_prompt(...), add_special_tokens=True) > lora_max_length
  - bucket by num_hops, Random(lora_seed)-shuffle each bucket, take
    proportional n_select per bucket, then Random(lora_seed)-shuffle the
    final list
Items that pass the LoRA filter but fail chunk-detect are still dropped
(router cannot use them) but no NEW items are pulled in to fill — the router
trains on a strict subset of LoRA's 10k. The selected example IDs are saved
to <manifest_path> if provided.
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.musique import format_musique_prompt
sys.path.insert(0, str(Path(__file__).parent))
from chunk_detector import detect_hotpotqa_chunks


class ChunkedMuSiQueDataset(Dataset):
    """MuSiQue with per-example chunk_ends + is_supporting precomputed."""

    def __init__(self, tokenizer, max_length=4096, split="train", max_samples=None,
                 shuffle_paragraphs=False, shuffle_seed=0,
                 match_lora_samples=True, lora_max_length=4096, lora_seed=42,
                 manifest_path=None, lora_pool_path=None):
        assert tokenizer.is_fast, "Need a fast tokenizer (return_offsets_mapping)"
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_paragraphs = shuffle_paragraphs
        rng = random.Random(shuffle_seed)

        # Canonical pool: load from JSON instead of HF + filter
        if lora_pool_path is not None and split == "train":
            print(f"Loading MuSiQue from canonical pool {lora_pool_path}")
            with open(lora_pool_path) as _f:
                _pool = json.load(_f)
            ds_iter = [
                {
                    "id": ex.get("id", ""),
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "answerable": True,
                    "paragraphs": ex["paragraphs"],
                    "question_decomposition": ex.get("question_decomposition", []),
                }
                for ex in _pool["examples"]
            ]
        elif match_lora_samples and max_samples is not None and split == "train":
            print(f"Loading MuSiQue {split} split (match_lora_samples={match_lora_samples}) ...")
            ds = load_dataset("dgslibisey/MuSiQue", split=split)
            ds_iter = self._select_lora_aligned(
                ds, tokenizer, max_samples, lora_max_length, lora_seed,
                manifest_path,
            )
        else:
            print(f"Loading MuSiQue {split} split ...")
            ds = load_dataset("dgslibisey/MuSiQue", split=split)
            ds_iter = ds

        self.examples = []
        n_skipped_unanswerable = 0
        n_skipped_noanswer = 0
        n_skipped_chunk = 0
        n_skipped_toolong = 0
        # In match_lora_samples mode, we iterate the LoRA-selected list and
        # never break early on max_samples (we keep all that pass chunk-detect).
        early_stop_on_max = not (match_lora_samples and max_samples is not None and split == "train")

        for item in ds_iter:
            if early_stop_on_max and max_samples is not None and len(self.examples) >= max_samples:
                break
            if not item.get("answerable", True):
                n_skipped_unanswerable += 1
                continue
            q = item["question"]
            a = item["answer"]
            if not a or not a.strip():
                n_skipped_noanswer += 1
                continue

            paragraphs = list(item["paragraphs"])  # list of dicts with title/paragraph_text/is_supporting
            is_supporting = [bool(p["is_supporting"]) for p in paragraphs]

            if self.shuffle_paragraphs and len(paragraphs) > 1:
                perm = list(range(len(paragraphs)))
                rng.shuffle(perm)
                paragraphs = [paragraphs[i] for i in perm]
                is_supporting = [is_supporting[i] for i in perm]

            text = format_musique_prompt(q, paragraphs, answer=a)

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

            # Hard requirement: σ-injector + tracker_head are built for fixed 20.
            # MuSiQue is ~99.9% 20-paragraph; the rare 17/18/19 cases must be
            # skipped (not just match-checked) so SigmaInjector never sees them.
            n_p_chunks = sum(1 for n in chunk_names if n.startswith("P"))
            if n_p_chunks != 20 or len(paragraphs) != 20:
                n_skipped_chunk += 1
                continue

            answer_match = re.search(r"\nAnswer:\s+", text)
            if answer_match is None:
                n_skipped_chunk += 1
                continue
            answer_start_char = answer_match.end()
            answer_start_tok = None
            for i, (s, e) in enumerate(offsets):
                if e > answer_start_char:
                    answer_start_tok = i
                    break
            if answer_start_tok is None:
                n_skipped_chunk += 1
                continue

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
        if n_skipped_unanswerable: print(f"  Skipped (unanswerable): {n_skipped_unanswerable}")
        if n_skipped_noanswer:    print(f"  Skipped (no answer): {n_skipped_noanswer}")
        if n_skipped_chunk:       print(f"  Skipped (chunk detect failed): {n_skipped_chunk}")
        if n_skipped_toolong:     print(f"  Skipped (truncated): {n_skipped_toolong}")

    @staticmethod
    def _select_lora_aligned(ds, tokenizer, max_samples, lora_max_length, lora_seed,
                             manifest_path):
        """Replicate data/musique.py:MuSiQueDataset selection logic.

        Returns: list of HF rows, in the same order LoRA training saw them.
        """
        print(f"  [LoRA-aligned MQ] filtering with token<={lora_max_length}, "
              f"stratified by hop, Random({lora_seed}) shuffle ...")
        hop_buckets = defaultdict(list)
        n_lora_unanswerable = 0
        n_lora_noanswer = 0
        n_lora_toolong = 0
        for item in ds:
            if not item.get("answerable", True):
                n_lora_unanswerable += 1
                continue
            q = item["question"]
            a = item["answer"]
            if not a or not a.strip():
                n_lora_noanswer += 1
                continue
            paragraphs = item["paragraphs"]
            full_text = format_musique_prompt(q, paragraphs, answer=a)
            token_len = len(tokenizer.encode(full_text, add_special_tokens=True))
            if token_len > lora_max_length:
                n_lora_toolong += 1
                continue
            num_hops = len(item.get("question_decomposition", []))
            hop_buckets[num_hops].append(item)

        total_available = sum(len(v) for v in hop_buckets.values())
        rng = random.Random(lora_seed)
        selected = []
        for hop_count in sorted(hop_buckets.keys()):
            bucket = hop_buckets[hop_count]
            rng.shuffle(bucket)
            n_select = int(max_samples * len(bucket) / total_available)
            n_select = max(1, min(n_select, len(bucket)))
            selected.extend(bucket[:n_select])
        rng.shuffle(selected)

        print(f"  [LoRA-aligned MQ] LoRA pool: {total_available} (skipped {n_lora_unanswerable}u/{n_lora_noanswer}na/{n_lora_toolong}long)")
        print(f"  [LoRA-aligned MQ] selected {len(selected)} from buckets: "
              f"{ {k: len(v) for k, v in hop_buckets.items()} }")

        if manifest_path is not None:
            ids = [it.get("id", "") for it in selected]
            Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w") as f:
                json.dump({
                    "dataset": "musique",
                    "tokenizer": getattr(tokenizer, "name_or_path", "unknown"),
                    "lora_max_length": lora_max_length,
                    "lora_seed": lora_seed,
                    "n_selected": len(selected),
                    "ids": ids,
                }, f, indent=2)
            print(f"  [LoRA-aligned MQ] wrote manifest to {manifest_path}")
        return selected

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
