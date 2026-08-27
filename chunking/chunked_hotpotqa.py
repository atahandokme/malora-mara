"""
Chunked HotpotQA dataset.

Extends HotpotQADataset with chunk boundary info per example.
Each item has:
    input_ids       : (T,) long
    attention_mask  : (T,) long
    labels          : (T,) long, -100 everywhere except answer region
    chunk_ends      : List[int] — last-token index per chunk (ascending)
    chunk_names     : List[str] — e.g. ["P1", "P2", ..., "P10", "Q", "A"]
    is_supporting   : List[bool] — length 10, True iff that paragraph is in
                      HotpotQA's gold supporting_facts (matched by title).
                      Aligned with the P1..P10 chunks (Q/A excluded).
"""

import random
import re
import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

# path to repo root for data import
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.hotpotqa import format_hotpotqa_prompt
sys.path.insert(0, str(Path(__file__).parent))
from chunk_detector import detect_hotpotqa_chunks


class ChunkedHotpotQADataset(Dataset):
    """HotpotQA with per-example chunk_ends precomputed."""

    def __init__(self, tokenizer, max_length=4096, split="train", max_samples=None,
                 shuffle_paragraphs=False, shuffle_seed=0, dynamic_shuffle=False,
                 lora_pool_path=None):
        """Args:
            shuffle_paragraphs: per-example permutation of P1..P10 (with
                is_supporting permuted accordingly). If dynamic_shuffle is False
                the permutation is fixed per example (set at load time).
            dynamic_shuffle: if True, re-shuffle paragraphs and re-tokenize on
                every __getitem__ call. Implies shuffle_paragraphs=True.
                Reproducible across replicas via per-call rng seeded by
                shuffle_seed + idx + epoch-counter; here we use Python's
                random.Random seeded by (shuffle_seed, idx, time-derived nonce)
                so each access is fresh."""
        assert tokenizer.is_fast, "Need a fast tokenizer (return_offsets_mapping)"
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dynamic_shuffle = dynamic_shuffle
        self.shuffle_paragraphs = shuffle_paragraphs or dynamic_shuffle
        self.shuffle_seed = shuffle_seed
        static_rng = random.Random(shuffle_seed)

        # Optional: load from a pre-built canonical pool instead of streaming HF
        if lora_pool_path is not None and split == "train":
            import json as _json
            print(f"Loading HotpotQA training examples from canonical pool {lora_pool_path}")
            with open(lora_pool_path) as _f:
                _pool = _json.load(_f)
            ds = _pool.get('examples', _pool) if isinstance(_pool, dict) else _pool
            print(f"  {len(ds)} examples in canonical pool")
        else:
            print(f"Loading HotpotQA {split} split ...")
            ds = load_dataset("hotpot_qa", "distractor", split=split)

        self.examples = []        # pre-tokenized items (used when dynamic_shuffle=False)
        self.raw_items = []       # (q, context_list, is_supporting, a) for dynamic re-tokenization
        n_skipped_noanswer = 0
        n_skipped_chunk = 0
        n_skipped_toolong = 0

        for item in ds:
            if max_samples is not None and len(self.examples) >= max_samples:
                break
            q = item["question"]
            a = item["answer"]
            if not a or not a.strip():
                n_skipped_noanswer += 1
                continue

            paragraph_titles = list(item["context"]["title"])
            context = list(zip(paragraph_titles, item["context"]["sentences"]))

            supporting_titles = set(item["supporting_facts"]["title"])
            is_supporting = [t in supporting_titles for t in paragraph_titles]

            if self.shuffle_paragraphs and not self.dynamic_shuffle and len(context) > 1:
                perm = list(range(len(context)))
                static_rng.shuffle(perm)
                context = [context[i] for i in perm]
                is_supporting = [is_supporting[i] for i in perm]
                paragraph_titles = [paragraph_titles[i] for i in perm]

            built = self._build_item(q, context, is_supporting, a)
            if built is None:
                # _build_item returned None — figure out which counter to bump
                # based on its own logic; we just bump the chunk counter for now.
                n_skipped_chunk += 1
                continue

            # If dynamic_shuffle, also stash raw context for re-tokenization
            self.raw_items.append({
                "q": q, "context": context, "is_supporting": is_supporting, "a": a,
            })
            self.examples.append(built)
            continue

        print(f"  Loaded {len(self.examples)} examples")
        if n_skipped_noanswer: print(f"  Skipped (no answer): {n_skipped_noanswer}")
        if n_skipped_chunk:   print(f"  Skipped (chunk detect failed): {n_skipped_chunk}")
        if n_skipped_toolong: print(f"  Skipped (truncated): {n_skipped_toolong}")
        if self.dynamic_shuffle:
            print(f"  Dynamic per-getitem paragraph shuffling ENABLED")

    def _build_item(self, q, context, is_supporting, a):
        """Tokenize, detect chunks, build labels. Returns dict or None on skip."""
        text = format_hotpotqa_prompt(q, context, answer=a)
        enc = self.tokenizer(
            text, return_tensors="pt", return_offsets_mapping=True,
            truncation=True, max_length=self.max_length,
        )
        input_ids = enc["input_ids"][0]
        offsets = enc["offset_mapping"][0].tolist()
        if offsets[-1][1] < len(text) - 2:
            return None  # truncated
        try:
            chunks = detect_hotpotqa_chunks(text, offsets)
        except AssertionError:
            return None
        chunk_ends = [c["end_tok"] for c in chunks]
        chunk_names = [c["name"] for c in chunks]
        n_p = sum(1 for n in chunk_names if n.startswith("P"))
        if n_p != 10:
            return None
        answer_match = re.search(r"\nAnswer:\s+", text)
        if answer_match is None:
            return None
        answer_start_char = answer_match.end()
        answer_start_tok = None
        for i, (s, _) in enumerate(offsets):
            if s >= answer_start_char:
                answer_start_tok = i
                break
        if answer_start_tok is None:
            return None

        # ===== EOS-fix (matches linear-bf16fix's HotpotQADataset behavior) =====
        # Without this, the model never sees <eos> as a label and at generation
        # rambles past the answer (we get "Chief of Protocol of the United States.
        # As an adult, she was named..." instead of just "Chief of Protocol").
        #
        # 1. Append EOS to input_ids so labels can include it.
        # 2. Count answer span by tokenizing answer alone (handles partial-prefix
        #    BPE artifacts: counting from the END is robust to leading-space
        #    variants like " Chief" vs "Chief").
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            input_ids = torch.cat([
                input_ids,
                torch.tensor([eos_id], dtype=input_ids.dtype),
            ])
        answer_tokens_alone = self.tokenizer.encode(a, add_special_tokens=False)
        answer_plus_eos_len = len(answer_tokens_alone) + (1 if eos_id is not None else 0)
        answer_start_tok = max(0, len(input_ids) - answer_plus_eos_len)

        labels = torch.full_like(input_ids, -100)
        labels[answer_start_tok:] = input_ids[answer_start_tok:]
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
            "chunk_ends": chunk_ends,  # unchanged; A chunk ends BEFORE the appended EOS
            "chunk_names": chunk_names,
            "is_supporting": is_supporting,
            "answer_text": a,
            "question": q,
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if not self.dynamic_shuffle:
            return self.examples[idx]
        # Dynamic re-shuffle path: re-permute the paragraphs of raw_items[idx]
        # and re-tokenize. Each access yields a fresh permutation.
        raw = self.raw_items[idx]
        context = list(raw["context"])
        is_supporting = list(raw["is_supporting"])
        if len(context) > 1:
            perm = list(range(len(context)))
            random.shuffle(perm)
            context = [context[i] for i in perm]
            is_supporting = [is_supporting[i] for i in perm]
        item = self._build_item(raw["q"], context, is_supporting, raw["a"])
        if item is None:
            # Should not happen for items that loaded successfully — fall back
            # to the cached static-shuffled version.
            return self.examples[idx]
        return item



from collate import collate_chunked  # noqa: E402,F401
