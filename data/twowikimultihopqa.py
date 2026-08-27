"""
2WikiMultihopQA dataset loader for training.

HF source: tries `xanhho/2WikiMultihopQA` first, then falls back to local JSON.
Format: Multi-hop QA with multiple context paragraphs (same shape as HotpotQA).
Each example has:
    question, answer, context: List[(title, sentences)],
    supporting_facts: list of {title, sent_id}, type: reasoning type

Selection: defaults to *stratified-by-type* with Random(42) shuffling and
proportional `n_select` per bucket — same scheme as MuSiQue's stratified-by-hop
selection. 2Wiki has 4 types (compositional / comparison / inference /
bridge_comparison). Pass `stratify_by_type=False` to recover the legacy
linear-first-N behavior used by pre-2026-05-09 checkpoints.
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

# Directory holding train.json / dev.json / test.json. Set TWOWIKI_DIR to use an
# existing local copy; otherwise the snapshot is fetched from the Hub on first use.
_TWOWIKI_DIR_ENV = "TWOWIKI_DIR"
_TWOWIKI_REPO = "voidful/2WikiMultihopQA"
_SPLIT_TO_FILE = {"train": "train.json", "validation": "dev.json",
                  "dev": "dev.json", "test": "test.json"}


def format_twowikimultihopqa_prompt(question, context_paragraphs, answer=None):
    """Same prompt shape as HotpotQA — context block, question, answer marker."""
    parts = []
    for title, sentences in context_paragraphs:
        paragraph = ' '.join(sentences) if isinstance(sentences, list) else sentences
        parts.append(f"{title}: {paragraph}")
    context = '\n\n'.join(parts)

    if answer is not None:
        return f"""Context:
{context}

Question: {question}

Answer: {answer}"""
    return f"""Context:
{context}

Question: {question}

Answer:"""


def _twowiki_dir():
    """Resolve the 2WikiMultihopQA snapshot directory, downloading it if needed."""
    local = os.environ.get(_TWOWIKI_DIR_ENV)
    if local:
        return Path(local)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(_TWOWIKI_REPO, repo_type="dataset"))


def _load_2wiki(split):
    """Load one 2WikiMultihopQA split from the voidful snapshot."""
    fname = _SPLIT_TO_FILE.get(split)
    if fname is None:
        raise ValueError(f"Unknown 2WikiMultihopQA split: {split!r}")
    path = _twowiki_dir() / fname
    if not path.exists():
        raise RuntimeError(
            f"2WikiMultihopQA file not found at {path}. Set ${_TWOWIKI_DIR_ENV} to a "
            "directory containing train.json / dev.json / test.json."
        )
    with open(path) as f:
        return json.load(f)


class TwoWikiMultihopQADataset(Dataset):
    """2WikiMultihopQA SFT dataset — mirrors HotpotQADataset interface."""

    def __init__(self, tokenizer=None, max_length=2048, split='train',
                 max_samples=None, stratify_by_type=True, seed=42,
                 lora_pool_path=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Canonical pool path: load from JSON instead of HF + filter
        if lora_pool_path is not None and split == 'train':
            print(f"Loading 2WikiMultihopQA from canonical pool {lora_pool_path}")
            with open(lora_pool_path) as _f:
                _pool = json.load(_f)
            self.examples = []
            for it in _pool['examples']:
                ctx = it['context']
                if isinstance(ctx, dict) and 'title' in ctx:
                    context = list(zip(ctx['title'], ctx['sentences']))
                else:
                    context = [(c[0], c[1]) for c in ctx]
                full_text = format_twowikimultihopqa_prompt(
                    it['question'], context, it['answer'],
                )
                self.examples.append({
                    'question': it['question'],
                    'answer': it['answer'],
                    'context': context,
                    'full_text': full_text,
                    'question_type': it.get('type', 'unknown'),
                })
            print(f"  Loaded {len(self.examples)} canonical examples "
                  f"(filter_rule={_pool.get('filter_rule', 'unknown')})")
            return

        print(f"Loading 2WikiMultihopQA {split} split (stratify_by_type={stratify_by_type})...")
        dataset = _load_2wiki(split)

        skipped_no_answer = 0
        skipped_too_long = 0

        if stratify_by_type and max_samples is not None and split == 'train':
            # Stratified-by-type pass: bucket → Random(seed) shuffle → proportional n_select → final shuffle
            type_buckets = defaultdict(list)
            for item in dataset:
                question = item.get('question', '').strip()
                answer = item.get('answer', '').strip()
                if not answer:
                    skipped_no_answer += 1
                    continue
                ctx = item['context']
                if isinstance(ctx, dict) and 'title' in ctx:
                    context = list(zip(ctx['title'], ctx['sentences']))
                else:
                    context = [(c[0], c[1]) for c in ctx]
                full_text = format_twowikimultihopqa_prompt(question, context, answer)
                token_len = len(self.tokenizer.encode(full_text, add_special_tokens=True))
                if token_len > max_length:
                    skipped_too_long += 1
                    continue
                t = item.get('type', 'unknown')
                type_buckets[t].append({
                    'question': question, 'answer': answer, 'context': context,
                    'full_text': full_text, 'question_type': t,
                })
            total_available = sum(len(v) for v in type_buckets.values())
            rng = random.Random(seed)
            self.examples = []
            for type_name in sorted(type_buckets.keys()):
                bucket = type_buckets[type_name]
                rng.shuffle(bucket)
                n_select = int(max_samples * len(bucket) / total_available)
                n_select = max(1, min(n_select, len(bucket)))
                self.examples.extend(bucket[:n_select])
            rng.shuffle(self.examples)
            bucket_sizes = {k: len(v) for k, v in type_buckets.items()}
            print(f"  Stratified pool: {total_available}; type buckets: {bucket_sizes}")
            print(f"  Selected {len(self.examples)} examples (Random({seed}), proportional)")
        else:
            # Legacy linear-first-N pass
            self.examples = []
            for item in dataset:
                if max_samples is not None and len(self.examples) >= max_samples:
                    break
                question = item.get('question', '').strip()
                answer = item.get('answer', '').strip()
                if not answer:
                    skipped_no_answer += 1
                    continue
                ctx = item['context']
                if isinstance(ctx, dict) and 'title' in ctx:
                    context = list(zip(ctx['title'], ctx['sentences']))
                else:
                    context = [(c[0], c[1]) for c in ctx]
                full_text = format_twowikimultihopqa_prompt(question, context, answer)
                token_len = len(self.tokenizer.encode(full_text, add_special_tokens=True))
                if token_len > max_length:
                    skipped_too_long += 1
                    continue
                self.examples.append({
                    'question': question, 'answer': answer, 'context': context,
                    'full_text': full_text, 'question_type': item.get('type', 'unknown'),
                })
            print(f"  Loaded {len(self.examples)} examples (linear-first-N, legacy)")

        if skipped_no_answer > 0:
            print(f"  Skipped {skipped_no_answer} no-answer examples")
        if skipped_too_long > 0:
            print(f"  Skipped {skipped_too_long} examples > {max_length} tokens")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        question_text = f"""Question: {example['question']}

Answer: {example['answer']}"""
        question_encoded = self.tokenizer(
            question_text, add_special_tokens=False, padding=False,
        )
        question_tokens = len(question_encoded['input_ids'])
        reserved_tokens = question_tokens + 10
        available_for_context = self.max_length - reserved_tokens

        if available_for_context < 100:
            full_encoded = self.tokenizer(
                example['full_text'], add_special_tokens=True,
                truncation=True, max_length=self.max_length,
                padding=False, return_tensors='pt',
            )
            input_ids = full_encoded['input_ids'].squeeze(0)
            attention_mask = full_encoded['attention_mask'].squeeze(0)
            labels = input_ids.clone()
            labels[:-20] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            context_parts = []
            for title, sentences in example['context']:
                paragraph = ' '.join(sentences) if isinstance(sentences, list) else sentences
                context_parts.append(f"{title}: {paragraph}")
            context_str = '\n\n'.join(context_parts)

            context_encoded = self.tokenizer(
                f"Context:\n{context_str}",
                add_special_tokens=False, truncation=True,
                max_length=available_for_context, padding=False,
            )

            eos = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
            combined_ids = context_encoded['input_ids'] + question_encoded['input_ids'] + eos
            if self.tokenizer.bos_token_id is not None:
                combined_ids = [self.tokenizer.bos_token_id] + combined_ids

            input_ids = torch.tensor(combined_ids, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)

            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]

            labels = input_ids.clone()
            answer_tokens = self.tokenizer.encode(example['answer'], add_special_tokens=False)
            answer_plus_eos = len(answer_tokens) + len(eos)
            answer_start = len(input_ids) - answer_plus_eos
            labels[:answer_start] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
